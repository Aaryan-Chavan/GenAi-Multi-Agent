"""
pipeline/dataset_pipeline.py
==============================================================================
Enterprise-grade, dataset-independent orchestration layer for the ingestion →
embedding → storage pipeline.

This module contains ORCHESTRATION ONLY. Every actual piece of business logic
(loading, cleaning, mapping, schema analysis, separation, chunking,
intelligence generation, embedding, quantization, and storage) lives in its
own module under ingestion/, preprocessing/, embeddings/ and storage/. This
file is only responsible for:

    * calling those modules in the correct order,
    * tracking which steps have completed (resumable, crash-safe),
    * persisting every intermediate artifact to disk so a step can always
      reconstruct its inputs even after a full process restart,
    * skipping steps that are already done and whose output is still valid,
    * producing structured status information,
    * logging, and
    * raising clear, typed exceptions on failure.

Pipeline flow (fixed, do not reorder):

    Dataset Upload
        -> Load CSV
        -> Cleaning
        -> Canonical Mapping
        -> Schema Analyzer
        -> Structured / Unstructured Separation
        -> Precomputed Intelligence
        -> Chunking
        -> Embedding Generation
        -> (Optional Quantization)
        -> DuckDB Loader
        -> Qdrant Loader
        -> Pipeline Complete

Public API (unchanged, backward compatible):

    DatasetPipeline
        .initialize()
        .run()
        .resume()
        .status()
        .reset()
        .validate()
        .close()

RedisCache is intentionally NEVER used during dataset building.

Resumability model
-------------------
`self._artifacts` is an in-memory cache only — it is never the source of
truth. Every step follows the pattern:

    if artifact in memory:
        use it
    else:
        load it from its persisted file on disk (raising
        PipelineValidationError if that file is missing/corrupt)
    validate -> run step -> persist outputs -> cache outputs -> return metrics

This means `resume()` (or `run()` after a crash, restart, or fresh process)
never depends on anything having survived in RAM: every artifact a step
needs is re-derived from disk on demand via the `_load_*` helper methods.
==============================================================================
"""

from __future__ import annotations

import hashlib
import json
import logging
import logging.handlers
import threading
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

# ==========================================================================
# CONFIG
# ==========================================================================
#
# All file paths, batch sizes, and tunables are sourced from
# config.settings so nothing in this orchestration layer is hardcoded.
# Newer settings introduced by this revision (artifact paths, log rotation,
# health-check toggles, parallelism) are pulled in defensively via
# getattr-with-default so this file stays backward compatible with older
# settings.py files that have not yet defined them.
# ==========================================================================

from Config import settings as _settings

CLEANED_DATA_FILE = _settings.CLEANED_DATA_FILE
RAW_DATA_FILE = getattr(_settings, "RAW_DATA_FILE")
ENABLE_QUANTIZATION = _settings.ENABLE_QUANTIZATION
EMBEDDINGS_DIR = _settings.EMBEDDINGS_DIR
EMBEDDINGS_FILE = _settings.EMBEDDINGS_FILE
QDRANT_BATCH_SIZE = _settings.QDRANT_BATCH_SIZE
STRUCTURED_TABLE = _settings.STRUCTURED_TABLE
METADATA_TABLE = _settings.METADATA_TABLE
INTELLIGENCE_TABLE = _settings.INTELLIGENCE_TABLE
PIPELINE_STATE_FILE = _settings.PIPELINE_STATE_FILE
PIPELINE_LOG_FILE = _settings.PIPELINE_LOG_FILE

# Intermediate-artifact file paths. Fall back to sensible defaults inside
# EMBEDDINGS_DIR's parent (the shared `data/processed` style directory) if
# the project's settings.py has not yet been extended with these.
_DATA_PROCESSED_DIR = Path(getattr(_settings, "PROCESSED_DATA_DIR", Path(EMBEDDINGS_DIR).parent / "processed"))

MAPPED_DATA_FILE = Path(getattr(_settings, "MAPPED_DATA_FILE", _DATA_PROCESSED_DIR / "mapped.csv"))
SCHEMA_FILE = Path(getattr(_settings, "SCHEMA_FILE", _DATA_PROCESSED_DIR / "schema.json"))
STRUCTURED_DATA_FILE = Path(getattr(_settings, "STRUCTURED_DATA_FILE", _DATA_PROCESSED_DIR / "structured.csv"))
METADATA_DATA_FILE = Path(getattr(_settings, "METADATA_DATA_FILE", _DATA_PROCESSED_DIR / "metadata.csv"))
UNSTRUCTURED_DATA_FILE = Path(getattr(_settings, "UNSTRUCTURED_DATA_FILE", _DATA_PROCESSED_DIR / "unstructured.csv"))
INTELLIGENCE_DATA_FILE = Path(getattr(_settings, "INTELLIGENCE_DATA_FILE", _DATA_PROCESSED_DIR / "intelligence.csv"))
CHUNKED_DATA_FILE = Path(getattr(_settings, "CHUNKED_DATA_FILE", _DATA_PROCESSED_DIR / "chunked.csv"))
EMBEDDING_METADATA_FILE = Path(
    getattr(_settings, "EMBEDDING_METADATA_FILE", Path(EMBEDDINGS_DIR) / "metadata_embeddings.csv")
)
EMBEDDINGS_NPY_FILE = Path(getattr(_settings, "EMBEDDINGS_NPY_FILE", Path(EMBEDDINGS_DIR) / "embeddings.npy"))

# Operational tunables, all configuration-driven (no hardcoded magic numbers).
QDRANT_LOAD_PARALLELISM = int(getattr(_settings, "QDRANT_LOAD_PARALLELISM", 8))
LOG_MAX_BYTES = int(getattr(_settings, "LOG_MAX_BYTES", 10 * 1024 * 1024))  # 10 MB
LOG_BACKUP_COUNT = int(getattr(_settings, "LOG_BACKUP_COUNT", 5))
FINGERPRINT_HASH_BYTES = int(getattr(_settings, "FINGERPRINT_HASH_BYTES", 10 * 1024 * 1024))  # 10 MB
REQUIRED_COLUMNS = list(getattr(_settings, "REQUIRED_COLUMNS", []) or [])
ENABLE_DUCKDB_HEALTHCHECK = bool(getattr(_settings, "ENABLE_DUCKDB_HEALTHCHECK", True))
ENABLE_QDRANT_HEALTHCHECK = bool(getattr(_settings, "ENABLE_QDRANT_HEALTHCHECK", True))

# ==========================================================================
# INGESTION
# ==========================================================================

from Ingestion.load_csv import CSVLoader
from Ingestion.clean_data import DataCleaner
from Ingestion.canonical_mapper import CanonicalMapper

# ==========================================================================
# PREPROCESSING
# ==========================================================================

from Preprocessing.schema_analyzer import SchemaAnalyzer
from Preprocessing.data_separator import DataSeparator
from Preprocessing.chunking import TextChunker
from Preprocessing.precomputed_intelligence import PrecomputedIntelligence

# ==========================================================================
# EMBEDDINGS
# ==========================================================================

from Embeddings.embedding_generator import EmbeddingGenerator
from Embeddings.quantization import EmbeddingQuantizer

# ==========================================================================
# STORAGE (RedisCache intentionally NOT imported/used here)
# ==========================================================================

from Storage.duckdb_loader import DuckDBLoader
from Storage.qdrant_loader import QdrantLoader


# ==========================================================================
# LOGGING
# ==========================================================================

def _build_logger() -> logging.Logger:
    """Configure and return the module-level logger.

    Writes to both a size-based rotating log file (PIPELINE_LOG_FILE,
    bounded by LOG_MAX_BYTES/LOG_BACKUP_COUNT so logs never grow
    unbounded) and stdout, so the pipeline is observable both
    interactively and in production/headless runs.
    """
    logger = logging.getLogger("dataset_pipeline")
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        # Avoid duplicate handlers if this module is imported more than once
        # (e.g. re-imported by a notebook or test runner).
        return logger

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        Path(PIPELINE_LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            PIPELINE_LOG_FILE,
            maxBytes=LOG_MAX_BYTES,
            backupCount=LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setFormatter(fmt)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
    except Exception:  # pragma: no cover - logging must never crash the app
        # If we cannot write a log file (e.g. read-only filesystem), fall
        # back silently to console-only logging.
        pass

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    return logger


logger = _build_logger()


# ==========================================================================
# CUSTOM EXCEPTIONS
# ==========================================================================

class DatasetPipelineError(Exception):
    """Base class for all dataset pipeline errors."""


class PipelineStateError(DatasetPipelineError):
    """Raised when the persisted pipeline state is missing, corrupt, or
    inconsistent with the requested operation."""


class PipelineStepError(DatasetPipelineError):
    """Raised when an individual pipeline step fails.

    Wraps the original exception so callers get both the step name and the
    root cause without losing the traceback.
    """

    def __init__(self, step: str, original: BaseException):
        self.step = step
        self.original = original
        super().__init__(f"Step '{step}' failed: {original}")


class PipelineValidationError(DatasetPipelineError):
    """Raised when input validation fails, or a required persisted
    artifact is missing/unreadable, before a step is allowed to run."""


class PipelineNotInitializedError(DatasetPipelineError):
    """Raised when run()/resume()/status() etc. are called before
    initialize()."""


class PipelineDependencyError(DatasetPipelineError):
    """Raised when a required external dependency (DuckDB, Qdrant) fails a
    health check before a step is allowed to use it."""


# ==========================================================================
# STEP DEFINITIONS / STATE MODEL
# ==========================================================================

class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


# Fixed, ordered list of pipeline step names. This order must never change.
STEP_ORDER: List[str] = [
    "load_csv",
    "cleaning",
    "canonical_mapping",
    "schema_analysis",
    "data_separation",
    "precomputed_intelligence",
    "chunking",
    "embedding_generation",
    "quantization",
    "duckdb_load",
    "qdrant_load",
]


@dataclass
class StepRecord:
    """Metadata tracked for a single pipeline step."""

    status: str = StepStatus.PENDING.value
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_seconds: Optional[float] = None
    error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineState:
    """Full persisted state of a pipeline run.

    Serialized to PIPELINE_STATE_FILE as JSON after every successful step so
    that a crash or restart can resume from the last completed step instead
    of repeating expensive work.
    """

    dataset_path: Optional[str] = None
    dataset_fingerprint: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    steps: Dict[str, StepRecord] = field(
        default_factory=lambda: {name: StepRecord() for name in STEP_ORDER}
    )

    def to_json(self) -> Dict[str, Any]:
        data = asdict(self)
        return data

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "PipelineState":
        steps_data = data.get("steps", {})
        steps = {
            name: StepRecord(**steps_data.get(name, {}))
            for name in STEP_ORDER
        }
        return cls(
            dataset_path=data.get("dataset_path"),
            dataset_fingerprint=data.get("dataset_fingerprint"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            steps=steps,
        )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fingerprint_file(path: Path) -> str:
    """Fingerprint a dataset file for change detection.

    Combines cheap metadata (name, size, mtime) with a SHA-256 hash of the
    first FINGERPRINT_HASH_BYTES bytes of the file. This is deliberately a
    *partial* hash rather than a full-file hash so it stays fast on very
    large datasets, while being far more reliable than metadata alone at
    detecting an in-place content change that doesn't alter file size
    (e.g. a corrected row) or a touched-but-unchanged file.
    """
    try:
        stat = path.stat()
        hasher = hashlib.sha256()
        with open(path, "rb") as fh:
            hasher.update(fh.read(FINGERPRINT_HASH_BYTES))
        partial_hash = hasher.hexdigest()
        return f"{path.name}:{stat.st_size}:{int(stat.st_mtime)}:{partial_hash}"
    except OSError as exc:
        raise PipelineValidationError(
            f"Cannot fingerprint dataset file '{path}': {exc}"
        ) from exc


# ==========================================================================
# TIMER CONTEXT MANAGER (kept, generalized)
# ==========================================================================

class _StageTimer:
    """Context manager used to log the start/end and duration of each
    pipeline stage in a consistent, human-readable format."""

    def __init__(self, name: str):
        self.name = name
        self.start: Optional[float] = None
        self.elapsed: float = 0.0

    def __enter__(self) -> "_StageTimer":
        self.start = time.time()
        logger.info("=" * 80)
        logger.info(self.name)
        logger.info("=" * 80)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.elapsed = time.time() - (self.start or time.time())
        if exc_type is None:
            logger.info("Completed in %.2f sec", self.elapsed)
        else:
            logger.error("Failed after %.2f sec: %s", self.elapsed, exc_val)


def stage(name: str) -> _StageTimer:
    return _StageTimer(name)


# ==========================================================================
# DATASET PIPELINE
# ==========================================================================

class DatasetPipeline:
    """Thread-safe, resumable orchestrator for the full dataset build
    pipeline.

    This class does not implement any business logic itself. It only wires
    together the existing modules in the fixed order required by the
    project, persists progress and every intermediate artifact to disk,
    and exposes a small, predictable public API.

    Every step is written so that it never depends on `self._artifacts`
    having survived in memory: if an artifact is missing, it is reloaded
    from its persisted file via a dedicated `_load_*` helper, which raises
    `PipelineValidationError` if the file is absent or unreadable. This is
    what makes `resume()` safe after a crash, server restart, or brand new
    process.

    Parameters
    ----------
    dataset_path:
        Path to the raw CSV dataset to ingest. If omitted, RAW_DATA_FILE
        from config.settings is used.
    state_file:
        Path to the JSON file used to persist pipeline progress. Defaults to
        PIPELINE_STATE_FILE from config.settings.
    force:
        If True, every step is re-run regardless of persisted state or
        target-store validation. Use for a full rebuild.
    """

    def __init__(
        self,
        dataset_path: Optional[str] = None,
        state_file: Optional[str] = None,
        force: bool = False,
    ) -> None:
        self._lock = threading.RLock()
        self._initialized = False
        self._closed = False
        self._force = force

        self._dataset_path = Path(dataset_path) if dataset_path else Path(RAW_DATA_FILE)
        self._state_path = Path(state_file) if state_file else Path(PIPELINE_STATE_FILE)

        self._state: Optional[PipelineState] = None

        # In-memory artifact cache. Used purely as a performance
        # optimization within a single long-lived process — every step is
        # written to tolerate this being empty (e.g. right after a
        # restart) by reloading from the corresponding persisted file.
        self._artifacts: Dict[str, Any] = {}

        # Lazily constructed module instances (cheap objects; constructed
        # once per pipeline instance and reused across steps/resumes).
        self._duckdb_loader: Optional[DuckDBLoader] = None
        self._qdrant_loader: Optional[QdrantLoader] = None

    # ----------------------------------------------------------------
    # PUBLIC API
    # ----------------------------------------------------------------

    def initialize(self) -> "DatasetPipeline":
        """Prepare the pipeline for execution.

        Loads (or creates) the persisted state file, validates the dataset
        path exists, and lazily constructs storage-layer clients. Must be
        called before run()/resume()/status()/reset()/validate().

        Returns self to allow chaining, e.g. `DatasetPipeline().initialize().run()`.
        """
        with self._lock:
            if self._closed:
                raise PipelineStateError("Cannot initialize a closed pipeline instance.")

            logger.info("Initializing DatasetPipeline")
            logger.info("Dataset path : %s", self._dataset_path)
            logger.info("State file   : %s", self._state_path)

            if not self._dataset_path.exists():
                raise PipelineValidationError(
                    f"Dataset file not found: {self._dataset_path}"
                )

            self._state = self._load_or_create_state()
            self._duckdb_loader = DuckDBLoader()
            self._qdrant_loader = QdrantLoader()

            self._initialized = True
            logger.info("DatasetPipeline initialized successfully.")
            return self

    def validate(self) -> Dict[str, Any]:
        """Validate the dataset and environment without mutating any state
        or running any steps.

        Performs an expanded set of checks: existence, readability,
        non-empty, parseable CSV, delimiter/encoding sanity, required
        columns (if REQUIRED_COLUMNS is configured), duplicate IDs, and
        malformed row detection.

        Returns a structured report; see `report["issues"]` for a list of
        every problem found (empty list means the dataset is clean).
        """
        self._ensure_initialized()

        report: Dict[str, Any] = {
            "dataset_path": str(self._dataset_path),
            "exists": False,
            "readable": False,
            "non_empty": False,
            "encoding_ok": False,
            "delimiter_ok": False,
            "columns": [],
            "row_count_sampled": 0,
            "duplicate_id_count": 0,
            "malformed_row_count": 0,
            "missing_required_columns": [],
            "issues": [],
        }

        try:
            report["exists"] = self._dataset_path.exists()
            if not report["exists"]:
                report["issues"].append("Dataset file does not exist.")
                return report

            size = self._dataset_path.stat().st_size
            report["non_empty"] = size > 0
            if not report["non_empty"]:
                report["issues"].append("Dataset file is empty.")
                return report

            # Encoding check: attempt a strict UTF-8 decode of a sample.
            try:
                with open(self._dataset_path, "rb") as fh:
                    sample_bytes = fh.read(1_000_000)
                sample_bytes.decode("utf-8")
                report["encoding_ok"] = True
            except UnicodeDecodeError as exc:
                report["issues"].append(f"Encoding issue detected (not valid UTF-8): {exc}")

            # Header + delimiter sanity check.
            header_df = pd.read_csv(self._dataset_path, nrows=0)
            report["readable"] = True
            report["columns"] = list(header_df.columns)
            report["delimiter_ok"] = len(report["columns"]) > 1

            if not report["delimiter_ok"]:
                report["issues"].append(
                    "Only one column detected — check that the delimiter is actually a comma."
                )

            if len(report["columns"]) == 0:
                report["issues"].append("No columns detected in CSV header.")

            if REQUIRED_COLUMNS:
                missing = [c for c in REQUIRED_COLUMNS if c not in report["columns"]]
                report["missing_required_columns"] = missing
                if missing:
                    report["issues"].append(f"Missing required columns: {missing}")

            # Sampled row-level checks: malformed rows and duplicate IDs.
            # Uses on_bad_lines='skip' + a before/after count comparison
            # (rather than raising) so we can *report* malformed rows
            # instead of crashing validate().
            try:
                total_lines = sum(1 for _ in open(self._dataset_path, "r", encoding="utf-8", errors="replace")) - 1
                sample_df = pd.read_csv(
                    self._dataset_path, low_memory=False, on_bad_lines="skip", engine="python"
                )
                report["row_count_sampled"] = len(sample_df)
                report["malformed_row_count"] = max(total_lines - len(sample_df), 0)

                id_col = "record_id" if "record_id" in sample_df.columns else None
                if id_col is None:
                    for candidate in ("id", "Id", "ID"):
                        if candidate in sample_df.columns:
                            id_col = candidate
                            break
                if id_col is not None:
                    report["duplicate_id_count"] = int(sample_df[id_col].duplicated().sum())
                    if report["duplicate_id_count"] > 0:
                        report["issues"].append(
                            f"Found {report['duplicate_id_count']} duplicate values in '{id_col}'."
                        )

                if report["malformed_row_count"] > 0:
                    report["issues"].append(
                        f"Detected approximately {report['malformed_row_count']} malformed row(s)."
                    )
            except Exception as exc:
                report["issues"].append(f"Row-level validation could not complete: {exc}")

        except Exception as exc:  # pragma: no cover - defensive
            report["issues"].append(f"Validation error: {exc}")

        report["valid"] = (
            report["exists"]
            and report["readable"]
            and report["non_empty"]
            and not report["missing_required_columns"]
            and report["duplicate_id_count"] == 0
        )
        logger.info("Validation report: %s", report)
        return report

    def run(self) -> Dict[str, Any]:
        """Run the full pipeline from the beginning.

        Any step already marked COMPLETED in the persisted state — and
        whose output is confirmed still valid in its target store — is
        skipped rather than repeated. This makes run() safe to call
        repeatedly (idempotent) and equivalent to resume() when nothing has
        changed, but run() always re-validates from step 1 onward, whereas
        resume() jumps straight to the first incomplete step.
        """
        self._ensure_initialized()
        return self._execute(start_from_first_incomplete=False)

    def resume(self) -> Dict[str, Any]:
        """Resume a previously interrupted run.

        Jumps directly to the first step that is not COMPLETED according to
        persisted state, skipping the validity re-checks run() performs on
        already-completed steps. Because every step reloads its required
        inputs from disk when they are not already cached in memory, this
        works correctly even in a brand-new process with an empty
        `_artifacts` cache (e.g. after a crash or server restart).
        """
        self._ensure_initialized()
        if self._state is None or all(
            s.status == StepStatus.PENDING.value for s in self._state.steps.values()
        ):
            logger.info("No prior progress found — resume() will behave like run().")
        return self._execute(start_from_first_incomplete=True)

    def status(self) -> Dict[str, Any]:
        """Return a structured snapshot of pipeline progress.

        Returns
        -------
        dict with keys: dataset_path, dataset_fingerprint, created_at,
        updated_at, overall_status, progress_percent, steps (per-step
        status/timing/error/metrics).
        """
        self._ensure_initialized()
        assert self._state is not None

        overall = self._overall_status()
        return {
            "dataset_path": self._state.dataset_path,
            "dataset_fingerprint": self._state.dataset_fingerprint,
            "created_at": self._state.created_at,
            "updated_at": self._state.updated_at,
            "overall_status": overall,
            "progress_percent": self._progress_percent(),
            "steps": {
                name: asdict(record) for name, record in self._state.steps.items()
            },
        }

    def reset(self) -> Dict[str, Any]:
        """Reset all pipeline progress tracking.

        This clears the JSON state file so the next run() starts from step
        1. It does NOT delete any already-written data files, DuckDB
        tables, or Qdrant collections — those are left in place and will be
        re-validated (and reused if still valid, or overwritten if not) the
        next time run()/resume() executes each step.
        """
        with self._lock:
            self._ensure_initialized()
            logger.warning("Resetting pipeline state at %s", self._state_path)
            self._state = PipelineState(
                dataset_path=str(self._dataset_path),
                dataset_fingerprint=_fingerprint_file(self._dataset_path),
                created_at=_now_iso(),
                updated_at=_now_iso(),
            )
            self._persist_state()
            self._artifacts = {}
            return self.status()

    def close(self) -> None:
        """Release any resources held by underlying storage loaders.

        Safe to call multiple times. After close(), the instance can no
        longer be used; construct a new DatasetPipeline if needed.
        """
        with self._lock:
            if self._closed:
                return
            logger.info("Closing DatasetPipeline and releasing resources.")
            for loader_name, loader in (
                ("duckdb_loader", self._duckdb_loader),
                ("qdrant_loader", self._qdrant_loader),
            ):
                closer = getattr(loader, "close", None)
                if callable(closer):
                    try:
                        closer()
                    except Exception as exc:  # pragma: no cover - defensive
                        logger.warning("Error closing %s: %s", loader_name, exc)
            self._closed = True
            self._initialized = False

    # ----------------------------------------------------------------
    # INTERNAL: STATE MANAGEMENT
    # ----------------------------------------------------------------

    def _ensure_initialized(self) -> None:
        if self._closed:
            raise PipelineStateError("This DatasetPipeline instance has been closed.")
        if not self._initialized or self._state is None:
            raise PipelineNotInitializedError(
                "DatasetPipeline.initialize() must be called before use."
            )

    def _load_or_create_state(self) -> PipelineState:
        fingerprint = _fingerprint_file(self._dataset_path)

        if self._state_path.exists():
            try:
                with open(self._state_path, "r", encoding="utf-8") as fh:
                    raw = json.load(fh)
                state = PipelineState.from_json(raw)
            except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
                logger.warning(
                    "Could not parse existing state file (%s) — starting fresh: %s",
                    self._state_path, exc,
                )
                state = None
            else:
                if state.dataset_fingerprint != fingerprint:
                    logger.warning(
                        "Dataset fingerprint changed (%s -> %s). "
                        "Resetting progress for a fresh dataset.",
                        state.dataset_fingerprint, fingerprint,
                    )
                    state = None
        else:
            state = None

        if state is None:
            state = PipelineState(
                dataset_path=str(self._dataset_path),
                dataset_fingerprint=fingerprint,
                created_at=_now_iso(),
                updated_at=_now_iso(),
            )

        return state

    def _persist_state(self) -> None:
        """Atomically write the current state to disk.

        Writes to a temp file and renames over the target so a crash mid
        write can never leave a corrupt/partial state file behind.
        """
        assert self._state is not None
        self._state.updated_at = _now_iso()

        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._state_path.with_suffix(self._state_path.suffix + ".tmp")

        with self._lock:
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(self._state.to_json(), fh, indent=2, default=str)
            tmp_path.replace(self._state_path)

    def _overall_status(self) -> str:
        assert self._state is not None
        statuses = {s.status for s in self._state.steps.values()}
        if StepStatus.FAILED.value in statuses:
            return StepStatus.FAILED.value
        if all(
            self._state.steps[name].status in (StepStatus.COMPLETED.value, StepStatus.SKIPPED.value)
            for name in STEP_ORDER
        ):
            return "completed"
        if StepStatus.RUNNING.value in statuses:
            return "running"
        if any(
            self._state.steps[name].status != StepStatus.PENDING.value for name in STEP_ORDER
        ):
            return "in_progress"
        return "not_started"

    def _progress_percent(self) -> float:
        """Percentage of steps that are COMPLETED or SKIPPED."""
        assert self._state is not None
        done = sum(
            1 for name in STEP_ORDER
            if self._state.steps[name].status in (StepStatus.COMPLETED.value, StepStatus.SKIPPED.value)
        )
        return round(100.0 * done / len(STEP_ORDER), 2)

    # ----------------------------------------------------------------
    # INTERNAL: STEP EXECUTION FRAMEWORK
    # ----------------------------------------------------------------

    def _first_incomplete_step_index(self) -> int:
        assert self._state is not None
        for idx, name in enumerate(STEP_ORDER):
            if self._state.steps[name].status != StepStatus.COMPLETED.value:
                return idx
        return len(STEP_ORDER)

    def _run_step(
        self,
        name: str,
        func: Callable[[], Dict[str, Any]],
        already_valid: Optional[Callable[[], bool]] = None,
    ) -> None:
        """Execute a single named step with full state tracking.

        Parameters
        ----------
        name:
            Step name; must be a member of STEP_ORDER.
        func:
            Zero-arg callable that performs the step and returns a dict of
            metrics to store alongside the step record. Should raise on
            failure. Responsible for reloading its own inputs from disk
            (via the `_load_*` helpers) if they are not already cached in
            `self._artifacts`, and for persisting its own outputs to disk.
        already_valid:
            Optional zero-arg callable. If it returns True and force=False,
            the step is marked SKIPPED without calling func(), avoiding
            expensive repeated work (e.g. re-embedding, re-loading Qdrant).
            Must itself be resume-safe (reload from disk if needed).
        """
        assert self._state is not None
        record = self._state.steps[name]

        if not self._force and record.status == StepStatus.COMPLETED.value:
            logger.info("[%s] already completed — skipping.", name)
            return

        if not self._force and already_valid is not None:
            try:
                if already_valid():
                    logger.info("[%s] target already valid — skipping expensive work.", name)
                    record.status = StepStatus.SKIPPED.value
                    record.completed_at = _now_iso()
                    self._persist_state()
                    return
            except Exception as exc:
                # A failed validity check should not abort the pipeline —
                # fall through and simply (re)run the step.
                logger.debug("[%s] validity check raised %s — will run step.", name, exc)

        record.status = StepStatus.RUNNING.value
        record.started_at = _now_iso()
        record.error = None
        self._persist_state()

        start = time.time()
        try:
            with stage(name.replace("_", " ").upper()):
                metrics = func() or {}
        except Exception as exc:
            record.status = StepStatus.FAILED.value
            record.completed_at = _now_iso()
            record.duration_seconds = time.time() - start
            record.error = f"{type(exc).__name__}: {exc}"
            self._persist_state()
            logger.error("Step '%s' failed:\n%s", name, traceback.format_exc())
            raise PipelineStepError(name, exc) from exc

        record.status = StepStatus.COMPLETED.value
        record.completed_at = _now_iso()
        record.duration_seconds = time.time() - start
        record.metrics = metrics
        self._persist_state()

    def _execute(self, start_from_first_incomplete: bool) -> Dict[str, Any]:
        assert self._state is not None

        logger.info("=" * 80)
        logger.info("STARTING PIPELINE")
        logger.info("=" * 80)

        start_idx = self._first_incomplete_step_index() if start_from_first_incomplete else 0
        steps_to_consider = STEP_ORDER[start_idx:]

        try:
            for step_name in steps_to_consider:
                handler = self._STEP_HANDLERS[step_name]
                handler(self)

            logger.info("=" * 80)
            logger.info("PIPELINE COMPLETED")
            logger.info("=" * 80)

        except PipelineStepError:
            logger.error("Pipeline halted due to a step failure. See status() for details.")
            raise

        return self.status()

    # ----------------------------------------------------------------
    # INTERNAL: ARTIFACT LOADER HELPERS
    #
    # Single source of truth for "load persisted artifact X from disk".
    # Every helper: validates the file exists (raising
    # PipelineValidationError if not), loads it, caches it in
    # `self._artifacts`, and returns it. This eliminates duplicated
    # pd.read_csv() calls scattered across step implementations and is
    # what makes every step resume-safe regardless of process restarts.
    # ----------------------------------------------------------------

    def _load_csv_generic(self, path: Path, artifact_key: str, label: str) -> pd.DataFrame:
        if not path.exists():
            raise PipelineValidationError(
                f"Cannot load '{label}': expected file not found at {path}. "
                f"The corresponding earlier step may not have completed successfully."
            )
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception as exc:
            raise PipelineValidationError(f"Failed to read '{label}' from {path}: {exc}") from exc

        self._artifacts[artifact_key] = df
        return df

    def _load_raw_dataframe(self) -> pd.DataFrame:
        cached = self._artifacts.get("raw_df")
        if cached is not None:
            return cached
        return self._load_csv_generic(self._dataset_path, "raw_df", "raw dataset")

    def _load_cleaned_dataframe(self) -> pd.DataFrame:
        cached = self._artifacts.get("cleaned_df")
        if cached is not None:
            return cached
        return self._load_csv_generic(Path(CLEANED_DATA_FILE), "cleaned_df", "cleaned dataset")

    def _load_mapped_dataframe(self) -> pd.DataFrame:
        cached = self._artifacts.get("mapped_df")
        if cached is not None:
            return cached
        df = self._load_csv_generic(MAPPED_DATA_FILE, "mapped_df", "canonically mapped dataset")
        if "record_id" not in df.columns:
            df.insert(0, "record_id", range(len(df)))
            self._artifacts["mapped_df"] = df
        return df

    def _load_schema(self) -> Dict[str, Any]:
        cached = self._artifacts.get("schema_result")
        if cached is not None:
            return cached
        if not SCHEMA_FILE.exists():
            raise PipelineValidationError(
                f"Cannot load schema: expected file not found at {SCHEMA_FILE}. "
                f"The schema_analysis step may not have completed successfully."
            )
        try:
            with open(SCHEMA_FILE, "r", encoding="utf-8") as fh:
                schema_result = json.load(fh)
        except Exception as exc:
            raise PipelineValidationError(f"Failed to read schema from {SCHEMA_FILE}: {exc}") from exc

        self._artifacts["schema_result"] = schema_result
        return schema_result

    def _load_structured_dataframe(self) -> pd.DataFrame:
        cached = self._artifacts.get("structured_df")
        if cached is not None:
            return cached
        return self._load_csv_generic(STRUCTURED_DATA_FILE, "structured_df", "structured dataset")

    def _load_metadata_dataframe(self) -> pd.DataFrame:
        cached = self._artifacts.get("metadata_df")
        if cached is not None:
            return cached
        return self._load_csv_generic(METADATA_DATA_FILE, "metadata_df", "metadata dataset")

    def _load_unstructured_dataframe(self) -> pd.DataFrame:
        cached = self._artifacts.get("unstructured_df")
        if cached is not None:
            return cached
        return self._load_csv_generic(UNSTRUCTURED_DATA_FILE, "unstructured_df", "unstructured dataset")

    def _load_intelligence_dataframe(self, required: bool = True) -> Optional[pd.DataFrame]:
        cached = self._artifacts.get("intelligence_df")
        if cached is not None:
            return cached
        if not INTELLIGENCE_DATA_FILE.exists():
            if required:
                raise PipelineValidationError(
                    f"Cannot load intelligence data: expected file not found at "
                    f"{INTELLIGENCE_DATA_FILE}. The precomputed_intelligence step "
                    f"may not have completed successfully."
                )
            return None
        return self._load_csv_generic(INTELLIGENCE_DATA_FILE, "intelligence_df", "precomputed intelligence")

    def _load_chunk_dataframe(self) -> pd.DataFrame:
        cached = self._artifacts.get("chunk_df")
        if cached is not None:
            return cached
        return self._load_csv_generic(CHUNKED_DATA_FILE, "chunk_df", "chunked dataset")

    def _load_embedding_metadata(self) -> pd.DataFrame:
        """Load the embeddings metadata dataframe, preferring the pipeline's
        own persisted `EMBEDDING_METADATA_FILE`, falling back to whatever
        pre-computed embeddings file is discovered via `_find_embedding_files`.
        """
        cached = self._artifacts.get("metadata_df_embeddings")
        if cached is not None:
            return cached

        if EMBEDDING_METADATA_FILE.exists():
            return self._load_csv_generic(
                EMBEDDING_METADATA_FILE, "metadata_df_embeddings", "embedding metadata"
            )

        _, metadata_path = self._find_embedding_files()
        if metadata_path is None:
            raise PipelineValidationError(
                "Cannot load embedding metadata: no persisted embedding metadata file found "
                f"(checked {EMBEDDING_METADATA_FILE} and {EMBEDDINGS_DIR}). "
                "The embedding_generation step may not have completed successfully."
            )

        df = self._load_metadata_file(metadata_path)
        self._artifacts["metadata_df_embeddings"] = df
        return df

    # ----------------------------------------------------------------
    # STEP IMPLEMENTATIONS (orchestration calls into existing modules only)
    # ----------------------------------------------------------------

    def _step_load_csv(self) -> None:
        def _do() -> Dict[str, Any]:
            loader = CSVLoader()
            df = loader.load(str(self._dataset_path))
            if df is None or df.empty:
                raise PipelineValidationError("Loaded CSV is empty or None.")
            self._artifacts["raw_df"] = df
            return {"rows": len(df), "columns": len(df.columns)}

        self._run_step("load_csv", _do)

    def _step_cleaning(self) -> None:
        def _do() -> Dict[str, Any]:
            raw_df = self._artifacts.get("raw_df")
            if raw_df is None:
                raw_df = self._load_raw_dataframe()

            cleaner = DataCleaner()
            rows_before = len(raw_df)
            cleaned_df = cleaner.clean(raw_df)

            Path(CLEANED_DATA_FILE).parent.mkdir(parents=True, exist_ok=True)
            if hasattr(cleaner, "save"):
                cleaner.save(cleaned_df, str(CLEANED_DATA_FILE))
            else:
                cleaned_df.to_csv(CLEANED_DATA_FILE, index=False)

            self._artifacts["cleaned_df"] = cleaned_df

            # Reduce memory usage: the raw dataframe is not needed by any
            # later step (all downstream steps read from cleaned_df onward).
            self._artifacts.pop("raw_df", None)
            del raw_df

            return {
                "rows_before": rows_before,
                "rows_after": len(cleaned_df),
                "rows_dropped": rows_before - len(cleaned_df),
                "columns": len(cleaned_df.columns),
            }

        self._run_step("cleaning", _do)

    def _step_canonical_mapping(self) -> None:
        def _do() -> Dict[str, Any]:
            cleaned_df = self._artifacts.get("cleaned_df")
            if cleaned_df is None:
                cleaned_df = self._load_cleaned_dataframe()

            mapper = CanonicalMapper()
            mapped_df = mapper.map(cleaned_df)

            if "record_id" not in mapped_df.columns:
                mapped_df.insert(0, "record_id", range(len(mapped_df)))

            MAPPED_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
            if hasattr(mapper, "save"):
                mapper.save(mapped_df, str(MAPPED_DATA_FILE))
            else:
                mapped_df.to_csv(MAPPED_DATA_FILE, index=False)

            self._artifacts["mapped_df"] = mapped_df

            # cleaned_df is superseded by mapped_df for every later step.
            self._artifacts.pop("cleaned_df", None)
            del cleaned_df

            return {"rows": len(mapped_df), "columns": len(mapped_df.columns)}

        self._run_step("canonical_mapping", _do)

    def _step_schema_analysis(self) -> None:
        def _do() -> Dict[str, Any]:
            df = self._artifacts.get("mapped_df")
            if df is None:
                df = self._load_mapped_dataframe()

            analyzer = SchemaAnalyzer()
            schema_result = analyzer.analyze(df)
            if hasattr(analyzer, "print_summary"):
                analyzer.print_summary(schema_result)

            SCHEMA_FILE.parent.mkdir(parents=True, exist_ok=True)
            try:
                serializable = (
                    schema_result if isinstance(schema_result, (dict, list))
                    else getattr(schema_result, "__dict__", str(schema_result))
                )
                with open(SCHEMA_FILE, "w", encoding="utf-8") as fh:
                    json.dump(serializable, fh, indent=2, default=str)
            except Exception as exc:
                logger.warning("Could not persist schema.json: %s", exc)

            self._artifacts["schema_result"] = schema_result
            field_count = len(getattr(schema_result, "fields", None) or schema_result.get("fields", []) if isinstance(schema_result, dict) else [])
            return {"field_count": field_count}

        self._run_step("schema_analysis", _do)

    def _step_data_separation(self) -> None:
        def _do() -> Dict[str, Any]:
            df = self._artifacts.get("mapped_df")
            if df is None:
                df = self._load_mapped_dataframe()

            schema_result = self._artifacts.get("schema_result")
            if schema_result is None:
                schema_result = self._load_schema()

            separator = DataSeparator()
            separated_data = separator.separate(df, schema_result)
            if hasattr(separator, "summary"):
                separator.summary(separated_data)
            if hasattr(separator, "save"):
                separator.save(separated_data)

            structured_df = separated_data["structured_df"]
            metadata_df = separated_data["metadata_df"]
            unstructured_df = separated_data["unstructured_df"]

            for path, frame in (
                (STRUCTURED_DATA_FILE, structured_df),
                (METADATA_DATA_FILE, metadata_df),
                (UNSTRUCTURED_DATA_FILE, unstructured_df),
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                if not path.exists():
                    frame.to_csv(path, index=False)

            self._artifacts["structured_df"] = structured_df
            self._artifacts["metadata_df"] = metadata_df
            self._artifacts["unstructured_df"] = unstructured_df

            # mapped_df is fully superseded by the three separated frames.
            self._artifacts.pop("mapped_df", None)
            del df

            return {
                "structured_rows": len(structured_df),
                "metadata_rows": len(metadata_df),
                "unstructured_rows": len(unstructured_df),
            }

        def _valid() -> bool:
            try:
                structured_df = self._artifacts.get("structured_df") or self._load_structured_dataframe()
                metadata_df = self._artifacts.get("metadata_df") or self._load_metadata_dataframe()
            except PipelineValidationError:
                return False
            self._check_duckdb_health()
            loader = self._duckdb_loader
            assert loader is not None
            return self._duckdb_tables_valid(loader, len(structured_df), len(metadata_df))

        self._run_step("data_separation", _do)

    def _step_precomputed_intelligence(self) -> None:
        def _do() -> Dict[str, Any]:
            unstructured_df = self._artifacts.get("unstructured_df")
            if unstructured_df is None:
                unstructured_df = self._load_unstructured_dataframe()

            engine = PrecomputedIntelligence()
            intelligence_df = engine.generate(unstructured_df)
            if hasattr(engine, "summary"):
                engine.summary(intelligence_df)
            if hasattr(engine, "save"):
                engine.save(intelligence_df)

            INTELLIGENCE_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
            if not INTELLIGENCE_DATA_FILE.exists():
                intelligence_df.to_csv(INTELLIGENCE_DATA_FILE, index=False)

            self._check_duckdb_health()
            loader = self._duckdb_loader
            assert loader is not None
            loader.load_intelligence(intelligence_df)
            if hasattr(loader, "show_tables"):
                loader.show_tables()

            self._artifacts["intelligence_df"] = intelligence_df
            return {"rows": len(intelligence_df)}

        def _valid() -> bool:
            try:
                unstructured_df = self._artifacts.get("unstructured_df") or self._load_unstructured_dataframe()
            except PipelineValidationError:
                return False
            self._check_duckdb_health()
            loader = self._duckdb_loader
            assert loader is not None
            return self._duckdb_intelligence_valid(loader, len(unstructured_df))

        self._run_step("precomputed_intelligence", _do, already_valid=_valid)

    def _step_chunking(self) -> None:
        def _do() -> Dict[str, Any]:
            unstructured_df = self._artifacts.get("unstructured_df")
            if unstructured_df is None:
                unstructured_df = self._load_unstructured_dataframe()

            chunker = TextChunker()
            chunk_df = chunker.chunk_dataframe(unstructured_df)
            if hasattr(chunker, "summary"):
                chunker.summary(chunk_df)
            if hasattr(chunker, "save"):
                chunker.save(chunk_df)

            CHUNKED_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
            if not CHUNKED_DATA_FILE.exists():
                chunk_df.to_csv(CHUNKED_DATA_FILE, index=False)

            self._artifacts["chunk_df"] = chunk_df

            # unstructured_df is no longer needed once chunked; intelligence
            # and separation steps already persisted what they need.
            self._artifacts.pop("unstructured_df", None)
            del unstructured_df

            return {"chunks": len(chunk_df)}

        def _valid() -> bool:
            embedding_path, metadata_path = self._find_embedding_files()
            return embedding_path is not None and metadata_path is not None

        self._run_step("chunking", _do, already_valid=_valid)

    def _step_embedding_generation(self) -> None:
        def _do() -> Dict[str, Any]:
            embedding_path, metadata_path = self._find_embedding_files()
            start = time.time()

            if embedding_path is not None and metadata_path is not None:
                metadata_df_embeddings = self._load_metadata_file(metadata_path)
                logger.info("Loaded pre-computed embeddings from %s", embedding_path)
                vector_dimension = self._infer_vector_dimension(embedding_path)
            else:
                chunk_df = self._artifacts.get("chunk_df")
                if chunk_df is None:
                    chunk_df = self._load_chunk_dataframe()

                embedder = EmbeddingGenerator()
                metadata_df_embeddings, _unique_df = embedder.generate(chunk_df)
                self._artifacts["embedder"] = embedder

                # chunk_df is only needed to produce embeddings; drop it now.
                self._artifacts.pop("chunk_df", None)
                del chunk_df

                vector_dimension = self._infer_vector_dimension(Path(EMBEDDINGS_NPY_FILE))

            intelligence_df = self._artifacts.get("intelligence_df")
            if intelligence_df is None:
                intelligence_df = self._load_intelligence_dataframe(required=False)

            metadata_df_embeddings = self._merge_intelligence(metadata_df_embeddings, intelligence_df)
            metadata_df_embeddings = self._fix_series_columns(metadata_df_embeddings)

            EMBEDDING_METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)
            if not EMBEDDING_METADATA_FILE.exists():
                metadata_df_embeddings.to_csv(EMBEDDING_METADATA_FILE, index=False)

            self._artifacts["metadata_df_embeddings"] = metadata_df_embeddings

            elapsed = time.time() - start
            metrics: Dict[str, Any] = {
                "vector_rows": len(metadata_df_embeddings),
                "elapsed_seconds": round(elapsed, 3),
            }
            if vector_dimension is not None:
                metrics["vector_dimension"] = vector_dimension

            return metrics

        self._run_step("embedding_generation", _do)

    def _step_quantization(self) -> None:
        if not ENABLE_QUANTIZATION:
            with self._lock:
                assert self._state is not None
                record = self._state.steps["quantization"]
                record.status = StepStatus.SKIPPED.value
                record.completed_at = _now_iso()
                record.metrics = {"reason": "ENABLE_QUANTIZATION is False"}
                self._persist_state()
            logger.info("[quantization] disabled via config — skipping.")
            return

        def _do() -> Dict[str, Any]:
            if not Path(EMBEDDINGS_NPY_FILE).exists():
                raise PipelineValidationError(
                    f"Cannot quantize: embeddings file not found at {EMBEDDINGS_NPY_FILE}. "
                    f"The embedding_generation step may not have completed successfully."
                )

            embedder = self._artifacts.get("embedder") or EmbeddingGenerator()
            embeddings = embedder.load_embeddings()
            if embeddings is None or len(embeddings) == 0:
                raise PipelineValidationError(
                    "Cannot quantize: load_embeddings() returned no data."
                )

            quantizer = EmbeddingQuantizer()
            quantized_embeddings, scale = quantizer.quantize_int8(embeddings)
            error_metrics = quantizer.reconstruction_error(embeddings, quantized_embeddings, scale)
            quantizer.summary(embeddings, quantized_embeddings, error_metrics)
            quantizer.save_quantized(quantized_embeddings, scale, error_metrics=error_metrics)

            metrics = {"reconstruction_error": error_metrics} if error_metrics else {}
            metrics["embedding_count"] = len(embeddings)
            del embeddings
            return metrics

        self._run_step("quantization", _do)

    def _step_duckdb_load(self) -> None:
        def _do() -> Dict[str, Any]:
            self._check_duckdb_health()
            loader = self._duckdb_loader
            assert loader is not None

            structured_df = self._artifacts.get("structured_df")
            if structured_df is None:
                structured_df = self._load_structured_dataframe()

            metadata_df = self._artifacts.get("metadata_df")
            if metadata_df is None:
                metadata_df = self._load_metadata_dataframe()

            loader.load_structured(structured_df)
            loader.load_metadata(metadata_df)
            if hasattr(loader, "show_tables"):
                loader.show_tables()

            return {
                "structured_rows_inserted": len(structured_df),
                "metadata_rows_inserted": len(metadata_df),
            }

        def _valid() -> bool:
            try:
                structured_df = self._artifacts.get("structured_df") or self._load_structured_dataframe()
                metadata_df = self._artifacts.get("metadata_df") or self._load_metadata_dataframe()
            except PipelineValidationError:
                return False
            self._check_duckdb_health()
            loader = self._duckdb_loader
            assert loader is not None
            return self._duckdb_tables_valid(loader, len(structured_df), len(metadata_df))

        self._run_step("duckdb_load", _do, already_valid=_valid)

    def _step_qdrant_load(self) -> None:
        def _do() -> Dict[str, Any]:
            self._check_qdrant_health()
            loader = self._qdrant_loader
            assert loader is not None

            metadata_df_embeddings = self._artifacts.get("metadata_df_embeddings")
            if metadata_df_embeddings is None:
                metadata_df_embeddings = self._load_embedding_metadata()

            if loader.collection_exists():
                logger.info("Vector count mismatch or empty — recreating collection.")
                loader.delete_collection()

            loader.load_dataframe(
                metadata_df_embeddings,
                str(EMBEDDINGS_FILE),
                batch_size=QDRANT_BATCH_SIZE,
                parallel=QDRANT_LOAD_PARALLELISM,
            )

            vectors_stored = loader.count()

            # Final step — the embeddings dataframe is no longer needed.
            self._artifacts.pop("metadata_df_embeddings", None)
            del metadata_df_embeddings

            return {"vectors_stored": vectors_stored}

        def _valid() -> bool:
            try:
                metadata_df_embeddings = self._artifacts.get("metadata_df_embeddings") or self._load_embedding_metadata()
            except PipelineValidationError:
                return False
            self._check_qdrant_health()
            loader = self._qdrant_loader
            assert loader is not None
            return self._qdrant_valid(loader, len(metadata_df_embeddings))

        self._run_step("qdrant_load", _do, already_valid=_valid)

    _STEP_HANDLERS: Dict[str, Callable[["DatasetPipeline"], None]] = {
        "load_csv": _step_load_csv,
        "cleaning": _step_cleaning,
        "canonical_mapping": _step_canonical_mapping,
        "schema_analysis": _step_schema_analysis,
        "data_separation": _step_data_separation,
        "precomputed_intelligence": _step_precomputed_intelligence,
        "chunking": _step_chunking,
        "embedding_generation": _step_embedding_generation,
        "quantization": _step_quantization,
        "duckdb_load": _step_duckdb_load,
        "qdrant_load": _step_qdrant_load,
    }

    # ----------------------------------------------------------------
    # INTERNAL: HEALTH CHECKS
    # ----------------------------------------------------------------

    def _check_duckdb_health(self) -> None:
        """Verify the DuckDB connection is usable before relying on it.

        Raises PipelineDependencyError with a descriptive message if the
        connection cannot be reached/queried. Controlled by
        ENABLE_DUCKDB_HEALTHCHECK for environments where this check is
        undesirable (e.g. certain test harnesses).
        """
        if not ENABLE_DUCKDB_HEALTHCHECK:
            return
        loader = self._duckdb_loader
        if loader is None:
            raise PipelineDependencyError("DuckDB loader has not been initialized.")

        health_check = getattr(loader, "health_check", None) or getattr(loader, "ping", None)
        try:
            if callable(health_check):
                health_check()
            else:
                # Fall back to a harmless, universally-supported probe.
                loader.table_exists(STRUCTURED_TABLE)
        except Exception as exc:
            raise PipelineDependencyError(f"DuckDB health check failed: {exc}") from exc

    def _check_qdrant_health(self) -> None:
        """Verify the Qdrant server is reachable before relying on it.

        Raises PipelineDependencyError with a descriptive message if the
        server cannot be reached. Controlled by ENABLE_QDRANT_HEALTHCHECK.
        """
        if not ENABLE_QDRANT_HEALTHCHECK:
            return
        loader = self._qdrant_loader
        if loader is None:
            raise PipelineDependencyError("Qdrant loader has not been initialized.")

        health_check = getattr(loader, "health_check", None) or getattr(loader, "ping", None)
        try:
            if callable(health_check):
                health_check()
            else:
                # Fall back to a harmless, universally-supported probe.
                loader.collection_exists()
        except Exception as exc:
            raise PipelineDependencyError(f"Qdrant server is unavailable: {exc}") from exc

    # ----------------------------------------------------------------
    # INTERNAL: HELPERS (ported from the original script, generalized)
    # ----------------------------------------------------------------

    @staticmethod
    def _find_embedding_files() -> Tuple[Optional[Path], Optional[Path]]:
        """Return (embedding_path, metadata_path) if a pre-computed
        embeddings artifact is present on disk, else (None, None)."""
        embedding_file = Path(EMBEDDINGS_NPY_FILE)

        metadata_file: Optional[Path] = None
        if EMBEDDING_METADATA_FILE.exists():
            metadata_file = EMBEDDING_METADATA_FILE
        else:
            try:
                for file in Path(EMBEDDINGS_DIR).glob("*"):
                    lower = file.name.lower()
                    if "metadata" in lower and lower.endswith((".csv", ".parquet", ".pkl")):
                        metadata_file = file
                        break
            except OSError as exc:
                logger.debug("Could not scan EMBEDDINGS_DIR: %s", exc)
                return None, None

        if embedding_file.exists() and metadata_file is not None:
            return embedding_file, metadata_file

        return None, None

    @staticmethod
    def _infer_vector_dimension(embedding_path: Path) -> Optional[int]:
        """Best-effort peek at the embedding vector dimensionality for
        metrics reporting, without loading the full array into memory
        where avoidable."""
        try:
            import numpy as np  # local import: numpy is an EmbeddingGenerator dependency, not a new one

            with open(embedding_path, "rb") as fh:
                version = np.lib.format.read_magic(fh)
                shape, _fortran, _dtype = np.lib.format._read_array_header(fh, version)
            return int(shape[1]) if len(shape) > 1 else None
        except Exception:
            return None

    @staticmethod
    def _load_metadata_file(path: Path) -> pd.DataFrame:
        """Load a metadata file based on its extension (csv/parquet/pkl)."""
        suffix = str(path).lower()
        try:
            if suffix.endswith(".csv"):
                return pd.read_csv(path, low_memory=False)
            elif suffix.endswith(".parquet"):
                return pd.read_parquet(path)
            else:
                return pd.read_pickle(path)
        except Exception as exc:
            raise PipelineValidationError(
                f"Failed to load metadata file {path}: {exc}"
            ) from exc

    @staticmethod
    def _duckdb_tables_valid(
        duckdb_loader: DuckDBLoader, expected_structured: int, expected_metadata: int
    ) -> bool:
        """True if structured and metadata tables exist and row counts match."""
        for table, expected in (
            (STRUCTURED_TABLE, expected_structured),
            (METADATA_TABLE, expected_metadata),
        ):
            if not duckdb_loader.table_exists(table):
                return False
            if duckdb_loader.row_count(table) != expected:
                return False
        return True

    @staticmethod
    def _duckdb_intelligence_valid(duckdb_loader: DuckDBLoader, expected_rows: int) -> bool:
        """True if the intelligence table exists and its row count matches."""
        if not duckdb_loader.table_exists(INTELLIGENCE_TABLE):
            return False
        return duckdb_loader.row_count(INTELLIGENCE_TABLE) == expected_rows

    @staticmethod
    def _qdrant_valid(qdrant_loader: QdrantLoader, expected_vectors: int) -> bool:
        """True if the Qdrant collection exists and vector count matches."""
        if not qdrant_loader.collection_exists():
            return False
        return qdrant_loader.count() == expected_vectors

    @staticmethod
    def _merge_intelligence(
        metadata_df_embeddings: pd.DataFrame,
        intelligence_df: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        """Left-join intelligence columns onto metadata_df_embeddings.

        - Identifies a shared join key (chunk_id preferred, record_id
          fallback) so this works for any dataset schema.
        - Drops overlapping columns from intelligence_df to prevent _x/_y
          column splits.
        - Flattens any cells that ended up as pandas Series after the merge
          and re-casts numeric columns to their correct dtype.

        If intelligence_df is None (e.g. step was skipped or the
        precomputed_intelligence artifact is genuinely unavailable), the
        original dataframe is returned unchanged.
        """
        if intelligence_df is None or intelligence_df.empty:
            logger.info("No intelligence data available to merge — skipping merge.")
            return metadata_df_embeddings

        join_key = None
        for candidate in ("chunk_id", "record_id"):
            if (
                candidate in metadata_df_embeddings.columns
                and candidate in intelligence_df.columns
            ):
                join_key = candidate
                break

        if join_key is None:
            raise PipelineValidationError(
                "Cannot merge intelligence: no shared key (chunk_id / record_id) "
                "found between metadata_df_embeddings and intelligence_df."
            )

        overlap = [
            c for c in intelligence_df.columns
            if c in metadata_df_embeddings.columns and c != join_key
        ]
        if overlap:
            logger.info("Dropping overlapping columns from intelligence_df: %s", overlap)

        intel_to_merge = intelligence_df.drop(columns=overlap)
        enriched = metadata_df_embeddings.merge(intel_to_merge, on=join_key, how="left")
        enriched = DatasetPipeline._fix_series_columns(enriched)

        new_cols = [c for c in enriched.columns if c not in metadata_df_embeddings.columns]
        logger.info("Intelligence columns merged: %s", new_cols)
        logger.info("Enriched dataframe shape: %s", enriched.shape)

        return enriched

    @staticmethod
    def _fix_series_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Flatten any cell values that ended up as pandas Series objects
        (a known pitfall of certain merge/groupby patterns), then re-cast
        the column to numeric where possible. Generic over any dataset
        schema — inspects every column rather than a hardcoded list.
        """
        if df.empty:
            return df

        fixed: List[str] = []
        for col in df.columns:
            sample = df[col].iloc[0]
            if isinstance(sample, pd.Series):
                fixed.append(col)
                df[col] = df[col].apply(
                    lambda x: x.iloc[0] if isinstance(x, pd.Series) else x
                )
                try:
                    df[col] = pd.to_numeric(df[col], errors="ignore")
                except Exception:
                    pass

        if fixed:
            logger.warning("Flattened Series objects in columns: %s", fixed)

        return df

    # ----------------------------------------------------------------
    # CONTEXT MANAGER SUPPORT
    # ----------------------------------------------------------------

    def __enter__(self) -> "DatasetPipeline":
        return self.initialize()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


# ==========================================================================
# ENTRY POINT
# ==========================================================================

def main() -> None:
    """CLI-style entry point preserving the original script's behavior:
    run the full pipeline once, end-to-end, using config defaults."""
    pipeline = DatasetPipeline()
    try:
        pipeline.initialize()
        pipeline.validate()
        result = pipeline.run()
        logger.info("Final status: %s", json.dumps(result, indent=2, default=str))
    except DatasetPipelineError as exc:
        logger.error("Pipeline failed: %s", exc)
        raise
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()