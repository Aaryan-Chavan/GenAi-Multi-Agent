"""
Storage/qdrant_loader.py
=========================
Universal, production-grade, schema-agnostic Qdrant ingestion engine.

Ingests ANY tabular dataset (Amazon reviews, support tickets, medical
notes, financial transactions, research papers, emails, news articles,
HR records, manufacturing logs, or anything else) without requiring any
code changes. There are no hardcoded field registries, no hardcoded
filesystem paths, and no assumptions about column names beyond dynamic
auto-detection heuristics that can always be overridden via configuration.

Public API (preserved for backward compatibility with existing pipelines):

    QdrantLoader.create_collection(vector_size)
    QdrantLoader.build_payload(row_values, columns)
    QdrantLoader.load_dataframe(metadata_df, embedding_file=None, ...)
    QdrantLoader.load_from_sources(chunks_csv=None, ...)
    QdrantLoader.search(query_vector, limit=5, filters=None)
    QdrantLoader.count()
    QdrantLoader.collection_exists()
    QdrantLoader.delete_collection()
    QdrantLoader.collection_info()

MULTI-GRANULARITY SOURCE MERGING
---------------------------------
This revision fixes a data-corruption bug in multi-source ingestion.

A very common real-world layout is:

    chunks.csv        -- one row PER CHUNK        (chunk_id, document_id, text, ...)
    intelligence.csv  -- one row PER CHUNK         (chunk_id, document_id, entities, ...)
    metadata.csv       -- one row PER DOCUMENT      (document_id, title, author, ...)

i.e. chunks/intelligence are at *chunk* granularity and metadata is at
*document* granularity, so `document_id` repeats many times in the
chunk-level frames but is unique in metadata.csv. The previous merge
strategy picked a join key only if it was highly unique in *every*
frame being merged -- which is never true here, since `document_id`
is not unique in chunks.csv/intelligence.csv. That caused the loader
to silently fall back to *positional* row alignment (zipping rows by
index and truncating to the shortest frame), which corrupts the data:
chunk 5 of document A could end up tagged with document B's metadata.

`merge_sources` now:
  1. Treats the LARGEST frame (by row count) as the base -- this is
     naturally the chunk-level data, since there are always at least
     as many chunks as documents.
  2. For every other frame, looks for a common key column that is
     unique WITHIN THAT SMALLER FRAME (not necessarily unique in the
     base). This correctly supports both 1:1 joins (chunks <-> chunk
     intelligence, on chunk_id) and many:1 joins (chunks <-> document
     metadata, on document_id).
  3. Falls back to positional alignment ONLY when frame lengths match
     exactly and no shared key exists -- and raises a clear,
     actionable error instead of silently corrupting data when
     lengths differ and no key can be found.

New capabilities from prior revisions remain documented inline in each
section header below.
"""

from __future__ import annotations

import ast
import base64
import hashlib
import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
    MatchExcept,
    Range,
    PayloadSchemaType,
)

try:
    import psutil  # optional, only used for memory-usage reporting
except ImportError:  # pragma: no cover - optional dependency
    psutil = None

# ======================================================================
# CONFIGURATION (fully dynamic -- no hardcoded paths / names / sizes)
# ======================================================================
#
# All previously hardcoded values (Qdrant host/port/collection/batch
# size, and any project-specific source paths) are now resolved from,
# in order of priority:
#
#   1. Explicit constructor / method arguments (highest priority).
#   2. A `LoaderConfig` object passed to the constructor.
#   3. Environment variables (QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION,
#      QDRANT_BATCH_SIZE, QDRANT_CHECKPOINT_PATH, QDRANT_FAILED_ROWS_LOG,
#      QDRANT_LOG_FILE).
#   4. The project's Config.settings module, if importable (kept for
#      backward compatibility with existing pipelines that already rely
#      on it) -- this is a *soft* dependency: if the module or any name
#      inside it is missing, the loader falls back to safe defaults
#      instead of raising an ImportError at import time.
#   5. Safe, non-project-specific defaults.
#
# Nothing in this module assumes a specific project directory layout.

def _soft_import_settings() -> Dict[str, Any]:
    try:
        from Config import settings as _settings  # type: ignore
    except Exception:
        return {}
    keys = (
        "QDRANT_HOST", "QDRANT_PORT", "QDRANT_COLLECTION", "QDRANT_BATCH_SIZE",
        "QDRANT_CHECKPOINT_PATH", "QDRANT_FAILED_ROWS_LOG", "QDRANT_LOG_FILE",
    )
    return {k: getattr(_settings, k) for k in keys if hasattr(_settings, k)}


_SETTINGS = _soft_import_settings()


def _default(name: str, fallback: Any) -> Any:
    if name in _SETTINGS:
        return _SETTINGS[name]
    env_val = os.environ.get(name)
    if env_val is not None:
        return env_val
    return fallback


@dataclass
class LoaderConfig:
    """Central, injectable configuration object.

    Every value has a safe, project-agnostic default and can be
    overridden per-instance, per-call, or via environment variables /
    an optional ``Config.settings`` module. No value is a magic number
    baked into the ingestion logic itself.
    """

    host: str = field(default_factory=lambda: str(_default("QDRANT_HOST", "localhost")))
    port: int = field(default_factory=lambda: int(_default("QDRANT_PORT", 6333)))
    collection_name: str = field(default_factory=lambda: str(_default("QDRANT_COLLECTION", "default_collection")))
    batch_size: int = field(default_factory=lambda: int(_default("QDRANT_BATCH_SIZE", 256)))
    checkpoint_path: str = field(default_factory=lambda: str(_default("QDRANT_CHECKPOINT_PATH", "checkpoint.json")))
    failed_rows_log: str = field(default_factory=lambda: str(_default("QDRANT_FAILED_ROWS_LOG", "failed_rows.log")))
    log_file: Optional[str] = field(default_factory=lambda: _default("QDRANT_LOG_FILE", None))
    parallel: int = 8
    max_batch_retries: int = 3
    csv_chunksize: Optional[int] = None  # e.g. 100_000 to stream large CSVs
    duplicate_strategy: str = "primary_key_or_hash"  # see DuplicateDetector
    distance_metric: Distance = Distance.COSINE
    create_payload_indexes: bool = True
    validate_collection_compatibility: bool = True
    verify_upload: bool = True
    # Uniqueness threshold used when deciding whether a shared column can
    # safely act as a join key against a given (typically smaller) frame.
    merge_key_uniqueness_threshold: float = 0.98


# ======================================================================
# LOGGING
# ======================================================================

logger = logging.getLogger("qdrant_loader")


def configure_logging(log_file: Optional[Union[str, Path]] = None, level: int = logging.INFO) -> None:
    """Configure console (+ optional file) logging with timestamps.

    Safe to call multiple times; existing handlers of the same type are
    not duplicated.
    """
    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in logger.handlers):
        console = logging.StreamHandler()
        console.setFormatter(fmt)
        logger.addHandler(console)

    if log_file:
        log_path = Path(log_file)
        already = any(
            isinstance(h, logging.FileHandler) and Path(h.baseFilename) == log_path.resolve()
            for h in logger.handlers
        )
        if not already:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setFormatter(fmt)
            logger.addHandler(file_handler)


configure_logging()


# ======================================================================
# EXCEPTIONS -- every failure explains what/where/why/how to fix
# ======================================================================

class LoaderError(Exception):
    """Base class for all loader exceptions.

    Formats a structured message so failures are self-explanatory:
    what failed, where, why, and a suggested resolution.
    """

    def __init__(self, what: str, where: str, why: str, resolution: str):
        self.what = what
        self.where = where
        self.why = why
        self.resolution = resolution
        message = (
            f"[{what}] failed in [{where}]. Reason: {why}. "
            f"Suggested resolution: {resolution}"
        )
        super().__init__(message)


class ValidationError(LoaderError):
    """Raised when pre-ingestion data validation fails."""


class EmbeddingValidationError(LoaderError):
    """Raised when embedding validation fails."""


class CollectionCompatibilityError(LoaderError):
    """Raised when an existing collection is incompatible with incoming data."""


class UploadIntegrityError(LoaderError):
    """Raised when post-upload verification detects data loss or corruption."""


# ======================================================================
# MISSING-VALUE HANDLING
# ======================================================================

_MISSING_STRINGS = {"", "nan", "null", "none", "n/a", "na", "unknown"}


def is_missing(value: Any) -> bool:
    """Return True if ``value`` should be treated as absent."""
    if value is None:
        return True
    if value is pd.NA:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    if isinstance(value, np.floating) and np.isnan(value):
        return True
    if isinstance(value, str) and value.strip().lower() in _MISSING_STRINGS:
        return True
    return False


# ======================================================================
# DYNAMIC TYPE INFERENCE
# ======================================================================
# Fields whose raw string representation must be preserved exactly (no
# numeric coercion) because they are typically used as stable
# identifiers/dedup keys and losing formatting (e.g. leading zeros)
# would break identity guarantees. This list is a heuristic default,
# not a hard schema -- it is extended dynamically at runtime by
# `detect_identifier_columns` for any dataset.
_RAW_PRESERVE_FIELDS = frozenset({
    "record_id", "chunk_id", "document_id", "text_hash", "id", "uuid", "guid",
})

_ID_LIKE_SUFFIXES = ("_id", "_uuid", "_guid", "_key", "_code", "_sku", "_asin")
_ID_LIKE_NAMES = {
    "id", "uuid", "guid", "asin", "sku", "key",
}

_LARGE_TEXT_HINTS = (
    "text", "chunk", "body", "content", "description", "review", "summary",
    "notes", "comment", "message", "abstract", "transcript",
)

# Legacy / conventional key names kept as join-key candidates even if the
# generic name-scoring heuristic would otherwise miss them.
_LEGACY_MERGE_KEY_CANDIDATES = ("record_id", "chunk_id", "document_id", "doc_id", "id")


def infer_native(value: Any) -> Any:
    """Convert a raw cell value into a JSON/Qdrant-safe native Python type.

    Handles int, float, bool, str, list, dict, datetime, and numpy/pandas
    scalar types. Lists and dicts are preserved as native structures
    (never stringified). JSON- or Python-literal-encoded strings (e.g. a
    list or dict serialized into a CSV cell) are parsed back into their
    native structure.
    """
    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, bool):
        return value

    if isinstance(value, int):
        return int(value)

    if isinstance(value, float):
        return float(value)

    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()

    if isinstance(value, np.ndarray):
        return [infer_native(v) for v in value.tolist()]

    if isinstance(value, (list, tuple)):
        return [infer_native(v) for v in value]

    if isinstance(value, (set, frozenset)):
        return [infer_native(v) for v in value]

    if isinstance(value, dict):
        return {str(k): infer_native(v) for k, v in value.items()}

    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return base64.b64encode(value).decode("ascii")

    if isinstance(value, str):
        return _infer_from_string(value)

    return str(value)


def _infer_from_string(value: str) -> Any:
    stripped = value.strip()
    low = stripped.lower()

    if low in ("true", "false"):
        return low == "true"

    is_bracketed = (
        (stripped.startswith("{") and stripped.endswith("}"))
        or (stripped.startswith("[") and stripped.endswith("]"))
    )
    if is_bracketed:
        parsed = _try_parse_structure(stripped)
        if parsed is not None:
            return infer_native(parsed)

    try:
        return int(stripped)
    except ValueError:
        pass

    try:
        return float(stripped)
    except ValueError:
        pass

    return value


def _try_parse_structure(stripped: str) -> Any:
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, TypeError):
        pass
    try:
        return ast.literal_eval(stripped)
    except (ValueError, SyntaxError):
        return None


def _classify_value_type(value: Any) -> str:
    """Classify an already-inferred native value into a coarse type tag
    used for payload-index creation decisions."""
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "float"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    if isinstance(value, str):
        if _looks_like_datetime(value):
            return "datetime"
        return "keyword"
    return "unknown"


_DATETIME_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}([T ]\d{2}:\d{2}(:\d{2})?(\.\d+)?(Z|[+-]\d{2}:?\d{2})?)?$"
)


def _looks_like_datetime(value: str) -> bool:
    return bool(_DATETIME_RE.match(value.strip()))


# ======================================================================
# PAYLOAD SANITIZATION / FLATTENING
# ======================================================================

def _sanitize_value(value: Any) -> Any:
    """Ensure a value is JSON-serializable (handles bytes/sets/objects)."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return base64.b64encode(value).decode("ascii")
    if isinstance(value, (set, frozenset)):
        return [_sanitize_value(v) for v in value]
    if isinstance(value, (list, tuple)):
        return [_sanitize_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _sanitize_value(v) for k, v in value.items()}
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value)


def sanitize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively sanitize an entire payload dict."""
    return {k: _sanitize_value(v) for k, v in payload.items()}


def flatten_payload(payload: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """Flatten nested dicts (and lists of dicts) using dot-notation keys so
    every leaf field is independently filterable in Qdrant.

    Lists of scalars are preserved as-is (Qdrant natively supports
    match-any / contains queries against array payload fields). Lists of
    dicts are both flattened element-wise (``field.0.subfield``) AND kept
    as the original array under ``field``, so callers can filter either
    the whole array or a specific element's sub-field.
    """
    flat: Dict[str, Any] = {}

    for key, value in payload.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key

        if isinstance(value, dict):
            flat.update(flatten_payload(value, new_key, sep=sep))
        elif isinstance(value, list) and value and all(isinstance(el, dict) for el in value):
            for i, element in enumerate(value):
                flat.update(flatten_payload(element, f"{new_key}.{i}", sep=sep))
            flat[new_key] = value
        else:
            flat[new_key] = value

    return flat


# ======================================================================
# HASHING / DETERMINISTIC IDS
# ======================================================================

def _stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_text_hash(payload: Dict[str, Any]) -> str:
    """Compute a stable content hash for a payload.

    Priority: existing ``text_hash`` field -> hash of the largest
    "text-like" field present -> hash of the full sorted payload.
    """
    existing = payload.get("text_hash")
    if existing and not is_missing(existing):
        return str(existing)

    text_like_keys = [
        k for k in payload
        if any(hint in k.lower() for hint in _LARGE_TEXT_HINTS)
    ]
    for key in text_like_keys:
        value = payload.get(key)
        if isinstance(value, str) and value and not is_missing(value):
            return _stable_hash(value)

    return _stable_hash(json.dumps(payload, sort_keys=True, default=str))


def compute_composite_hash(payload: Dict[str, Any], keys: Sequence[str]) -> str:
    """Compute a hash over a specific, ordered subset of payload fields."""
    parts = [str(payload.get(k, "")) for k in keys]
    return _stable_hash("|".join(parts))


# Fixed namespace so deterministic UUIDs are stable across processes/runs.
_ID_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


def _coerce_qdrant_id(value: Any) -> Union[int, str]:
    if isinstance(value, bool):
        value = int(value)
    if isinstance(value, (int, np.integer)):
        return int(value)

    text = str(value).strip()
    try:
        return str(uuid.UUID(text))
    except (ValueError, AttributeError):
        return str(uuid.uuid5(_ID_NAMESPACE, text))


# ======================================================================
# 5. AUTOMATIC PRIMARY / IDENTIFIER KEY DETECTION
# ======================================================================
#
# Instead of only checking a fixed list of names (record_id / chunk_id /
# id), columns are scored dynamically based on:
#   - name heuristics (ends with _id/_uuid/_key/_code, or is exactly
#     id/uuid/guid/sku/asin/key)
#   - uniqueness ratio (fraction of non-null values that are unique)
#   - value shape (looks like a UUID, is integer-typed, etc.)
#
# The highest-scoring column(s) become identifier candidates. When no
# single column is sufficiently unique, a composite key (concatenation
# of the best partial candidates) is used instead, and only when no
# usable identifier exists at all does the loader fall back to
# content-hash-derived deterministic IDs.
#
# NOTE: after a chunk-level + document-level merge, the merged frame
# will contain BOTH `chunk_id` (unique per row -- the correct primary
# key for point identity) and `document_id` (repeated across every
# chunk of the same document -- NOT a valid primary key). The scoring
# below naturally prefers `chunk_id` because it scores highest on
# uniqueness; `document_id` will simply fail the uniqueness threshold
# and be skipped as a primary-key candidate, even though it remains a
# perfectly normal, filterable payload field.

_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


def _uniqueness_ratio(series: pd.Series) -> float:
    non_null = series.dropna()
    if len(non_null) == 0:
        return 0.0
    return non_null.nunique() / len(non_null)


def _name_score(col: str) -> float:
    low = col.lower()
    if low in _ID_LIKE_NAMES:
        return 1.0
    if any(low.endswith(suf) for suf in _ID_LIKE_SUFFIXES):
        return 0.9
    if "id" in low.split("_"):
        return 0.6
    return 0.0


def detect_identifier_columns(
    df: pd.DataFrame, uniqueness_threshold: float = 0.98, sample_size: int = 5000
) -> List[str]:
    """Return columns ranked as identifier candidates, best first.

    A column qualifies if it scores well on naming heuristics AND has a
    high uniqueness ratio (computed on a sample for performance on very
    large frames).
    """
    sample = df.sample(min(sample_size, len(df)), random_state=0) if len(df) > sample_size else df
    scored: List[Tuple[float, str]] = []

    for col in df.columns:
        name_score = _name_score(col)
        if name_score == 0.0:
            continue
        try:
            uniq = _uniqueness_ratio(sample[col])
        except TypeError:
            continue
        if uniq < uniqueness_threshold:
            continue
        combined = name_score * 0.4 + uniq * 0.6
        scored.append((combined, col))

    scored.sort(key=lambda t: t[0], reverse=True)
    return [col for _, col in scored]


def detect_primary_key(
    df: pd.DataFrame, uniqueness_threshold: float = 0.98
) -> Tuple[Optional[List[str]], str]:
    """Detect the best primary key for a dataframe.

    Returns a tuple ``(key_columns, strategy)`` where ``strategy`` is one
    of: "single_column", "composite", or "none" (meaning deterministic
    hash-based IDs must be used).
    """
    candidates = detect_identifier_columns(df, uniqueness_threshold=uniqueness_threshold)
    if candidates:
        return [candidates[0]], "single_column"

    # No single column is unique enough on its own -- try a composite of
    # the best partially-unique id-like columns.
    id_like = [c for c in df.columns if _name_score(c) > 0.0]
    if len(id_like) >= 2:
        sample = df.sample(min(5000, len(df)), random_state=0) if len(df) > 5000 else df
        combo_uniqueness = sample[id_like].astype(str).agg("|".join, axis=1).nunique() / max(len(sample), 1)
        if combo_uniqueness >= uniqueness_threshold:
            return id_like, "composite"

    return None, "none"


def resolve_point_id(
    payload: Dict[str, Any], text_hash: str, key_columns: Optional[Sequence[str]] = None
) -> Union[int, str]:
    """Resolve a stable, deterministic Qdrant point ID.

    Priority: dynamically detected primary key column(s) -> legacy
    record_id/chunk_id fields (kept for backward compatibility) ->
    text_hash -> random UUID.
    """
    if key_columns:
        values = [payload.get(k) for k in key_columns]
        if all(v is not None and not is_missing(v) for v in values):
            joined = "|".join(str(v) for v in values)
            return _coerce_qdrant_id(joined)

    for legacy_key in ("record_id", "chunk_id"):
        value = payload.get(legacy_key)
        if value is not None and not is_missing(value):
            return _coerce_qdrant_id(value)

    if text_hash:
        return _coerce_qdrant_id(text_hash)

    return str(uuid.uuid4())


# ======================================================================
# 6. DUPLICATE DETECTION (configurable strategy)
# ======================================================================

class DuplicateDetector:
    """Configurable, multi-strategy duplicate detector.

    Strategies:
        - "primary_key": duplicate iff the resolved point ID repeats.
        - "text_hash": duplicate iff the content hash repeats (legacy
          behavior).
        - "composite_payload_hash": duplicate iff a hash over a chosen
          subset of payload fields repeats.
        - "primary_key_or_hash" (default): duplicate iff EITHER the
          point ID or the content hash has been seen before -- the
          strictest, safest default since it also catches accidental
          hash collisions or ID reuse across differently-shaped rows.
    """

    VALID_STRATEGIES = {
        "primary_key", "text_hash", "composite_payload_hash", "primary_key_or_hash",
    }

    def __init__(
        self,
        strategy: str = "primary_key_or_hash",
        composite_keys: Optional[Sequence[str]] = None,
    ):
        if strategy not in self.VALID_STRATEGIES:
            raise ValidationError(
                what="DuplicateDetector initialization",
                where="DuplicateDetector.__init__",
                why=f"Unknown duplicate_strategy '{strategy}'",
                resolution=f"Use one of {sorted(self.VALID_STRATEGIES)}",
            )
        self.strategy = strategy
        self.composite_keys = list(composite_keys) if composite_keys else None
        self._seen_ids: Set[Any] = set()
        self._seen_hashes: Set[str] = set()

    def seed_existing(self, ids: Iterable[Any] = (), hashes: Iterable[str] = ()) -> None:
        self._seen_ids.update(ids)
        self._seen_hashes.update(h for h in hashes if h)

    def is_duplicate_and_record(self, point_id: Any, payload: Dict[str, Any], text_hash: str) -> bool:
        """Check whether the row is a duplicate under the configured
        strategy, and if not, record it as seen."""
        if self.strategy == "primary_key":
            if point_id in self._seen_ids:
                return True
            self._seen_ids.add(point_id)
            return False

        if self.strategy == "text_hash":
            if text_hash in self._seen_hashes:
                return True
            self._seen_hashes.add(text_hash)
            return False

        if self.strategy == "composite_payload_hash":
            keys = self.composite_keys or sorted(payload.keys())
            chash = compute_composite_hash(payload, keys)
            if chash in self._seen_hashes:
                return True
            self._seen_hashes.add(chash)
            return False

        # primary_key_or_hash (default, strictest)
        if point_id in self._seen_ids or text_hash in self._seen_hashes:
            return True
        self._seen_ids.add(point_id)
        self._seen_hashes.add(text_hash)
        return False


# ======================================================================
# EMBEDDING INDEX DETECTION
# ======================================================================

_EMBEDDING_INDEX_CANDIDATES = ("embedding_index", "vector_index", "emb_idx")


def detect_embedding_index_column(df: pd.DataFrame) -> Optional[str]:
    for candidate in _EMBEDDING_INDEX_CANDIDATES:
        if candidate in df.columns:
            return candidate
    return None


def detect_named_vector_columns(df: pd.DataFrame) -> List[str]:
    """Detect columns that hold inline embedding vectors, supporting
    multiple named vectors (e.g. title_embedding, body_embedding).

    A column qualifies if its name ends with ``_embedding``/``_vector``
    or is exactly ``embedding``/``vector``, AND at least one non-null
    value in a sample parses as a numeric vector.
    """
    candidates = []
    for col in df.columns:
        low = col.lower()
        if low in ("embedding", "vector") or low.endswith(("_embedding", "_vector")):
            candidates.append(col)

    confirmed = []
    for col in candidates:
        sample = df[col].dropna().head(25)
        for value in sample:
            if _parse_vector_value(value) is not None:
                confirmed.append(col)
                break
    return confirmed


def _parse_vector_value(value: Any) -> Optional[List[float]]:
    if value is None or is_missing(value):
        return None
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, (list, tuple)):
        try:
            return [float(v) for v in value]
        except (TypeError, ValueError):
            return None
    if isinstance(value, str):
        parsed = _try_parse_structure(value.strip())
        if isinstance(parsed, (list, tuple)):
            try:
                return [float(v) for v in parsed]
            except (TypeError, ValueError):
                return None
    return None


def _vector_field_name(column: str) -> str:
    """Derive the Qdrant named-vector name from a source column name,
    e.g. 'title_embedding' -> 'title', 'embedding' -> 'default'."""
    low = column.lower()
    for suffix in ("_embedding", "_vector"):
        if low.endswith(suffix):
            return low[: -len(suffix)]
    if low in ("embedding", "vector"):
        return "default"
    return low


# ======================================================================
# MULTI-GRANULARITY MERGE (chunk-level 1:1 + document-level many:1)
# ======================================================================
#
# See the module docstring for the full explanation of why this is
# necessary. In short: a naive "join key must be unique in every
# frame" rule breaks the extremely common case of chunk-level sources
# (chunks.csv, intelligence.csv -- one row per chunk) plus a
# document-level metadata source (metadata.csv -- one row per
# document), because the shared key (document_id) is intentionally
# NOT unique in the chunk-level frames.
#
# The strategy implemented here instead:
#   1. Sorts frames largest-to-smallest by row count. The largest
#      frame(s) are, by definition, at the finest granularity (e.g.
#      chunks) and become the accumulating base.
#   2. For each remaining frame, finds a key column common with the
#      base that is unique WITHIN THAT FRAME (the "one" side of a
#      one-to-many or one-to-one relationship) and left-joins it onto
#      the base. This correctly handles both:
#        - chunk_id: unique in chunks.csv AND intelligence.csv -> 1:1 join
#        - document_id: unique in metadata.csv only -> many:1 join
#   3. If no such key exists, falls back to positional alignment ONLY
#      when the two frames have identical row counts (the only case in
#      which position-based zipping is unambiguous). Otherwise raises a
#      descriptive ``ValidationError`` rather than silently truncating
#      or misaligning rows.


def detect_join_key_for_frame(
    base_columns: Sequence[str],
    frame: pd.DataFrame,
    uniqueness_threshold: float = 0.98,
) -> Optional[str]:
    """Find the best column shared between ``base_columns`` and
    ``frame`` that is unique WITHIN ``frame`` itself.

    This intentionally does NOT require uniqueness in the base frame,
    which is what allows correct many-to-one joins (e.g. many chunk
    rows -> one document row via ``document_id``).
    """
    common = set(base_columns) & set(frame.columns)
    candidates: List[Tuple[float, str]] = []

    for col in common:
        is_legacy = col in _LEGACY_MERGE_KEY_CANDIDATES
        name_score = _name_score(col)
        if name_score == 0.0 and not is_legacy:
            continue
        try:
            uniq = _uniqueness_ratio(frame[col])
        except TypeError:
            continue
        if uniq < uniqueness_threshold:
            continue
        combined = max(name_score, 0.5 if is_legacy else 0.0) * 0.4 + uniq * 0.6
        candidates.append((combined, col))

    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0], reverse=True)
    return candidates[0][1]


def merge_sources(
    frames: Dict[str, pd.DataFrame],
    uniqueness_threshold: float = 0.98,
) -> pd.DataFrame:
    """Dynamically merge any number of named source dataframes,
    correctly supporting mixed granularities (e.g. per-chunk sources
    merged with a per-document metadata source).

    Args:
        frames: mapping of ``source_name -> dataframe``. Entries that
            are ``None`` or empty are ignored. Order matters only as a
            tie-breaker when multiple frames share the same (maximum)
            row count.
        uniqueness_threshold: minimum uniqueness ratio (within the
            *smaller* frame) for a shared column to be treated as a
            valid join key.

    Returns:
        The merged dataframe, at the granularity of the largest input
        frame (one row per finest-grained unit, e.g. per chunk), with
        every other frame's columns joined on.
    """
    present = {k: v for k, v in frames.items() if v is not None and not v.empty}
    if not present:
        raise ValidationError(
            what="Source merge",
            where="merge_sources",
            why="All provided source dataframes are empty or None",
            resolution="Provide at least one non-empty source dataframe/CSV",
        )

    if len(present) == 1:
        return next(iter(present.values())).reset_index(drop=True)

    # Largest frame first (ties broken by original insertion order) --
    # this is the finest-grained frame (e.g. chunks) and becomes the base
    # that every other frame is left-joined onto.
    ordered = sorted(present.items(), key=lambda kv: -len(kv[1]))
    base_name, merged = ordered[0]
    merged = merged.reset_index(drop=True)
    logger.info(
        "Using source '%s' (%d rows) as the merge base (finest granularity).",
        base_name, len(merged),
    )

    for name, frame in ordered[1:]:
        key = detect_join_key_for_frame(merged.columns, frame, uniqueness_threshold=uniqueness_threshold)

        if key:
            overlap = [c for c in frame.columns if c in merged.columns and c != key]
            frame_reduced = frame.drop(columns=overlap) if overlap else frame
            before_rows = len(merged)
            merged = merged.merge(frame_reduced, on=key, how="left")
            logger.info(
                "Merged source '%s' (%d rows) onto base via key '%s' -> %d rows.",
                name, len(frame), key, len(merged),
            )
            if len(merged) != before_rows:
                logger.warning(
                    "Row count changed while merging '%s' (%d -> %d rows). This "
                    "usually means '%s' has duplicate values for key '%s' causing "
                    "row fan-out -- verify this is expected.",
                    name, before_rows, len(merged), name, key,
                )
            if overlap:
                logger.info("Dropped overlapping columns from '%s' before merge: %s", name, overlap)
            continue

        # No safe join key found. Positional alignment is only safe when
        # the row counts already match exactly (unambiguous 1:1 zip).
        if len(frame) == len(merged):
            logger.warning(
                "No common unique key found between base and '%s'; both have "
                "%d rows, so aligning positionally by row order. Verify the "
                "row order in your source files actually corresponds.",
                name, len(frame),
            )
            overlap = [c for c in frame.columns if c in merged.columns]
            frame_reduced = frame.drop(columns=overlap) if overlap else frame
            merged = pd.concat(
                [merged.reset_index(drop=True), frame_reduced.reset_index(drop=True)], axis=1
            )
            continue

        raise ValidationError(
            what="Source merge",
            where="merge_sources",
            why=(
                f"No common, sufficiently-unique key column found to join source "
                f"'{name}' (shape {frame.shape}) onto the base frame '{base_name}' "
                f"(shape {merged.shape}), and their row counts differ, so "
                f"positional alignment would silently misalign or truncate data."
            ),
            resolution=(
                "Ensure the source files share a common identifier column whose "
                "values are unique within the smaller/more-aggregated file (e.g. "
                "'chunk_id' shared between chunk-level sources, or 'document_id' "
                "unique within a document-level metadata file). If your column "
                "uses a different name, rename it to something matching *_id/id/"
                "*_key, or pre-merge the files yourself before calling "
                "load_dataframe()."
            ),
        )

    return merged.reset_index(drop=True)


# ======================================================================
# CSV READING (optionally chunked, no hardcoded paths)
# ======================================================================

def _read_csv_safe(path: Union[str, Path], chunksize: Optional[int] = None) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:
    path = Path(path)
    if not path.exists():
        raise ValidationError(
            what="Source file check",
            where=f"_read_csv_safe({path})",
            why="Required source file does not exist on disk",
            resolution="Verify the path passed to the loader / your configuration is correct",
        )
    if chunksize:
        return pd.read_csv(path, low_memory=False, chunksize=chunksize)
    return pd.read_csv(path, low_memory=False)


def read_csv_in_chunks(path: Union[str, Path], chunksize: int) -> Iterator[pd.DataFrame]:
    """Stream a large CSV in chunks instead of loading it entirely into
    memory. Each yielded chunk can be passed independently through
    ``QdrantLoader.load_dataframe``."""
    for chunk in pd.read_csv(Path(path), low_memory=False, chunksize=chunksize):
        yield chunk


# ======================================================================
# 3 & 4. DATA / EMBEDDING VALIDATION
# ======================================================================

@dataclass
class ValidationReport:
    """Summary of pre-ingestion validation checks."""

    row_count: int
    column_count: int
    duplicate_columns: List[str]
    duplicate_ids: int
    missing_embedding_rows: int
    vector_dimension: Optional[int]
    mixed_dimensions_found: bool
    nan_vectors: int
    zero_vectors: int
    infinite_vectors: int
    duplicate_vectors: int
    warnings: List[str]


def validate_dataframe_structure(df: pd.DataFrame) -> List[str]:
    """Validate structural integrity of a dataframe before ingestion.

    Checks: duplicate columns, empty dataframe, and (if present) invalid
    JSON-like string fields that fail to parse when they look like they
    should be structured data. Raises ``ValidationError`` on hard
    failures and returns a list of non-fatal warnings.
    """
    warnings: List[str] = []

    if df is None:
        raise ValidationError(
            what="Dataframe validation", where="validate_dataframe_structure",
            why="Input dataframe is None", resolution="Pass a valid, non-None pandas DataFrame",
        )

    duplicate_columns = df.columns[df.columns.duplicated()].tolist()
    if duplicate_columns:
        raise ValidationError(
            what="Column validation", where="validate_dataframe_structure",
            why=f"Duplicate column names found: {duplicate_columns}",
            resolution="Rename or drop duplicate columns before ingestion",
        )

    if df.empty:
        warnings.append("Dataframe is empty (0 rows).")

    for col in df.columns:
        low = col.lower()
        if "json" in low or low.endswith(("_obj", "_data", "_meta")):
            sample = df[col].dropna().astype(str).head(50)
            bad = 0
            for value in sample:
                stripped = value.strip()
                if stripped.startswith(("{", "[")) and _try_parse_structure(stripped) is None:
                    bad += 1
            if bad:
                warnings.append(
                    f"Column '{col}' looks JSON-like but {bad}/{len(sample)} sampled "
                    f"values failed to parse; they will be stored as raw strings."
                )

    return warnings


def validate_ids(ids: Sequence[Any]) -> int:
    """Return the number of duplicate IDs found in a sequence of
    resolved point IDs (does not raise -- duplicates are handled by the
    DuplicateDetector during ingestion; this is purely diagnostic)."""
    seen: Set[Any] = set()
    duplicates = 0
    for i in ids:
        if i in seen:
            duplicates += 1
        else:
            seen.add(i)
    return duplicates


def validate_embeddings(
    embeddings: np.ndarray,
    expected_dimension: Optional[int] = None,
    check_duplicates: bool = False,
) -> Tuple[int, int, int, int, Optional[int]]:
    """Validate an embedding matrix.

    Checks: shape/dimension consistency, NaN values, infinite values,
    zero vectors, and (optionally, expensive) exact duplicate vectors.

    Returns:
        (nan_count, zero_count, inf_count, duplicate_count, dimension)

    Raises ``EmbeddingValidationError`` immediately on dimension
    mismatch or if the matrix has zero rows/columns -- these are hard
    stops, never silently skipped.
    """
    if embeddings is None or embeddings.size == 0:
        raise EmbeddingValidationError(
            what="Embedding validation", where="validate_embeddings",
            why="Embedding matrix is empty or None",
            resolution="Verify the embedding file/column actually contains vectors",
        )

    if embeddings.ndim != 2:
        raise EmbeddingValidationError(
            what="Embedding validation", where="validate_embeddings",
            why=f"Embedding matrix has unexpected shape {embeddings.shape} (expected 2D)",
            resolution="Ensure embeddings are stored as a 2D array of shape (n_rows, dim)",
        )

    dimension = int(embeddings.shape[1])
    if expected_dimension is not None and dimension != expected_dimension:
        raise EmbeddingValidationError(
            what="Embedding dimension check", where="validate_embeddings",
            why=f"Embedding dimension {dimension} does not match expected {expected_dimension}",
            resolution="Do not mix embeddings from different models/dimensions in one ingest run",
        )

    sample = embeddings if embeddings.shape[0] <= 200_000 else embeddings[:200_000]
    finite_mask = np.isfinite(sample.astype(np.float64, copy=False))
    nan_count = int(np.isnan(sample.astype(np.float64, copy=False)).any(axis=1).sum())
    inf_count = int(np.isinf(sample.astype(np.float64, copy=False)).any(axis=1).sum())
    zero_count = int((~sample.any(axis=1)).sum())

    duplicate_count = 0
    if check_duplicates:
        _, counts = np.unique(sample, axis=0, return_counts=True)
        duplicate_count = int((counts > 1).sum())

    return nan_count, zero_count, inf_count, duplicate_count, dimension


def run_full_validation(
    df: pd.DataFrame,
    embeddings: Optional[np.ndarray],
    key_columns: Optional[Sequence[str]],
    embedding_index_col: Optional[str] = None,
) -> ValidationReport:
    """Run all pre-ingestion validation checks and return a structured
    report. Raises on any hard failure; stops immediately."""
    warnings = validate_dataframe_structure(df)

    dimension: Optional[int] = None
    nan_vectors = zero_vectors = inf_vectors = dup_vectors = 0
    missing_embedding_rows = 0
    mixed_dimensions = False

    if embeddings is not None:
        nan_vectors, zero_vectors, inf_vectors, dup_vectors, dimension = validate_embeddings(embeddings)
        if embedding_index_col and embedding_index_col in df.columns:
            indices = pd.to_numeric(df[embedding_index_col], errors="coerce")
            missing_embedding_rows = int(indices.isna().sum())
            out_of_range = int(((indices < 0) | (indices >= embeddings.shape[0])).sum())
            missing_embedding_rows += out_of_range
            if embeddings.shape[0] != len(df):
                warnings.append(
                    f"Row count ({len(df)}) does not match embedding count "
                    f"({embeddings.shape[0]}); ensure the embedding index column "
                    f"correctly maps rows to vectors."
                )

    if key_columns:
        try:
            dup_ids = int(len(df) - df.drop_duplicates(subset=key_columns).shape[0])
        except Exception:
            dup_ids = 0
    else:
        dup_ids = 0

    report = ValidationReport(
        row_count=len(df),
        column_count=len(df.columns),
        duplicate_columns=[],
        duplicate_ids=dup_ids,
        missing_embedding_rows=missing_embedding_rows,
        vector_dimension=dimension,
        mixed_dimensions_found=mixed_dimensions,
        nan_vectors=nan_vectors,
        zero_vectors=zero_vectors,
        infinite_vectors=inf_vectors,
        duplicate_vectors=dup_vectors,
        warnings=warnings,
    )

    logger.info(
        "Validation summary: rows=%d cols=%d dup_ids=%d missing_emb_rows=%d "
        "nan_vectors=%d zero_vectors=%d inf_vectors=%d warnings=%d",
        report.row_count, report.column_count, report.duplicate_ids,
        report.missing_embedding_rows, report.nan_vectors, report.zero_vectors,
        report.infinite_vectors, len(report.warnings),
    )
    for w in warnings:
        logger.warning("Validation warning: %s", w)

    return report


# ======================================================================
# CHECKPOINTING
# ======================================================================

@dataclass
class Checkpoint:
    """Resumable upload progress, persisted to disk as JSON."""

    collection: str
    total_rows: int
    completed_upto: int = 0
    uploaded: int = 0
    skipped: int = 0
    duplicates: int = 0
    invalid_vectors: int = 0
    failed: int = 0
    batch_retries: int = 0

    @classmethod
    def load(cls, path: Path) -> Optional["Checkpoint"]:
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            known = {f: data[f] for f in cls.__dataclass_fields__ if f in data}
            return cls(**known)
        except Exception as exc:
            logger.warning("Could not parse checkpoint file %s (%s); ignoring.", path, exc)
            return None

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self)))


# ======================================================================
# 10. STRUCTURED STATISTICS OBJECT
# ======================================================================

@dataclass
class IngestionStats:
    """Structured result object returned by ``load_dataframe`` /
    ``load_from_sources`` instead of relying purely on log output."""

    collection: str
    total_rows: int = 0
    uploaded: int = 0
    duplicates: int = 0
    skipped: int = 0
    failed: int = 0
    invalid_vectors: int = 0
    batch_retries: int = 0
    elapsed_seconds: float = 0.0
    throughput_rows_per_sec: float = 0.0
    average_batch_seconds: float = 0.0
    eta_seconds: float = 0.0
    peak_memory_mb: Optional[float] = None
    started_at: str = ""
    finished_at: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _current_memory_mb() -> Optional[float]:
    if psutil is None:
        return None
    try:
        process = psutil.Process(os.getpid())
        return round(process.memory_info().rss / (1024 * 1024), 2)
    except Exception:
        return None


# ======================================================================
# 7. AUTOMATIC PAYLOAD INDEX CREATION
# ======================================================================

_INDEX_TYPE_MAP = {
    "keyword": PayloadSchemaType.KEYWORD,
    "integer": PayloadSchemaType.INTEGER,
    "float": PayloadSchemaType.FLOAT,
    "bool": PayloadSchemaType.BOOL,
    "datetime": PayloadSchemaType.DATETIME,
}


def infer_indexable_fields(
    sample_payloads: Sequence[Dict[str, Any]], max_text_len: int = 256
) -> Dict[str, str]:
    """Infer which flattened payload fields should get a Qdrant payload
    index, and with what schema type.

    Large free-text fields (chunk text, descriptions, reviews, etc.) are
    deliberately excluded -- indexing them would be expensive and
    provides little filtering value. A field is excluded if its name
    matches a text hint AND its sampled string values are long, or if
    its values are inconsistent in type across the sample.
    """
    field_types: Dict[str, Set[str]] = {}
    field_lengths: Dict[str, List[int]] = {}

    for payload in sample_payloads:
        for key, value in payload.items():
            if value is None:
                continue
            if isinstance(value, list):
                continue  # array fields are not indexed automatically
            vtype = _classify_value_type(value)
            field_types.setdefault(key, set()).add(vtype)
            if isinstance(value, str):
                field_lengths.setdefault(key, []).append(len(value))

    indexable: Dict[str, str] = {}
    for key, types in field_types.items():
        low = key.lower()

        if any(hint in low for hint in _LARGE_TEXT_HINTS):
            continue
        if low.endswith(("_embedding", "_vector")) or low in ("embedding", "vector"):
            continue
        if len(types) != 1:
            continue  # inconsistent type across rows -- unsafe to index

        vtype = next(iter(types))
        if vtype == "keyword":
            lengths = field_lengths.get(key, [])
            if lengths and (sum(lengths) / len(lengths)) > max_text_len:
                continue
        if vtype in _INDEX_TYPE_MAP:
            indexable[key] = vtype

    return indexable


# ======================================================================
# 8. COLLECTION VALIDATION (compatibility checks)
# ======================================================================

def validate_collection_compatibility(
    client: QdrantClient,
    collection_name: str,
    expected_size: int,
    expected_distance: Distance,
    named_vectors: Optional[Sequence[str]] = None,
) -> None:
    """Verify an existing collection is compatible with incoming data.

    Checks vector dimension, distance metric, and (for named-vector
    collections) that every expected named vector exists with a
    matching configuration. Raises ``CollectionCompatibilityError`` with
    a descriptive message on any mismatch.
    """
    try:
        info = client.get_collection(collection_name)
    except Exception as exc:
        raise CollectionCompatibilityError(
            what="Collection compatibility check",
            where="validate_collection_compatibility",
            why=f"Could not fetch collection info: {exc}",
            resolution="Verify the collection name and Qdrant connectivity",
        )

    vectors_config = info.config.params.vectors

    def _check_single(size: int, distance: Distance, label: str) -> None:
        if size != expected_size:
            raise CollectionCompatibilityError(
                what="Vector dimension check", where="validate_collection_compatibility",
                why=f"Collection '{collection_name}' vector '{label}' has size {size}, "
                    f"incoming data has size {expected_size}",
                resolution="Use a different collection name, recreate the collection, "
                           "or fix the embedding model producing mismatched dimensions",
            )
        if distance != expected_distance:
            raise CollectionCompatibilityError(
                what="Distance metric check", where="validate_collection_compatibility",
                why=f"Collection '{collection_name}' vector '{label}' uses distance "
                    f"{distance}, but {expected_distance} was requested",
                resolution="Match the distance metric to the existing collection, or recreate it",
            )

    if isinstance(vectors_config, dict):
        if named_vectors:
            for name in named_vectors:
                if name not in vectors_config:
                    raise CollectionCompatibilityError(
                        what="Named vector check", where="validate_collection_compatibility",
                        why=f"Collection '{collection_name}' has no named vector '{name}'",
                        resolution="Recreate the collection with all required named vectors, "
                                   "or rename the incoming vector field",
                    )
                vp = vectors_config[name]
                _check_single(vp.size, vp.distance, name)
    else:
        _check_single(vectors_config.size, vectors_config.distance, "default")

    logger.info("Collection '%s' compatibility verified.", collection_name)


# ======================================================================
# 9. UPLOAD VERIFICATION
# ======================================================================

def verify_upload_integrity(stats: IngestionStats, actual_collection_count: Optional[int]) -> None:
    """Cross-check ingestion statistics against the collection's actual
    point count and raise ``UploadIntegrityError`` if they diverge
    beyond what duplicates/skips can account for."""
    accounted_for = stats.uploaded + stats.duplicates + stats.skipped + stats.failed + stats.invalid_vectors
    if accounted_for != stats.total_rows:
        raise UploadIntegrityError(
            what="Upload integrity verification", where="verify_upload_integrity",
            why=(
                f"Row accounting mismatch: uploaded({stats.uploaded}) + "
                f"duplicates({stats.duplicates}) + skipped({stats.skipped}) + "
                f"failed({stats.failed}) + invalid_vectors({stats.invalid_vectors}) = "
                f"{accounted_for}, but total_rows was {stats.total_rows}"
            ),
            resolution="Inspect the failed_rows log and re-run ingestion; do not trust "
                       "the collection state until this is resolved",
        )

    if actual_collection_count is not None and actual_collection_count < stats.uploaded:
        logger.warning(
            "Collection point count (%d) is lower than the number of points this "
            "run reported as uploaded (%d); this can happen with concurrent writers "
            "or upserts overwriting existing IDs -- verify manually if unexpected.",
            actual_collection_count, stats.uploaded,
        )


# ======================================================================
# QDRANT LOADER
# ======================================================================

class QdrantLoader:
    """Universal, schema-agnostic ingestion engine for Qdrant.

    Works with any tabular dataset without code changes: identifier
    columns, merge keys, vector fields, and payload index fields are all
    detected dynamically at runtime. Behavior is fully driven by
    ``LoaderConfig`` (or constructor overrides) -- no hardcoded paths,
    collection names, or batch sizes.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        collection_name: Optional[str] = None,
        failed_rows_log: Optional[Union[str, Path]] = None,
        config: Optional[LoaderConfig] = None,
    ):
        self.config = config or LoaderConfig()

        self.collection_name = collection_name or self.config.collection_name
        self.checkpoint_path = Path(checkpoint_path or self.config.checkpoint_path)
        self.failed_rows_log = Path(failed_rows_log or self.config.failed_rows_log)

        configure_logging(self.config.log_file)

        self.client = QdrantClient(
            host=host or self.config.host,
            port=int(port or self.config.port),
            timeout=120,
        )

    # =========================================================
    # CREATE COLLECTION
    # =========================================================

    def create_collection(
        self,
        vector_size: Union[int, Dict[str, int]],
        distance: Optional[Distance] = None,
        recreate: bool = False,
    ) -> None:
        """Create the collection if it does not exist.

        ``vector_size`` may be a single int (single unnamed vector) or a
        dict of ``{vector_name: size}`` for named-vector collections. If
        the collection already exists, its configuration is validated
        for compatibility (unless disabled via config), raising a
        descriptive exception on mismatch.
        """
        distance = distance or self.config.distance_metric

        if self.collection_exists():
            if recreate:
                self.delete_collection()
            else:
                if self.config.validate_collection_compatibility:
                    if isinstance(vector_size, dict):
                        validate_collection_compatibility(
                            self.client, self.collection_name,
                            expected_size=next(iter(vector_size.values())),
                            expected_distance=distance,
                            named_vectors=list(vector_size.keys()),
                        )
                    else:
                        validate_collection_compatibility(
                            self.client, self.collection_name,
                            expected_size=vector_size, expected_distance=distance,
                        )
                logger.info("Collection exists and is compatible: %s", self.collection_name)
                return

        if isinstance(vector_size, dict):
            vectors_config = {
                name: VectorParams(size=size, distance=distance)
                for name, size in vector_size.items()
            }
        else:
            vectors_config = VectorParams(size=vector_size, distance=distance)

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=vectors_config,
        )
        logger.info("Collection created: %s (vectors=%s)", self.collection_name, vector_size)

    # =========================================================
    # BUILD PAYLOAD (dynamic, type-aware)
    # =========================================================

    def build_payload(
        self,
        row_values: Sequence[Any],
        columns: Sequence[str],
        raw_preserve_fields: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        """Build a type-aware, flattened, sanitized payload for one row.

        Args:
            row_values: Positional values for a single row (e.g. from
                ``DataFrame.itertuples(index=False, name=None)``).
            columns: Column names, aligned positionally with ``row_values``.
            raw_preserve_fields: Optional extra set of column names whose
                exact string representation must be preserved (used for
                dynamically detected identifier columns, in addition to
                the built-in legacy defaults).

        Returns:
            A JSON/Qdrant-safe payload dict. Missing values are skipped
            entirely; identifier-like fields keep their exact string
            representation; every other field is dynamically type-inferred.
        """
        preserve = _RAW_PRESERVE_FIELDS if raw_preserve_fields is None else (_RAW_PRESERVE_FIELDS | raw_preserve_fields)
        payload: Dict[str, Any] = {}

        for col, value in zip(columns, row_values):
            if is_missing(value):
                continue

            if col in preserve:
                payload[col] = str(value)
                continue

            try:
                native = infer_native(value)
            except Exception:
                native = str(value)

            if is_missing(native):
                continue

            payload[col] = native

        return flatten_payload(sanitize_payload(payload))

    # =========================================================
    # LOAD FROM RAW SOURCE FILES
    # =========================================================

    def load_from_sources(
        self,
        chunks_csv: Optional[Union[str, Path]] = None,
        intelligence_csv: Optional[Union[str, Path]] = None,
        metadata_csv: Optional[Union[str, Path]] = None,
        unstructured_csv: Optional[Union[str, Path]] = None,
        embedding_file: Optional[Union[str, Path]] = None,
        extra_sources: Optional[Dict[str, Union[str, Path]]] = None,
        batch_size: Optional[int] = None,
        parallel: Optional[int] = None,
        resume: bool = True,
        max_batch_retries: Optional[int] = None,
    ) -> IngestionStats:
        """Read and dynamically merge available source CSVs, then ingest.

        Every path is optional and must be supplied by the caller or
        configuration -- there are no built-in default paths. At least
        one of ``chunks_csv``, ``metadata_csv``, or ``intelligence_csv``
        (or an entry in ``extra_sources``) must be provided.

        Typical layout this is designed for:
            chunks_csv        -- one row PER CHUNK (chunk text + chunk_id
                                  + document_id, embeddings usually keyed
                                  off this file's row order)
            intelligence_csv  -- one row PER CHUNK (precomputed
                                  intelligence/enrichment, keyed by
                                  chunk_id, usually also carrying
                                  document_id)
            metadata_csv       -- one row PER DOCUMENT (document-level
                                  metadata, keyed by document_id -- there
                                  will always be fewer metadata rows than
                                  chunk rows, since each document is
                                  split into multiple chunks)

        These are merged via ``merge_sources``, which correctly performs
        a 1:1 join between chunk-level sources and a many:1 join of the
        document-level metadata onto every chunk -- see the module
        docstring for details. ``unstructured_csv`` is accepted as a
        deprecated alias for ``chunks_csv`` for backward compatibility
        with older pipelines. ``extra_sources`` allows passing any
        additional named CSVs (e.g. a second enrichment file) without
        requiring code changes.
        """
        chunks_csv = chunks_csv or unstructured_csv

        sources: Dict[str, Union[str, Path]] = {}
        if chunks_csv:
            sources["chunks"] = chunks_csv
        if intelligence_csv:
            sources["intelligence"] = intelligence_csv
        if metadata_csv:
            sources["metadata"] = metadata_csv
        if extra_sources:
            sources.update(extra_sources)

        if not sources:
            raise ValidationError(
                what="Source configuration", where="load_from_sources",
                why="No source CSV paths were provided",
                resolution=(
                    "Pass at least one of chunks_csv/metadata_csv/intelligence_csv "
                    "(or an entry in extra_sources)"
                ),
            )

        logger.info("Reading source CSVs: %s", list(sources.keys()))
        frames = {name: _read_csv_safe(path) for name, path in sources.items()}
        for name, frame in frames.items():
            logger.info("Source '%s': %d rows x %d columns", name, len(frame), len(frame.columns))

        merged_df = merge_sources(frames, uniqueness_threshold=self.config.merge_key_uniqueness_threshold)
        logger.info("Merged dataset: %d rows x %d columns", len(merged_df), len(merged_df.columns))

        return self.load_dataframe(
            merged_df,
            embedding_file=embedding_file,
            batch_size=batch_size,
            parallel=parallel,
            resume=resume,
            max_batch_retries=max_batch_retries,
        )

    # =========================================================
    # LOAD DATAFRAME (core ingestion pipeline)
    # =========================================================

    def load_dataframe(
        self,
        metadata_df: pd.DataFrame,
        embedding_file: Optional[Union[str, Path]] = None,
        batch_size: Optional[int] = None,
        parallel: Optional[int] = None,
        resume: bool = True,
        max_batch_retries: Optional[int] = None,
    ) -> IngestionStats:
        """Ingest a pre-merged dataframe (+ embeddings) into Qdrant.

        Fully dynamic: identifier columns, named vector columns, and
        payload index fields are all detected automatically. Handles
        validation, deduplication (in-memory, against both the current
        run and the existing collection), deterministic point IDs,
        batch upload with retry, checkpointed resume, automatic payload
        index creation, and post-upload verification. Returns a
        structured ``IngestionStats`` object.

        The incoming dataframe is expected to be at the finest
        granularity you want stored as individual points (typically one
        row per chunk). If it was produced by ``load_from_sources``,
        document-level metadata columns will already be repeated across
        every chunk row belonging to that document -- this is correct
        and intentional, not a bug.
        """
        batch_size = int(batch_size or self.config.batch_size)
        parallel = int(parallel or self.config.parallel)
        max_batch_retries = int(max_batch_retries or self.config.max_batch_retries)

        if metadata_df is None:
            raise ValidationError(
                what="Dataframe check", where="load_dataframe",
                why="metadata_df is None", resolution="Pass a valid, non-None pandas DataFrame",
            )
        if metadata_df.empty:
            logger.info("Empty dataframe — nothing to upload.")
            return IngestionStats(collection=self.collection_name)

        df = metadata_df.reset_index(drop=True)
        started_at = datetime.utcnow().isoformat()

        # ------------------------------------------------------------
        # Identifier / merge-key style detection for this dataset.
        # ------------------------------------------------------------
        key_columns, key_strategy = detect_primary_key(df)
        if key_strategy != "none":
            logger.info("Detected primary key (%s): %s", key_strategy, key_columns)
        else:
            logger.info("No reliable identifier column found; deterministic hash-based IDs will be used.")

        raw_preserve = set(key_columns) if key_columns else set()

        # ------------------------------------------------------------
        # Vector source resolution: supports single or multiple named
        # inline embedding columns, with embeddings.npy as a fallback
        # for single-vector datasets that store vectors out-of-band.
        # ------------------------------------------------------------
        named_vector_columns = detect_named_vector_columns(df)
        embedding_index_col: Optional[str] = None
        embeddings_matrix = None
        n_vectors: Optional[int] = None
        vector_sizes: Dict[str, int] = {}

        if named_vector_columns:
            logger.info("Inline vector column(s) detected: %s (priority over embeddings.npy)", named_vector_columns)
            for col in named_vector_columns:
                vname = _vector_field_name(col)
                vector_sizes[vname] = self._first_valid_inline_vector_size(df[col])
        else:
            embedding_index_col = detect_embedding_index_column(df)
            if embedding_index_col is None:
                logger.warning(
                    "No embedding index column found (checked %s); falling back "
                    "to row position as the embedding index.",
                    _EMBEDDING_INDEX_CANDIDATES,
                )
                df = df.copy()
                df["__embedding_index__"] = np.arange(len(df))
                embedding_index_col = "__embedding_index__"

            if embedding_file is None:
                raise ValidationError(
                    what="Embedding source resolution", where="load_dataframe",
                    why="No inline embedding column was found and no embedding_file was provided",
                    resolution="Provide an inline '*_embedding' column or pass embedding_file=path/to/embeddings.npy",
                )

            logger.info("Loading embeddings from disk: %s", embedding_file)
            embeddings_matrix = np.load(embedding_file, mmap_mode="r")
            vector_sizes["default"] = int(embeddings_matrix.shape[1])
            n_vectors = int(embeddings_matrix.shape[0])
            logger.info("Embedding matrix: %s", embeddings_matrix.shape)

        # ------------------------------------------------------------
        # 3 & 4. Full data + embedding validation -- stop immediately
        # on any hard failure.
        # ------------------------------------------------------------
        run_full_validation(
            df,
            embeddings_matrix,
            key_columns,
            embedding_index_col=embedding_index_col if not named_vector_columns else None,
        )

        # ------------------------------------------------------------
        # 8. Collection creation / compatibility validation.
        # ------------------------------------------------------------
        if len(vector_sizes) == 1 and "default" in vector_sizes:
            self.create_collection(vector_sizes["default"])
        else:
            self.create_collection(vector_sizes)

        embedding_col_indices = {col: df.columns.get_loc(col) for col in named_vector_columns}
        columns = df.columns.tolist()
        payload_columns = [c for c in columns if c not in named_vector_columns]
        total_rows = len(df)

        checkpoint = Checkpoint.load(self.checkpoint_path) if resume else None
        if checkpoint and (
            checkpoint.collection != self.collection_name or checkpoint.total_rows != total_rows
        ):
            logger.warning("Checkpoint does not match this run; starting fresh.")
            checkpoint = None
        if checkpoint is None:
            checkpoint = Checkpoint(collection=self.collection_name, total_rows=total_rows)

        start_row = checkpoint.completed_upto
        if start_row:
            logger.info("Resuming upload from row %d/%d", start_row, total_rows)

        logger.info("Loading existing identifiers/hashes for duplicate detection...")
        existing_hashes, existing_ids = self._load_existing_dedup_state()
        detector = DuplicateDetector(strategy=self.config.duplicate_strategy)
        detector.seed_existing(ids=existing_ids, hashes=existing_hashes)
        logger.info("Loaded %d existing dedup fingerprints from collection", len(existing_hashes))

        uploaded = checkpoint.uploaded
        skipped = checkpoint.skipped
        duplicates = checkpoint.duplicates
        invalid_vectors = checkpoint.invalid_vectors
        failed = checkpoint.failed
        batch_retries_total = checkpoint.batch_retries

        sample_payloads: List[Dict[str, Any]] = []

        logger.info(
            "Uploading %s rows (batch=%s, parallel=%s)...",
            f"{total_rows:,}", f"{batch_size:,}", parallel,
        )

        t_start = time.perf_counter()
        batch_durations: List[float] = []

        for batch_start in range(start_row, total_rows, batch_size):
            batch_t0 = time.perf_counter()
            batch_end = min(batch_start + batch_size, total_rows)
            batch_df = df.iloc[batch_start:batch_end]

            points: List[PointStruct] = []
            row_index_map: Dict[Any, int] = {}

            for offset, row_values in enumerate(batch_df.itertuples(index=False, name=None)):
                row_index = batch_start + offset

                if named_vector_columns:
                    payload_values = tuple(
                        v for i, v in enumerate(row_values) if columns[i] not in named_vector_columns
                    )
                else:
                    payload_values = row_values

                try:
                    payload = self.build_payload(payload_values, payload_columns, raw_preserve_fields=raw_preserve)
                except Exception as exc:
                    skipped += 1
                    self._log_failed_row(row_index, None, f"bad payload: {exc}")
                    continue

                vector_obj: Union[List[float], np.ndarray, Dict[str, Any]]
                dimension_ok = True

                if named_vector_columns:
                    named_vectors: Dict[str, Any] = {}
                    for col in named_vector_columns:
                        vname = _vector_field_name(col)
                        raw_val = row_values[embedding_col_indices[col]]
                        parsed = _parse_vector_value(raw_val)
                        if parsed is None:
                            invalid_vectors += 1
                            skipped += 1
                            self._log_failed_row(row_index, None, f"missing/unparsable vector in '{col}'")
                            dimension_ok = False
                            break
                        if len(parsed) != vector_sizes[vname]:
                            invalid_vectors += 1
                            skipped += 1
                            self._log_failed_row(
                                row_index, None,
                                f"vector dimension mismatch in '{col}' "
                                f"(expected {vector_sizes[vname]}, got {len(parsed)})",
                            )
                            dimension_ok = False
                            break
                        named_vectors[vname] = parsed
                    if not dimension_ok:
                        continue
                    vector_obj = named_vectors if len(named_vectors) > 1 else next(iter(named_vectors.values()))
                else:
                    emb_idx = self._safe_int(payload.get(embedding_index_col))
                    if emb_idx is None or n_vectors is None or not (0 <= emb_idx < n_vectors):
                        invalid_vectors += 1
                        skipped += 1
                        self._log_failed_row(row_index, None, "invalid embedding index")
                        continue
                    vector_obj = embeddings_matrix[emb_idx]
                    if len(vector_obj) != vector_sizes["default"]:
                        invalid_vectors += 1
                        skipped += 1
                        self._log_failed_row(
                            row_index, None,
                            f"vector dimension mismatch (expected {vector_sizes['default']}, got {len(vector_obj)})",
                        )
                        continue

                text_hash = compute_text_hash(payload)
                point_id = resolve_point_id(payload, text_hash, key_columns=key_columns)

                if detector.is_duplicate_and_record(point_id, payload, text_hash):
                    duplicates += 1
                    continue

                payload.setdefault("text_hash", text_hash)
                row_index_map[point_id] = row_index

                if len(sample_payloads) < 500:
                    sample_payloads.append(payload)

                points.append(PointStruct(id=point_id, vector=vector_obj, payload=payload))

            if points:
                ok, retries_used = self._upload_batch_with_retry(points, max_batch_retries, parallel, row_index_map)
                uploaded += ok
                failed += len(points) - ok
                batch_retries_total += retries_used

            checkpoint.completed_upto = batch_end
            checkpoint.uploaded = uploaded
            checkpoint.skipped = skipped
            checkpoint.duplicates = duplicates
            checkpoint.invalid_vectors = invalid_vectors
            checkpoint.failed = failed
            checkpoint.batch_retries = batch_retries_total
            checkpoint.save(self.checkpoint_path)

            batch_durations.append(time.perf_counter() - batch_t0)
            self._log_progress(
                batch_end, total_rows, uploaded, skipped, duplicates, invalid_vectors, failed, t_start
            )

        total_elapsed = time.perf_counter() - t_start

        # ------------------------------------------------------------
        # 7. Automatic payload index creation.
        # ------------------------------------------------------------
        if self.config.create_payload_indexes and sample_payloads:
            self._create_payload_indexes(sample_payloads)

        stats = IngestionStats(
            collection=self.collection_name,
            total_rows=total_rows,
            uploaded=uploaded,
            duplicates=duplicates,
            skipped=skipped,
            failed=failed,
            invalid_vectors=invalid_vectors,
            batch_retries=batch_retries_total,
            elapsed_seconds=round(total_elapsed, 3),
            throughput_rows_per_sec=round(uploaded / total_elapsed, 2) if total_elapsed > 0 else 0.0,
            average_batch_seconds=round(sum(batch_durations) / len(batch_durations), 3) if batch_durations else 0.0,
            eta_seconds=0.0,
            peak_memory_mb=_current_memory_mb(),
            started_at=started_at,
            finished_at=datetime.utcnow().isoformat(),
        )

        logger.info(
            "Finished. uploaded=%d skipped=%d duplicates=%d invalid_vectors=%d failed=%d "
            "in %.1fm (%.1f rows/s)",
            stats.uploaded, stats.skipped, stats.duplicates, stats.invalid_vectors,
            stats.failed, total_elapsed / 60, stats.throughput_rows_per_sec,
        )

        # ------------------------------------------------------------
        # 9. Upload verification.
        # ------------------------------------------------------------
        if self.config.verify_upload:
            actual_count = None
            try:
                actual_count = self.count()
            except Exception as exc:
                logger.warning("Could not fetch collection count for verification: %s", exc)
            verify_upload_integrity(stats, actual_count)

        if checkpoint.completed_upto >= total_rows:
            self._clear_checkpoint()

        return stats

    # =========================================================
    # PAYLOAD INDEX CREATION
    # =========================================================

    def _create_payload_indexes(self, sample_payloads: Sequence[Dict[str, Any]]) -> None:
        indexable = infer_indexable_fields(sample_payloads)
        if not indexable:
            logger.info("No payload fields qualified for automatic indexing.")
            return

        for field_name, vtype in indexable.items():
            schema_type = _INDEX_TYPE_MAP.get(vtype)
            if schema_type is None:
                continue

            for attempt in range(3):
                try:
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name=field_name,
                        field_schema=schema_type,
                    )
                    logger.info("Created payload index: %s (%s)", field_name, vtype)
                    break
                except Exception as exc:
                    if attempt == 2:
                        logger.warning("Could not create payload index for '%s': %s", field_name, exc)
                    else:
                        logger.warning(
                            "Payload index attempt %d/3 failed for '%s': %s -- retrying",
                            attempt + 1, field_name, exc,
                        )
                        time.sleep(2 ** attempt)

    # =========================================================
    # HELPERS
    # =========================================================

    @staticmethod
    def _safe_int(value: Any) -> Optional[int]:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _first_valid_inline_vector_size(self, series: pd.Series) -> int:
        for value in series:
            vector = _parse_vector_value(value)
            if vector is not None:
                return len(vector)
        raise ValidationError(
            what="Inline vector size detection", where="_first_valid_inline_vector_size",
            why="Column contains no valid/parsable vector values",
            resolution="Verify the embedding column contains numeric arrays or JSON-encoded arrays",
        )

    def _log_failed_row(self, row_index: Optional[int], point_id: Any, reason: str) -> None:
        entry = (
            f"{datetime.utcnow().isoformat()} | row={row_index} "
            f"| point_id={point_id} | reason={reason}\n"
        )
        try:
            self.failed_rows_log.parent.mkdir(parents=True, exist_ok=True)
            with open(self.failed_rows_log, "a", encoding="utf-8") as fh:
                fh.write(entry)
        except OSError as exc:
            logger.warning("Could not write to failed rows log %s: %s", self.failed_rows_log, exc)

    @staticmethod
    def _log_progress(
        batch_end: int, total_rows: int, uploaded: int, skipped: int,
        duplicates: int, invalid_vectors: int, failed: int, t_start: float,
    ) -> None:
        elapsed = time.perf_counter() - t_start
        rate = uploaded / elapsed if elapsed > 0 else 0.0
        remaining = total_rows - batch_end
        eta_minutes = (remaining / rate / 60) if rate > 0 else 0.0

        logger.info(
            "Progress %s/%s | uploaded=%d skipped=%d duplicates=%d invalid_vectors=%d failed=%d "
            "| %.1f rows/s | ETA %.1fm",
            f"{batch_end:,}", f"{total_rows:,}", uploaded, skipped,
            duplicates, invalid_vectors, failed, rate, eta_minutes,
        )

    # =========================================================
    # BATCH / POINT UPLOAD WITH RETRY
    # =========================================================

    def _upload_batch_with_retry(
        self,
        points: List[PointStruct],
        max_retries: int,
        parallel: int,
        row_index_map: Optional[Dict[Any, int]] = None,
    ) -> Tuple[int, int]:
        """Retry a full batch upload with exponential backoff; on
        persistent failure, fall back to per-point upload so only the
        genuinely bad rows are skipped instead of the whole batch.

        Returns ``(successful_count, retries_used)``.
        """
        retries_used = 0
        for attempt in range(1, max_retries + 1):
            try:
                self.client.upload_points(
                    collection_name=self.collection_name,
                    points=points,
                    parallel=parallel,
                    max_retries=1,
                    wait=True,
                )
                return len(points), retries_used
            except Exception as exc:
                retries_used += 1
                logger.warning(
                    "Batch upload attempt %d/%d failed (%d points): %s",
                    attempt, max_retries, len(points), exc,
                )
                if attempt < max_retries:
                    time.sleep(2 ** (attempt - 1))

        logger.warning(
            "Batch upload failed after %d attempts; retrying %d points individually.",
            max_retries, len(points),
        )
        ok = self._upload_points_individually(points, row_index_map)
        return ok, retries_used

    def _upload_points_individually(
        self, points: List[PointStruct], row_index_map: Optional[Dict[Any, int]] = None
    ) -> int:
        ok = 0
        for point in points:
            try:
                self.client.upsert(collection_name=self.collection_name, points=[point], wait=True)
                ok += 1
            except Exception as exc:
                row_index = (row_index_map or {}).get(point.id)
                logger.error("Skipping bad point id=%s due to error: %s", point.id, exc)
                self._log_failed_row(row_index, point.id, f"upload failed: {exc}")
        return ok

    # =========================================================
    # CHECKPOINT HELPERS
    # =========================================================

    def _clear_checkpoint(self) -> None:
        try:
            if self.checkpoint_path.exists():
                self.checkpoint_path.unlink()
        except OSError as exc:
            logger.warning("Could not remove checkpoint file %s: %s", self.checkpoint_path, exc)

    # =========================================================
    # EXISTING-STATE LOADING (bulk duplicate detection)
    # =========================================================

    def _load_existing_dedup_state(self) -> Tuple[Set[str], Set[Any]]:
        """Scroll the existing collection into memory to build dedup
        fingerprint sets (``text_hash`` payload values and point IDs),
        used for O(1) duplicate lookups during ingest."""
        hashes: Set[str] = set()
        ids: Set[Any] = set()

        if not self.collection_exists():
            return hashes, ids

        try:
            if self.count() == 0:
                return hashes, ids
        except Exception:
            return hashes, ids

        next_offset = None
        while True:
            records, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                with_payload=["text_hash"],
                with_vectors=False,
                limit=10000,
                offset=next_offset,
            )
            for record in records:
                text_hash = (record.payload or {}).get("text_hash")
                if text_hash:
                    hashes.add(text_hash)
                ids.add(record.id)
            if next_offset is None:
                break

        return hashes, ids

    # =========================================================
    # SEARCH (vector + generic metadata filters)
    # =========================================================

    def search(
        self,
        query_vector,
        limit: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        vector_name: Optional[str] = None,
    ):
        """Hybrid search: vector similarity + optional metadata filters.

        Args:
            query_vector: The query embedding.
            limit: Number of results to return.
            filters: Optional dict of ``field -> spec``. Supported specs:
                - scalar / bool -> exact match
                - ``(min, max)`` tuple -> inclusive range
                - list/tuple/set -> match-any / "contains"
                - ``{"gte": ..., "lte": ..., "gt": ..., "lt": ...}`` -> range
                - ``{"any": [...]}`` -> explicit match-any
                - ``{"except": [...]}`` -> match-except
                - dotted keys (e.g. ``"entities.type"``) -> nested field
                  filtering, since payloads are stored flattened.
            vector_name: For collections with multiple named vectors,
                the name of the vector to search against.
        """
        qdrant_filter = self._build_filter(filters)

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            using=vector_name,
            query_filter=qdrant_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        return response.points

    def _build_filter(self, filters: Optional[Dict[str, Any]]) -> Optional[Filter]:
        if not filters:
            return None

        conditions = []
        for field_key, value in filters.items():
            condition = self._build_condition(field_key, value)
            if condition is not None:
                conditions.append(condition)

        return Filter(must=conditions) if conditions else None

    @staticmethod
    def _build_condition(field_key: str, value: Any) -> Optional[FieldCondition]:
        if value is None:
            return None

        if isinstance(value, tuple) and len(value) == 2:
            return FieldCondition(key=field_key, range=Range(gte=value[0], lte=value[1]))

        if isinstance(value, dict):
            if any(k in value for k in ("gt", "gte", "lt", "lte")):
                return FieldCondition(
                    key=field_key,
                    range=Range(
                        gt=value.get("gt"), gte=value.get("gte"),
                        lt=value.get("lt"), lte=value.get("lte"),
                    ),
                )
            if "any" in value:
                return FieldCondition(key=field_key, match=MatchAny(any=list(value["any"])))
            if "except" in value:
                return FieldCondition(
                    key=field_key, match=MatchExcept(**{"except": list(value["except"])})
                )
            raise ValidationError(
                what="Filter validation", where="_build_condition",
                why=f"Unsupported filter specification for '{field_key}': {value!r}",
                resolution="Use a scalar, (min, max) tuple, list, or one of the "
                           "supported dict specs: gt/gte/lt/lte, any, except",
            )

        if isinstance(value, (list, tuple, set)):
            return FieldCondition(key=field_key, match=MatchAny(any=list(value)))

        return FieldCondition(key=field_key, match=MatchValue(value=value))

    # =========================================================
    # COUNT
    # =========================================================

    def count(self) -> int:
        info = self.client.get_collection(self.collection_name)
        return info.points_count

    # =========================================================
    # COLLECTION EXISTS
    # =========================================================

    def collection_exists(self) -> bool:
        try:
            return bool(self.client.collection_exists(self.collection_name))
        except AttributeError:
            collections = [c.name for c in self.client.get_collections().collections]
            return self.collection_name in collections

    # =========================================================
    # DELETE COLLECTION
    # =========================================================

    def delete_collection(self) -> None:
        self.client.delete_collection(collection_name=self.collection_name)
        self._clear_checkpoint()
        logger.info("Deleted: %s", self.collection_name)

    # =========================================================
    # COLLECTION INFO
    # =========================================================

    def collection_info(self):
        return self.client.get_collection(self.collection_name)


# ======================================================================
# CLI ENTRY POINT
# ======================================================================
#
# Allows the module to be run directly, e.g.:
#
#   python qdrant_loader.py --action ingest \
#       --chunks-csv data/chunks.csv \
#       --intelligence-csv data/intelligence.csv \
#       --metadata-csv data/metadata.csv \
#       --embedding-file data/embeddings.npy \
#       --collection my_collection --host localhost --port 6333
#
#   python qdrant_loader.py --action ingest --metadata-csv data/chunks_only.csv \
#       --embedding-file data/embeddings.npy --collection my_collection
#
#   python qdrant_loader.py --action search --collection my_collection \
#       --query-vector-file data/query.npy --limit 5
#
#   python qdrant_loader.py --action info --collection my_collection
#   python qdrant_loader.py --action count --collection my_collection
#   python qdrant_loader.py --action delete --collection my_collection
#
# All CLI flags are optional overrides on top of LoaderConfig defaults
# (env vars / Config.settings / built-in fallbacks) -- nothing here is
# hardcoded to a specific project layout.
#
# NOTE: --metadata-csv is reused as the single-file input when no other
# source flags are given (e.g. one pre-merged CSV), for backward
# compatibility with older invocations of this CLI.

def _build_arg_parser() -> "argparse.ArgumentParser":
    import argparse

    parser = argparse.ArgumentParser(
        prog="qdrant_loader",
        description=(
            "Universal, schema-agnostic Qdrant ingestion engine. "
            "Single entry point -- behavior is selected via --action "
            "(default: ingest)."
        ),
    )

    # ---- connection / runtime configuration -----------------------
    parser.add_argument("--host", default=None, help="Qdrant host (default: config/env/localhost)")
    parser.add_argument("--port", type=int, default=None, help="Qdrant port (default: config/env/6333)")
    parser.add_argument("--collection", default=None, help="Target collection name")
    parser.add_argument("--checkpoint-path", default=None, help="Path to checkpoint JSON file")
    parser.add_argument("--failed-rows-log", default=None, help="Path to failed-rows log file")
    parser.add_argument("--log-file", default=None, help="Optional path to also log to a file")
    parser.add_argument("--batch-size", type=int, default=None, help="Upload batch size")
    parser.add_argument("--parallel", type=int, default=None, help="Parallel upload workers")
    parser.add_argument("--max-batch-retries", type=int, default=None, help="Max retries per batch")
    parser.add_argument(
        "--duplicate-strategy", default=None,
        choices=sorted(DuplicateDetector.VALID_STRATEGIES),
        help="Duplicate detection strategy",
    )
    parser.add_argument("--no-verify-upload", action="store_true", help="Skip post-upload integrity verification")
    parser.add_argument("--no-payload-index", action="store_true", help="Skip automatic payload index creation")
    parser.add_argument(
        "--no-collection-validation", action="store_true",
        help="Skip existing-collection compatibility validation",
    )

    # ---- single action selector (replaces subcommands) -------------
    parser.add_argument(
        "--action", default="ingest",
        choices=["ingest", "search", "info", "count", "delete", "exists"],
        help=(
            "What to do (default: ingest). 'ingest' auto-detects whether to run "
            "load_from_sources (if any of --chunks-csv/--intelligence-csv is given "
            "alongside/instead of --metadata-csv) or load_dataframe on a single "
            "pre-merged CSV (if only --metadata-csv is given)."
        ),
    )

    # ---- ingestion inputs -------------------------------------------
    parser.add_argument(
        "--chunks-csv", default=None,
        help="Path to the per-chunk unstructured/raw-text CSV (one row per chunk)",
    )
    parser.add_argument(
        "--unstructured-csv", default=None,
        help="Deprecated alias for --chunks-csv",
    )
    parser.add_argument(
        "--metadata-csv", default=None,
        help="Path to the document-level metadata CSV, OR a single pre-merged CSV "
             "if it's the only source flag given",
    )
    parser.add_argument(
        "--intelligence-csv", default=None,
        help="Path to the per-chunk precomputed intelligence/enrichment CSV",
    )
    parser.add_argument("--embedding-file", default=None, help="Path to an embeddings .npy file")
    parser.add_argument("--no-resume", action="store_true", help="Ignore any existing checkpoint")
    parser.add_argument("--csv-chunksize", type=int, default=None, help="Stream --metadata-csv in chunks of this size")

    # ---- search inputs ------------------------------------------------
    parser.add_argument("--query-vector-file", default=None, help="Path to a .npy file containing one query vector")
    parser.add_argument("--limit", type=int, default=5, help="Number of search results to return")
    parser.add_argument("--vector-name", default=None, help="Named vector to search against, if applicable")

    return parser


def _loader_from_args(args) -> "QdrantLoader":
    config = LoaderConfig()
    if args.host:
        config.host = args.host
    if args.port:
        config.port = args.port
    if args.collection:
        config.collection_name = args.collection
    if args.checkpoint_path:
        config.checkpoint_path = args.checkpoint_path
    if args.failed_rows_log:
        config.failed_rows_log = args.failed_rows_log
    if args.log_file:
        config.log_file = args.log_file
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.parallel:
        config.parallel = args.parallel
    if args.max_batch_retries:
        config.max_batch_retries = args.max_batch_retries
    if args.duplicate_strategy:
        config.duplicate_strategy = args.duplicate_strategy
    if args.no_verify_upload:
        config.verify_upload = False
    if args.no_payload_index:
        config.create_payload_indexes = False
    if args.no_collection_validation:
        config.validate_collection_compatibility = False

    return QdrantLoader(config=config)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Single CLI entry point. Behavior is chosen via --action (default:
    'ingest'). Returns a process exit code (0 = success).

    Examples:
        python qdrant_loader.py --chunks-csv chunks.csv --intelligence-csv intel.csv --metadata-csv meta.csv --embedding-file emb.npy --collection my_collection
        python qdrant_loader.py --metadata-csv premerged.csv --embedding-file emb.npy --collection my_collection
        python qdrant_loader.py --action search --collection my_collection --query-vector-file q.npy --limit 5
        python qdrant_loader.py --action info --collection my_collection
        python qdrant_loader.py --action count --collection my_collection
        python qdrant_loader.py --action exists --collection my_collection
        python qdrant_loader.py --action delete --collection my_collection
    """
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    try:
        loader = _loader_from_args(args)

        if args.action == "ingest":
            chunks_csv = args.chunks_csv or args.unstructured_csv
            if not args.metadata_csv and not chunks_csv and not args.intelligence_csv:
                parser.error(
                    "--action ingest requires at least one of "
                    "--metadata-csv / --chunks-csv / --intelligence-csv"
                )

            multi_source = bool(chunks_csv or args.intelligence_csv)

            if multi_source:
                stats = loader.load_from_sources(
                    chunks_csv=chunks_csv,
                    intelligence_csv=args.intelligence_csv,
                    metadata_csv=args.metadata_csv,
                    embedding_file=args.embedding_file,
                    resume=not args.no_resume,
                )
                print(json.dumps(stats.as_dict(), indent=2))
            elif args.csv_chunksize:
                aggregate: Optional[IngestionStats] = None
                for chunk in read_csv_in_chunks(args.metadata_csv, args.csv_chunksize):
                    stats = loader.load_dataframe(
                        chunk, embedding_file=args.embedding_file, resume=not args.no_resume,
                    )
                    if aggregate is None:
                        aggregate = stats
                    else:
                        aggregate.total_rows += stats.total_rows
                        aggregate.uploaded += stats.uploaded
                        aggregate.duplicates += stats.duplicates
                        aggregate.skipped += stats.skipped
                        aggregate.failed += stats.failed
                        aggregate.invalid_vectors += stats.invalid_vectors
                        aggregate.batch_retries += stats.batch_retries
                        aggregate.elapsed_seconds += stats.elapsed_seconds
                        aggregate.finished_at = stats.finished_at
                print(json.dumps((aggregate or IngestionStats(collection=loader.collection_name)).as_dict(), indent=2))
            else:
                df = pd.read_csv(args.metadata_csv, low_memory=False)
                stats = loader.load_dataframe(
                    df, embedding_file=args.embedding_file, resume=not args.no_resume,
                )
                print(json.dumps(stats.as_dict(), indent=2))

        elif args.action == "search":
            if not args.query_vector_file:
                parser.error("--action search requires --query-vector-file")
            query_vector = np.load(args.query_vector_file)
            results = loader.search(query_vector, limit=args.limit, vector_name=args.vector_name)
            output = [{"id": r.id, "score": r.score, "payload": r.payload} for r in results]
            print(json.dumps(output, indent=2, default=str))

        elif args.action == "info":
            print(loader.collection_info())

        elif args.action == "count":
            print(loader.count())

        elif args.action == "delete":
            loader.delete_collection()
            print(f"Deleted collection: {loader.collection_name}")

        elif args.action == "exists":
            print(loader.collection_exists())

        return 0

    except LoaderError as exc:
        logger.error(str(exc))
        return 1
    except Exception as exc:  # pragma: no cover - top-level safety net
        logger.exception("Unhandled error while running qdrant_loader CLI: %s", exc)
        return 1


if __name__ == "__main__":
    import sys as _sys
    _sys.exit(main())