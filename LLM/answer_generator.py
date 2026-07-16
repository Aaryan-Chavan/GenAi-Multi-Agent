"""
llm/answer_generator.py
============================================================

Enterprise Answer Generator (Schema-Agnostic, Dataset-Independent)

Single responsibility:

    Receive Context -> Validate/Normalize -> Build Prompt
    -> Call QwenClient -> Return Standardized Answer

This class performs no retrieval, no SQL generation, no semantic
search, no orchestration, and no reranking -- it only turns whatever
context it's handed (from HybridAgent, a raw list of rows/chunks, or
anything else) into a grounded answer.

Context tolerance
------------------
`generate(context=...)` accepts, without raising:
    - None
    - a list / tuple / set of dicts (or dataclasses, namedtuples,
      strings, arbitrary objects)
    - a single dict (including a `HybridResult.to_dict()`-shaped
      payload -- any list-of-dict values inside it are merged
      generically, without assuming specific key names)
    - a dict containing a SQL-style `columns` + `rows` pair, where
      `rows` may be a list of tuples/lists (e.g. DuckDB `fetchall()`
      output) rather than pre-zipped dicts -- these are paired up
      into `List[Dict[str, Any]]` here, at any nesting depth
    - a plain string (e.g. an already-compressed context block)
    - a generator/iterator (consumed exactly once)
    - a pandas DataFrame (duck-typed, so pandas is not a hard import)

Everything is normalized internally into `List[Dict[str, Any]]`
before any further processing.

Prompt construction
--------------------
If a working `PromptTemplateManager` is available, it is used first
(preserving prior behavior exactly). If it is unavailable, wasn't
supplied, or raises, this class falls back to its own built-in,
domain-agnostic prompt builder -- so it never depends on another
module to function. Either way, the *context records* it hands off
have already been deduplicated, capped, and length-truncated here,
so both paths benefit from the same token-efficiency work.
============================================================
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

LOGGER = logging.getLogger(__name__)

# Matches 2+ consecutive repeats of a single punctuation character
# (e.g. "!!!", "??", "..."). Used only for cheap dedup-key normalization
# below -- not used to alter any text that reaches the LLM or the user.
_REPEATED_PUNCTUATION_RE = re.compile(r"([^\w\s])\1+")


def _collapse_repeated_punctuation(text: str) -> str:
    """Collapse runs of the same punctuation character down to one
    occurrence (e.g. "great!!!" -> "great!"). Pure string operation,
    O(n) in text length, no external dependencies -- used only to
    build a slightly more forgiving dedup key."""
    return _REPEATED_PUNCTUATION_RE.sub(r"\1", text)

# ------------------------------------------------------------------
# Defensive imports: this file must remain importable and usable
# even if optional collaborators are missing/broken. QwenClient is
# the one true hard dependency (without it there is nothing to call).
# ------------------------------------------------------------------

try:
    from Config import settings  # type: ignore
except Exception as exc:  # pragma: no cover - environment dependent
    LOGGER.warning("Config.settings not importable (%s); using built-in defaults.", exc)

    class _SettingsFallback:
        MODEL_NAME = "Unknown"
        MAX_CONTEXT_CHUNKS = 12
        MAX_CONTEXT_CHARS_PER_ITEM = 500
        MAX_SOURCES = 20
        TEMPERATURE = None
        TOP_P = None
        MAX_RESPONSE_TOKENS = None

    settings = _SettingsFallback()  # type: ignore

try:
    from LLM.prompt_templates import PromptTemplateManager  # type: ignore
except Exception as exc:  # pragma: no cover - environment dependent
    LOGGER.warning(
        "LLM.prompt_templates.PromptTemplateManager not importable (%s); "
        "the built-in generic prompt builder will be used instead.", exc,
    )
    PromptTemplateManager = None  # type: ignore

from LLM.qwen_client import GenerationResult, QwenClient  # hard dependency, unchanged


class AnswerStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"   # answered, but with no grounded context
    FAILED = "failed"
    EMPTY = "empty"


class AnswerType(str, Enum):
    GENERAL = "general"
    STRUCTURED = "structured"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


# ============================================================
# SCHEMA CONFIG -- tunable discovery rules, not hardcoded fields
# ============================================================

@dataclass(slots=True)
class SchemaConfig:
    """
    Controls how AnswerGenerator discovers fields in context records.
    No field names are assumed to exist -- these are ordered
    candidate lists checked at runtime, so this stays dataset- and
    domain-independent (works for reviews, tickets, legal docs, DB
    rows, logs, ...).
    """

    text_field_candidates: Sequence[str] = field(
        default_factory=lambda: (
            "text", "content", "chunk", "chunk_text", "body",
            "passage", "value", "description", "summary",
        )
    )
    score_field_candidates: Sequence[str] = field(
        default_factory=lambda: (
            "score", "similarity", "relevance", "confidence", "rank_score",
        )
    )
    excluded_source_fields: Sequence[str] = field(
        default_factory=lambda: ("embedding", "vector", "raw", "_id", "_qdrant_id")
    )


# ============================================================
# ANSWER RESULT -- standardized, success/failure alike
# ============================================================

@dataclass
class AnswerResult:
    answer: str
    answer_type: str
    status: str
    confidence: float
    sources: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    latency: float = 0.0
    timestamp: float = field(default_factory=time.time)
    # Additive fields (appended at the end -> positional construction
    # of the original 8-field shape still works unchanged).
    success: bool = True
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Lightweight dict conversion (avoids dataclasses.asdict's
        recursive deep-copy of sources/metadata, which is unnecessary
        here since both are already plain dict/list structures)."""
        return {
            "answer": self.answer,
            "answer_type": self.answer_type,
            "status": self.status,
            "confidence": self.confidence,
            "sources": self.sources,
            "metadata": self.metadata,
            "latency": self.latency,
            "timestamp": self.timestamp,
            "success": self.success,
            "error": self.error,
            "warnings": self.warnings,
        }


# ============================================================
# ANSWER GENERATOR
# ============================================================

class AnswerGenerator:
    """
    Schema-agnostic Answer Generator. Works with any retrieval
    dataset shape by discovering relevant fields at runtime instead
    of assuming fixed key names, and by normalizing any reasonable
    context input shape before use.
    """

    _GENERIC_INSTRUCTIONS: Dict["AnswerType", str] = {
        AnswerType.STRUCTURED: "Use the structured records below to answer with precise facts and figures.",
        AnswerType.SEMANTIC: "Use the evidence passages below to answer, synthesizing the relevant information.",
        AnswerType.HYBRID: "Use both the structured records and the evidence passages below to answer.",
        AnswerType.GENERAL: "Use the context below, if any, to answer the question.",
    }

    # --- Column/row pairing candidates -------------------------------
    # SQL engines (DuckDB included) commonly return column names and
    # row values as two separate structures (e.g. `cursor.description`
    # + `fetchall()` tuples). These candidate keys let us recognize
    # that shape wherever it appears in a context payload and zip it
    # back into `List[Dict[str, Any]]` records, without hardcoding any
    # dataset-specific column name.
    _COLUMN_KEY_CANDIDATES: Sequence[str] = (
        "columns", "column_names", "fields", "field_names", "headers",
    )
    _ROW_KEY_CANDIDATES: Sequence[str] = (
        "rows", "data", "results", "records", "values",
    )

    # --- Pre-computed confidence signal candidates --------------------
    # A HybridResult.to_dict()-shaped payload may carry these as
    # top-level scalars alongside its list-of-dict fields. They must
    # never be silently dropped just because they aren't part of any
    # single context record.
    _EXTERNAL_CONFIDENCE_KEYS: Sequence[str] = (
        "execution_confidence", "sql_confidence", "structured_confidence",
        "planner_confidence", "routing_confidence", "semantic_confidence",
        "overall_confidence", "confidence",
    )

    def __init__(
        self,
        qwen_client: Optional[QwenClient] = None,
        prompt_manager: Optional[Any] = None,
        schema_config: Optional[SchemaConfig] = None,
    ) -> None:
        self.is_test_mode = os.getenv("TESTING", "0") == "1"

        # Always available, regardless of test mode -- this is a
        # cheap, pure-Python object with no external dependencies,
        # and every field-discovery helper relies on it existing.
        self.schema = schema_config or SchemaConfig()

        if self.is_test_mode:
            # Lightweight dummy dependencies (no HF, no GPU, no network).
            self.qwen = None
            self.prompt_manager = None
        else:
            self.qwen = qwen_client or QwenClient()
            if prompt_manager is not None:
                self.prompt_manager = prompt_manager
            elif PromptTemplateManager is not None:
                try:
                    self.prompt_manager = PromptTemplateManager()
                except Exception as exc:
                    LOGGER.warning(
                        "PromptTemplateManager could not be constructed (%s); "
                        "falling back to the built-in generic prompt builder.", exc,
                    )
                    self.prompt_manager = None
            else:
                self.prompt_manager = None

        self.total_requests = 0
        self.total_latency = 0.0
        self.model_name = getattr(settings, "MODEL_NAME", getattr(settings, "LLM_MODEL", "Unknown"))

        LOGGER.info(
            "AnswerGenerator initialized | model=%s | test_mode=%s | prompt_manager=%s",
            self.model_name, self.is_test_mode, "external" if self.prompt_manager else "built-in",
        )

    # ============================================================
    # LIMITS (token / memory budgeting -- tunable via settings)
    # ============================================================

    def _max_context_items(self) -> int:
        try:
            return max(1, int(getattr(settings, "MAX_CONTEXT_CHUNKS", 12)))
        except (TypeError, ValueError):
            return 12

    def _max_item_chars(self) -> int:
        try:
            return max(0, int(getattr(settings, "MAX_CONTEXT_CHARS_PER_ITEM", 500)))
        except (TypeError, ValueError):
            return 500

    def _max_sources(self) -> int:
        try:
            return max(1, int(getattr(settings, "MAX_SOURCES", 20)))
        except (TypeError, ValueError):
            return 20

    # ============================================================
    # INPUT VALIDATION / NORMALIZATION
    # ============================================================

    @staticmethod
    def _validate_query(query: str) -> str:
        if not isinstance(query, str):
            raise TypeError("Query must be a string.")
        query = query.strip()
        if not query:
            raise ValueError("Query cannot be empty.")
        return query

    @staticmethod
    def _coerce_record(item: Any) -> Dict[str, Any]:
        """Best-effort coercion of a single context element into a
        dict. Never raises -- unrecognized shapes degrade to a
        `{"text": str(item)}` record instead."""
        if isinstance(item, dict):
            return item
        if is_dataclass(item) and not isinstance(item, type):
            try:
                return asdict(item)
            except Exception:
                pass
        if hasattr(item, "_asdict"):  # namedtuple
            try:
                return dict(item._asdict())
            except Exception:
                pass
        if isinstance(item, str):
            return {"text": item}
        if isinstance(item, (tuple, list)):
            return {f"value_{i}": v for i, v in enumerate(item)}
        return {"text": str(item)}

    @classmethod
    def _zip_columns_rows(
        cls, columns: Sequence[Any], rows: Sequence[Any],
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Pair a list of column names with a list of rows into
        `List[Dict[str, Any]]`.

        This is the shape most SQL drivers (DuckDB's `fetchall()`
        included) naturally return: column names kept *separate* from
        row tuples. Prior to this fix, that separation meant SQL
        result rows never survived `_validate_context`'s "is this a
        list of dicts?" check (tuples aren't dicts), so every
        aggregated/aliased column silently vanished before reaching
        the prompt.

        Each row may itself already be a dict (passed through as-is,
        so already-zipped input is unaffected), or a tuple/list
        (zipped positionally against `columns`), or a bare scalar
        (wrapped using the single column name, for single-column
        result sets). Returns None if `columns` is empty or `rows`
        isn't list-shaped, so callers can safely fall back to other
        extraction strategies.
        """
        if not columns or not isinstance(rows, (list, tuple)):
            return None
        col_names = [str(c) for c in columns]
        zipped: List[Dict[str, Any]] = []
        for row in rows:
            if isinstance(row, dict):
                zipped.append(row)
            elif isinstance(row, (list, tuple)):
                zipped.append(
                    {col_names[i]: value for i, value in enumerate(row) if i < len(col_names)}
                )
            else:
                zipped.append({(col_names[0] if col_names else "value"): row})
        return zipped

    @classmethod
    def _extract_records_from_mapping(
        cls,
        mapping: Dict[str, Any],
        _depth: int = 0,
        _seen: Optional[set] = None,
        _max_depth: int = 4,
    ) -> List[Dict[str, Any]]:
        """
        Recursively discover every context record hiding inside a
        dict-shaped payload (typically a `HybridResult.to_dict()`
        output), regardless of how deeply structured/semantic results
        are nested.

        Three extraction strategies are combined, all schema-agnostic
        (no dataset column names, and no fixed wrapper-key names other
        than generic container labels like "structured"/"rows" are
        ever assumed to be *present* -- they're just checked for):

          1. columns/rows pairing at this level (SQL-result shape --
             see `_zip_columns_rows`), e.g.
             {"columns": [...], "rows": [(...), (...)]}
          2. any value that is already a `List[dict]` (original
             behavior, preserved as-is for backward compatibility)
          3. recursing into nested dict values, so a wrapper such as
             {"structured": {"columns": [...], "rows": [...]},
              "semantic": {"chunks": [...]}}
             still yields every row/chunk instead of being silently
             skipped because the top level itself isn't list-shaped.

        `_seen` guards against reference cycles (defensive only --
        ordinary JSON-shaped payloads can't cycle, but this is cheap
        insurance for arbitrary Python objects passed in directly).
        Depth is capped (default 4) purely as a sanity bound; realistic
        HybridResult payloads are 1-2 levels deep.
        """
        if _depth > _max_depth or not isinstance(mapping, dict):
            return []
        if _seen is None:
            _seen = set()
        mapping_id = id(mapping)
        if mapping_id in _seen:
            return []
        _seen.add(mapping_id)

        collected: List[Dict[str, Any]] = []

        # Strategy 1: columns/rows pairing at this level.
        columns_val = None
        for key in cls._COLUMN_KEY_CANDIDATES:
            candidate = mapping.get(key)
            if isinstance(candidate, (list, tuple)) and candidate:
                columns_val = candidate
                break
        if columns_val is not None:
            for key in cls._ROW_KEY_CANDIDATES:
                rows_val = mapping.get(key)
                if isinstance(rows_val, (list, tuple)) and rows_val:
                    zipped = cls._zip_columns_rows(columns_val, rows_val)
                    if zipped:
                        collected.extend(zipped)
                    break

        # Strategy 2: values that are already List[dict] (original
        # top-level-only behavior, now applied at every level visited).
        for value in mapping.values():
            if isinstance(value, (list, tuple)) and value and all(isinstance(x, dict) for x in value):
                collected.extend(value)

        # Strategy 3: recurse into nested dict values so wrapped
        # structured/semantic payloads aren't skipped just because the
        # container itself isn't directly list-shaped.
        for value in mapping.values():
            if isinstance(value, dict):
                collected.extend(
                    cls._extract_records_from_mapping(value, _depth + 1, _seen, _max_depth)
                )

        return collected

    @classmethod
    def _extract_external_confidences(cls, context: Any) -> Dict[str, float]:
        """
        Pull any pre-computed, top-level confidence signals out of a
        dict-shaped context payload (execution_confidence,
        sql_confidence, planner_confidence, routing_confidence,
        semantic_confidence, ...) before that dict is reduced down to
        its list-of-dict fields.

        Previously these scalars sat right next to the
        structured_facts/semantic_chunks lists and were discarded
        the moment `_validate_context` found list fields to merge --
        meaning a real `execution_confidence=1.00` from a successful
        SQL run never influenced the final confidence at all, and was
        silently replaced by a naive per-record-score average (which
        SQL rows typically don't have, since scores are a semantic-
        search concept). That produced exactly the
        `execution_confidence=1.00` / `overall_confidence=0.41`
        mismatch this fix addresses.

        Never assumes a key is present; only includes ones that
        actually exist and are numeric, each clamped into [0, 1].
        """
        found: Dict[str, float] = {}
        if not isinstance(context, dict):
            return found
        for key in cls._EXTERNAL_CONFIDENCE_KEYS:
            value = context.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                found[key] = max(0.0, min(float(value), 1.0))
        return found

    @classmethod
    def _validate_context(cls, context: Any) -> List[Dict[str, Any]]:
        """
        Normalize arbitrary context input into `List[Dict[str, Any]]`.

        Accepts (without ever raising): None, dict (including
        `HybridResult.to_dict()`-shaped payloads, with structured
        results discovered at any nesting depth and SQL-style
        columns/rows pairs zipped into records -- see
        `_extract_records_from_mapping`), list/tuple/set, string,
        generator/iterator, duck-typed pandas DataFrame,
        dataclasses/namedtuples, or any other object.
        """
        if context is None:
            return []

        # Duck-typed pandas DataFrame (avoids a hard pandas import).
        if hasattr(context, "to_dict") and hasattr(context, "columns") and hasattr(context, "iterrows"):
            try:
                records = context.to_dict(orient="records")
                return [r for r in records if isinstance(r, dict)]
            except Exception:
                LOGGER.debug("Context looked like a DataFrame but conversion failed; falling back.")

        if isinstance(context, dict):
            merged = cls._extract_records_from_mapping(context)
            if merged:
                return merged
            return [context]

        if isinstance(context, str):
            text = context.strip()
            return [{"text": text}] if text else []

        if hasattr(context, "__iter__"):
            try:
                iterator = iter(context)
            except TypeError:
                return [{"text": str(context)}]
            return [cls._coerce_record(item) for item in iterator]

        return [{"text": str(context)}]

    # ============================================================
    # DYNAMIC FIELD DISCOVERY
    # ============================================================

    def _find_text_field(self, chunk: Dict[str, Any]) -> Optional[str]:
        """Return the key holding this chunk's body text, if any."""
        for key in self.schema.text_field_candidates:
            value = chunk.get(key)
            if value and str(value).strip():
                return key
        return None

    def _find_score_field(self, chunk: Dict[str, Any]) -> Optional[str]:
        """Return the key holding a numeric score, if any."""
        for key in self.schema.score_field_candidates:
            value = chunk.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return key
        return None

    # ============================================================
    # SINGLE-PASS CONTEXT ANALYSIS
    # (dedup + truncation + capping + sources + confidence, together)
    # ============================================================

    def _analyze_context(
        self, records: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], float, bool]:
        """
        One pass over normalized context records producing:
          - clean_records: deduplicated, length-capped, count-capped
            records ready for prompt construction (best-scored first)
          - sources: cleaned metadata for the caller (embedding/vector
            fields excluded, count-capped)
          - confidence: average of any discovered per-record relevance
            scores (0.0 if none were found)
          - had_scores: whether at least one record actually carried a
            numeric score field. This is returned separately from
            `confidence` so callers (see `generate`) can tell "no
            score fields existed" apart from "scores existed and
            averaged to zero" -- conflating the two previously caused
            a real 0.0 to be blended into overall confidence even when
            the record set (e.g. plain SQL rows) simply never carries
            a per-row score to begin with.

        Exact-duplicate text is skipped cheaply via a hash set; items
        with no discoverable text field (e.g. pure structured rows)
        are never deduplicated against each other, since superficially
        similar rows may still be legitimately distinct records.
        """
        if not records:
            return [], [], 0.0, False

        max_items = self._max_context_items()
        max_chars = self._max_item_chars()
        max_sources = self._max_sources()

        scored: List[Tuple[Optional[float], Optional[str], Dict[str, Any]]] = []
        for item in records:
            if not isinstance(item, dict) or not item:
                continue
            text_key = self._find_text_field(item)
            score_key = self._find_score_field(item)
            score_val: Optional[float] = None
            if score_key is not None:
                try:
                    score_val = float(item[score_key])
                except (TypeError, ValueError):
                    score_val = None
            scored.append((score_val, text_key, item))

        if not scored:
            return [], [], 0.0, False

        # Best-scored evidence first, so truncation keeps what matters;
        # stable sort preserves relative order among unscored items.
        scored.sort(key=lambda t: (t[0] is None, -(t[0] or 0.0)))

        seen_hashes: set = set()
        clean_records: List[Dict[str, Any]] = []
        sources: List[Dict[str, Any]] = []
        conf_scores: List[float] = []

        for score_val, text_key, item in scored:
            if len(clean_records) >= max_items and len(sources) >= max_sources:
                break

            text_value = ""
            if text_key is not None:
                text_value = str(item.get(text_key, "")).strip()
                if max_chars and len(text_value) > max_chars:
                    text_value = text_value[:max_chars].rstrip() + "\u2026"

            if text_value:
                # --- Improvement: slightly stronger duplicate detection ---
                # Still O(n) (a single hash-set membership check/insert
                # per record, no embeddings or external similarity libs).
                # In addition to the prior whitespace-collapse + lowercase
                # normalization, we now also collapse runs of repeated
                # punctuation (e.g. "great!!!" vs "great!") so trivially
                # re-punctuated duplicates are also caught.
                normalized = " ".join(text_value.lower().split())
                dedup_key = _collapse_repeated_punctuation(normalized)
                if dedup_key in seen_hashes:
                    continue
                seen_hashes.add(dedup_key)

            if score_val is not None:
                conf_scores.append(max(0.0, min(score_val, 1.0)))

            if len(clean_records) < max_items:
                record = dict(item)
                if text_key is not None:
                    record[text_key] = text_value
                clean_records.append(record)

            if len(sources) < max_sources:
                cleaned_source = {
                    k: v for k, v in item.items()
                    if k != text_key
                    and k not in self.schema.excluded_source_fields
                    and v is not None
                    and v != ""
                }
                if cleaned_source:
                    sources.append(cleaned_source)

        # --- Improvement: robust confidence estimation ---
        # Previously this averaged EVERY discovered score, so a single
        # low-relevance result could drag confidence down even when the
        # top matches were strong. Now we:
        #   1. clamp scores into [0, 1] (already done above, when
        #      appended to conf_scores),
        #   2. sort descending,
        #   3. average only the top 3 (or fewer, if fewer exist),
        #   4. return 0.0 (and had_scores=False) if no valid scores
        #      exist at all, so the caller can distinguish "no signal"
        #      from "signal averaged to zero".
        # `conf_scores` is already populated in best-score-first order
        # because `scored` was sorted descending before this loop ran,
        # so no extra pass over the full record set is required here --
        # we simply take the first (up to) 3 entries.
        if conf_scores:
            top_scores = sorted(conf_scores, reverse=True)[:3]
            confidence = round(sum(top_scores) / len(top_scores), 3)
            had_scores = True
        else:
            confidence = 0.0
            had_scores = False
        return clean_records, sources, confidence, had_scores

    # Backward-compatible thin wrappers around _analyze_context, kept
    # in case anything calls these private helpers directly.

    def _format_sources(self, context: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        _, sources, _, _ = self._analyze_context(self._validate_context(context))
        return sources

    def _estimate_confidence(self, context: List[Dict[str, Any]]) -> float:
        _, _, confidence, _ = self._analyze_context(self._validate_context(context))
        return confidence

    # ============================================================
    # CONTEXT FORMATTER (built-in, generic, token-conscious)
    # ============================================================

    def _format_context(self, context: List[Dict[str, Any]]) -> str:
        """
        Render already-cleaned context records into a compact,
        schema-agnostic text block. Skips excluded/empty fields
        (embeddings, vectors, internal ids) so they never inflate the
        prompt with unusable tokens.
        """
        if not context:
            return "No external context available."

        formatted: List[str] = []
        for idx, item in enumerate(context, start=1):
            if not isinstance(item, dict) or not item:
                continue

            text_key = self._find_text_field(item)
            # --- Improvement: clearer per-record heading ---
            # A dedicated "Record N" heading (instead of folding the
            # index into the first line of body text) makes record
            # boundaries unambiguous once metadata lines are added
            # below, without assuming/hardcoding any dataset field name.
            lines: List[str] = [f"Record {idx}:"]
            if text_key is not None:
                text_value = str(item.get(text_key, "")).strip()
                if text_value:
                    lines.append(text_value)

            # --- Improvement: metadata visually distinguished ---
            # Metadata lines are indented and prefixed so they read as
            # supporting fields rather than body text, while still
            # skipping empty values and any excluded/internal fields
            # (embeddings, vectors, ids) -- schema-agnostic throughout.
            metadata_lines: List[str] = []
            for key, value in item.items():
                if key == text_key or value is None or value == "":
                    continue
                if key in self.schema.excluded_source_fields:
                    continue
                pretty_key = key.replace("_", " ").title()
                metadata_lines.append(f"    - {pretty_key}: {value}")

            if metadata_lines:
                lines.append("  Metadata:")
                lines.extend(metadata_lines)

            formatted.append("\n".join(lines))

        # --- Improvement: clearer separation between records ---
        # A visible rule between records (rather than a bare blank
        # line) makes record boundaries unambiguous when scanning a
        # long context block.
        separator = "\n" + ("-" * 40) + "\n"
        return separator.join(formatted) if formatted else "No external context available."

    # ============================================================
    # PROMPT BUILDER
    # ============================================================

    def _build_prompt(
        self,
        query: str,
        records: List[Dict[str, Any]],
        answer_type: "AnswerType",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Prefers the external PromptTemplateManager (preserving prior
        behavior exactly) if one is available and it succeeds; falls
        back to the built-in generic prompt builder otherwise. Either
        way, `records` has already been deduplicated/capped/truncated
        by `_analyze_context`, so both paths are token-efficient.

        `metadata` is optional (default None) and fully backward
        compatible -- existing callers that omit it get byte-for-byte
        identical behavior. When supplied, it's passed through to the
        built-in generic prompt builder, which may surface a small,
        high-level summary of it (retrieval mode, row/chunk counts,
        intent, entities, filters -- whichever keys actually exist).
        """
        if self.prompt_manager is not None:
            try:
                builder = {
                    AnswerType.STRUCTURED: self.prompt_manager.build_structured_prompt,
                    AnswerType.SEMANTIC: self.prompt_manager.build_semantic_prompt,
                    AnswerType.HYBRID: self.prompt_manager.build_hybrid_prompt,
                }.get(answer_type, self.prompt_manager.build_prompt)
                return builder(query=query, context=records)
            except Exception as exc:
                LOGGER.warning(
                    "External PromptTemplateManager failed (%s); using the built-in generic prompt builder.", exc,
                )
        return self._build_generic_prompt(query, records, answer_type, metadata=metadata)

    # Keys we'll surface from an optional `metadata` dict, if present.
    # Purely descriptive of *retrieval*, never domain-specific (no
    # products/reviews/customers/tickets assumptions).
    _METADATA_PROMPT_KEYS: Sequence[Tuple[str, str]] = (
        ("retrieval_mode", "Retrieval mode"),
        ("structured_rows", "Structured rows retrieved"),
        ("semantic_chunks", "Semantic chunks retrieved"),
        ("intent", "Detected intent"),
        ("entities", "Detected entities"),
        ("filters", "Applied filters"),
    )

    def _build_generic_prompt(
        self,
        query: str,
        records: List[Dict[str, Any]],
        answer_type: "AnswerType",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Self-contained, domain-agnostic prompt: never assumes
        products/reviews/customers or any other fixed domain.

        --- Improvement: strengthened instructions ---
        The instructions below now explicitly cover: answering only
        from supplied context, never fabricating information, stating
        insufficiency explicitly, preferring exact structured facts
        over inferred/semantic statements when both are present, and
        surfacing (rather than silently resolving) conflicting
        context. All of this remains domain-independent.

        --- Improvement: optional retrieval metadata (backward compatible) ---
        If `metadata` is provided, a short, high-level "Retrieval
        context" block is appended, listing only whichever of a fixed
        set of generic keys actually exist (never assuming any key is
        present). If `metadata` is omitted, this section is skipped
        entirely and the prompt is byte-for-byte identical to before.
        """
        instruction = self._GENERIC_INSTRUCTIONS.get(answer_type, self._GENERIC_INSTRUCTIONS[AnswerType.GENERAL])
        context_block = self._format_context(records)

        guidance_lines = [
            instruction,
            "Answer strictly from the context below. Never fabricate or assume "
            "information that is not present in it.",
            "If the context is insufficient to answer confidently, state that "
            "explicitly instead of guessing.",
            "When both exact structured records and semantic evidence passages "
            "are present, prefer the structured records for precise values "
            "(numbers, dates, identifiers, etc.) and use the semantic evidence "
            "for explanation and context around those values.",
            "If the context contains conflicting information, explain the "
            "conflict rather than silently picking one side.",
            "Keep the answer concise but complete.",
        ]
        guidance = "\n".join(guidance_lines)

        metadata_block = ""
        if metadata:
            summary_lines = []
            for key, label in self._METADATA_PROMPT_KEYS:
                if key in metadata and metadata[key] not in (None, "", [], {}):
                    summary_lines.append(f"- {label}: {metadata[key]}")
            if summary_lines:
                metadata_block = "Retrieval context:\n" + "\n".join(summary_lines) + "\n\n"

        return (
            f"{guidance}\n\n"
            f"{metadata_block}"
            f"Context:\n{context_block}\n\n"
            f"Question: {query}\n"
            "Answer:"
        )

    # ============================================================
    # LLM GENERATION
    # ============================================================

    def _generate_answer(self, prompt: str) -> GenerationResult:
        return self.qwen.generate(prompt=prompt)

    # ============================================================
    # RESULT BUILDERS
    # ============================================================

    def _build_result(
        self,
        llm_result: GenerationResult,
        answer_type: "AnswerType",
        confidence: float,
        sources: List[Dict[str, Any]],
        grounded: bool,
    ) -> AnswerResult:
        raw_text = llm_result.text
        answer = self._clean_answer(raw_text)

        if not answer:
            status = AnswerStatus.EMPTY
        elif not grounded:
            status = AnswerStatus.PARTIAL
        else:
            status = AnswerStatus.SUCCESS

        metadata = {
            "model": llm_result.model,
            "prompt_tokens": llm_result.prompt_tokens,
            "generated_tokens": llm_result.generated_tokens,
            "total_tokens": llm_result.total_tokens,
            "finish_reason": llm_result.finish_reason,
            "tokens_per_second": llm_result.tokens_per_second,
            "request_id": llm_result.request_id,
            "cache_hit": llm_result.cache_hit,
        }

        return AnswerResult(
            answer=answer,
            answer_type=answer_type.value,
            status=status.value,
            confidence=confidence,
            sources=sources,
            metadata=metadata,
            latency=llm_result.latency,
            success=status in (AnswerStatus.SUCCESS, AnswerStatus.PARTIAL),
            error=None,
        )

    def _error_result(self, answer_type: "AnswerType", exc: Exception, stage: str, start_time: float) -> AnswerResult:
        """Standardized failure response. Never re-raises. Deliberately
        keeps the exception text out of `answer` -- it lives in `error`
        instead, so callers can't mistake it for a generated answer."""
        LOGGER.exception("Answer generation failed at stage=%s", stage)
        latency = time.time() - start_time
        at = answer_type.value if isinstance(answer_type, AnswerType) else str(answer_type)
        return AnswerResult(
            answer="",
            answer_type=at,
            status=AnswerStatus.FAILED.value,
            confidence=0.0,
            sources=[],
            metadata={},
            latency=round(latency, 3),
            success=False,
            error=f"{stage}: {exc}",
        )

    # ============================================================
    # RESPONSE CLEANER
    # ============================================================

    @staticmethod
    def _clean_answer(answer: Optional[str]) -> str:
        if answer is None:
            return ""
        answer = answer.strip()
        prefixes = ["Answer:", "Final Answer:", "Response:", "Assistant:", "AI:"]
        lower_answer = answer.lower()
        for prefix in prefixes:
            if lower_answer.startswith(prefix.lower()):
                answer = answer[len(prefix):].strip()
                break
        return answer

    # ============================================================
    # PUBLIC GENERATE
    # ============================================================

    def generate(
        self,
        query: str,
        context: Optional[Any] = None,
        answer_type: "AnswerType" = AnswerType.HYBRID,
    ) -> AnswerResult:
        """
        Generate a grounded answer from `query` and `context`.

        `context` may be None, a list/tuple/set of dicts, a single
        dict (including a HybridResult.to_dict()-shaped payload -- now
        with structured results discovered at any nesting depth and
        SQL-style columns/rows pairs zipped into records), a string, a
        generator, a pandas DataFrame, or effectively anything else --
        it is normalized internally and this method never raises due
        to context shape. On any internal failure a structured
        `AnswerResult(success=False, error=...)` is returned instead of
        propagating an exception.
        """
        start_time = time.time()
        self.total_requests += 1

        if not isinstance(answer_type, AnswerType):
            try:
                answer_type = AnswerType(answer_type)
            except Exception:
                answer_type = AnswerType.GENERAL

        try:
            query = self._validate_query(query)
        except Exception as exc:
            return self._error_result(answer_type, exc, "invalid_query", start_time)

        records = self._validate_context(context)
        clean_records, sources, item_confidence, had_item_scores = self._analyze_context(records)

        # --- Fix for Bug 6: preserve pre-computed confidence signals ---
        # A dict-shaped payload may carry execution/planner/routing/
        # semantic confidence as top-level scalars. Those must be
        # blended into the final confidence instead of being discarded
        # (previously) or diluted by a spurious 0.0 derived from SQL
        # rows that simply don't have a per-row score field.
        external_confidences = self._extract_external_confidences(context)
        confidence_signals: List[float] = list(external_confidences.values())
        if had_item_scores:
            confidence_signals.append(item_confidence)

        if confidence_signals:
            confidence = round(sum(confidence_signals) / len(confidence_signals), 3)
        elif clean_records:
            # Grounded in real records, but no explicit confidence
            # signal (external or per-item) was ever supplied. This is
            # a neutral, non-zero default -- it must not be treated as
            # "answer succeeded but confidence is untrustworthy" (0.0),
            # since the presence of clean_records already means the
            # answer is grounded in retrieved evidence.
            confidence = 0.75
        else:
            confidence = 0.0

        warnings: List[str] = []
        if context is not None and not clean_records:
            warnings.append("Context was supplied but no usable evidence could be extracted from it.")
        elif not clean_records:
            warnings.append("No context supplied; the answer is not grounded in retrieved evidence.")

        try:
            t_prompt = time.time()
            prompt = self._build_prompt(query=query, records=clean_records, answer_type=answer_type)
            prompt_latency_ms = (time.time() - t_prompt) * 1000
        except Exception as exc:
            return self._error_result(answer_type, exc, "prompt_construction", start_time)

        try:
            llm_result = self._generate_answer(prompt)
        except Exception as exc:
            return self._error_result(answer_type, exc, "llm_generation", start_time)

        result = self._build_result(
            llm_result=llm_result,
            answer_type=answer_type,
            confidence=confidence,
            sources=sources,
            grounded=bool(clean_records),
        )
        result.warnings = warnings

        # --- Improvement: prompt/context diagnostics (additive only) ---
        # These are appended to the metadata dict already built by
        # _build_result; nothing existing is removed or overwritten.
        result.metadata["context_record_count"] = len(clean_records)
        result.metadata["source_count"] = len(sources)
        result.metadata["prompt_character_count"] = len(prompt)
        result.metadata["formatted_context_characters"] = len(self._format_context(clean_records)) \
            if clean_records else 0
        result.metadata["prompt_build_latency_ms"] = round(prompt_latency_ms, 3)
        result.metadata["context_confidence"] = item_confidence
        result.metadata["external_confidence_signals"] = external_confidences

        latency = time.time() - start_time
        result.latency = round(latency, 3)
        self.total_latency += latency

        # --- Improvement: richer INFO logging ---
        # Now also reports context record/source counts, confidence,
        # answer type, and prompt size -- while still never logging
        # the prompt text itself or any raw user/context content.
        LOGGER.info(
            "Answer generated | query_hash=%s | answer_len=%d | answer_type=%s | "
            "context_records=%d | sources=%d | confidence=%.3f | external_confidence=%s | "
            "prompt_chars=%d | prompt_build_ms=%.1f | total_latency=%.3f | prompt_tokens=%s | "
            "generated_tokens=%s | status=%s",
            QwenClient.hash_prompt(query), len(result.answer), result.answer_type,
            len(clean_records), len(sources), confidence, external_confidences, len(prompt),
            prompt_latency_ms, latency,
            result.metadata.get("prompt_tokens"), result.metadata.get("generated_tokens"), result.status,
        )
        return result

    # ============================================================
    # STATISTICS / HEALTH / RESET / DIAGNOSTICS
    # ============================================================

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "model": self.model_name,
            "total_requests": self.total_requests,
            "average_latency": round(self.total_latency / self.total_requests, 4)
            if self.total_requests > 0 else 0.0,
            "total_latency": round(self.total_latency, 4),
            "max_context_chunks": self._max_context_items(),
            "max_context_chars_per_item": self._max_item_chars(),
            "max_sources": self._max_sources(),
            "temperature": getattr(settings, "TEMPERATURE", None),
            "top_p": getattr(settings, "TOP_P", None),
            "max_response_tokens": getattr(settings, "MAX_RESPONSE_TOKENS", None),
        }

    def health_check(self) -> Dict[str, Any]:
        health = {
            "answer_generator": True,
            "qwen_client": self.qwen is not None,
            "prompt_templates": self.prompt_manager is not None,
            "status": "healthy",
        }
        try:
            if self.qwen is not None and hasattr(self.qwen, "health_check"):
                health["qwen"] = self.qwen.health_check()
        except Exception:
            health["status"] = "degraded"
        return health

    def reset(self) -> None:
        self.total_requests = 0
        self.total_latency = 0.0
        LOGGER.info("AnswerGenerator reset.")

    def diagnostics(self) -> Dict[str, Any]:
        diag = {"generator": self.get_statistics(), "health": self.health_check()}
        if self.qwen is not None and hasattr(self.qwen, "diagnostics"):
            diag["qwen"] = self.qwen.diagnostics()
        return diag

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name})"

    def __str__(self) -> str:
        stats = self.get_statistics()
        return (
            f"AnswerGenerator(model={stats['model']}, "
            f"requests={stats['total_requests']}, "
            f"avg_latency={stats['average_latency']:.4f}s)"
        )


_GENERATOR: Optional[AnswerGenerator] = None


def get_answer_generator() -> AnswerGenerator:
    """Process-wide singleton accessor. In TESTING mode, returns a
    dummy generator that never touches the LLM/network."""
    global _GENERATOR

    if os.getenv("TESTING", "0") == "1":
        class DummyAnswerGenerator:
            def generate(self, *args, **kwargs):
                return "TEST_MODE_RESPONSE"

        return DummyAnswerGenerator()  # type: ignore[return-value]

    if _GENERATOR is None:
        _GENERATOR = AnswerGenerator()

    return _GENERATOR


# ============================================================
# SMOKE TEST -- demonstrates tolerance for multiple context shapes
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    generator = AnswerGenerator()

    list_of_dicts = [{"chunk_id": 1, "text": "Qwen3-14B is a multilingual large language model.", "score": 0.96}]
    hybrid_result_shaped = {
        "structured_facts": [{"id": 1, "avg_rating": 4.2}],
        "semantic_chunks": [{"text": "Users report strong performance on multilingual tasks.", "score": 0.81}],
    }
    # Regression case for Bug 1/2/3: SQL rows nested under a wrapper
    # key, with columns/rows kept separate (DuckDB fetchall() shape),
    # plus top-level execution confidence that must not be discarded.
    sql_result_shaped = {
        "structured": {
            "columns": ["brand", "avg_rating"],
            "rows": [("Acme", 4.6), ("Zenith", 3.8), ("Nova", 3.2)],
        },
        "execution_confidence": 1.0,
    }
    plain_string_context = "Qwen3-14B was released as part of the Qwen3 model family."
    empty_context = None

    for label, ctx in (
        ("list[dict]", list_of_dicts),
        ("hybrid-result-shaped dict", hybrid_result_shaped),
        ("nested SQL columns/rows dict", sql_result_shaped),
        ("plain string", plain_string_context),
        ("None", empty_context),
    ):
        result = generator.generate(query="What is Qwen3?", context=ctx, answer_type=AnswerType.HYBRID)
        print(f"\n--- context={label} ---")
        print(result.to_dict())

    print("\nStatistics:", generator.get_statistics())