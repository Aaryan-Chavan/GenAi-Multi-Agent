from __future__ import annotations

import asyncio
import difflib
import hashlib
import json
import logging
import math
import re
import time
import threading
from collections import OrderedDict
from dataclasses import dataclass, field, asdict, replace
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import duckdb

# =========================================================
# LOGGING
# =========================================================
#
# Library-style logging: a NullHandler is attached by default so this
# module never emits "no handlers configured" warnings, and the host
# application can attach its own handlers/level via `logging.getLogger
# ("structured_agent")` without any change here.

logger = logging.getLogger("structured_agent")
if not logger.handlers:
    logger.addHandler(logging.NullHandler())


# =========================================================
# CONFIGURATION
# =========================================================

@dataclass
class LLMConfig:
    endpoint: str = "http://localhost:11434/api/generate"
    model: str = "qwen2.5-coder:7b"
    timeout: float = 30.0
    temperature: float = 0.0
    max_tokens: int = 512

    # ---- Timeout strategy (spec #12) ----
    # "development" uses `dev_timeout` (short, fail-fast for local
    # iteration); "production" uses `timeout` above (more generous,
    # tolerant of a loaded/cold-start local model). Selected via
    # `AgentConfig.timeout_mode`, resolved by `effective_timeout()`.
    dev_timeout: float = 10.0

    # ---- Retry mechanism (spec #5, #12) ----
    # A single timeout/connection error is never treated as fatal: the
    # request is retried up to `max_retries` additional times with
    # exponential backoff before the caller sees an LLMGenerationError.
    max_retries: int = 2
    retry_backoff_seconds: float = 0.5
    retry_backoff_multiplier: float = 2.0

    # ---- Streaming (spec #5, #15) ----
    # When True, requests are made with stream=True and the newline-
    # delimited JSON chunks Ollama emits are read and concatenated
    # incrementally rather than waiting for one large response body.
    stream: bool = False

    def effective_timeout(self, mode: str) -> float:
        return self.dev_timeout if mode == "development" else self.timeout


@dataclass
class AgentConfig:
    """
    Every tunable value lives here. Nothing dataset-specific is embedded
    in business logic — thresholds below are *fallback* defaults used
    only when adaptive statistics can't be computed (e.g. empty table).
    """
    default_row_limit: int = 10
    max_row_limit: int = 500
    schema_cache_ttl_seconds: float = 30.0

    # Fallback heuristics — used only when a table has too few rows to
    # derive adaptive percentile-based thresholds.
    fallback_categorical_distinct_ratio: float = 0.5
    fallback_categorical_distinct_abs: int = 500
    min_rows_for_adaptive_stats: int = 20

    # Minimum score (0-1) a candidate column must reach to be selected
    # via semantic ranking before we fall back to statistical ranking.
    min_semantic_match_score: float = 0.15

    # Minimum score for a table to be considered an explicit multi-table
    # participant (in addition to a literal name mention in the query).
    min_table_join_score: float = 0.35

    # ---- Semantic Query Planner (spec #2-#10) ----
    # When True, a QueryPlan (intents, metrics, derived metrics,
    # dimensions, row/aggregate filters, sort, joins, relevant schema)
    # is built once per query and handed to the LLM prompt builder and
    # SQL validator, instead of each stage re-deriving structure from
    # raw natural language independently. Purely additive: when False,
    # behavior is identical to the pre-planner pipeline.
    enable_query_planner: bool = True
    # Business/derived-metric glossary (spec #4): maps a generic metric
    # noun-phrase (e.g. "complaint") to a list of extra search terms
    # used -- ALONGSIDE ordinary semantic/fuzzy column matching -- to
    # find the schema column that best indicates that concept. This is
    # linguistic scaffolding only; it never names a specific dataset's
    # table/column, and callers are free to override/extend it per
    # deployment (e.g. add domain terms for their own business metrics).
    derived_metric_glossary: Dict[str, List[str]] = field(default_factory=lambda: {
        "complaint": ["complaint", "negative", "issue", "problem", "dissatisfied", "unhappy"],
        "verified purchase": ["verified", "purchase", "confirmed"],
        "return": ["return", "returned", "refund", "refunded"],
        "conversion": ["converted", "purchase", "checkout", "order"],
        "repeat customer": ["repeat", "returning", "loyal", "recurring"],
        "churn": ["churn", "churned", "cancelled", "canceled", "inactive"],
        "satisfaction": ["satisfaction", "satisfied", "happy", "positive"],
        "resolution": ["resolution", "resolved", "closed"],
        "growth": ["growth", "increase", "change"],
    })


    # SQL validation / automatic repair.
    validate_sql_before_execution: bool = True
    enable_sql_auto_repair: bool = True
    # Spec: "if SQL execution fails because of a recoverable error,
    # attempt one intelligent correction" -- bounded to a single retry
    # by default. Still caller-configurable for environments that want
    # more resilience at the cost of extra latency on the failure path.
    max_repair_attempts: int = 1

    # Statistics engine.
    top_values_sample_size: int = 5

    # LLM-driven SQL generation. When True, the LLM remains AVAILABLE as
    # a generation strategy (see `prefer_deterministic_sql` below for
    # which strategy is tried first).
    use_llm_sql_generation: bool = True
    # If the LLM call fails outright (network/timeout/parse error) or
    # produces SQL that doesn't validate/execute even after repair,
    # fall back to the deterministic rule-based SQLBuilder rather than
    # failing the whole request. Set False to make LLM failures hard
    # errors (useful while debugging prompts).
    llm_fallback_to_rules: bool = True
    # Cap how many columns per table are described in the LLM prompt,
    # to keep very wide tables from blowing up the context window.
    llm_max_columns_per_table: int = 60
    # Cap how many TABLES are described in the LLM prompt. For schemas
    # with many tables, only the most query-relevant ones (by the same
    # SchemaLinker ranking used elsewhere) plus any table literally
    # named in the query are included -- keeps prompt size (and LLM
    # latency/cost) from scaling with total schema size.
    llm_max_tables_in_prompt: int = 12
    # Include a few sample/top values per column in the prompt — helps
    # the LLM pick correct literal values (e.g. exact category spelling)
    # at the cost of a larger prompt.
    llm_include_sample_values: bool = True

    # ---- SQL-attempt confidence gating (Change 9) ----
    # If LLM-generated SQL validates with confidence below this
    # threshold, execution is skipped entirely and the rule-based
    # SQLBuilder fallback is invoked immediately -- avoids spending a
    # DuckDB round-trip (and, on failure, a repair round-trip) on SQL
    # the validator already doesn't trust.
    min_sql_validation_confidence: float = 0.5

    # ---- DEPRECATED (kept for backward compatibility only) ----
    # These no longer gate generation order: per current requirements,
    # LLM SQL generation is always attempted FIRST, with the rule-based
    # SQLBuilder used strictly as a fallback after LLM generation,
    # validation, or execution fails (see StructuredAgent._attempt_llm_sql /
    # _attempt_rule_sql). Retained only so callers that read/set these
    # fields directly don't break.
    prefer_deterministic_sql: bool = True
    deterministic_confidence_threshold: float = 0.55

    # ---- result caching ----
    # Avoids re-running intent detection, re-calling the LLM, and
    # re-executing SQL for identical, repeated requests. Keyed on
    # (query, filters, entities, needs_time_series, schema.version), so
    # it auto-invalidates whenever the schema changes.
    enable_result_cache: bool = True
    result_cache_ttl_seconds: float = 60.0
    result_cache_maxsize: int = 256

    # ---- Timeout strategy (spec #12) ----
    # "development" | "production" — selects LLMConfig.dev_timeout vs
    # LLMConfig.timeout for every Ollama call. Production is the
    # conservative default; switch to "development" for fast-fail local
    # iteration against a model that's already warm.
    timeout_mode: str = "production"

    # ---- Debug mode (spec #17) ----
    # When True (or when `run(..., debug=True)` is passed per-call),
    # every pipeline stage appends a human-readable line to
    # `result.metadata["debug_trace"]` and the trace is also emitted via
    # logger.debug(). See StructuredAgent._DebugTrace.
    debug_mode: bool = False

    # ---- Result row cap (Fix 1) ----
    # Controls the *defensive* outer LIMIT applied to already-generated
    # SQL in `_ensure_row_cap`, independent of `default_row_limit` /
    # `max_row_limit` (which only affect the LIMIT the SQLBuilder/LLM
    # put *inside* a TOP-N style query, e.g. "top 5 products").
    #   - None (default) -> no defensive wrapping at all; the complete
    #     result set the generated SQL produces is returned, e.g.
    #         SELECT DISTINCT "brand" FROM "metadata_data";
    #     executes exactly as generated, with no outer LIMIT added.
    #   - a positive int  -> generated SQL is wrapped in
    #         SELECT * FROM (<sql>) AS _capped_result LIMIT <N>;
    #     whenever the generated SQL doesn't already contain its own
    #     LIMIT clause.
    # This replaces the previous behavior where the outer cap silently
    # reused `default_row_limit` (10), truncating every unbounded query
    # (DISTINCT listings, full result sets, etc.) to 10 rows.
    max_result_rows: Optional[int] = None

    llm: LLMConfig = field(default_factory=LLMConfig)


# =========================================================
# STRUCTURED ERRORS (spec #9)
# =========================================================

@dataclass
class StructuredError:
    """
    Replaces generic strings like "SQL generation failed" with a
    machine- and human-readable breakdown of exactly which pipeline
    stage failed, why, and (when derivable) what to do about it.

    `result.error` (a plain string) is still populated for backward
    compatibility -- it's simply `str(this)` -- but callers that want
    the structured breakdown can read `result.metadata["structured_error"]`.
    """
    stage: str            # e.g. "Schema Discovery", "LLM SQL Generation", "Validation", "Execution"
    reason: str            # short, specific cause, e.g. "Timeout", "Unknown column 'Customer'"
    recommendation: str = ""   # what the caller/system should do next
    suggested_fix: Optional[str] = None  # e.g. "Did you mean 'CustomerName'?"
    detail: Optional[str] = None         # full underlying exception/message, if any
    # NEW (spec #11): coarse, stable failure category -- e.g.
    # "binder_error", "timeout", "llm_timeout", "no_sql_returned".
    # Additive: existing consumers reading stage/reason/detail are
    # unaffected by this always-present-with-default field.
    category: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:
        parts = [f"Stage: {self.stage}", f"Reason: {self.reason}"]
        if self.suggested_fix:
            parts.append(f"Suggested Fix: {self.suggested_fix}")
        if self.recommendation:
            parts.append(f"Recommendation: {self.recommendation}")
        return " | ".join(parts)


# =========================================================
# REQUEST METRICS (spec #14)
# =========================================================

@dataclass
class RequestMetrics:
    """Per-request timing/size metrics for one `StructuredAgent.run()`
    call. Attached to `result.metadata["metrics"]`. All *_ms fields are
    0.0 when the corresponding stage did not run (e.g. cache hit)."""
    schema_discovery_ms: float = 0.0
    intent_detection_ms: float = 0.0
    schema_linking_ms: float = 0.0
    prompt_build_ms: float = 0.0
    llm_latency_ms: float = 0.0
    validation_ms: float = 0.0
    repair_ms: float = 0.0
    execution_ms: float = 0.0
    total_latency_ms: float = 0.0

    prompt_chars: int = 0
    prompt_tokens_estimate: int = 0
    sql_chars: int = 0
    rows_returned: int = 0
    token_usage: Optional[Dict[str, Any]] = None  # populated if Ollama reports eval_count/prompt_eval_count

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentResult:
    success: bool = False
    query_type: str = "structured"

    # Fix 8: `success` is preserved exactly as before (True iff the
    # final SQL executed against DuckDB without an error) so existing
    # callers that only check `if result.success:` are unaffected. This
    # NEW field adds the finer-grained, semantically-aware signal the
    # spec asked for, additively:
    #   "success"          -- executed AND matches the QueryPlan (all
    #                          requested metrics/grouping present, no
    #                          fallback/repair was needed, a non-empty
    #                          result where one was expected).
    #   "partial_success"   -- executed and returned a result, but only
    #                          after repair and/or the rule-based
    #                          fallback, or with a minor plan/SQL
    #                          mismatch (e.g. an optional filter not
    #                          reflected) -- usable, but reviewed with
    #                          reduced confidence.
    #   "failed"            -- execution failed outright, OR the result
    #                          is semantically wrong (queried the wrong
    #                          table, or is missing every requested
    #                          metric).
    status: str = "failed"

    sql: Optional[str] = None

    rows: List[Dict[str, Any]] = field(default_factory=list)
    row_count: int = 0
    columns: List[str] = field(default_factory=list)

    error: Optional[str] = None
    latency_ms: float = 0.0

    metadata: Dict[str, Any] = field(default_factory=dict)

    # ---- confidence / provenance (additive, backward compatible) ----
    planner_confidence: float = 0.0
    sql_confidence: float = 0.0
    schema_confidence: float = 0.0
    execution_confidence: float = 0.0
    repair_confidence: float = 1.0
    # NEW (Change 7): single weighted confidence score combining all
    # of the above, for callers that don't want to hand-weight the
    # individual components themselves.
    overall_confidence: float = 0.0
    tables_used: List[str] = field(default_factory=list)
    join_path: List[str] = field(default_factory=list)

    # NEW (spec #9, #14, #17): structured error breakdown, metrics, and
    # debug trace. All additive/optional -- existing consumers that only
    # read `error`/`metadata` are unaffected.
    structured_error: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict:
        return asdict(self)


# =========================================================
# CONFIDENCE CALIBRATION (spec #14)
# =========================================================
#
# Every confidence value in the pipeline (planner, SQL, validation,
# semantic, execution, repair, overall) is expected to already be
# derived as a value in [0.0, 1.0], but several are weighted sums or
# ratios computed in different places -- a bug in any one of those
# formulas (see the QueryPlanner weight-normalization fix below) can
# silently push a "confidence" outside that range. `_clamp01` is a
# single, cheap defensive backstop applied at every point a confidence
# value is written onto a public result object, so a formula bug
# degrades to a clamped extreme value instead of an out-of-range
# number leaking to callers.

def _clamp01(value: Optional[float]) -> float:
    """Clamps `value` to [0.0, 1.0]. None/NaN/non-numeric input is
    treated as 0.0 rather than raising, since confidence fields must
    never make `AgentResult` construction fail."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    if v != v:  # NaN check without importing math for one comparison
        return 0.0
    return max(0.0, min(1.0, v))


# =========================================================
# SQL STATEMENT SPLITTING / NORMALIZATION / MUTATION TRACKING
# (spec #1, #2, #3, #10)
# =========================================================
#
# One centralized, string-literal-aware utility for (a) splitting raw
# LLM output into individual SQL statements without being fooled by a
# semicolon inside a string literal, quoted identifier, or comment,
# and (b) turning that into a single, clean, normalized statement.
# Every stage downstream of SQL generation (validator, repair engine,
# executor) is expected to operate on SQL that has already passed
# through `SQLNormalizer.extract_and_normalize` exactly once, instead
# of each stage re-implementing its own ad hoc regex-based cleanup.

def _split_sql_statements(text: str) -> List[str]:
    """
    Splits `text` on top-level statement-terminating semicolons.
    Tracks single-quoted string literals, double-quoted identifiers,
    line comments (`-- ...`), and block comments (`/* ... */`) so a
    semicolon *inside* any of those is never mistaken for a statement
    boundary -- e.g. `SELECT * FROM t WHERE name = 'a;b';` is correctly
    seen as one statement, not two. Returns the raw (unstripped)
    segments; a trailing empty segment results from a trailing ';'.
    Purely syntactic (no schema/semantic knowledge) and safe to call on
    any SQL dialect DuckDB accepts.
    """
    statements: List[str] = []
    buf: List[str] = []
    i = 0
    n = len(text)
    in_single = False
    in_double = False
    in_line_comment = False
    in_block_comment = False

    while i < n:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < n else ""

        if in_line_comment:
            buf.append(ch)
            if ch == "\n":
                in_line_comment = False
            i += 1
            continue

        if in_block_comment:
            buf.append(ch)
            if ch == "*" and nxt == "/":
                buf.append(nxt)
                i += 2
                in_block_comment = False
                continue
            i += 1
            continue

        if in_single:
            buf.append(ch)
            if ch == "'" and nxt == "'":  # escaped '' inside a literal
                buf.append(nxt)
                i += 2
                continue
            if ch == "'":
                in_single = False
            i += 1
            continue

        if in_double:
            buf.append(ch)
            if ch == '"' and nxt == '"':  # escaped "" inside an identifier
                buf.append(nxt)
                i += 2
                continue
            if ch == '"':
                in_double = False
            i += 1
            continue

        # Not currently inside any literal/comment.
        if ch == "'":
            in_single = True
            buf.append(ch)
            i += 1
            continue
        if ch == '"':
            in_double = True
            buf.append(ch)
            i += 1
            continue
        if ch == "-" and nxt == "-":
            in_line_comment = True
            buf.append(ch)
            i += 1
            continue
        if ch == "/" and nxt == "*":
            in_block_comment = True
            buf.append(ch)
            i += 1
            continue
        if ch == ";":
            statements.append("".join(buf))
            buf = []
            i += 1
            continue

        buf.append(ch)
        i += 1

    if buf:
        statements.append("".join(buf))
    return statements


@dataclass
class NormalizedSQL:
    """Result of `SQLNormalizer.extract_and_normalize`. `sql` is None
    whenever no single clean executable statement could be produced;
    `rejected_reason` explains why. `extra_statements_dropped` is a
    mutation-tracking signal (spec #10): when > 0, the raw LLM output
    contained more than one executable statement and every statement
    after the first was discarded -- never silently concatenated or
    executed."""
    sql: Optional[str]
    statement_count: int = 0
    extra_statements_dropped: int = 0
    had_markdown: bool = False
    rejected_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SQLNormalizer:
    """
    Centralized SQL extraction + normalization stage (spec #1, #2).

    Turns raw LLM output into exactly one clean SQL statement:
      - strips ``<think>...</think>`` blocks and markdown code fences
      - discards leading explanation/prose (anything before the first
        SELECT/WITH keyword)
      - splits on statement boundaries in a string/comment-aware way
        (see `_split_sql_statements`) and keeps only the FIRST
        executable statement, recording how many extras were dropped
      - trims whitespace and normalizes line endings
      - collapses duplicate/trailing semicolons and guarantees the
        result ends with exactly one

    Normalization here NEVER changes SQL semantics: every step above
    only touches framing (prose, markdown, statement boundary,
    whitespace, semicolon count), never the SQL body's tokens.
    """

    _THINK_BLOCK_RE = re.compile(r"(?is)<think>.*?</think>")
    _FENCE_RE = re.compile(r"```[a-zA-Z]*")
    _STATEMENT_START_RE = re.compile(r"\b(SELECT|WITH)\b", re.IGNORECASE)

    @classmethod
    def extract_and_normalize(cls, raw_text: Optional[str]) -> NormalizedSQL:
        if not raw_text or not raw_text.strip():
            return NormalizedSQL(sql=None, rejected_reason="empty_input")

        text = raw_text.strip()
        had_markdown = "```" in text or "<think>" in text.lower()

        # 1. Remove <think>...</think> blocks (model "reasoning" prose).
        text = cls._THINK_BLOCK_RE.sub("", text)

        # 2. Remove markdown code fences (```sql / ```), keeping the
        #    fenced content itself intact.
        text = cls._FENCE_RE.sub("", text)

        # 3. Normalize line endings.
        text = text.replace("\r\n", "\n").replace("\r", "\n").strip()

        # 4. Discard everything before the first SELECT/WITH keyword --
        #    that's explanatory prose, not SQL.
        match = cls._STATEMENT_START_RE.search(text)
        if not match:
            return NormalizedSQL(sql=None, had_markdown=had_markdown, rejected_reason="no_sql_keyword_found")
        tail = text[match.start():]

        # 5. String/comment-aware statement split; keep only the first
        #    non-empty executable statement. Empty statements (e.g. a
        #    stray trailing ";;") are dropped entirely, never counted
        #    as "extra statements".
        raw_statements = _split_sql_statements(tail)
        statements = [s.strip() for s in raw_statements if s.strip()]
        if not statements:
            return NormalizedSQL(sql=None, had_markdown=had_markdown, rejected_reason="no_executable_statement")

        first = statements[0]
        # Only segments that themselves look like a genuine SQL
        # statement (start with SELECT/WITH) count as "extra
        # statements" -- trailing explanatory prose after the closing
        # semicolon (e.g. "Hope this helps!") is surrounding text, not
        # a second executable statement, and is silently discarded
        # like any other prose rather than inflating the dropped count.
        extra = sum(1 for s in statements[1:] if cls._STATEMENT_START_RE.match(s))

        # 6. Guarantee exactly one trailing semicolon.
        first = first.rstrip().rstrip(";").rstrip()
        normalized = first + ";"

        return NormalizedSQL(
            sql=normalized,
            statement_count=len(statements),
            extra_statements_dropped=extra,
            had_markdown=had_markdown,
        )


class SQLMutationKind(str, Enum):
    """Classification of a SQL edit (spec #10), from least to most
    concerning. Only FORMATTING and SYNTACTIC_REPAIR are things a
    downstream stage is allowed to apply automatically; SEMANTIC means
    the edit changed what the query computes and must be rejected
    rather than silently applied."""
    NONE = "none"
    FORMATTING = "formatting"
    SYNTACTIC_REPAIR = "syntactic_repair"
    SEMANTIC = "semantic"


_STRUCTURAL_TOKEN_RE = re.compile(
    r"\b(SELECT|FROM|WHERE|GROUP\s+BY|HAVING|ORDER\s+BY|LIMIT|JOIN|LEFT|RIGHT|"
    r"INNER|OUTER|UNION|INTERSECT|EXCEPT|DISTINCT|SUM|AVG|COUNT|MIN|MAX|MEDIAN|"
    r"STDDEV|VARIANCE|ASC|DESC|AND|OR|NOT|CASE|WHEN|THEN|ELSE|END)\b"
    r"|(!=|<>|<=|>=|=|<|>)",
    re.IGNORECASE,
)


def classify_sql_mutation(original: Optional[str], modified: Optional[str]) -> str:
    """
    Compares `original` SQL against `modified` SQL (spec #10) and
    returns a `SQLMutationKind` value. This is a conservative
    structural heuristic, not a SQL parser or equivalence checker --
    it is meant to flag likely-semantic edits for logging/rejection,
    not to *prove* two queries are equivalent.

    Method: build a "structural fingerprint" of each query -- the
    ordered sequence of clause keywords, aggregate functions, boolean/
    comparison operators, and join types, ignoring identifiers,
    literals, aliases, and whitespace/case. If the fingerprints match,
    only names changed (a rename, re-qualification, or an added GROUP
    BY/HAVING entry that doesn't introduce a new clause keyword) --
    classified as a repair. If the fingerprints differ, the query's
    shape changed -- classified as semantic.
    """
    if original is None or modified is None:
        return SQLMutationKind.NONE.value

    def _collapse_ws(s: str) -> str:
        return re.sub(r"\s+", " ", s.strip()).lower()

    if _collapse_ws(original) == _collapse_ws(modified):
        return SQLMutationKind.NONE.value

    def _fingerprint(s: str) -> List[str]:
        out = []
        for m in _STRUCTURAL_TOKEN_RE.finditer(s):
            tok = m.group(1) or m.group(2)
            out.append(re.sub(r"\s+", " ", tok).upper())
        return out

    orig_fp = _fingerprint(original)
    mod_fp = _fingerprint(modified)

    if orig_fp == mod_fp:
        return SQLMutationKind.SYNTACTIC_REPAIR.value

    # A fingerprint that is a superset of the original (e.g. one
    # GROUP BY/HAVING entry appended, nothing removed) is still a
    # structural repair, not a semantic rewrite -- the original
    # clauses are all still present, in the same order, just with
    # additions.
    if len(mod_fp) >= len(orig_fp) and mod_fp[: len(orig_fp)] == orig_fp:
        return SQLMutationKind.SYNTACTIC_REPAIR.value

    return SQLMutationKind.SEMANTIC.value


# =========================================================
# EXECUTION ERROR CLASSIFICATION (spec #11)
# =========================================================
#
# Maps a raw backend (DuckDB) exception message to a coarse, stable
# category. The ORIGINAL exception text is never discarded or replaced
# -- callers keep `result.error` / `StructuredError.detail` verbatim --
# this classification is attached alongside it, purely additively, so
# metadata/dashboards/alerting can group failures by category instead
# of pattern-matching raw exception strings themselves.

_EXECUTION_ERROR_PATTERNS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("timeout", ("timeout", "timed out")),
    ("connection_error", ("connection refused", "could not connect", "connection reset", "broken pipe", "connection closed")),
    ("permission_error", ("permission denied", "access denied", "not authorized", "unauthorized")),
    ("parser_error", ("parser error",)),
    ("binder_error", ("binder error",)),
    ("catalog_error", ("catalog error",)),
    ("syntax_error", ("syntax error",)),
    ("constraint_violation", ("constraint error", "constraint violation")),
    ("unsupported_sql", ("not implemented", "not supported", "unsupported")),
    ("out_of_memory", ("out of memory", "memory limit")),
)


def classify_execution_error(message: Optional[str]) -> str:
    """
    Maps a raw execution-exception message to one of: syntax_error,
    parser_error, binder_error, catalog_error, constraint_violation,
    timeout, connection_error, permission_error, unsupported_sql,
    out_of_memory, or the generic fallback "execution_error". Returns
    "unknown_error" only when there is no message at all to classify.
    """
    if not message:
        return "unknown_error"
    m = message.lower()
    for category, needles in _EXECUTION_ERROR_PATTERNS:
        if any(needle in m for needle in needles):
            return category
    return "execution_error"


# =========================================================
# TOKENIZATION / SEMANTIC UTILITIES
# (small, dependency-free — no dataset knowledge baked in)
# =========================================================

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    return _TOKEN_PATTERN.findall(text.lower())


def split_identifier(name: str) -> List[str]:
    """snake_case / camelCase / kebab-case -> tokens."""
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name)
    spaced = spaced.replace("_", " ").replace("-", " ")
    return tokenize(spaced)


# A small, generic synonym table. This is *linguistic* knowledge (words
# that mean the same thing in English), not dataset knowledge — it lets
# the agent connect "revenue" in a question to a column named "sales"
# without hardcoding any table/column name.
_SYNONYM_GROUPS: List[Set[str]] = [
    {"sales", "revenue", "income", "earnings", "turnover"},
    {"price", "cost", "amount", "value", "fee", "charge"},
    {"rating", "score", "grade", "rank"},
    {"count", "quantity", "qty", "number", "volume", "total"},
    {"customer", "client", "user", "buyer", "account"},
    {"product", "item", "sku", "good"},
    {"date", "day", "time", "period", "timestamp"},
    {"category", "type", "class", "group", "segment"},
    {"region", "location", "area", "zone", "territory"},
    {"employee", "staff", "worker", "personnel"},
    {"name", "title", "label"},
    {"id", "identifier", "key", "code"},
]

_SYNONYM_LOOKUP: Dict[str, Set[str]] = {}
for _group in _SYNONYM_GROUPS:
    for _word in _group:
        _SYNONYM_LOOKUP[_word] = _group


def expand_with_synonyms(tokens: Sequence[str]) -> Set[str]:
    expanded: Set[str] = set(tokens)
    for tok in tokens:
        expanded |= _SYNONYM_LOOKUP.get(tok, set())
    return expanded


def token_overlap_score(a_tokens: Set[str], b_tokens: Set[str]) -> float:
    """Jaccard-style overlap, normalized to [0, 1]."""
    if not a_tokens or not b_tokens:
        return 0.0
    intersection = a_tokens & b_tokens
    if not intersection:
        return 0.0
    return len(intersection) / min(len(a_tokens), len(b_tokens))


def fuzzy_name_score(a: str, b: str) -> float:
    """
    General-purpose fuzzy identifier similarity, used (in addition to
    token-overlap) to connect naming variants that don't tokenize the
    same way — e.g. "customerID", "customer_id", "cust_id", "custId" —
    without ever hardcoding any of those specific spellings. Combines
    a character-sequence ratio with token overlap so both "close
    spelling" and "close meaning" naming styles are caught.
    """
    a_norm = "".join(split_identifier(a))
    b_norm = "".join(split_identifier(b))
    if not a_norm or not b_norm:
        return 0.0
    seq_score = difflib.SequenceMatcher(None, a_norm, b_norm).ratio()
    tok_score = token_overlap_score(
        expand_with_synonyms(split_identifier(a)), expand_with_synonyms(split_identifier(b))
    )
    return max(seq_score, tok_score)


# =========================================================
# SCHEMA DISCOVERY
# =========================================================

class ColumnKind(str, Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    TEMPORAL = "temporal"
    BOOLEAN = "boolean"
    IDENTIFIER = "identifier"
    TEXTUAL = "textual"
    UNKNOWN = "unknown"


_NUMERIC_TYPE_HINTS = ("int", "decimal", "numeric", "float", "double", "real", "hugeint")
_TEMPORAL_TYPE_HINTS = ("timestamp", "date", "time", "interval")
_BOOLEAN_TYPE_HINTS = ("bool",)
_TEXTUAL_TYPE_HINTS = ("varchar", "char", "text", "string", "blob", "json")

# Weak secondary signal only — never the sole basis for classification.
_TEMPORAL_NAME_HINTS = ("date", "time", "timestamp", "period")


@dataclass
class ColumnProfile:
    """Runtime statistical profile of a single column."""
    row_count: int = 0
    null_count: int = 0
    distinct_count: int = 0
    min_value: Any = None
    max_value: Any = None
    avg_length: Optional[float] = None
    numeric_stddev: Optional[float] = None
    numeric_avg: Optional[float] = None
    numeric_variance: Optional[float] = None
    top_values: List[Tuple[Any, int]] = field(default_factory=list)

    @property
    def null_ratio(self) -> float:
        return self.null_count / self.row_count if self.row_count else 0.0

    @property
    def distinct_ratio(self) -> float:
        return self.distinct_count / self.row_count if self.row_count else 0.0

    @property
    def is_unique(self) -> bool:
        return self.row_count > 1 and self.distinct_count == self.row_count


@dataclass
class ColumnInfo:
    name: str
    sql_type: str
    nullable: bool
    kind: ColumnKind = ColumnKind.UNKNOWN
    profile: ColumnProfile = field(default_factory=ColumnProfile)
    semantic_tokens: Set[str] = field(default_factory=set)
    # True for candidate-key columns that are unique but were not chosen
    # as *the* primary key (e.g. a secondary unique code column).
    is_candidate_key: bool = False

    def matches_query(self, query_tokens: Set[str]) -> float:
        return token_overlap_score(self.semantic_tokens, query_tokens)

    def fuzzy_matches(self, name_or_phrase: str) -> float:
        return fuzzy_name_score(self.name, name_or_phrase)


@dataclass
class TableInfo:
    name: str
    columns: List[ColumnInfo] = field(default_factory=list)
    row_count: Optional[int] = None
    primary_key: Optional[str] = None
    semantic_tokens: Set[str] = field(default_factory=set)
    candidate_keys: List[str] = field(default_factory=list)
    # Marks synthetic, joined "virtual" tables produced by the join
    # planner; `name` for these already contains a full FROM/JOIN
    # expression and must NOT be re-quoted as a plain identifier.
    is_virtual: bool = False

    def columns_of(self, *kinds: ColumnKind) -> List[ColumnInfo]:
        return [c for c in self.columns if c.kind in kinds]

    def column_names(self) -> List[str]:
        return [c.name for c in self.columns]

    def get_column(self, name: str) -> Optional[ColumnInfo]:
        for c in self.columns:
            if c.name == name:
                return c
        return None

    def best_fuzzy_column(self, phrase: str, cutoff: float = 0.55) -> Optional[ColumnInfo]:
        """Best column whose name fuzzily matches an arbitrary phrase
        (e.g. "customerID", "cust id", "customer_id")."""
        scored = [(c.fuzzy_matches(phrase), c) for c in self.columns]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        if scored and scored[0][0] >= cutoff:
            return scored[0][1]
        return None


@dataclass
class Relationship:
    """A candidate foreign-key relationship inferred at runtime."""
    from_table: str
    from_column: str
    to_table: str
    to_column: str
    confidence: float
    cardinality: str  # "one_to_one" | "one_to_many" | "many_to_one"


@dataclass
class DiscoveredSchema:
    tables: Dict[str, TableInfo] = field(default_factory=dict)
    relationships: List[Relationship] = field(default_factory=list)
    discovered_at: float = 0.0
    version: int = 0

    def is_empty(self) -> bool:
        return len(self.tables) == 0

    def get_table(self, name: Optional[str]) -> Optional[TableInfo]:
        return self.tables.get(name) if name else None

    def related_tables(self, table_name: str) -> List[Tuple[TableInfo, Relationship]]:
        found = []
        for rel in self.relationships:
            other_name = None
            if rel.from_table == table_name:
                other_name = rel.to_table
            elif rel.to_table == table_name:
                other_name = rel.from_table
            if other_name and other_name in self.tables:
                found.append((self.tables[other_name], rel))
        return found

    def connectivity(self, table_name: str) -> int:
        """How many relationships touch this table — used as a relevance signal."""
        return sum(
            1 for rel in self.relationships
            if table_name in (rel.from_table, rel.to_table)
        )

    def confidence_score(self) -> float:
        """
        A rough measure of how confidently the schema was understood:
        fraction of columns with a non-UNKNOWN kind, weighted by whether
        tables have any rows to profile at all. Used to temper downstream
        planner/SQL confidence when the underlying schema is thin/empty.
        """
        if self.is_empty():
            return 0.0
        total_cols = 0
        known_cols = 0
        populated_tables = 0
        for t in self.tables.values():
            if (t.row_count or 0) > 0:
                populated_tables += 1
            for c in t.columns:
                total_cols += 1
                if c.kind != ColumnKind.UNKNOWN:
                    known_cols += 1
        col_score = known_cols / total_cols if total_cols else 0.0
        pop_score = populated_tables / len(self.tables)
        return round(0.7 * col_score + 0.3 * pop_score, 3)

    def as_registry_dict(self) -> Dict[str, Any]:
        """
        Flat, JSON-serializable view of the discovered schema — table
        names, columns, data types, per-kind column buckets, row counts,
        and primary-key candidates. Handy for debugging/introspection
        or exposing schema info to a caller/UI without leaking internal
        dataclasses.
        """
        registry: Dict[str, Any] = {}
        for name, t in self.tables.items():
            registry[name] = {
                "row_count": t.row_count,
                "primary_key": t.primary_key,
                "candidate_keys": list(t.candidate_keys),
                "columns": [
                    {"name": c.name, "sql_type": c.sql_type, "kind": c.kind.value}
                    for c in t.columns
                ],
                "numeric_columns": [c.name for c in t.columns_of(ColumnKind.NUMERIC)],
                "textual_columns": [c.name for c in t.columns_of(ColumnKind.TEXTUAL)],
                "categorical_columns": [c.name for c in t.columns_of(ColumnKind.CATEGORICAL)],
                "boolean_columns": [c.name for c in t.columns_of(ColumnKind.BOOLEAN)],
                "temporal_columns": [c.name for c in t.columns_of(ColumnKind.TEMPORAL)],
                "identifier_columns": [c.name for c in t.columns_of(ColumnKind.IDENTIFIER)],
            }
        return registry


class SchemaInspector:
    """
    Introspects a live DuckDB connection to build a fully profiled
    DiscoveredSchema, including inferred column roles and cross-table
    relationships. Contains zero references to any specific table or
    column name.

    Discovery works against *whatever tables already exist* on the
    connection — whether they were created via seed_dataframe()/
    seed_csv(), or already existed in the underlying DuckDB database
    file before the agent was constructed (e.g. a persistent .duckdb
    file containing `structured_data`, `metadata_data`, etc.). SHOW
    TABLES / DESCRIBE are used as a fallback/cross-check alongside
    information_schema so pre-existing tables are always picked up.
    """

    def __init__(self, conn: "duckdb.DuckDBPyConnection", config: AgentConfig):
        self.conn = conn
        self.config = config

    def discover(self) -> DiscoveredSchema:
        t0 = time.perf_counter()
        tables: Dict[str, TableInfo] = {}

        for table_name in self._list_table_names():
            try:
                tables[table_name] = self._discover_table(table_name)
            except Exception:
                logger.exception("schema_discovery: failed to profile table %r", table_name)

        relationships = self._discover_relationships(tables)

        schema = DiscoveredSchema(
            tables=tables, relationships=relationships, discovered_at=time.time()
        )
        logger.info(
            "schema_discovery: tables=%d relationships=%d elapsed_ms=%.1f",
            len(tables), len(relationships), (time.perf_counter() - t0) * 1000,
        )
        return schema

    # -----------------------------------------------------
    # Table listing (auto-discovery, no seeding required)
    # -----------------------------------------------------

    def _list_table_names(self) -> List[str]:
        names: Set[str] = set()

        try:
            rows = self.conn.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema NOT IN ('information_schema', 'pg_catalog')
                """
            ).fetchall()
            names.update(r[0] for r in rows)
        except Exception:
            logger.debug("schema_discovery: information_schema.tables query failed", exc_info=True)

        # Fallback / cross-check: SHOW TABLES also surfaces tables that
        # exist directly on the connection (e.g. a persistent .duckdb
        # file opened without ever calling seed_dataframe()), which is
        # what makes the agent usable immediately against an existing
        # database rather than only against explicitly-seeded frames.
        try:
            rows = self.conn.execute("SHOW TABLES").fetchall()
            names.update(r[0] for r in rows)
        except Exception:
            logger.debug("schema_discovery: SHOW TABLES failed", exc_info=True)

        return sorted(names)

    # -----------------------------------------------------
    # Table / column discovery
    # -----------------------------------------------------

    def _discover_table(self, table_name: str) -> TableInfo:
        col_rows = self._describe_columns(table_name)
        row_count = self._safe_row_count(table_name)
        columns: List[ColumnInfo] = []

        for col_name, data_type, is_nullable in col_rows:
            profile = self._profile_column(table_name, col_name, data_type, row_count)
            kind = self._classify_column(col_name, data_type, profile, row_count)
            columns.append(
                ColumnInfo(
                    name=col_name,
                    sql_type=data_type,
                    nullable=(str(is_nullable).upper() in ("YES", "TRUE")),
                    kind=kind,
                    profile=profile,
                    semantic_tokens=expand_with_synonyms(split_identifier(col_name)),
                )
            )

        primary_key = self._infer_primary_key(columns, row_count)
        candidate_keys = self._infer_candidate_keys(columns, primary_key)
        for c in columns:
            if c.name in candidate_keys:
                c.is_candidate_key = True

        return TableInfo(
            name=table_name,
            columns=columns,
            row_count=row_count,
            primary_key=primary_key,
            candidate_keys=candidate_keys,
            semantic_tokens=expand_with_synonyms(split_identifier(table_name)),
        )

    def _describe_columns(self, table_name: str) -> List[Tuple[str, str, Any]]:
        """
        Column metadata via information_schema first (gives a clean
        is_nullable flag); falls back to DESCRIBE, which works for any
        table/view DuckDB knows about even if information_schema is for
        some reason unavailable or lagging (e.g. certain attached DBs).
        """
        try:
            rows = self.conn.execute(
                """
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_name = ?
                ORDER BY ordinal_position
                """,
                [table_name],
            ).fetchall()
            if rows:
                return [(r[0], r[1], r[2]) for r in rows]
        except Exception:
            logger.debug("describe_columns: information_schema.columns failed for %r", table_name, exc_info=True)

        try:
            rows = self.conn.execute(f'DESCRIBE "{table_name}"').fetchall()
            # DESCRIBE returns: column_name, column_type, null, key, default, extra
            return [(r[0], r[1], r[2]) for r in rows]
        except Exception:
            logger.debug("describe_columns: DESCRIBE failed for %r", table_name, exc_info=True)
            return []

    def _safe_row_count(self, table_name: str) -> int:
        try:
            result = self.conn.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()
            return int(result[0]) if result else 0
        except Exception:
            return 0

    # -----------------------------------------------------
    # Statistical profiling
    # -----------------------------------------------------

    def _profile_column(
        self, table_name: str, col_name: str, data_type: str, row_count: int
    ) -> ColumnProfile:
        profile = ColumnProfile(row_count=row_count)
        if row_count == 0:
            return profile

        dt = data_type.lower()
        is_numeric_type = any(h in dt for h in _NUMERIC_TYPE_HINTS)

        try:
            base_stats = self.conn.execute(
                f'''
                SELECT
                    COUNT(*) - COUNT("{col_name}") AS nulls,
                    COUNT(DISTINCT "{col_name}") AS distinct_count,
                    MIN("{col_name}") AS min_val,
                    MAX("{col_name}") AS max_val
                FROM "{table_name}"
                '''
            ).fetchone()

            if base_stats:
                profile.null_count = int(base_stats[0] or 0)
                profile.distinct_count = int(base_stats[1] or 0)
                profile.min_value = base_stats[2]
                profile.max_value = base_stats[3]

        except Exception:
            pass

        if is_numeric_type:
            try:
                stat_row = self.conn.execute(
                    f'''
                    SELECT STDDEV("{col_name}"), AVG("{col_name}"), VARIANCE("{col_name}")
                    FROM "{table_name}"
                    '''
                ).fetchone()
                if stat_row:
                    profile.numeric_stddev = float(stat_row[0]) if stat_row[0] is not None else None
                    profile.numeric_avg = float(stat_row[1]) if stat_row[1] is not None else None
                    profile.numeric_variance = float(stat_row[2]) if stat_row[2] is not None else None
            except Exception:
                pass
        else:
            try:
                len_row = self.conn.execute(
                    f'SELECT AVG(LENGTH(CAST("{col_name}" AS VARCHAR))) FROM "{table_name}"'
                ).fetchone()
                profile.avg_length = float(len_row[0]) if len_row and len_row[0] is not None else None
            except Exception:
                pass

        # Top-value frequency sample — informs categorical/grouping
        # decisions and entity-resolution ("most popular X").
        try:
            top_rows = self.conn.execute(
                f'''
                SELECT "{col_name}", COUNT(*) AS freq
                FROM "{table_name}"
                WHERE "{col_name}" IS NOT NULL
                GROUP BY "{col_name}"
                ORDER BY freq DESC
                LIMIT {self.config.top_values_sample_size}
                '''
            ).fetchall()
            profile.top_values = [(r[0], int(r[1])) for r in top_rows]
        except Exception:
            pass

        return profile

    # -----------------------------------------------------
    # Adaptive classification
    # -----------------------------------------------------

    def _classify_column(
        self, col_name: str, data_type: str, profile: ColumnProfile, row_count: int
    ) -> ColumnKind:
        dt = data_type.lower()
        name_tokens = set(split_identifier(col_name))

        # 1) Strong signal: SQL type family.
        if any(h in dt for h in _BOOLEAN_TYPE_HINTS):
            return ColumnKind.BOOLEAN

        if any(h in dt for h in _TEMPORAL_TYPE_HINTS):
            return ColumnKind.TEMPORAL

        base_kind: ColumnKind
        if any(h in dt for h in _NUMERIC_TYPE_HINTS):
            base_kind = ColumnKind.NUMERIC
        elif any(h in dt for h in _TEXTUAL_TYPE_HINTS):
            base_kind = ColumnKind.TEXTUAL
        else:
            base_kind = ColumnKind.UNKNOWN

        # 2) Identifier detection via statistics. Uniqueness alone is not
        #    sufficient — a continuous measurement (e.g. a precise price
        #    or an age sampled without repeats) can be 100% unique in a
        #    small table without being a surrogate key. We require a
        #    second, independent signal before classifying as IDENTIFIER:
        #      - textual: short/uniform value length (typical of codes),
        #      - numeric: a "dense" value range (max-min+1 close to the
        #        distinct count, i.e. behaves like a generated sequence)
        #        combined with an integer type, since real surrogate keys
        #        are overwhelmingly sequential/dense integers whereas
        #        continuous measurements are not.
        if row_count > 1 and profile.is_unique and profile.null_ratio == 0.0:
            if base_kind == ColumnKind.TEXTUAL and (
                profile.avg_length is None or profile.avg_length <= 40
            ):
                return ColumnKind.IDENTIFIER
            if base_kind == ColumnKind.NUMERIC and self._looks_like_sequence(
                dt, profile
            ):
                return ColumnKind.IDENTIFIER

        # 3) Textual columns that look like dates (weak name hint +
        #    parseability would be the strongest signal, but we avoid an
        #    expensive per-value parse pass; name hint is a reasonable
        #    fallback signal here since type metadata alone is ambiguous).
        if base_kind == ColumnKind.TEXTUAL and name_tokens & set(_TEMPORAL_NAME_HINTS):
            return ColumnKind.TEMPORAL

        # 4) Categorical vs. free text / high-cardinality numeric, decided
        #    adaptively from the observed distinct ratio rather than a
        #    single global constant.
        if row_count >= self.config.min_rows_for_adaptive_stats:
            threshold = self._adaptive_categorical_threshold(row_count)
        else:
            threshold = self.config.fallback_categorical_distinct_ratio

        if base_kind in (ColumnKind.TEXTUAL, ColumnKind.NUMERIC):
            distinct_abs_ok = profile.distinct_count <= max(
                self.config.fallback_categorical_distinct_abs,
                int(row_count * threshold),
            )
            if profile.distinct_ratio <= threshold and distinct_abs_ok:
                return ColumnKind.CATEGORICAL

        return base_kind if base_kind != ColumnKind.UNKNOWN else ColumnKind.TEXTUAL

    def _looks_like_sequence(self, sql_type: str, profile: ColumnProfile) -> bool:
        """
        True when a unique numeric column's values look like a generated
        surrogate key (dense integer range) rather than a continuous
        measurement. Continuous values (prices, ages, scores, ...) are
        typically not dense integer sequences even when unique.
        """
        is_integer_type = "int" in sql_type and not any(
            h in sql_type for h in ("float", "double", "decimal", "numeric", "real")
        )
        if not is_integer_type:
            return False
        try:
            value_range = float(profile.max_value) - float(profile.min_value)
        except (TypeError, ValueError):
            return False
        if value_range <= 0:
            return False
        density = profile.distinct_count / (value_range + 1)
        return density >= 0.9

    def _adaptive_categorical_threshold(self, row_count: int) -> float:
        """
        Larger tables can tolerate a lower distinct-ratio ceiling for
        "categorical" (100 distinct values out of 100 rows is not a
        category; 100 distinct values out of 1,000,000 rows is). This
        scales the ratio down logarithmically with table size instead of
        using one fixed constant for every dataset size.
        """
        scale = 1.0 / (1.0 + math.log10(max(row_count, 10) / 10))
        return max(0.02, min(0.5, scale))

    def _infer_primary_key(
        self, columns: List[ColumnInfo], row_count: int
    ) -> Optional[str]:
        # Prefer columns already classified as IDENTIFIER, but fall back
        # to any fully-unique, fully-populated column (numeric or text)
        # so tables without an obvious surrogate key still get a usable
        # join anchor for relationship discovery.
        identifier_kind = [c for c in columns if c.kind == ColumnKind.IDENTIFIER]
        candidates = identifier_kind or [
            c for c in columns
            if c.profile.is_unique and c.profile.null_ratio == 0.0
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda c: (c.profile.avg_length or 0))
        return candidates[0].name

    def _infer_candidate_keys(
        self, columns: List[ColumnInfo], primary_key: Optional[str]
    ) -> List[str]:
        """
        Any fully-unique, fully-populated column other than the chosen
        primary key is a candidate key — useful both for join discovery
        (a foreign key may reference a candidate key rather than the
        primary key) and for statistics/validation.
        """
        return [
            c.name for c in columns
            if c.name != primary_key and c.profile.is_unique and c.profile.null_ratio == 0.0
        ]

    # -----------------------------------------------------
    # Relationship discovery
    # -----------------------------------------------------

    def _discover_relationships(
        self, tables: Dict[str, TableInfo]
    ) -> List[Relationship]:
        relationships: List[Relationship] = []
        table_list = list(tables.values())

        for table in table_list:
            identifier_like = [
                c for c in table.columns
                if c.kind in (ColumnKind.IDENTIFIER, ColumnKind.NUMERIC, ColumnKind.CATEGORICAL, ColumnKind.TEXTUAL)
            ]
            for col in identifier_like:
                for other in table_list:
                    if other.name == table.name:
                        continue
                    key_columns = [other.primary_key] if other.primary_key else []
                    key_columns += [k for k in other.candidate_keys if k not in key_columns]
                    for key_name in key_columns:
                        pk = other.get_column(key_name)
                        if pk is None:
                            continue
                        rel = self._evaluate_relationship(table, col, other, pk)
                        if rel is not None:
                            relationships.append(rel)

        # Deduplicate: keep the highest-confidence relationship per
        # (from_table, from_column, to_table) triple.
        best: Dict[Tuple[str, str, str], Relationship] = {}
        for rel in relationships:
            key = (rel.from_table, rel.from_column, rel.to_table)
            if key not in best or rel.confidence > best[key].confidence:
                best[key] = rel
        return list(best.values())

    def _evaluate_relationship(
        self, from_table: TableInfo, from_col: ColumnInfo,
        to_table: TableInfo, to_col: ColumnInfo,
    ) -> Optional[Relationship]:
        # Semantic name similarity between the candidate foreign key and
        # the target primary/candidate key (e.g. "customer_id" <-> "id"
        # in "customers"). Combines token overlap with general fuzzy
        # string similarity so naming variants like "customerID" /
        # "cust_id" / "CustomerId" are matched without any hardcoded
        # spelling.
        name_score = max(
            token_overlap_score(
                from_col.semantic_tokens, to_col.semantic_tokens | to_table.semantic_tokens
            ),
            fuzzy_name_score(from_col.name, to_col.name),
            fuzzy_name_score(from_col.name, f"{to_table.name}_{to_col.name}"),
        )

        # Value-overlap signal: independent of naming, a real foreign key
        # column should contain values that are (mostly) a subset of the
        # target key's observed value range/top values. This is a cheap,
        # sample-based proxy (using min/max/top-values already profiled)
        # rather than an expensive full containment scan, but it means a
        # foreign key can be discovered even when naming gives no hint at
        # all (e.g. "region_id" -> "code" in "regions").
        value_score = self._value_overlap_score(from_col, to_col)

        combined_score = max(name_score, value_score)
        if combined_score < 0.2:
            return None

        # Cardinality compatibility: a full cross-table value scan would
        # be exact but expensive; as a lightweight proxy we compare
        # distinct-value counts to classify the likely cardinality.
        if from_col.profile.is_unique and to_col.profile.is_unique:
            cardinality = "one_to_one"
        elif from_col.profile.distinct_count <= to_col.profile.distinct_count:
            cardinality = "many_to_one"
        else:
            cardinality = "one_to_many"

        # Datatype compatibility is a hard requirement, not just a signal.
        if not self._types_compatible(from_col.sql_type, to_col.sql_type):
            return None

        confidence = round(min(1.0, 0.6 * name_score + 0.4 * value_score + (0.15 if to_col.kind == ColumnKind.IDENTIFIER else 0.0)), 3)
        confidence = min(1.0, max(confidence, 0.2))

        return Relationship(
            from_table=from_table.name,
            from_column=from_col.name,
            to_table=to_table.name,
            to_column=to_col.name,
            confidence=confidence,
            cardinality=cardinality,
        )

     
    def _types_compatible(self, type_a, type_b):
        a, b = type_a.lower(), type_b.lower()

        if a == b:
            return True

        a_numeric = any(h in a for h in _NUMERIC_TYPE_HINTS)
        b_numeric = any(h in b for h in _NUMERIC_TYPE_HINTS)
        if a_numeric and b_numeric:
            return True

        a_text = any(h in a for h in _TEXTUAL_TYPE_HINTS)
        b_text = any(h in b for h in _TEXTUAL_TYPE_HINTS)
        if a_text and b_text:
            return True

        a_temporal = any(h in a for h in _TEMPORAL_TYPE_HINTS)
        b_temporal = any(h in b for h in _TEMPORAL_TYPE_HINTS)

        # FIX: allow partial temporal compatibility
        if a_temporal or b_temporal:
            return True

        return False

     
    def _value_overlap_score(self, from_col, to_col):
        """
        Robust statistical overlap score between two columns.

        Combines:
        1. Range containment (min/max overlap)
        2. Top-value intersection

        Returns score in [0, 1].
        """

        score = 0.0

        # ============================================================
        # FIX 1: SAFE RANGE COMPARISON
        # ============================================================
        try:
            fmin = from_col.profile.min_value
            fmax = from_col.profile.max_value
            tmin = to_col.profile.min_value
            tmax = to_col.profile.max_value

            # Ensure numeric comparability
            if all(v is not None for v in (fmin, fmax, tmin, tmax)):
                try:
                    fmin, fmax = float(fmin), float(fmax)
                    tmin, tmax = float(tmin), float(tmax)

                    # proper overlap check (not strict containment only)
                    ranges_overlap = not (fmax < tmin or fmin > tmax)

                    if ranges_overlap:
                        score += 0.5
                except (TypeError, ValueError):
                    pass

        except Exception:
            pass

        # ============================================================
        # FIX 2: SAFE TOP VALUE OVERLAP
        # ============================================================
        try:
            from_top = {v for v, _ in (from_col.profile.top_values or [])}
            to_top = {v for v, _ in (to_col.profile.top_values or [])}

            if from_top and to_top:
                intersection = from_top.intersection(to_top)

                if intersection:
                    # normalize contribution by overlap strength
                    score += 0.5 * (len(intersection) / max(len(from_top), 1))

        except Exception:
            pass

        # ============================================================
        # FIX 3: clamp final score
        # ============================================================
        return max(0.0, min(1.0, score))

class SchemaCache:
    """Thin TTL cache around SchemaInspector so discovery doesn't run per-query."""

    def __init__(self, config: AgentConfig):
        self._config = config
        self._schema: Optional[DiscoveredSchema] = None
        self._lock = threading.Lock()
        self._version = 0

    def get(self, conn: "duckdb.DuckDBPyConnection") -> DiscoveredSchema:
        with self._lock:
            if self._schema is None or self._is_stale():
                self._version += 1
                self._schema = SchemaInspector(conn, self._config).discover()
                self._schema.version = self._version
            return self._schema

    def invalidate(self) -> None:
        with self._lock:
            self._schema = None

    def _is_stale(self) -> bool:
        if self._schema is None:
            return True
        return (time.time() - self._schema.discovered_at) > self._config.schema_cache_ttl_seconds


# =========================================================
# INTENT DETECTION
# =========================================================

class Intent(str, Enum):
    COUNT = "count"
    SUM = "sum"
    AVERAGE = "average"
    MIN = "min"
    MAX = "max"
    MEDIAN = "median"
    STDDEV = "stddev"
    TOP_N = "top_n"
    BOTTOM_N = "bottom_n"
    DISTINCT = "distinct"
    FREQUENCY = "frequency"
    TREND = "trend"
    PERCENTAGE = "percentage"
    COMPARISON = "comparison"
    SEARCH = "search"
    JOIN = "join"
    LIST = "list"


# Synonym sets per intent — this is what replaces rigid, order-dependent
# regex priority with a scored match across natural-language variations.
_INTENT_SYNONYMS: Dict[Intent, Set[str]] = {
    Intent.TOP_N: {"top", "best", "highest", "largest", "greatest", "most"},
    Intent.BOTTOM_N: {"bottom", "worst", "lowest", "smallest", "least"},
    Intent.MEDIAN: {"median", "middle"},
    Intent.STDDEV: {"stddev", "deviation", "variance", "spread", "volatility"},
    Intent.AVERAGE: {"average", "avg", "mean", "typical"},
    Intent.SUM: {"total", "sum", "aggregate", "combined"},
    Intent.MIN: {"minimum", "min", "smallest", "lowest"},
    Intent.MAX: {"maximum", "max", "highest", "largest", "peak"},
    Intent.DISTINCT: {"distinct", "unique", "different"},
    Intent.FREQUENCY: {"frequency", "frequent", "common", "popular", "occurrences"},
    Intent.TREND: {
        "trend", "trends", "overtime", "time", "series", "hourly", "daily",
        "weekly", "monthly", "quarterly", "yearly", "annually", "history",
        "historical", "growth", "change", "moving", "rolling",
    },
    Intent.PERCENTAGE: {"percentage", "percent", "proportion", "ratio", "share"},
    Intent.COMPARISON: {"compare", "comparison", "versus", "vs", "difference", "between"},
    Intent.SEARCH: {"search", "find", "contains", "containing", "matching", "like"},
    Intent.JOIN: {"join", "combined", "together", "across", "related"},
    Intent.COUNT: {"count", "how", "many", "number"},
}

_GROUPING_HINTS = {"by", "per", "each", "group", "grouped", "across", "breakdown", "segment"}

_N_PATTERN = re.compile(
    r"\btop\s+(\d+)\b|\bfirst\s+(\d+)\b|\bbottom\s+(\d+)\b|\blast\s+(\d+)\b"
    r"|\bhighest\s+(\d+)\b|\blowest\s+(\d+)\b|\blatest\s+(\d+)\b"
)

_GRANULARITY_HINTS: Dict[str, str] = {
    "hour": "hour", "hourly": "hour",
    "day": "day", "daily": "day",
    "week": "week", "weekly": "week",
    "month": "month", "monthly": "month",
    "quarter": "quarter", "quarterly": "quarter",
    "year": "year", "yearly": "year", "annually": "year", "annual": "year",
}


@dataclass
class DetectedIntent:
    intent: Intent
    requested_n: Optional[int] = None
    wants_grouping: bool = False
    granularity: Optional[str] = None
    raw_query: str = ""
    query_tokens: Set[str] = field(default_factory=set)
    confidence: float = 0.0
    wants_moving_average: bool = False
    wants_growth_rate: bool = False


class IntentDetector:
    """
    Scores every candidate intent against the query's token set (expanded
    with synonyms) and returns the highest-scoring match, rather than the
    first regex that happens to fire.
    """

    def detect(self, query: str, needs_time_series_hint: bool = False) -> DetectedIntent:
        tokens = set(tokenize(query))
        expanded = expand_with_synonyms(tokens)

        n = self._extract_n(query.lower())
        wants_grouping = bool(tokens & _GROUPING_HINTS)
        granularity = self._extract_granularity(tokens)
        wants_moving_average = bool(tokens & {"moving", "rolling"})
        wants_growth_rate = bool(tokens & {"growth", "change"}) and "percentage" not in tokens

        if needs_time_series_hint:
            return DetectedIntent(
                Intent.TREND, n, wants_grouping, granularity, query, tokens,
                confidence=0.9, wants_moving_average=wants_moving_average,
                wants_growth_rate=wants_growth_rate,
            )

        best_intent = Intent.LIST
        best_score = 0.0

        for intent, synonyms in _INTENT_SYNONYMS.items():
            score = token_overlap_score(expanded, synonyms)
            # Slight boost for direct (non-synonym) token hits so exact
            # phrasing like "top 5" doesn't get out-scored by loose matches.
            if tokens & synonyms:
                score += 0.25
            if score > best_score:
                best_score = score
                best_intent = intent

        if granularity is not None and best_score < 0.25:
            best_intent = Intent.TREND
            best_score = max(best_score, 0.5)

        return DetectedIntent(
            best_intent, n, wants_grouping, granularity, query, tokens,
            confidence=round(min(1.0, best_score), 3),
            wants_moving_average=wants_moving_average,
            wants_growth_rate=wants_growth_rate,
        )

    def _extract_n(self, q: str) -> Optional[int]:
        match = _N_PATTERN.search(q)
        if not match:
            return None
        for group in match.groups():
            if group is not None:
                return int(group)
        return None

    def _extract_granularity(self, tokens: Set[str]) -> Optional[str]:
        for tok in tokens:
            if tok in _GRANULARITY_HINTS:
                return _GRANULARITY_HINTS[tok]
        return None


# =========================================================
# NATURAL-LANGUAGE FILTER EXTRACTION
# =========================================================

@dataclass
class ExtractedFilter:
    """
    A filter parsed from natural language (or supplied explicitly via
    the `filters=`/`entities=` kwargs on `StructuredAgent.run`), with the
    resolved SQL operator and value(s), plus the surrounding tokens used
    to later resolve which column it refers to (via the same semantic
    scoring as everything else — no hardcoded field names).
    """
    operator: str  # gt | gte | lt | lte | between | eq | neq | contains | startswith | endswith | in_list
    value: Any
    value2: Any = None
    context_tokens: Set[str] = field(default_factory=set)
    is_temporal_hint: bool = False


# Each pattern captures: (optional context words)(operator phrase)(value).
# Context tokens are taken from the words immediately preceding the match
# so "price greater than 100" resolves the filter to the "price" column.
_FILTER_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("between", re.compile(r"between\s+([\w.\-/]+)\s+and\s+([\w.\-/]+)", re.IGNORECASE)),
    ("in_list", re.compile(r"(?:in|one of)\s*[\[\(]?\s*([\w.\-]+(?:\s*,\s*[\w.\-]+)+)\s*[\]\)]?", re.IGNORECASE)),
    ("gte", re.compile(r"at least\s+([\w.\-/]+)", re.IGNORECASE)),
    ("lte", re.compile(r"at most\s+([\w.\-/]+)", re.IGNORECASE)),
    ("gt", re.compile(r"(?:greater than|more than|above|over|after)\s+([\w.\-/]+)", re.IGNORECASE)),
    ("lt", re.compile(r"(?:less than|below|under|before)\s+([\w.\-/]+)", re.IGNORECASE)),
    ("neq", re.compile(r"not equal(?:s)? to\s+([\w.\-/]+)|!=\s*([\w.\-/]+)", re.IGNORECASE)),
    ("eq", re.compile(r"equal(?:s)? to\s+([\w.\-/]+)|equals\s+([\w.\-/]+)", re.IGNORECASE)),
    ("startswith", re.compile(r"start(?:s|ing)? with\s+['\"]?([\w.\- ]+?)['\"]?(?:\s|$)", re.IGNORECASE)),
    ("endswith", re.compile(r"end(?:s|ing)? with\s+['\"]?([\w.\- ]+?)['\"]?(?:\s|$)", re.IGNORECASE)),
    ("contains", re.compile(r"contain(?:s|ing)?\s+['\"]?([\w.\- ]+?)['\"]?(?:\s|$)", re.IGNORECASE)),
]

_TEMPORAL_VALUE_HINT = re.compile(r"^\d{4}(-\d{1,2}(-\d{1,2})?)?$")

# Maps the operator spellings a caller might reasonably pass through the
# explicit `filters={"col": {"gte": 4}}` API (including common symbolic
# aliases) onto the same internal operator vocabulary used by NL parsing.
_OPERATOR_ALIASES: Dict[str, str] = {
    "gt": "gt", ">": "gt",
    "gte": "gte", ">=": "gte", "min": "gte",
    "lt": "lt", "<": "lt",
    "lte": "lte", "<=": "lte", "max": "lte",
    "eq": "eq", "=": "eq", "==": "eq", "equals": "eq",
    "neq": "neq", "!=": "neq", "ne": "neq", "not_equals": "neq",
    "between": "between", "range": "between",
    "in": "in_list", "in_list": "in_list", "one_of": "in_list",
    "contains": "contains", "like": "contains",
    "startswith": "startswith", "starts_with": "startswith",
    "endswith": "endswith", "ends_with": "endswith",
}


def _coerce_filter_value(raw: str) -> Tuple[Any, bool]:
    """Best-effort coercion of a raw matched value to int/float/str, plus
    whether it looks like a date (used to steer column-kind resolution)."""
    raw = raw.strip().strip("'\"")
    if _TEMPORAL_VALUE_HINT.match(raw):
        return raw, True
    try:
        return int(raw), False
    except ValueError:
        pass
    try:
        return float(raw), False
    except ValueError:
        pass
    return raw, False


class QueryFilterExtractor:
    """
    Extracts WHERE-clause-worthy filters from natural language without
    any hardcoded field names — only generic comparison/containment
    phrasing ("greater than", "before", "contains", "in [...]" ...).
    Also accepts explicitly-structured filters/entities (e.g. from an
    API caller that already knows exactly which field and operator it
    wants) via `from_structured()`, using the same ExtractedFilter shape
    so both sources flow through identical column-resolution logic.
    Column resolution happens later, in SQLBuilder, using the same
    semantic scoring used everywhere else in this module.
    """

    CONTEXT_WINDOW = 4

    def extract(self, query: str) -> List[ExtractedFilter]:
        filters: List[ExtractedFilter] = []
        lowered = query.lower()
        tokens = tokenize(query)

        for operator, pattern in _FILTER_PATTERNS:
            for match in pattern.finditer(lowered):
                groups = [g for g in match.groups() if g is not None]
                if not groups:
                    continue

                context_tokens = set(self._context_before(tokens, match.start(), lowered))

                if operator == "between" and len(groups) >= 2:
                    v1, is_temporal1 = _coerce_filter_value(groups[0])
                    v2, is_temporal2 = _coerce_filter_value(groups[1])
                    filters.append(ExtractedFilter(
                        "between", v1, v2, context_tokens, is_temporal1 or is_temporal2
                    ))
                    continue

                if operator == "in_list":
                    raw_items = [item.strip() for item in groups[0].split(",") if item.strip()]
                    if len(raw_items) < 2:
                        continue
                    values: List[Any] = []
                    any_temporal = False
                    for item in raw_items:
                        v, is_temporal = _coerce_filter_value(item)
                        values.append(v)
                        any_temporal = any_temporal or is_temporal
                    filters.append(ExtractedFilter("in_list", values, None, context_tokens, any_temporal))
                    continue

                value, is_temporal = _coerce_filter_value(groups[0])
                if operator in ("contains", "startswith", "endswith"):
                    filters.append(ExtractedFilter(operator, str(value), None, context_tokens, False))
                else:
                    filters.append(ExtractedFilter(operator, value, None, context_tokens, is_temporal))

        return filters

    def from_structured(
        self,
        filters: Optional[Dict[str, Any]] = None,
        entities: Optional[Dict[str, Any]] = None,
    ) -> List[ExtractedFilter]:
        """
        Convert explicitly-supplied `filters=`/`entities=` dicts (as
        passed to `StructuredAgent.run`) into ExtractedFilter objects.

        Supported `filters` shapes, keyed by field name:
          - {"rating": {"gte": 4}}            -> single operator/value
          - {"rating": {"between": [1, 5]}}    -> range
          - {"category": {"in": ["a", "b"]}}   -> membership
          - {"category": "Books"}               -> shorthand for equality

        `entities` is a simpler convenience shape for exact-match values
        extracted upstream (e.g. by an NER step), e.g. {"brand": "Apple"}
        — each entry becomes an equality filter.

        The field name itself (e.g. "rating") is used as the context
        phrase for column resolution, exactly like the words surrounding
        a natural-language filter phrase are used for NL-extracted ones
        — so an explicit filter on "brand" resolves to whichever real
        column is semantically/fuzzily closest to "brand", without any
        hardcoded column name.
        """
        out: List[ExtractedFilter] = []

        if filters:
            for field_name, spec in filters.items():
                context_tokens = set(split_identifier(str(field_name)))
                if isinstance(spec, dict):
                    for raw_op, raw_val in spec.items():
                        op = _OPERATOR_ALIASES.get(str(raw_op).lower())
                        if op is None:
                            logger.debug("from_structured: unrecognized filter operator %r for field %r", raw_op, field_name)
                            continue
                        if op == "between":
                            if isinstance(raw_val, (list, tuple)) and len(raw_val) == 2:
                                out.append(ExtractedFilter("between", raw_val[0], raw_val[1], context_tokens))
                            else:
                                logger.debug("from_structured: 'between' requires a 2-item list/tuple, got %r", raw_val)
                        elif op == "in_list":
                            values = list(raw_val) if isinstance(raw_val, (list, tuple, set)) else [raw_val]
                            out.append(ExtractedFilter("in_list", values, None, context_tokens))
                        else:
                            out.append(ExtractedFilter(op, raw_val, None, context_tokens))
                else:
                    # Bare value shorthand -> equality filter.
                    out.append(ExtractedFilter("eq", spec, None, context_tokens))

        if entities:
            for field_name, value in entities.items():
                context_tokens = set(split_identifier(str(field_name)))
                out.append(ExtractedFilter("eq", value, None, context_tokens))

        return out

    def _context_before(self, tokens: List[str], char_offset: int, lowered_query: str) -> List[str]:
        # Approximate: take the tokens preceding the match's word position.
        prefix = lowered_query[:char_offset]
        prefix_tokens = tokenize(prefix)
        return prefix_tokens[-self.CONTEXT_WINDOW:]


# =========================================================
# TABLE / COLUMN SELECTION (semantic + statistical ranking)
# =========================================================

class SchemaLinker:
    """
    Ranks tables and columns by combining semantic relevance to the query
    with structural signals (connectivity, cardinality, variance) —
    replacing "first match" / "largest table" fallbacks with scored
    candidate selection. Also plans multi-table join paths using the
    relationships discovered by SchemaInspector.
    """

    def __init__(self, config: AgentConfig):
        self.config = config

    # -------------------- table selection --------------------

    def select_table(
        self, schema: DiscoveredSchema, detected: DetectedIntent
    ) -> Optional[TableInfo]:
        if schema.is_empty():
            return None

        tables = list(schema.tables.values())
        if len(tables) == 1:
            return tables[0]

        query_tokens = expand_with_synonyms(detected.query_tokens)

        # Direct, unambiguous table-name mention always wins.
        for t in tables:
            if t.name.lower() in detected.raw_query.lower():
                return t

        return max(tables, key=lambda t: self._table_score(t, schema, query_tokens))

    def select_best_table(
        self,
        tables: Sequence[TableInfo],
        detected: DetectedIntent,
        schema: Optional[DiscoveredSchema] = None,
    ) -> Optional[TableInfo]:
        """
        Picks the single best table from an already-narrowed candidate
        list (e.g. the participants resolved by
        `resolve_participating_tables` / `find_join_path`), rather than
        re-ranking the *entire* schema like `select_table` does. Used by
        `QueryPlanner.plan()` once join/table participation has already
        been decided. Falls back to the fuller schema-aware scoring in
        `_table_score` when a `DiscoveredSchema` is supplied (so
        connectivity signals are still available); otherwise uses a
        lighter, schema-free heuristic.
        """
        if not tables:
            return None
        if len(tables) == 1:
            return tables[0]

        query_tokens = expand_with_synonyms(detected.query_tokens)

        for t in tables:
            if t.name.lower() in detected.raw_query.lower():
                return t

        if schema is not None:
            return max(tables, key=lambda t: self._table_score(t, schema, query_tokens))

        def _score(t: TableInfo) -> float:
            name_score = token_overlap_score(t.semantic_tokens, query_tokens)
            column_hits = sum(1 for c in t.columns if c.matches_query(query_tokens) > 0)
            column_score = column_hits / max(len(t.columns), 1)
            size_score = math.log10((t.row_count or 0) + 10) / 10.0
            return name_score * 0.5 + column_score * 0.4 + size_score * 0.1

        return max(tables, key=_score)

    def rank_tables(
        self, schema: DiscoveredSchema, detected: DetectedIntent
    ) -> List[Tuple[float, TableInfo]]:
        """Every table, scored — used for multi-table detection instead
        of only picking the single best table."""
        query_tokens = expand_with_synonyms(detected.query_tokens)
        scored = [
            (self._table_score(t, schema, query_tokens), t)
            for t in schema.tables.values()
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return scored

    def _table_score(self, t: TableInfo, schema: DiscoveredSchema, query_tokens: Set[str]) -> float:
        name_score = token_overlap_score(t.semantic_tokens, query_tokens)
        column_hits = sum(1 for c in t.columns if c.matches_query(query_tokens) > 0)
        column_score = column_hits / max(len(t.columns), 1)
        connectivity_score = schema.connectivity(t.name) / max(len(schema.relationships) or 1, 1)
        size_score = math.log10((t.row_count or 0) + 10) / 10.0
        return (
            name_score * 0.4
            + column_score * 0.35
            + connectivity_score * 0.15
            + size_score * 0.10
        )

    # -------------------- multi-table / join planning --------------------

    def resolve_participating_tables(
        self, schema: DiscoveredSchema, detected: DetectedIntent
    ) -> List[TableInfo]:
        """
        Tables the query plausibly needs, in relevance order:
          1. Any table whose name is literally mentioned in the query.
          2. Any other table scoring above `min_table_join_score`.
        Falls back to just the single best table when nothing else
        clears the bar, so single-table behavior is unaffected.
        """
        lowered = detected.raw_query.lower()
        mentioned = [t for t in schema.tables.values() if t.name.lower() in lowered]

        ranked = self.rank_tables(schema, detected)
        scored_extra = [
            t for score, t in ranked
            if score >= self.config.min_table_join_score and t not in mentioned
        ]

        participants = mentioned + scored_extra
        if not participants:
            best = self.select_table(schema, detected)
            return [best] if best else []
        return participants

    def find_join_path(
        self, schema: DiscoveredSchema, tables: List[TableInfo]
    ) -> Tuple[List[TableInfo], List[Relationship]]:
        """
        BFS over the discovered relationship graph, starting from the
        first requested table, to find a connecting path that touches
        every requested table. Returns (ordered_tables, edges) where
        `edges[i]` connects `ordered_tables[i+1]` back to something
        already in the path. Returns a single-table result (no edges)
        if fewer than 2 tables are requested or no full path connects
        them — callers should treat an empty edge list as "no safe join
        available" and fall back to single-table behavior.
        """
        if len(tables) < 2:
            return tables, []

        names_needed = {t.name for t in tables}
        start = tables[0]

        adjacency: Dict[str, List[Relationship]] = {}
        for rel in schema.relationships:
            adjacency.setdefault(rel.from_table, []).append(rel)
            adjacency.setdefault(rel.to_table, []).append(
                Relationship(rel.to_table, rel.to_column, rel.from_table, rel.from_column, rel.confidence, rel.cardinality)
            )

        order = [start.name]
        seen = {start.name}
        parent_edge: Dict[str, Relationship] = {}
        queue = [start.name]

        while queue and not names_needed.issubset(seen):
            current = queue.pop(0)
            # Prefer the highest-confidence edges first.
            for rel in sorted(adjacency.get(current, []), key=lambda r: -r.confidence):
                if rel.to_table not in seen:
                    seen.add(rel.to_table)
                    parent_edge[rel.to_table] = rel
                    order.append(rel.to_table)
                    queue.append(rel.to_table)

        if not names_needed.issubset(seen):
            logger.debug(
                "join_planning: could not connect all requested tables %s (reached %s)",
                sorted(names_needed), sorted(seen),
            )
            return [start], []

        ordered_tables = [schema.tables[n] for n in order if n == start.name or n in names_needed or n in parent_edge]
        edges = [parent_edge[t.name] for t in ordered_tables if t.name in parent_edge]
        return ordered_tables, edges

    # -------------------- column selection --------------------

    def select_column(
        self,
        table: TableInfo,
        detected: DetectedIntent,
        kinds: Sequence[ColumnKind],
        purpose: str = "generic",
        context_tokens: Optional[Set[str]] = None,
    ) -> Optional[ColumnInfo]:
        candidates = table.columns_of(*kinds)
        if not candidates:
            return None

        query_tokens = expand_with_synonyms(context_tokens or detected.query_tokens)
        scored: List[Tuple[float, ColumnInfo]] = []

        for col in candidates:
            semantic = col.matches_query(query_tokens)
            # Fuzzy identifier match against the raw (non-tokenized)
            # context phrase catches naming variants like "customerID"
            # / "cust_id" that pure token overlap can miss.
            phrase = " ".join(context_tokens or detected.query_tokens)
            fuzzy = col.fuzzy_matches(phrase) if phrase else 0.0
            structural = self._structural_score(col, purpose)
            total = max(semantic, fuzzy * 0.8) * 0.7 + structural * 0.3
            scored.append((total, col))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        best_score, best_col = scored[0]

        if best_score >= self.config.min_semantic_match_score or len(scored) == 1:
            return best_col

        # No confident semantic match — fall back to the statistically
        # strongest candidate for the purpose rather than "first column".
        return max(candidates, key=lambda c: self._structural_score(c, purpose))

    def _structural_score(self, col: ColumnInfo, purpose: str) -> float:
        """
        Statistical fitness of a column for a given purpose, independent
        of naming. E.g. for ranking/aggregation we prefer higher-variance
        numeric columns (more informative spread) over near-constant ones.
        """
        profile = col.profile
        if purpose in ("rank", "aggregate"):
            if profile.numeric_stddev is None or profile.max_value is None or profile.min_value is None:
                return 0.0
            try:
                value_range = float(profile.max_value) - float(profile.min_value)
            except (TypeError, ValueError):
                return 0.0
            if value_range <= 0:
                return 0.0
            # Coefficient-of-variation-like score, bounded to [0, 1].
            return min(1.0, profile.numeric_stddev / value_range)

        if purpose == "group":
            # Prefer categorical columns with a moderate number of
            # distinct values — too few is uninformative, too many is
            # effectively an identifier.
            if profile.row_count == 0:
                return 0.0
            ideal = min(20, max(2, profile.row_count // 10))
            distance = abs(profile.distinct_count - ideal)
            return max(0.0, 1.0 - distance / max(ideal, 1))

        if purpose == "temporal":
            return 1.0 if profile.null_ratio < 0.1 else 0.5

        if purpose == "filter":
            return 1.0 - profile.null_ratio

        return 1.0 - profile.null_ratio


# =========================================================
# SEMANTIC QUERY PLANNING (spec #2-#10)
# =========================================================
#
# Intermediate representation between "natural language" and "SQL".
# Every planning decision -- which metrics, which derived metrics,
# which GROUP BY columns, which filters are row-level vs. aggregate-
# level, how to sort, which joins are needed, and which slice of the
# schema is actually relevant -- is made HERE, once, deterministically,
# and handed to the LLM (and the rule-based SQLBuilder / SQLValidator)
# as an already-reasoned QueryPlan. The LLM's job becomes "translate
# this plan into correct DuckDB syntax", not "re-derive the plan from
# the question", which is what makes multi-metric / HAVING-vs-WHERE /
# derived-metric questions reliable.

_AGG_FUNC_KEYWORDS: Dict[str, List[str]] = {
    "AVG": ["average", "avg", "mean", "typical"],
    "SUM": ["total", "sum", "combined", "aggregate"],
    "COUNT": ["count", "number of", "how many", "total number"],
    "COUNT_DISTINCT": ["distinct", "unique", "different"],
    "MIN": ["minimum", "min", "lowest", "smallest", "earliest"],
    "MAX": ["maximum", "max", "highest", "largest", "latest", "peak"],
    "MEDIAN": ["median", "middle value"],
    "STDDEV": ["standard deviation", "stddev", "std dev"],
    "VARIANCE": ["variance"],
}

_PHRASE_TO_FUNC: Dict[str, str] = {
    phrase: func for func, phrases in _AGG_FUNC_KEYWORDS.items() for phrase in phrases
}

# Longer phrases first, so "standard deviation" matches before a bare
# "deviation" substring would otherwise mislead the regex.
_AGG_PHRASES_ORDERED: List[str] = sorted(_PHRASE_TO_FUNC.keys(), key=len, reverse=True)

_METRIC_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(p) for p in _AGG_PHRASES_ORDERED) + r")\s+(?:of\s+)?([a-z][a-z0-9_ ]*)",
    re.IGNORECASE,
)

_STOPWORDS_AFTER_METRIC = {
    "and", "for", "per", "by", "with", "having", "where", "order",
    "ordered", "group", "grouped", "return", "only", "should", "at",
    "least", "most", "than", "each",
    # Fix 5: verb/comparison words that follow a metric noun in phrases
    # like "average rating is above the X" or "total sales greater
    # than 100" -- without these, the noun-phrase capture kept going
    # past the actual metric target and produced garbage aliases like
    # "avg_rating_is_above_the".
    "is", "are", "was", "were", "has", "have", "having", "above",
    "below", "greater", "less", "more", "over", "under", "before",
    "after", "not", "no", "without", "show", "showing", "display",
    "list", "give", "return", "find", "get",
}


@dataclass
class MetricSpec:
    """One requested aggregate metric, e.g. AVG(rating)."""
    agg_func: str  # AVG | SUM | COUNT | COUNT_DISTINCT | MIN | MAX | MEDIAN | STDDEV | VARIANCE
    phrase: str  # raw noun phrase, e.g. "rating"
    context_tokens: Set[str] = field(default_factory=set)
    column: Optional[str] = None  # resolved column name (post schema-linking)
    alias: str = ""
    confidence: float = 0.0

    def sql_expression(self) -> str:
        col = f'"{self.column}"' if self.column else "*"
        if self.agg_func == "COUNT_DISTINCT":
            return f"COUNT(DISTINCT {col})"
        if self.agg_func == "COUNT" and not self.column:
            return "COUNT(*)"
        return f"{self.agg_func}({col})"


@dataclass
class DerivedMetricSpec:
    """A computed/business metric not stored directly, e.g. "complaint
    percentage" -> 100.0 * SUM(CASE WHEN <indicator> THEN 1 ELSE 0 END)
    / COUNT(*). Resolved semantically against the schema at plan time;
    never hardcoded to any specific dataset."""
    name: str  # e.g. "complaint percentage"
    kind: str = "percentage"  # percentage | rate | ratio
    indicator_column: Optional[str] = None
    indicator_value_hint: Optional[str] = None
    alias: str = ""
    confidence: float = 0.0

    def sql_expression(self) -> Optional[str]:
        if not self.indicator_column:
            return None
        col = f'"{self.indicator_column}"'
        if self.indicator_value_hint:
            value = self.indicator_value_hint.replace("'", "''")
            condition = f"CAST({col} AS VARCHAR) ILIKE '%{value}%'"
        else:
            condition = f"{col} = TRUE"
        return f"100.0 * SUM(CASE WHEN {condition} THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0)"


@dataclass
class DimensionSpec:
    phrase: str
    column: Optional[str] = None
    confidence: float = 0.0


@dataclass
class RowFilterSpec:
    extracted: "ExtractedFilter"
    column: Optional[str] = None
    confidence: float = 0.0


@dataclass
class AggregateFilterSpec:
    """A HAVING-clause filter, e.g. "at least 500 reviews" ->
    HAVING COUNT(*) >= 500. Bound to the specific metric it filters on
    (never emitted as a WHERE clause against a raw column)."""
    metric: MetricSpec
    operator: str  # gt | gte | lt | lte | eq | neq
    value: Any


@dataclass
class SortSpec:
    target_phrase: str
    direction: str = "desc"  # asc | desc
    metric: Optional[MetricSpec] = None
    dimension: Optional[DimensionSpec] = None
    confidence: float = 0.0


@dataclass
class JoinPlan:
    tables: List[str] = field(default_factory=list)
    edges: List[Relationship] = field(default_factory=list)


@dataclass
class QueryPlan:
    """The single intermediate object every downstream stage (prompt
    builder, validator, rule-based builder) consumes instead of
    re-parsing the raw question independently."""
    raw_query: str
    intents: List[Tuple[str, float]] = field(default_factory=list)
    primary_intent: str = "list"
    metrics: List[MetricSpec] = field(default_factory=list)
    derived_metrics: List[DerivedMetricSpec] = field(default_factory=list)
    dimensions: List[DimensionSpec] = field(default_factory=list)
    row_filters: List[RowFilterSpec] = field(default_factory=list)
    aggregate_filters: List[AggregateFilterSpec] = field(default_factory=list)
    sort: Optional[SortSpec] = None
    limit: Optional[int] = None
    relevant_tables: List[str] = field(default_factory=list)
    relevant_columns: Dict[str, List[str]] = field(default_factory=dict)
    joins: JoinPlan = field(default_factory=JoinPlan)
    confidence: Dict[str, float] = field(default_factory=dict)
    overall_confidence: float = 0.0
    # Free-form, best-effort diagnostic info about how the plan was
    # constructed (candidate/selected tables, counts, etc.) — additive
    # only, never read by downstream SQL generation logic.
    debug: Dict[str, Any] = field(default_factory=dict)

    def requires_grouping(self) -> bool:
        return bool(self.dimensions)

    def requires_having(self) -> bool:
        return bool(self.aggregate_filters)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intents": self.intents,
            "primary_intent": self.primary_intent,
            "metrics": [
                {"agg_func": m.agg_func, "phrase": m.phrase, "column": m.column,
                 "alias": m.alias, "confidence": m.confidence}
                for m in self.metrics
            ],
            "derived_metrics": [
                {"name": d.name, "kind": d.kind, "indicator_column": d.indicator_column,
                 "indicator_value_hint": d.indicator_value_hint, "alias": d.alias,
                 "confidence": d.confidence}
                for d in self.derived_metrics
            ],
            "dimensions": [
                {"phrase": d.phrase, "column": d.column, "confidence": d.confidence}
                for d in self.dimensions
            ],
            "row_filters": [
                {"operator": f.extracted.operator, "value": f.extracted.value, "column": f.column}
                for f in self.row_filters
            ],
            "aggregate_filters": [
                {"metric": f.metric.phrase, "operator": f.operator, "value": f.value}
                for f in self.aggregate_filters
            ],
            "sort": (
                {"target": self.sort.target_phrase, "direction": self.sort.direction}
                if self.sort else None
            ),
            "limit": self.limit,
            "relevant_tables": self.relevant_tables,
            "joins": {
                "tables": self.joins.tables,
                "edges": [
                    f"{r.from_table}.{r.from_column}->{r.to_table}.{r.to_column}"
                    for r in self.joins.edges
                ],
            },
            "confidence": self.confidence,
            "overall_confidence": self.overall_confidence,
        }

    def to_prompt_block(self) -> str:
        """Renders this plan as an explicit, already-reasoned
        instruction block for the LLM prompt. The model's job is to
        translate this plan into correct SQL syntax -- not to re-derive
        intent, metrics, grouping, or filters from the raw question."""
        lines: List[str] = [f"Primary intent: {self.primary_intent}"]
        if len(self.intents) > 1:
            others = ", ".join(f"{name}({score:.2f})" for name, score in self.intents[1:4])
            lines.append(f"Secondary intents: {others}")

        if self.metrics:
            lines.append("Requested metrics (include EVERY one of these as its own output column -- do not drop or merge any):")
            for m in self.metrics:
                col_desc = f'"{m.column}"' if m.column else "(no confident column match -- use best judgement)"
                lines.append(f'  - {m.agg_func}({col_desc}) AS "{m.alias or m.phrase.replace(" ", "_")}"')
        else:
            lines.append("Requested metrics: none explicitly detected -- infer the single most relevant metric.")

        if self.derived_metrics:
            lines.append("Derived/business metrics (compute via CASE WHEN -- do not assume a stored column exists):")
            for d in self.derived_metrics:
                expr = d.sql_expression()
                if expr:
                    lines.append(f'  - "{d.name}": {expr} AS "{d.alias or d.name.replace(" ", "_")}"')
                else:
                    lines.append(f'  - "{d.name}": no confident indicator column found -- best-effort or omit.')

        if self.dimensions:
            lines.append("GROUP BY these dimensions, in this order (hierarchical grouping):")
            for dim in self.dimensions:
                col_desc = f'"{dim.column}"' if dim.column else f"(unresolved phrase: {dim.phrase!r})"
                lines.append(f"  - {col_desc}")

        if self.row_filters:
            lines.append("WHERE (row-level) filters:")
            for rf in self.row_filters:
                col_desc = f'"{rf.column}"' if rf.column else f"(unresolved: {sorted(rf.extracted.context_tokens)})"
                lines.append(f"  - {col_desc} {rf.extracted.operator} {rf.extracted.value!r}")

        if self.aggregate_filters:
            lines.append("HAVING (aggregate-level) filters -- these filter an AGGREGATE, never a raw column, and must NOT become WHERE clauses:")
            for af in self.aggregate_filters:
                lines.append(f"  - {af.metric.sql_expression()} {af.operator} {af.value!r}")

        if self.sort:
            direction = "DESC" if self.sort.direction == "desc" else "ASC"
            if self.sort.metric:
                lines.append(f"ORDER BY {self.sort.metric.sql_expression()} {direction}")
            elif self.sort.dimension and self.sort.dimension.column:
                lines.append(f'ORDER BY "{self.sort.dimension.column}" {direction}')
            else:
                lines.append(f"ORDER BY the primary requested metric {direction}")

        if self.limit:
            lines.append(f"LIMIT {self.limit}")

        if self.joins.edges:
            lines.append("Required joins (use exactly these join conditions, in this order):")
            for rel in self.joins.edges:
                lines.append(f'  - "{rel.from_table}"."{rel.from_column}" = "{rel.to_table}"."{rel.to_column}"')

        return "\n".join(lines)


class MultiIntentAnalyzer:
    """Scores every candidate intent independently (spec #1) instead of
    returning only the single best match -- a question can simultaneously
    be an aggregation, a grouping, AND a filtering request, and every
    detected label carries its own confidence score."""

    def analyze(self, query: str, min_confidence: float = 0.12) -> List[Tuple[str, float]]:
        tokens = set(tokenize(query))
        expanded = expand_with_synonyms(tokens)
        scores: List[Tuple[str, float]] = []

        for intent, synonyms in _INTENT_SYNONYMS.items():
            score = token_overlap_score(expanded, synonyms)
            if tokens & synonyms:
                score += 0.25
            if score >= min_confidence:
                scores.append((intent.value, round(min(1.0, score), 3)))

        # Structural signals independent of the single-intent synonym
        # table -- these can co-occur with any of the above.
        if _GROUPING_HINTS & tokens:
            scores.append(("grouping", 0.7))
        if {"join", "combined", "across", "together"} & tokens:
            scores.append(("joins", 0.5))
        if re.search(r"\bat least\b|\bmore than\b|\bhaving\b", query, re.IGNORECASE):
            scores.append(("aggregation_filter", 0.6))
        if re.search(r"%|percentage|percent\b", query, re.IGNORECASE):
            scores.append(("percentage_calculation", 0.6))
        if re.search(r"\bratio\b", query, re.IGNORECASE):
            scores.append(("ratio_calculation", 0.6))
        if re.search(r"\bcorrelat", query, re.IGNORECASE):
            scores.append(("correlation", 0.6))
        if re.search(r"\bstatistical|significance|hypothesis\b", query, re.IGNORECASE):
            scores.append(("statistical_analysis", 0.5))

        merged: Dict[str, float] = {}
        for name, score in scores:
            merged[name] = max(merged.get(name, 0.0), score)
        ranked = sorted(merged.items(), key=lambda p: -p[1])
        return ranked or [("list", 0.3)]


class MetricPlanner:
    """Extracts every explicitly requested aggregate metric (spec #3).
    Never collapses "average rating AND average sentiment AND total
    reviews" into a single COUNT -- each matched aggregation phrase
    becomes its own independent MetricSpec."""

    def extract(self, query: str) -> List[MetricSpec]:
        metrics: List[MetricSpec] = []
        lowered = query.lower()
        used_char_ranges: List[Tuple[int, int]] = []

        for match in _METRIC_PATTERN.finditer(lowered):
            span = match.span()
            if any(a < span[1] and span[0] < b for a, b in used_char_ranges):
                continue  # don't let overlapping matches double-claim text

            agg_func = _PHRASE_TO_FUNC.get(match.group(1).strip())
            if not agg_func:
                continue

            # Fix: `match.group(2)`'s capturing group is greedy and, by
            # construction, always extends to the end of the sentence
            # (there's no in-regex stop condition) -- the *actual* noun
            # phrase is decided afterward, token-by-token, by the
            # `_STOPWORDS_AFTER_METRIC` loop below. Tracking word
            # boundaries via `word_matches` lets us compute a TIGHT
            # claimed span (agg phrase + only the tokens we actually
            # used) instead of the whole greedy match -- otherwise a
            # second metric phrase later in the same sentence (e.g.
            # "average rating AND total number of reviews") would fall
            # inside the first match's full span and be incorrectly
            # skipped as "overlapping", silently dropping a requested
            # metric.
            group2 = match.group(2)
            word_matches = list(re.finditer(r"[a-z0-9_]+", group2))
            noun_tokens: List[str] = []
            consumed_end_in_group2 = 0
            for wm in word_matches:
                tok = wm.group(0)
                if tok in _STOPWORDS_AFTER_METRIC:
                    break
                noun_tokens.append(tok)
                consumed_end_in_group2 = wm.end()
                if len(noun_tokens) >= 4:
                    break
            if not noun_tokens and agg_func != "COUNT":
                continue

            context_tokens = set(noun_tokens)
            phrase_display = " ".join(noun_tokens) or "records"
            alias = re.sub(r"[^a-z0-9]+", "_", f"{agg_func.lower()}_{'_'.join(noun_tokens) or 'value'}").strip("_")

            metrics.append(MetricSpec(
                agg_func=agg_func, phrase=phrase_display, context_tokens=context_tokens,
                alias=alias, confidence=0.6,
            ))
            tight_end = (match.start(2) + consumed_end_in_group2) if noun_tokens else match.end(1)
            used_char_ranges.append((span[0], tight_end))

        if re.search(r"\bhow many\b", lowered) and not any(m.agg_func == "COUNT" for m in metrics):
            metrics.append(MetricSpec(agg_func="COUNT", phrase="records", alias="count", confidence=0.7))

        # De-duplicate near-identical metrics (same func + same leading
        # noun token) that different phrasings might both match.
        deduped: List[MetricSpec] = []
        seen: Set[Tuple[str, str]] = set()
        for m in metrics:
            leading = sorted(m.context_tokens)[0] if m.context_tokens else ""
            sig = (m.agg_func, leading)
            if sig in seen:
                continue
            seen.add(sig)
            deduped.append(m)
        return deduped


_DERIVED_METRIC_PATTERN = re.compile(
    r"([a-z][a-z0-9_ ]*?)\s+(percentage|percent|rate|ratio)\b", re.IGNORECASE
)
_DERIVED_METRIC_STOPWORDS = {"the", "a", "an", "of", "and", "for", "average", "total", "each"}


class DerivedMetricPlanner:
    """Detects semantically-implied derived/business metrics (spec #4)
    like "complaint percentage" and resolves each to whichever schema
    column looks like the right indicator, via generic semantic/fuzzy
    matching plus a small CONFIGURABLE business glossary
    (AgentConfig.derived_metric_glossary). Never hardcodes any
    dataset-specific table or column name."""

    def __init__(self, config: AgentConfig):
        self.config = config

    def extract(self, query: str) -> List[DerivedMetricSpec]:
        specs: List[DerivedMetricSpec] = []
        for match in _DERIVED_METRIC_PATTERN.finditer(query.lower()):
            noun_tokens = [t for t in tokenize(match.group(1)) if t not in _DERIVED_METRIC_STOPWORDS]
            noun_tokens = noun_tokens[-3:]  # words closest to the suffix matter most
            if not noun_tokens:
                continue
            suffix = match.group(2).strip()
            name = f"{' '.join(noun_tokens)} {suffix}"
            kind = "rate" if suffix == "rate" else ("ratio" if suffix == "ratio" else "percentage")
            specs.append(DerivedMetricSpec(name=name, kind=kind, alias=name.replace(" ", "_"), confidence=0.5))
        return specs

    def resolve(self, specs: List[DerivedMetricSpec], table: TableInfo) -> List[DerivedMetricSpec]:
        candidates = table.columns_of(ColumnKind.BOOLEAN, ColumnKind.CATEGORICAL, ColumnKind.NUMERIC)
        if not candidates:
            return specs

        for spec in specs:
            noun = spec.name.rsplit(" ", 1)[0].strip()
            search_tokens = expand_with_synonyms(set(tokenize(spec.name)))
            for hint in self.config.derived_metric_glossary.get(noun, []):
                search_tokens |= set(tokenize(hint))

            best_col, best_score = None, 0.0
            for col in candidates:
                score = max(col.matches_query(search_tokens), col.fuzzy_matches(spec.name) * 0.8)
                if score > best_score:
                    best_score, best_col = score, col

            if best_col and best_score >= self.config.min_semantic_match_score:
                spec.indicator_column = best_col.name
                spec.confidence = round(min(0.95, 0.4 + best_score), 3)
                if best_col.kind == ColumnKind.CATEGORICAL and best_col.profile.top_values:
                    best_value = max(
                        (str(v) for v, _ in best_col.profile.top_values),
                        key=lambda v: fuzzy_name_score(v, noun),
                        default=None,
                    )
                    if best_value and fuzzy_name_score(best_value, noun) >= 0.3:
                        spec.indicator_value_hint = best_value
        return specs


# Fix 4: cue phrases that introduce a GROUP BY dimension. A bare "by"
# is deliberately guarded with a negative lookbehind so "ordered by
# rating" / "sorted by rating" (a SORT instruction, handled by
# SortDetector) is never also captured here as a bogus dimension.
_DIMENSION_CUE_RE = re.compile(
    r"\b(?:for each|for every|grouped by|group by|broken down by|each"
    r"|(?<!order )(?<!ordered )(?<!sort )(?<!sorted )(?<!rank )(?<!ranked )by|per)\b",
    re.IGNORECASE,
)
_DIMENSION_STOPWORDS = {"the", "a", "an"}

# Fix 4/5: tokens that end a dimension-phrase capture -- common verbs
# that introduce the REST of the sentence ("show", "list", ...) and
# every token that appears in one of the aggregation-keyword phrases
# ("average", "total", "rating" is fine but "average"/"rating... is"
# is not -- see _AGG_PHRASES_ORDERED). Without this, "For each brand
# show average rating" previously captured the entire tail
# "brand show average rating" as ONE dimension phrase, which then
# fuzzy-matched to the "rating" column instead of "brand".
_DIMENSION_STOP_TOKENS = {
    "show", "shows", "showing", "display", "displays", "list", "lists",
    "give", "gives", "return", "returns", "returning", "find", "finds",
    "get", "gets", "present", "presents", "provide", "provides",
    "having", "order", "ordered", "sort", "sorted", "where", "with",
    "only", "should", "also", "then", "and", "the", "a", "an", "is",
    "are", "was", "were", "at", "least", "most",
} | set(tokenize(" ".join(_PHRASE_TO_FUNC.keys())))

_DIMENSION_MAX_TOKENS = 6  # generous cap for hierarchical "X and Y" grouping


class DimensionDetector:
    """Extracts GROUP BY candidates from phrases like "for each brand",
    "by category", "per region" (spec #5), including hierarchical
    multi-part grouping ("by region and category"). Captures only the
    short noun phrase immediately following the grouping cue -- it
    stops at the first verb or aggregation keyword rather than running
    to the end of the sentence, so "for each brand show average rating"
    correctly yields dimension="brand" (not "rating", and not the
    whole tail of the sentence)."""

    def extract(self, query: str) -> List[DimensionSpec]:
        specs: List[DimensionSpec] = []
        seen: Set[str] = set()
        for cue in _DIMENSION_CUE_RE.finditer(query):
            rest_tokens = tokenize(query[cue.end():])
            collected: List[str] = []
            for tok in rest_tokens:
                if tok == "and":
                    # Separator for hierarchical grouping ("by region
                    # and category") -- keep scanning past it.
                    collected.append(",")
                    continue
                if tok in _DIMENSION_STOP_TOKENS:
                    break
                collected.append(tok)
                if len(collected) >= _DIMENSION_MAX_TOKENS:
                    break

            phrase_blob = " ".join(collected)
            for part in phrase_blob.split(","):
                tokens = [t for t in tokenize(part) if t not in _DIMENSION_STOPWORDS]
                if not tokens:
                    continue
                phrase = " ".join(tokens)
                if phrase in seen:
                    continue
                seen.add(phrase)
                specs.append(DimensionSpec(phrase=phrase, confidence=0.6))
        return specs

    def resolve(self, specs: List[DimensionSpec], table: TableInfo) -> List[DimensionSpec]:
        candidates = table.columns_of(
            ColumnKind.CATEGORICAL, ColumnKind.TEMPORAL, ColumnKind.TEXTUAL, ColumnKind.IDENTIFIER
        )
        for spec in specs:
            tokens = expand_with_synonyms(set(tokenize(spec.phrase)))
            best_col, best_score = None, 0.0
            for col in candidates:
                score = max(col.matches_query(tokens), col.fuzzy_matches(spec.phrase) * 0.8)
                if score > best_score:
                    best_score, best_col = score, col
            if best_col and best_score >= 0.15:
                spec.column = best_col.name
                spec.confidence = round(min(0.95, 0.4 + best_score), 3)
        return specs


_AGGREGATE_FILTER_CONTEXT_HINTS = {
    "reviews", "review", "ratings", "rating", "records", "rows", "orders",
    "customers", "purchases", "transactions", "entries", "count", "total",
    "results", "matches", "items",
}


class AggregateFilterClassifier:
    """Classifies each extracted filter as a row-level WHERE filter or
    an aggregate-level HAVING filter (spec #7) -- the failure mode this
    directly targets is "at least 500 reviews" being misread as a raw
    column comparison instead of HAVING COUNT(*) >= 500. A numeric
    comparison filter becomes a HAVING filter, bound to the matching
    metric, whenever its context phrase either (a) overlaps a detected
    metric's own context tokens, or (b) generically describes "count of
    rows" (reviews/orders/records/...). Everything else stays a WHERE
    filter against a real column."""

    def classify(
        self, filters: List[ExtractedFilter], metrics: List[MetricSpec],
    ) -> Tuple[List[RowFilterSpec], List[AggregateFilterSpec]]:
        row_filters: List[RowFilterSpec] = []
        aggregate_filters: List[AggregateFilterSpec] = []
        count_metric = next((m for m in metrics if m.agg_func == "COUNT"), None)

        for f in filters:
            if f.operator in ("gt", "gte", "lt", "lte", "eq", "neq") and isinstance(f.value, (int, float)):
                matched_metric = self._match_metric(f.context_tokens, metrics)
                is_row_count_phrase = bool(f.context_tokens & _AGGREGATE_FILTER_CONTEXT_HINTS)

                if matched_metric is not None:
                    aggregate_filters.append(AggregateFilterSpec(matched_metric, f.operator, f.value))
                    continue
                if is_row_count_phrase:
                    metric = count_metric or MetricSpec(
                        agg_func="COUNT", phrase="records", alias="count", confidence=0.5,
                    )
                    if count_metric is None:
                        metrics.append(metric)
                        count_metric = metric
                    aggregate_filters.append(AggregateFilterSpec(metric, f.operator, f.value))
                    continue

            row_filters.append(RowFilterSpec(extracted=f, confidence=0.5))

        return row_filters, aggregate_filters

     
    @staticmethod
    def _match_metric(context_tokens: Set[str], metrics: List[MetricSpec]) -> Optional[MetricSpec]:
        best, best_score = None, 0.0
        for m in metrics:
            score = token_overlap_score(context_tokens, m.context_tokens)
            if score > best_score:
                best, best_score = m, score
        return best if best_score > 0 else None


_SORT_PATTERN = re.compile(
    r"\border(?:ed)?\s+by\s+([a-z][a-z0-9_ ]*?)(?:\s+(ascending|descending|asc|desc))?"
    r"(?=,|\.|$| and )", re.IGNORECASE,
)
_RANK_PATTERN = re.compile(r"\b(highest|lowest|top|bottom)\b", re.IGNORECASE)


class SortDetector:
    """Extracts ORDER BY intent (spec #8): explicit "ordered by X
    descending", or implicit "highest"/"top N"/"lowest"/"bottom N"."""

    def extract(
        self, query: str, metrics: List[MetricSpec], dimensions: List[DimensionSpec],
    ) -> Optional[SortSpec]:
        lowered = query.lower()

        match = _SORT_PATTERN.search(lowered)
        if match:
            phrase = match.group(1).strip()
            direction = "asc" if (match.group(2) or "").lower() in ("asc", "ascending") else "desc"
            tokens = set(tokenize(phrase))
            metric = self._match_metric(tokens, metrics)
            dimension = None if metric else self._match_dimension(tokens, dimensions)
            return SortSpec(phrase, direction, metric, dimension, confidence=0.8)

        rank_match = _RANK_PATTERN.search(lowered)
        if rank_match and metrics:
            word = rank_match.group(1).lower()
            direction = "asc" if word in ("lowest", "bottom") else "desc"
            return SortSpec(word, direction, metrics[0], None, confidence=0.5)

        return None

     
    @staticmethod
    def _match_metric(tokens: Set[str], metrics: List[MetricSpec]) -> Optional[MetricSpec]:
        best, best_score = None, 0.0
        for m in metrics:
            score = token_overlap_score(tokens, m.context_tokens)
            if score > best_score:
                best, best_score = m, score
        return best if best_score > 0 else None

     
    @staticmethod
    def _match_dimension(tokens: Set[str], dimensions: List[DimensionSpec]) -> Optional[DimensionSpec]:
        best, best_score = None, 0.0
        for d in dimensions:
            score = token_overlap_score(tokens, set(tokenize(d.phrase)))
            if score > best_score:
                best, best_score = d, score
        return best if best_score > 0 else None


class QueryPlanner:
    """Orchestrates the full semantic planning pipeline (spec #2):
    intent analysis, metric/derived-metric/dimension/filter/sort
    detection, join planning, and relevant-schema selection, all fused
    into a single QueryPlan. Every sub-planner above is independently
    instantiable and independently unit-testable; this class only
    wires them together in the documented pipeline order."""

    def __init__(self, config: AgentConfig, linker: SchemaLinker):
        self.config = config
        self.linker = linker
        self.intent_analyzer = MultiIntentAnalyzer()
        self.metric_planner = MetricPlanner()
        self.derived_metric_planner = DerivedMetricPlanner(config)
        self.dimension_detector = DimensionDetector()
        self.filter_classifier = AggregateFilterClassifier()
        self.sort_detector = SortDetector()

    def plan(
        self,
        schema: DiscoveredSchema,
        detected: DetectedIntent,
        explicit_filters: Optional[List[ExtractedFilter]] = None,
        nl_filters: Optional[List[ExtractedFilter]] = None,
        tables: Optional[Sequence[TableInfo]] = None,
        row_limit: Optional[int] = None,
    ) -> QueryPlan:

        query = detected.raw_query
        plan = QueryPlan(raw_query=query)

        # ============================================================
        # 0. SAFETY CHECKS
        # ============================================================
        if not query or not query.strip():
            raise ValueError("Empty query passed to planner")

        if not schema or not schema.tables:
            raise ValueError("Schema is empty or invalid")

        # ============================================================
        # 1. INTENT ANALYSIS
        # ============================================================
        plan.intents = self.intent_analyzer.analyze(query)
        plan.primary_intent = plan.intents[0][0] if plan.intents else detected.intent.value

        # ============================================================
        # 2. TABLE SELECTION (DO FIRST)
        # ============================================================
        candidate_tables = (
            list(tables)
            if tables
            else self.linker.resolve_participating_tables(schema, detected)
        )

        ordered_tables, edges = self.linker.find_join_path(schema, candidate_tables)

        plan.joins = JoinPlan(
            tables=[t.name for t in ordered_tables],
            edges=list(edges),
        )

        selected_tables = ordered_tables or candidate_tables or list(schema.tables.values())[:1]
        plan.relevant_tables = [t.name for t in selected_tables]

        # NOTE: `select_best_table` operates on the already-narrowed
        # candidate list (not the whole schema) and optionally uses the
        # richer, schema-aware scoring (connectivity, etc.) when a
        # DiscoveredSchema is supplied, so join/relationship signals
        # aren't lost just because we're picking from a shortlist.
        primary_table = self.linker.select_best_table(selected_tables, detected, schema)

        if primary_table is None:
            raise ValueError("Could not determine a primary table for planning")

        # ============================================================
        # 3. METRICS + RESOLUTION (AFTER TABLE IS KNOWN)
        # ============================================================
        plan.metrics = self.metric_planner.extract(query)
        for m in plan.metrics:
            self._resolve_metric_column(m, primary_table)

        plan.derived_metrics = self.derived_metric_planner.extract(query)
        plan.derived_metrics = self.derived_metric_planner.resolve(
            plan.derived_metrics,
            primary_table,
        )

        plan.dimensions = self.dimension_detector.extract(query)
        plan.dimensions = self.dimension_detector.resolve(plan.dimensions, primary_table)

        # ============================================================
        # 4. FILTER CLASSIFICATION (NOW SAFE)
        # ============================================================
        all_filters = list(nl_filters or []) + list(explicit_filters or [])

        plan.row_filters, plan.aggregate_filters = self.filter_classifier.classify(
            all_filters,
            plan.metrics,
        )

        for rf in plan.row_filters:
            self._resolve_filter_column(rf, primary_table)

        # ============================================================
        # 5. SORT + LIMIT
        # ============================================================
        plan.sort = self.sort_detector.extract(
            query,
            plan.metrics,
            plan.dimensions,
        )

        plan.limit = row_limit or detected.requested_n

        # ============================================================
        # 6. COLUMN SELECTION
        # ============================================================
        plan.relevant_columns = {
            t.name: self._select_relevant_columns(t, plan)
            for t in selected_tables
        }

        # ============================================================
        # 7. CONFIDENCE SCORING (WEIGHTED)
        # ============================================================
        plan.confidence = self._score_confidence(plan)

        # Fix (spec #14): `_score_confidence` returns SIX keys (intent,
        # metrics, derived_metrics, dimensions, filters, joins), but
        # this weight table previously only declared FOUR of them
        # (metrics/dimensions/filters/joins, summing to 1.0) and relied
        # on `weights.get(k, 0.1)` for the other two -- silently adding
        # another 0.2 of weight on top of the declared 1.0 or so
        # `overall_confidence` could come out as high as ~1.2 for a
        # perfectly-confident plan, violating the 0.0-1.0 contract.
        # Every key `_score_confidence` can return is now listed
        # explicitly, and the weights sum to exactly 1.0.
        weights = {
            "intent": 0.10,
            "metrics": 0.35,
            "derived_metrics": 0.10,
            "dimensions": 0.15,
            "filters": 0.15,
            "joins": 0.15,
        }

        plan.overall_confidence = _clamp01(round(
            sum(plan.confidence.get(k, 0.0) * weights.get(k, 0.0) for k in plan.confidence),
            3,
        ))

        # ============================================================
        # 8. FINAL DEBUG INFO
        # ============================================================
        plan.debug = {
            "candidate_tables": [t.name for t in candidate_tables],
            "selected_tables": [t.name for t in selected_tables],
            "primary_table": primary_table.name if primary_table else None,
            "num_metrics": len(plan.metrics),
            "num_filters": len(all_filters),
            "num_dimensions": len(plan.dimensions),
            "num_joins": len(edges),
        }

        return plan

    def _resolve_metric_column(self, metric: MetricSpec, table: TableInfo) -> None:
        if metric.agg_func == "COUNT" and not metric.context_tokens:
            return  # bare COUNT(*) needs no column
        kinds: Tuple[ColumnKind, ...] = (
            (ColumnKind.NUMERIC,) if metric.agg_func not in ("COUNT", "COUNT_DISTINCT")
            else (ColumnKind.NUMERIC, ColumnKind.CATEGORICAL, ColumnKind.IDENTIFIER, ColumnKind.TEXTUAL)
        )
        candidates = table.columns_of(*kinds) or table.columns
        tokens = expand_with_synonyms(metric.context_tokens)
        best_col, best_score = None, 0.0
        for col in candidates:
            score = max(col.matches_query(tokens), col.fuzzy_matches(metric.phrase) * 0.8)
            if score > best_score:
                best_score, best_col = score, col
        if best_col and best_score >= self.config.min_semantic_match_score:
            metric.column = best_col.name
            metric.confidence = round(min(0.95, 0.4 + best_score), 3)
            # Fix 5: once the metric is bound to a real column, rebuild
            # its alias from {agg_func}_{column} rather than trusting
            # the raw noun-phrase tokens captured from the sentence --
            # those can run past the intended metric (e.g. into a
            # trailing comparison clause) and produce a nonsense alias
            # like "avg_rating_is_above_the". The resolved column name
            # is always a clean, schema-backed identifier.
            metric.alias = self._clean_alias(f"{metric.agg_func.lower()}_{best_col.name}")
        elif metric.agg_func == "COUNT":
            metric.confidence = 0.7  # COUNT(*) remains safe even unresolved
            metric.alias = self._clean_alias(metric.alias or "record_count")

    @staticmethod
    def _clean_alias(raw: str) -> str:
        """Normalizes a candidate alias into a safe, readable SQL
        identifier fragment: lowercase, single underscores, no leading/
        trailing underscores, letters/digits/underscore only."""
        cleaned = re.sub(r"[^a-z0-9]+", "_", raw.lower()).strip("_")
        return cleaned or "value"

    def _resolve_filter_column(self, rf: RowFilterSpec, table: TableInfo) -> None:
        tokens = expand_with_synonyms(rf.extracted.context_tokens)
        best_col, best_score = None, 0.0
        for col in table.columns:
            score = max(col.matches_query(tokens), col.fuzzy_matches(" ".join(rf.extracted.context_tokens)) * 0.8)
            if score > best_score:
                best_score, best_col = score, col
        if best_col and best_score >= self.config.min_semantic_match_score:
            rf.column = best_col.name
            rf.confidence = round(min(0.95, 0.4 + best_score), 3)

    def _select_relevant_columns(self, table: TableInfo, plan: QueryPlan) -> List[str]:
        """Reduces the prompt-facing column list to only what the plan
        actually needs plus a couple of high-signal extras (primary
        key, a name/title/label-like column), instead of describing
        every column of every table to the LLM (spec #10)."""
        needed: Set[str] = set()
        for m in plan.metrics:
            if m.column:
                needed.add(m.column)
        for d in plan.derived_metrics:
            if d.indicator_column:
                needed.add(d.indicator_column)
        for dim in plan.dimensions:
            if dim.column:
                needed.add(dim.column)
        for rf in plan.row_filters:
            if rf.column:
                needed.add(rf.column)
        for af in plan.aggregate_filters:
            if af.metric.column:
                needed.add(af.metric.column)
        if table.primary_key:
            needed.add(table.primary_key)

        if not needed:
            return [c.name for c in table.columns[: self.config.llm_max_columns_per_table]]

        ordered = [c.name for c in table.columns if c.name in needed]
        extras = [
            c.name for c in table.columns
            if c.name not in needed and c.kind in (ColumnKind.TEXTUAL, ColumnKind.CATEGORICAL)
            and {"name", "title", "label"} & c.semantic_tokens
        ][:3]
        return ordered + extras

    def _score_confidence(self, plan: QueryPlan) -> Dict[str, float]:
        def avg(values: List[float], default: float) -> float:
            return round(sum(values) / len(values), 3) if values else default

        intent_conf = plan.intents[0][1] if plan.intents else 0.3
        metric_conf = avg([m.confidence for m in plan.metrics], 0.5)
        derived_conf = avg([d.confidence for d in plan.derived_metrics], 0.7)
        dimension_conf = avg([d.confidence for d in plan.dimensions], 0.6)
        filter_values = [rf.confidence for rf in plan.row_filters]
        filter_values += [
            getattr(af, "confidence", 0.7)
            for af in plan.aggregate_filters
        ]

        filter_conf = avg(filter_values, 0.8)

        join_conf = (
    1.0
    if len(plan.joins.tables) <= 1
    else (0.9 if plan.joins.edges else 0.4)
)
        return {
            "intent": round(intent_conf, 3),
            "metrics": metric_conf,
            "derived_metrics": derived_conf,
            "dimensions": dimension_conf,
            "filters": filter_conf,
            "joins": join_conf,
        }


# =========================================================
# SQL VALIDATION
# =========================================================

@dataclass
class ValidationResult:
    valid: bool
    issues: List[str] = field(default_factory=list)
    confidence: float = 1.0


_QUOTED_IDENTIFIER_RE = re.compile(r'"([^"]+)"')
_SQL_KEYWORDS = {
    "select", "from", "where", "group", "by", "order", "having", "limit",
    "as", "on", "join", "and", "or", "not", "null", "is", "desc", "asc",
    "distinct", "case", "when", "then", "else", "end", "with", "union",
    "intersect", "except", "over", "partition", "count", "sum", "avg",
    "min", "max", "median", "stddev", "variance", "cast", "try_cast",
    "double", "varchar", "timestamp", "extract", "month", "exists", "in",
}

# Matches `AS "alias"` (case-insensitive) so output aliases the query
# itself defines -- and any later *reference* to that same alias, e.g.
# in ORDER BY or an outer SELECT -- aren't flagged as unknown schema
# identifiers. DuckDB (like standard SQL) allows a SELECT alias to be
# referenced by name in ORDER BY / GROUP BY / an enclosing query, so an
# alias is a name the SQL *introduces*, not a schema identifier that
# needs to exist beforehand.
_ALIAS_AS_RE = re.compile(r'\bAS\s+"([^"]+)"', re.IGNORECASE)


class SQLValidator:
    """
    Lightweight, engine-agnostic pre-flight check: every quoted
    identifier in the generated SQL must correspond to either a known
    table name, a known column name on one of the tables involved, or
    a SELECT alias the query itself defines. This catches most "wrong
    column"/"wrong table" mistakes before they become a DuckDB
    execution error, without re-implementing a SQL parser.
    """

    def validate(
        self,
        schema: DiscoveredSchema,
        sql: str,
        tables: Sequence[TableInfo],
        plan: Optional["QueryPlan"] = None,
    ) -> ValidationResult:
        issues: List[str] = []

        # Fix (spec #3): reject SQL containing more than one executable
        # statement outright, rather than relying on DuckDB to reject
        # it. `SQLNormalizer` already collapses LLM output to a single
        # statement before this point, and `SQLBuilder`/`SQLRepairEngine`
        # never emit more than one -- this is a defense-in-depth check
        # so a bug or an unexpected code path can never let a
        # multi-statement string reach execution unnoticed. Detection is
        # string/comment-aware (a ';' inside a literal or comment is not
        # a statement boundary), so it can't be fooled the way a naive
        # `sql.count(";")` check could.
        statement_segments = [s.strip() for s in _split_sql_statements(sql) if s.strip()]
        if len(statement_segments) > 1:
            issues.append(
                f"SQL contains {len(statement_segments)} executable statements; "
                "only a single statement may be executed."
            )

        known_tables = set(schema.tables.keys())
        known_columns: Set[str] = set()
        for t in tables:
            known_columns.update(t.column_names())
        # Also allow columns from every table in the schema, since join
        # queries may reference tables discovered dynamically.
        for t in schema.tables.values():
            known_columns.update(t.column_names())

        # Fix 1: SELECT aliases (`... AS "alias"`) are names the SQL
        # itself introduces. Every alias the query defines is a valid
        # identifier for the REST of that query (ORDER BY, an outer
        # SELECT wrapping this one, etc.) -- not just at the exact spot
        # it's defined. Collecting all of them up front (rather than
        # only exempting the `AS "..."` span itself) is what lets
        # `ORDER BY "avg_rating"` resolve correctly against
        # `AVG("rating") AS "avg_rating"` earlier in the same query.
        alias_names: Set[str] = {m.group(1) for m in _ALIAS_AS_RE.finditer(sql)}

        for match in _QUOTED_IDENTIFIER_RE.finditer(sql):
            identifier = match.group(1)
            if identifier in known_tables or identifier in known_columns or identifier in alias_names:
                continue
            if identifier.lower() in _SQL_KEYWORDS:
                continue
            issues.append(f"Unrecognized identifier: \"{identifier}\"")

        if "SELECT" not in sql.upper():
            issues.append("Generated SQL has no SELECT clause.")

        if plan is not None:
            issues.extend(self._validate_against_plan(sql, plan))

        valid = len(issues) == 0
        confidence = 1.0 if valid else max(0.1, 1.0 - 0.2 * len(issues))
        return ValidationResult(valid=valid, issues=issues, confidence=round(confidence, 3))

     
    @staticmethod
    def _validate_against_plan(sql: str, plan: "QueryPlan") -> List[str]:
        """Checks generated SQL against the QueryPlan it was supposed to
        implement (spec #14): every requested metric is represented,
        GROUP BY exists when dimensions were requested, HAVING exists
        when aggregate filters were requested (and isn't a WHERE
        clause), and ORDER BY roughly matches the requested sort.
        Conservative by design -- flags only clear, structural
        mismatches so valid SQL that phrases things slightly
        differently (e.g. an equivalent CASE expression) isn't
        needlessly rejected."""
        issues: List[str] = []
        upper_sql = sql.upper()

        for m in plan.metrics:
            if m.column and f'"{m.column}"' not in sql:
                issues.append(
                    f'Requested metric {m.agg_func}("{m.column}") not found in generated SQL.'
                )
            elif not m.column and m.agg_func not in upper_sql:
                issues.append(f"Requested aggregation {m.agg_func} not found in generated SQL.")

        if plan.requires_grouping() and "GROUP BY" not in upper_sql:
            issues.append(
                "QueryPlan requested grouping by "
                f"{[d.phrase for d in plan.dimensions]} but SQL has no GROUP BY clause."
            )

        if plan.requires_having() and "HAVING" not in upper_sql:
            issues.append(
                "QueryPlan requested aggregate filter(s) "
                f"{[(f.metric.phrase, f.operator, f.value) for f in plan.aggregate_filters]} "
                "but SQL has no HAVING clause (aggregate filters must not be emitted as WHERE)."
            )

        if plan.sort is not None and "ORDER BY" not in upper_sql:
            issues.append(f'QueryPlan requested sorting by "{plan.sort.target_phrase}" but SQL has no ORDER BY clause.')

        return issues


# =========================================================
# SQL AUTOMATIC REPAIR
# =========================================================

_CANDIDATE_BINDINGS_RE = re.compile(r"Candidate bindings:\s*(.+)", re.IGNORECASE)
_MISSING_COLUMN_RE = re.compile(r'(?:Referenced|Binder Error:.*?column) "([^"]+)"', re.IGNORECASE)
_MISSING_TABLE_RE = re.compile(r'Table (?:with name )?"?([A-Za-z0-9_]+)"? (?:does not exist|not found)', re.IGNORECASE)
_AMBIGUOUS_COLUMN_RE = re.compile(r'Ambiguous reference to column "([^"]+)"', re.IGNORECASE)
_GROUP_BY_MISSING_RE = re.compile(r'column "([^"]+)" must appear in the GROUP BY clause', re.IGNORECASE)


class SQLRepairEngine:
    """
    Best-effort, bounded-attempt automatic repair for the most common
    execution failures: a misspelled/renamed column or table, an
    ambiguous unqualified column reference in a joined query, or a
    missing GROUP BY entry. Repairs are conservative — if no confident
    fix can be found, the original error is returned untouched rather
    than guessing.
    """

    def __init__(self, config: AgentConfig):
        self.config = config

    def attempt_repair(
        self, sql: str, error_message: str, schema: DiscoveredSchema, tables: Sequence[TableInfo]
    ) -> Optional[str]:
        known_tables = list(schema.tables.keys())
        known_columns: List[str] = []
        for t in schema.tables.values():
            known_columns.extend(t.column_names())

        # 1) DuckDB explicitly suggests candidate bindings — trust its
        #    own binder over our own fuzzy matching when available.
        binding_match = _CANDIDATE_BINDINGS_RE.search(error_message)
        missing_col_match = _MISSING_COLUMN_RE.search(error_message)
        if binding_match and missing_col_match:
            candidates = re.findall(r'"([^"]+)"', binding_match.group(1))
            offending = missing_col_match.group(1)
            if candidates and offending in sql:
                repaired = sql.replace(f'"{offending}"', f'"{candidates[0]}"', 1)
                if repaired != sql:
                    logger.info("sql_repair: replaced missing column \"%s\" -> \"%s\" (duckdb suggestion)", offending, candidates[0])
                    return repaired

        # 2) Missing column, no candidate bindings — fuzzy-match against
        #    every known column name.
        if missing_col_match:
            offending = missing_col_match.group(1)
            fixed = self._fuzzy_replace(sql, offending, known_columns)
            if fixed:
                logger.info("sql_repair: fuzzy-matched missing column \"%s\" -> \"%s\"", offending, fixed[1])
                return fixed[0]

        # 3) Missing/misnamed table.
        table_match = _MISSING_TABLE_RE.search(error_message)
        if table_match:
            offending = table_match.group(1)
            fixed = self._fuzzy_replace(sql, offending, known_tables, quoted=True)
            if fixed:
                logger.info("sql_repair: fuzzy-matched missing table \"%s\" -> \"%s\"", offending, fixed[1])
                return fixed[0]

        # 4) Ambiguous column in a multi-table query — qualify it with
        #    the first participating table's alias/name that actually
        #    has that column.
        ambiguous_match = _AMBIGUOUS_COLUMN_RE.search(error_message)
        if ambiguous_match:
            offending = ambiguous_match.group(1)
            for t in tables:
                if t.get_column(offending) is not None and not t.is_virtual:
                    qualified = f'"{t.name}"."{offending}"'
                    # Only qualify bare (unqualified) occurrences.
                    pattern = re.compile(r'(?<!\.)"' + re.escape(offending) + r'"')
                    repaired, n = pattern.subn(qualified, sql, count=1)
                    if n:
                        logger.info("sql_repair: qualified ambiguous column \"%s\" -> %s", offending, qualified)
                        return repaired

        # 5) A SELECT column is missing from GROUP BY — the safest
        #    automatic fix is to append it to the GROUP BY clause rather
        #    than guess at removing it from SELECT.
        group_missing_match = _GROUP_BY_MISSING_RE.search(error_message)
        if group_missing_match:
            offending = group_missing_match.group(1)
            group_idx = sql.upper().find("GROUP BY")
            if group_idx != -1 and f'"{offending}"' not in sql[group_idx:]:
                insert_at = sql.find(",", group_idx)
                clause_end = insert_at if insert_at != -1 else len(sql.rstrip(";"))
                repaired = sql[:clause_end] + f', "{offending}"' + sql[clause_end:]
                logger.info("sql_repair: added \"%s\" to GROUP BY", offending)
                return repaired

        return None

     
    @staticmethod
    def _fuzzy_replace(
        sql: str, offending: str, candidates: Sequence[str], quoted: bool = True
    ) -> Optional[Tuple[str, str]]:
        if not candidates:
            return None
        matches = difflib.get_close_matches(offending, candidates, n=1, cutoff=0.6)
        if not matches:
            # Fall back to our own fuzzy_name_score, which additionally
            # understands naming-convention variants (camelCase, synonym
            # tokens, etc.) that difflib's raw character ratio can miss.
            scored = sorted(candidates, key=lambda c: fuzzy_name_score(c, offending), reverse=True)
            if scored and fuzzy_name_score(scored[0], offending) >= 0.6:
                matches = [scored[0]]
        if not matches:
            return None
        best = matches[0]
        if quoted:
            repaired = sql.replace(f'"{offending}"', f'"{best}"')
        else:
            repaired = sql.replace(offending, best)
        if repaired == sql:
            return None
        return repaired, best


# =========================================================
# SQL GENERATION
# =========================================================

class SQLGenerationError(Exception):
    pass


# Fix 3: gt/gte/lt/lte/eq/neq -> SQL comparison operator, used to render
# QueryPlan aggregate (HAVING) filters in `SQLBuilder.build_from_plan`.
_HAVING_OPERATOR_SQL: Dict[str, str] = {
    "gt": ">", "gte": ">=", "lt": "<", "lte": "<=", "eq": "=", "neq": "!=",
}


class SQLBuilder:
    """
    Generates SQL purely from discovered schema + interpreted query.
    No table name, column name, alias, or grouping decision is ever
    hardcoded — every choice is delegated to SchemaLinker. Supports
    single-table analytics (aggregation, ranking, trends, ...) as well
    as automatically-planned multi-table joins.

    Filters can come from two sources, merged before SQL is built:
      1. Natural-language phrases parsed out of the query itself
         (QueryFilterExtractor.extract).
      2. Explicit, structured filters/entities passed in by the caller
         (QueryFilterExtractor.from_structured), e.g. from an API layer
         that already knows exactly which field/operator/value it wants.
    Both flow through identical column-resolution and predicate-
    rendering logic, so an explicit `{"brand": "Apple"}` filter is
    resolved to whichever real column best matches "brand" exactly the
    same way a natural-language "brand is Apple" phrase would be.
    """

    def __init__(self, config: AgentConfig, linker: SchemaLinker):
        self.config = config
        self.linker = linker
        self.filter_extractor = QueryFilterExtractor()

    def build(
        self,
        schema: DiscoveredSchema,
        detected: DetectedIntent,
        extra_filters: Optional[List[ExtractedFilter]] = None,
    ) -> Tuple[str, List[TableInfo], List[Relationship]]:
        """
        Returns (sql, participating_tables, join_edges). Callers that
        only care about the SQL string (e.g. existing single-table
        flows) can ignore the extra return values — but they're kept as
        part of the tuple, rather than stashed on hidden state, so the
        agent can report tables_used/join_path without re-deriving them.

        `extra_filters` are explicitly-supplied filters (already
        converted via QueryFilterExtractor.from_structured) that are
        merged with whatever natural-language filters are parsed out of
        `detected.raw_query`, and apply to both the single-table and
        multi-table join code paths.
        """
        if schema.is_empty():
            raise SQLGenerationError("No tables are available in the dataset.")

        # Fix 2: only impose a LIMIT when the user explicitly asked for
        # a bounded result (top/first/bottom/last/highest/lowest/latest N).
        # Otherwise every matching row is returned -- no more silent
        # `LIMIT 10` on plain listing queries like "show all brands".
        limit: Optional[int] = detected.requested_n
        if limit is not None:
            limit = min(limit, self.config.max_row_limit)

        filters = self.filter_extractor.extract(detected.raw_query) + (extra_filters or [])

        # ---- multi-table join attempt -------------------------------
        participants = self.linker.resolve_participating_tables(schema, detected)
        if len(participants) >= 2:
            ordered_tables, edges = self.linker.find_join_path(schema, participants)
            if edges:
                sql = self._build_joined_query(ordered_tables, edges, detected, limit, filters)
                return sql, ordered_tables, edges

        # ---- single-table path (unchanged behavior) ------------------
        table = self.linker.select_table(schema, detected)
        if table is None:
            raise SQLGenerationError("Could not determine a relevant table.")

        builder_map = {
            Intent.TOP_N: lambda t: self._build_ranked(t, detected, limit, True, filters),
            Intent.BOTTOM_N: lambda t: self._build_ranked(t, detected, limit, False, filters),
            Intent.AVERAGE: lambda t: self._build_aggregate(t, detected, "AVG", filters),
            Intent.MEDIAN: lambda t: self._build_aggregate(t, detected, "MEDIAN", filters),
            Intent.SUM: lambda t: self._build_aggregate(t, detected, "SUM", filters),
            Intent.MIN: lambda t: self._build_aggregate(t, detected, "MIN", filters),
            Intent.MAX: lambda t: self._build_aggregate(t, detected, "MAX", filters),
            Intent.STDDEV: lambda t: self._build_aggregate(t, detected, "STDDEV", filters),
            Intent.COUNT: lambda t: self._build_count(t, detected, filters),
            Intent.DISTINCT: lambda t: self._build_distinct(t, detected, limit, filters),
            Intent.FREQUENCY: lambda t: self._build_frequency(t, detected, limit, filters),
            Intent.TREND: lambda t: self._build_trend(t, detected),
            Intent.PERCENTAGE: lambda t: self._build_percentage(t, detected, limit, filters),
            Intent.COMPARISON: lambda t: self._build_aggregate(t, detected, "SUM", filters, force_group=True),
            Intent.SEARCH: lambda t: self._build_search(t, detected, limit, filters),
            Intent.JOIN: lambda t: self._build_list(t, limit, filters, detected=detected),
            Intent.LIST: lambda t: self._build_list(t, limit, filters, detected=detected),
        }

        handler = builder_map.get(
            detected.intent, lambda t: self._build_list(t, limit, filters, detected=detected)
        )
        sql = handler(table)
        return sql, [table], []

    # ---------------------------------------------------
    # Fix 3: QueryPlan-driven build (preferred rule-based fallback)
    # ---------------------------------------------------

    def build_from_plan(
        self, schema: DiscoveredSchema, plan: "QueryPlan",
    ) -> Tuple[str, List[TableInfo], List[Relationship]]:
        """
        Builds SQL directly from an already-computed `QueryPlan` instead
        of re-parsing `plan.raw_query` from scratch. This is now the
        PREFERRED rule-based fallback path whenever a usable plan is
        available (see `StructuredAgent._attempt_rule_sql`): it reuses
        exactly the metrics, dimensions, row/aggregate filters, joins,
        sort, and limit the planner already resolved, so the rule
        engine can never silently replace a semantically-correct LLM
        query with an unrelated one built from a fresh (and possibly
        much cruder) re-read of the question.

        Raises `SQLGenerationError` if the plan doesn't carry enough
        resolved information (e.g. zero resolved metrics/dimensions) to
        build a meaningful query -- the caller is expected to fall back
        to `build()` (the original NL-parsing path) in that case, so a
        thin/low-confidence plan never produces worse SQL than before.
        """
        if not plan.relevant_tables:
            raise SQLGenerationError("QueryPlan has no relevant tables to build from.")

        table = schema.get_table(plan.relevant_tables[0])
        if table is None:
            raise SQLGenerationError(f"QueryPlan table {plan.relevant_tables[0]!r} not found in schema.")

        dims_with_cols = [d for d in plan.dimensions if d.column]
        metrics_with_cols = [m for m in plan.metrics if m.column or (m.agg_func == "COUNT" and not m.context_tokens)]
        derived_with_expr = [d for d in plan.derived_metrics if d.sql_expression()]

        if not metrics_with_cols and not derived_with_expr and not dims_with_cols:
            # Nothing usable was resolved -- let the caller fall back to
            # the original NL-based builder rather than emit an empty
            # or meaningless SELECT.
            raise SQLGenerationError("QueryPlan has no resolved metrics/dimensions to build from.")

        select_parts: List[str] = [f'"{d.column}"' for d in dims_with_cols]

        for m in metrics_with_cols:
            alias = self._clean_identifier(m.alias) or f"{m.agg_func.lower()}_value"
            select_parts.append(f'{m.sql_expression()} AS "{alias}"')

        for d in derived_with_expr:
            alias = self._clean_identifier(d.alias) or self._clean_identifier(d.name)
            select_parts.append(f'{d.sql_expression()} AS "{alias}"')

        if not select_parts:
            raise SQLGenerationError("QueryPlan produced no selectable columns.")

        # ---- WHERE (row-level filters) --------------------------------
        where_predicates: List[str] = []
        for rf in plan.row_filters:
            if not rf.column:
                continue
            col = table.get_column(rf.column)
            if col is None:
                continue
            where_predicates.append(self._render_predicate(col, rf.extracted))
        where_sql = self._where_clause(where_predicates)

        # ---- GROUP BY ---------------------------------------------------
        group_sql = ""
        if dims_with_cols:
            group_sql = " GROUP BY " + ", ".join(f'"{d.column}"' for d in dims_with_cols)

        # ---- HAVING (aggregate-level filters) --------------------------
        having_sql = ""
        if plan.aggregate_filters:
            having_parts = []
            for af in plan.aggregate_filters:
                op_sql = _HAVING_OPERATOR_SQL.get(af.operator)
                if not op_sql:
                    continue
                value_sql = af.value if isinstance(af.value, (int, float)) else f"'{af.value}'"
                having_parts.append(f"{af.metric.sql_expression()} {op_sql} {value_sql}")
            if having_parts:
                having_sql = " HAVING " + " AND ".join(having_parts)

        # ---- ORDER BY ---------------------------------------------------
        order_sql = ""
        if plan.sort:
            direction = "DESC" if plan.sort.direction == "desc" else "ASC"
            if plan.sort.metric and (plan.sort.metric.column or plan.sort.metric.agg_func == "COUNT"):
                order_sql = f" ORDER BY {plan.sort.metric.sql_expression()} {direction}"
            elif plan.sort.dimension and plan.sort.dimension.column:
                order_sql = f' ORDER BY "{plan.sort.dimension.column}" {direction}'

        # ---- LIMIT (Fix 2: only when the plan actually carries one) ----
        limit_sql = f" LIMIT {plan.limit}" if plan.limit else ""

        sql = (
            f'SELECT {", ".join(select_parts)} '
            f'FROM {self._table_ref(table)}{where_sql}{group_sql}{having_sql}{order_sql}{limit_sql};'
        )
        return sql, [table], list(plan.joins.edges)

    @staticmethod
    def _clean_identifier(raw: Optional[str]) -> str:
        if not raw:
            return ""
        return re.sub(r"[^a-z0-9]+", "_", raw.lower()).strip("_")

    # ---------------------------------------------------
    # shared helpers
    # ---------------------------------------------------

    def _table_ref(self, table: TableInfo) -> str:
        return table.name if table.is_virtual else f'"{table.name}"'

    def _resolve_filter_predicates(
        self, table: TableInfo, detected: DetectedIntent, filters: List[ExtractedFilter],
        exclude_kinds: Sequence[ColumnKind] = (),
    ) -> List[str]:
        """
        Resolves each natural-language filter to a concrete column on
        `table` (via semantic scoring against the filter's own context
        tokens when available) and renders it as a SQL predicate.
        Filters that can't be confidently resolved to any column are
        silently skipped rather than guessed.
        """
        predicates: List[str] = []
        for f in filters:
            if f.operator in ("gt", "gte", "lt", "lte", "between"):
                kinds = [ColumnKind.NUMERIC, ColumnKind.TEMPORAL] if not f.is_temporal_hint else [ColumnKind.TEMPORAL, ColumnKind.NUMERIC]
            elif f.operator in ("contains", "startswith", "endswith"):
                kinds = [ColumnKind.TEXTUAL, ColumnKind.CATEGORICAL]
            else:
                kinds = [ColumnKind.CATEGORICAL, ColumnKind.TEXTUAL, ColumnKind.NUMERIC, ColumnKind.BOOLEAN]

            kinds = [k for k in kinds if k not in exclude_kinds]
            if not kinds:
                continue

            col = self.linker.select_column(
                table, detected, kinds, purpose="filter",
                context_tokens=f.context_tokens or None,
            )
            if col is None:
                continue

            predicates.append(self._render_predicate(col, f))

        return predicates

     
    @staticmethod
    def _render_predicate(col: ColumnInfo, f: ExtractedFilter) -> str:
        name = f'"{col.name}"'
        is_numeric = col.kind == ColumnKind.NUMERIC
        cast_expr = f'TRY_CAST({name} AS DOUBLE)' if is_numeric else name

        def lit(v: Any) -> str:
            if isinstance(v, (int, float)):
                return str(v)
            return "'" + str(v).replace("'", "''") + "'"

        if f.operator == "gt":
            return f"{cast_expr} > {lit(f.value)}"
        if f.operator == "gte":
            return f"{cast_expr} >= {lit(f.value)}"
        if f.operator == "lt":
            return f"{cast_expr} < {lit(f.value)}"
        if f.operator == "lte":
            return f"{cast_expr} <= {lit(f.value)}"
        if f.operator == "between":
            return f"{cast_expr} BETWEEN {lit(f.value)} AND {lit(f.value2)}"
        if f.operator == "in_list":
            rendered = ", ".join(lit(v) for v in f.value)
            return f"{name} IN ({rendered})"
        if f.operator == "eq":
            return f"{name} = {lit(f.value)}"
        if f.operator == "neq":
            return f"{name} != {lit(f.value)}"
        if f.operator == "contains":
            return f"{name} ILIKE '%{f.value}%'"
        if f.operator == "startswith":
            return f"{name} ILIKE '{f.value}%'"
        if f.operator == "endswith":
            return f"{name} ILIKE '%{f.value}'"
        return "TRUE"

     
    @staticmethod
    def _where_clause(predicates: Sequence[str]) -> str:
        if not predicates:
            return ""
        return " WHERE " + " AND ".join(predicates)

    # ---------------------------------------------------
    # Column-projection helpers (Changes 4/5)
    # ---------------------------------------------------

    def _build_select_columns(
        self, table: TableInfo, detected: DetectedIntent, purpose: str = "list",
        max_columns: int = 8,
    ) -> List[ColumnInfo]:
        """
        Chooses an informative, schema-agnostic subset of `table`'s
        columns instead of blindly returning every column (SELECT *).
        Priority order:
          1. The table's primary key -- lightweight context, not the
             point of the answer, so it's included but doesn't crowd
             out real content.
          2. Descriptive "label" columns (name/title/label-like, via
             the same synonym vocabulary used everywhere else).
          3. The grouping column, if the query implies grouping.
          4. Numeric/temporal columns the query's own tokens point at
             (the metric/date actually being asked about).
          5. For rank/aggregate purposes, the statistically strongest
             numeric column, if step 4 didn't already surface one.
          6. Remaining categorical columns, up to the column budget.
        No table/column name is ever hardcoded; every choice comes
        from ColumnKind classification and semantic/fuzzy scoring
        already computed during schema discovery. If fewer than 2
        columns are confidently selected, falls back to the full
        column list (capped) rather than returning an unhelpfully
        narrow projection.
        """
        query_tokens = expand_with_synonyms(detected.query_tokens)
        selected: List[ColumnInfo] = []
        seen: Set[str] = set()

        def _add(col: Optional[ColumnInfo]) -> None:
            if col is not None and col.name not in seen:
                selected.append(col)
                seen.add(col.name)

        if table.primary_key:
            _add(table.get_column(table.primary_key))

        label_tokens = {"name", "title", "label"}
        label_cols = [
            c for c in table.columns_of(ColumnKind.TEXTUAL, ColumnKind.CATEGORICAL)
            if c.semantic_tokens & label_tokens
        ]
        label_cols.sort(key=lambda c: c.matches_query(query_tokens), reverse=True)
        for c in label_cols[:2]:
            _add(c)

        if detected.wants_grouping:
            group_col = self.linker.select_column(
                table, detected, (ColumnKind.CATEGORICAL, ColumnKind.BOOLEAN), "group"
            )
            _add(group_col)

        for kinds in ((ColumnKind.NUMERIC,), (ColumnKind.TEMPORAL,)):
            candidates = [c for c in table.columns_of(*kinds) if c.matches_query(query_tokens) > 0]
            candidates.sort(key=lambda c: c.matches_query(query_tokens), reverse=True)
            for c in candidates[:2]:
                _add(c)

        if purpose in ("rank", "aggregate") and not any(c.kind == ColumnKind.NUMERIC for c in selected):
            numeric_col = self.linker.select_column(table, detected, (ColumnKind.NUMERIC,), "aggregate")
            _add(numeric_col)

        if len(selected) < max_columns:
            for c in table.columns_of(ColumnKind.CATEGORICAL, ColumnKind.BOOLEAN):
                if len(selected) >= max_columns:
                    break
                _add(c)

        if len(selected) < 2:
            return list(table.columns[:max_columns]) or list(table.columns)

        return selected[:max_columns]

     
    @staticmethod
    def _select_list_sql(columns: Sequence[ColumnInfo], alias: Optional[str] = None) -> str:
        prefix = f'{alias}.' if alias else ''
        return ", ".join(f'{prefix}"{c.name}"' for c in columns)

    def _build_joined_select_columns(
        self, pool: List[Tuple[str, ColumnInfo]], detected: DetectedIntent, max_columns: int = 10,
    ) -> List[Tuple[str, ColumnInfo]]:
        """
        Same informative-column heuristic as `_build_select_columns`,
        operating across every table in a joined query's column pool
        (Change 5). Each selected column keeps its owning alias so the
        caller can render an alias-qualified projection -- e.g.
        customer name / order date / product name / revenue instead of
        raw foreign-key IDs. Schema-agnostic; no table/column names are
        hardcoded.
        """
        query_tokens = expand_with_synonyms(detected.query_tokens)
        selected: List[Tuple[str, ColumnInfo]] = []
        seen: Set[Tuple[str, str]] = set()

        def _add(item: Optional[Tuple[str, ColumnInfo]]) -> None:
            if item is not None and (item[0], item[1].name) not in seen:
                selected.append(item)
                seen.add((item[0], item[1].name))

        label_tokens = {"name", "title", "label"}
        label_candidates = [
            (alias, c) for alias, c in pool
            if c.kind in (ColumnKind.TEXTUAL, ColumnKind.CATEGORICAL) and c.semantic_tokens & label_tokens
        ]
        label_candidates.sort(key=lambda ac: ac[1].matches_query(query_tokens), reverse=True)
        for item in label_candidates[:4]:
            _add(item)

        numeric_candidates = [
            (alias, c) for alias, c in pool
            if c.kind == ColumnKind.NUMERIC and c.matches_query(query_tokens) > 0
        ]
        numeric_candidates.sort(key=lambda ac: ac[1].matches_query(query_tokens), reverse=True)
        for item in numeric_candidates[:2]:
            _add(item)

        temporal_candidates = [
            (alias, c) for alias, c in pool
            if c.kind == ColumnKind.TEMPORAL and c.matches_query(query_tokens) > 0
        ]
        for item in temporal_candidates[:1]:
            _add(item)

        if len(selected) < max_columns:
            for alias, c in pool:
                if c.kind != ColumnKind.CATEGORICAL:
                    continue
                if len(selected) >= max_columns:
                    break
                _add((alias, c))

        if len(selected) < 2:
            return pool[:max_columns]

        return selected[:max_columns]

    # ---------------------------------------------------
    # Individual builders (single table)
    # ---------------------------------------------------

    def _build_list(
        self, table: TableInfo, limit: Optional[int],
        filters: Optional[List[ExtractedFilter]] = None,
        detected: Optional[DetectedIntent] = None,
    ) -> str:
        detected = detected or DetectedIntent(Intent.LIST, raw_query="")
        limit_clause = f" LIMIT {limit}" if limit is not None else ""
        predicates = self._resolve_filter_predicates(table, detected, filters or [])
        columns = self._build_select_columns(table, detected, purpose="list")
        select_sql = self._select_list_sql(columns)
        return f'SELECT {select_sql} FROM {self._table_ref(table)}{self._where_clause(predicates)}{limit_clause};'

    def _build_search(
        self, table: TableInfo, detected: DetectedIntent, limit: Optional[int], filters: List[ExtractedFilter]
    ) -> str:
        limit_clause = f" LIMIT {limit}" if limit is not None else ""
        predicates = self._resolve_filter_predicates(table, detected, filters)
        if not predicates:
            # No explicit "contains X" phrase parsed — fall back to a
            # broad ILIKE search across the best textual column using
            # the raw query tokens themselves.
            text_col = self.linker.select_column(table, detected, (ColumnKind.TEXTUAL, ColumnKind.CATEGORICAL), "filter")
            if text_col is None:
                return self._build_list(table, limit, detected=detected)
            keyword = detected.raw_query.strip().replace("'", "''")
            predicates = [f'"{text_col.name}" ILIKE \'%{keyword}%\'']
        columns = self._build_select_columns(table, detected, purpose="list")
        select_sql = self._select_list_sql(columns)
        return f'SELECT {select_sql} FROM {self._table_ref(table)}{self._where_clause(predicates)}{limit_clause};'

    def _build_count(self, table: TableInfo, detected: DetectedIntent, filters: List[ExtractedFilter]) -> str:
        predicates = self._resolve_filter_predicates(table, detected, filters)
        where_sql = self._where_clause(predicates)
        if detected.wants_grouping:
            group_col = self.linker.select_column(
                table, detected, (ColumnKind.CATEGORICAL, ColumnKind.BOOLEAN), "group"
            )
            if group_col is not None:
                return (
                    f'SELECT "{group_col.name}", COUNT(*) AS row_count '
                    f'FROM {self._table_ref(table)}{where_sql} '
                    f'GROUP BY "{group_col.name}" '
                    f'ORDER BY row_count DESC;'
                )
        return f'SELECT COUNT(*) AS row_count FROM {self._table_ref(table)}{where_sql};'

    def _build_distinct(
        self, table: TableInfo, detected: DetectedIntent, limit: Optional[int], filters: List[ExtractedFilter]
    ) -> str:
        col = self.linker.select_column(
            table, detected,
            (ColumnKind.CATEGORICAL, ColumnKind.TEXTUAL, ColumnKind.NUMERIC),
            "group",
        )
        if col is None:
            raise SQLGenerationError(
                f"No suitable column found for a distinct-values query on '{table.name}'."
            )
        predicates = self._resolve_filter_predicates(table, detected, filters)
        return (
            f'SELECT DISTINCT "{col.name}" '
            f'FROM {self._table_ref(table)}{self._where_clause(predicates)} '
            f'ORDER BY "{col.name}" '
            f'{"LIMIT " + str(limit) if limit is not None else ""};'
        )

    def _build_frequency(
        self, table: TableInfo, detected: DetectedIntent, limit: Optional[int], filters: List[ExtractedFilter]
    ) -> str:
        col = self.linker.select_column(
            table, detected, (ColumnKind.CATEGORICAL, ColumnKind.BOOLEAN), "group"
        )
        if col is None:
            raise SQLGenerationError(
                f"No categorical column found for a frequency query on '{table.name}'."
            )
        predicates = self._resolve_filter_predicates(table, detected, filters)
        return (
            f'SELECT "{col.name}", COUNT(*) AS frequency '
            f'FROM {self._table_ref(table)}{self._where_clause(predicates)} '
            f'GROUP BY "{col.name}" '
            f'ORDER BY frequency DESC '
            f'{"LIMIT " + str(limit) if limit is not None else ""};'
        )

    def _build_aggregate(
        self, table: TableInfo, detected: DetectedIntent, agg_fn: str,
        filters: List[ExtractedFilter], force_group: bool = False,
    ) -> str:
        numeric_col = self.linker.select_column(
            table, detected, (ColumnKind.NUMERIC,), "aggregate"
        )
        if numeric_col is None:
            raise SQLGenerationError(
                f"No numeric column found to compute {agg_fn} on '{table.name}'."
            )

        alias = f"{agg_fn.lower()}_{numeric_col.name}"

        # Grouping is only applied when the query actually implies it
        # ("by", "per", "each", ...) — not simply because a categorical
        # column happens to exist on the table.
        group_col = None
        if detected.wants_grouping or force_group:
            group_col = self.linker.select_column(
                table, detected, (ColumnKind.CATEGORICAL, ColumnKind.BOOLEAN), "group"
            )

        all_predicates = self._resolve_filter_predicates(table, detected, filters, exclude_kinds=())

        # Filters whose context tokens overlap the aggregate target's own
        # tokens are treated as post-aggregation (HAVING) constraints
        # rather than row-level (WHERE) constraints when grouping is
        # active — e.g. "total sales by region above 1000" filters the
        # per-region total, not individual rows.
        where_predicates: List[str] = []
        having_predicates: List[str] = []
        if group_col is not None:
            for f, predicate in zip(filters, self._resolve_filter_predicates_paired(table, detected, filters)):
                if predicate is None:
                    continue
                overlaps_target = bool(f.context_tokens & numeric_col.semantic_tokens)
                if overlaps_target and f.operator in ("gt", "gte", "lt", "lte", "between", "eq", "neq"):
                    having_predicates.append(self._rewrite_predicate_for_alias(predicate, numeric_col.name, alias))
                else:
                    where_predicates.append(predicate)
        else:
            where_predicates = all_predicates

        where_sql = self._where_clause(where_predicates)
        having_sql = f" HAVING {' AND '.join(having_predicates)}" if having_predicates else ""

        if group_col is not None:
            return (
                f'SELECT "{group_col.name}", '
                f'{agg_fn}(TRY_CAST("{numeric_col.name}" AS DOUBLE)) AS {alias} '
                f'FROM {self._table_ref(table)} '
                f'WHERE TRY_CAST("{numeric_col.name}" AS DOUBLE) IS NOT NULL'
                f'{(" AND " + " AND ".join(where_predicates)) if where_predicates else ""} '
                f'GROUP BY "{group_col.name}"'
                f'{having_sql} '
                f'ORDER BY {alias} DESC;'
            )

        return (
            f'SELECT {agg_fn}(TRY_CAST("{numeric_col.name}" AS DOUBLE)) AS {alias} '
            f'FROM {self._table_ref(table)} '
            f'WHERE TRY_CAST("{numeric_col.name}" AS DOUBLE) IS NOT NULL'
            f'{(" AND " + " AND ".join(where_predicates)) if where_predicates else ""};'
        )

    def _resolve_filter_predicates_paired(
        self, table: TableInfo, detected: DetectedIntent, filters: List[ExtractedFilter]
    ) -> List[Optional[str]]:
        """Same resolution as `_resolve_filter_predicates` but keeps a
        1:1 positional correspondence with `filters` (using None for
        unresolved ones) so callers can inspect each filter's own
        context alongside its rendered predicate."""
        out: List[Optional[str]] = []
        for f in filters:
            if f.operator in ("gt", "gte", "lt", "lte", "between"):
                kinds = [ColumnKind.NUMERIC, ColumnKind.TEMPORAL]
            elif f.operator in ("contains", "startswith", "endswith"):
                kinds = [ColumnKind.TEXTUAL, ColumnKind.CATEGORICAL]
            else:
                kinds = [ColumnKind.CATEGORICAL, ColumnKind.TEXTUAL, ColumnKind.NUMERIC, ColumnKind.BOOLEAN]
            col = self.linker.select_column(table, detected, kinds, purpose="filter", context_tokens=f.context_tokens or None)
            out.append(self._render_predicate(col, f) if col is not None else None)
        return out

     
    @staticmethod
    def _rewrite_predicate_for_alias(predicate: str, column_name: str, alias: str) -> str:
        return predicate.replace(f'TRY_CAST("{column_name}" AS DOUBLE)', alias).replace(f'"{column_name}"', alias)

    def _build_ranked(
        self, table: TableInfo, detected: DetectedIntent, limit: Optional[int], descending: bool,
        filters: List[ExtractedFilter],
    ) -> str:
        numeric_col = self.linker.select_column(
            table, detected, (ColumnKind.NUMERIC,), "rank"
        )
        if numeric_col is None:
            return self._build_list(table, limit, filters, detected=detected)

        columns = self._build_select_columns(table, detected, purpose="rank")
        # Guarantee the ranking metric itself is always visible in the
        # output, even if the general heuristic picked a different
        # numeric column.
        if numeric_col.name not in {c.name for c in columns}:
            columns = columns + [numeric_col]
        select_sql = self._select_list_sql(columns)

        predicates = self._resolve_filter_predicates(table, detected, filters)
        extra_where = (" AND " + " AND ".join(predicates)) if predicates else ""
        direction = "DESC" if descending else "ASC"
        return (
            f'SELECT {select_sql} FROM {self._table_ref(table)} '
            f'WHERE TRY_CAST("{numeric_col.name}" AS DOUBLE) IS NOT NULL{extra_where} '
            f'ORDER BY TRY_CAST("{numeric_col.name}" AS DOUBLE) {direction} '
            f'{"LIMIT " + str(limit) if limit is not None else ""};'
        )

    def _build_percentage(
        self, table: TableInfo, detected: DetectedIntent, limit: Optional[int], filters: List[ExtractedFilter]
    ) -> str:
        group_col = self.linker.select_column(
            table, detected, (ColumnKind.CATEGORICAL, ColumnKind.BOOLEAN), "group"
        )
        if group_col is None:
            raise SQLGenerationError(
                f"No categorical column found for a percentage breakdown on '{table.name}'."
            )
        predicates = self._resolve_filter_predicates(table, detected, filters)
        return (
            f'SELECT "{group_col.name}", '
            f'COUNT(*) AS row_count, '
            f'ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS percentage '
            f'FROM {self._table_ref(table)}{self._where_clause(predicates)} '
            f'GROUP BY "{group_col.name}" '
            f'ORDER BY percentage DESC '
            f'{"LIMIT " + str(limit) if limit is not None else ""};'
        )

    def _build_trend(self, table: TableInfo, detected: DetectedIntent) -> str:
        time_col = self.linker.select_column(
            table, detected, (ColumnKind.TEMPORAL,), "temporal"
        )
        if time_col is None:
            raise SQLGenerationError(
                f"'{table.name}' has no temporal column, so a trend/time-series "
                f"query cannot be answered."
            )

        granularity = detected.granularity or self._infer_granularity(time_col)
        bucket_expr = self._bucket_expression(time_col, granularity)

        # Optional numeric measure to aggregate per period, if the query
        # implies one (e.g. "monthly revenue trend"); otherwise falls
        # back to row counts, matching prior behavior exactly.
        numeric_col = self.linker.select_column(table, detected, (ColumnKind.NUMERIC,), "aggregate")
        measure_expr = (
            f'SUM(TRY_CAST("{numeric_col.name}" AS DOUBLE))'
            if numeric_col is not None and numeric_col.matches_query(expand_with_synonyms(detected.query_tokens)) > 0
            else "COUNT(*)"
        )
        measure_alias = "row_count" if measure_expr == "COUNT(*)" else "period_value"

        base_cte = (
            f'WITH period_totals AS ('
            f'SELECT {bucket_expr} AS period, {measure_expr} AS {measure_alias} '
            f'FROM "{table.name}" '
            f'GROUP BY period'
            f')'
        )

        if not (detected.wants_moving_average or detected.wants_growth_rate):
            return f'{base_cte} SELECT period, {measure_alias} FROM period_totals ORDER BY period;'

        select_extra = []
        if detected.wants_moving_average:
            select_extra.append(
                f'AVG({measure_alias}) OVER (ORDER BY period ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS moving_average'
            )
        if detected.wants_growth_rate:
            select_extra.append(
                f'ROUND(100.0 * ({measure_alias} - LAG({measure_alias}) OVER (ORDER BY period)) '
                f'/ NULLIF(LAG({measure_alias}) OVER (ORDER BY period), 0), 2) AS pct_change'
            )

        extra_sql = (", " + ", ".join(select_extra)) if select_extra else ""
        return (
            f'{base_cte} '
            f'SELECT period, {measure_alias}{extra_sql} '
            f'FROM period_totals ORDER BY period;'
        )

    def _infer_granularity(self, time_col: ColumnInfo) -> str:
        """
        When the user doesn't specify a granularity, pick one based on the
        actual span of the data: a few days of data shouldn't be bucketed
        by year, and a decade of data shouldn't be bucketed by hour.
        """
        min_v, max_v = time_col.profile.min_value, time_col.profile.max_value
        if min_v is None or max_v is None:
            return "month"

        try:
            span_days = abs((max_v - min_v).days)  # works for date/datetime objects
        except Exception:
            return "month"

        if span_days <= 3:
            return "hour"
        if span_days <= 90:
            return "day"
        if span_days <= 365 * 2:
            return "week"
        if span_days <= 365 * 8:
            return "month"
        if span_days <= 365 * 30:
            return "quarter"
        return "year"

    def _bucket_expression(self, time_col: ColumnInfo, granularity: str) -> str:
        fmt_by_granularity = {
            "hour": "%Y-%m-%d %H:00",
            "day": "%Y-%m-%d",
            "week": "%Y-%W",
            "month": "%Y-%m",
            "year": "%Y",
        }

        if granularity == "quarter":
            date_expr = f'CAST("{time_col.name}" AS TIMESTAMP)'
            return (
                f"strftime({date_expr}, '%Y') || '-Q' || "
                f"CAST(((EXTRACT(month FROM {date_expr}) - 1) / 3 + 1) AS VARCHAR)"
            )

        fmt = fmt_by_granularity.get(granularity, "%Y-%m")
        date_expr = f'CAST("{time_col.name}" AS TIMESTAMP)'
        return f"strftime({date_expr}, '{fmt}')"

    # ---------------------------------------------------
    # Multi-table join builder
    # ---------------------------------------------------

    def _build_joined_query(
        self,
        ordered_tables: List[TableInfo],
        edges: List[Relationship],
        detected: DetectedIntent,
        limit: Optional[int],
        filters: Optional[List[ExtractedFilter]] = None,
    ) -> str:
        limit_clause = f" LIMIT {limit}" if limit is not None else ""
        alias_map = {t.name: f"t{i}" for i, t in enumerate(ordered_tables)}

        from_sql = f'"{ordered_tables[0].name}" AS {alias_map[ordered_tables[0].name]}'
        for rel in edges:
            if rel.from_table not in alias_map or rel.to_table not in alias_map:
                continue
            from_sql += (
                f' JOIN "{rel.to_table}" AS {alias_map[rel.to_table]} '
                f'ON {alias_map[rel.from_table]}."{rel.from_column}" = '
                f'{alias_map[rel.to_table]}."{rel.to_column}"'
            )

        pool: List[Tuple[str, ColumnInfo]] = [
            (alias_map[t.name], c) for t in ordered_tables for c in t.columns
        ]
        query_tokens = expand_with_synonyms(detected.query_tokens)

        def best(kinds: Sequence[ColumnKind], purpose: str) -> Tuple[Optional[str], Optional[ColumnInfo]]:
            scored = []
            for alias, c in pool:
                if c.kind not in kinds:
                    continue
                semantic = c.matches_query(query_tokens)
                structural = self.linker._structural_score(c, purpose)
                scored.append((semantic * 0.7 + structural * 0.3, alias, c))
            if not scored:
                return None, None
            scored.sort(key=lambda x: x[0], reverse=True)
            return scored[0][1], scored[0][2]

        primary_alias = alias_map[ordered_tables[0].name]

        # Resolve filters (NL-extracted + explicit) against the joined
        # column pool, qualifying each predicate with its table alias.
        join_predicates = [p for p in (self._resolve_join_filter(pool, f) for f in (filters or [])) if p]
        where_sql = (" WHERE " + " AND ".join(join_predicates)) if join_predicates else ""

        aggregate_intents = {
            Intent.SUM: "SUM", Intent.AVERAGE: "AVG", Intent.MEDIAN: "MEDIAN",
            Intent.MIN: "MIN", Intent.MAX: "MAX", Intent.STDDEV: "STDDEV",
        }

        if detected.intent in aggregate_intents:
            agg_fn = aggregate_intents[detected.intent]
            n_alias, num_col = best((ColumnKind.NUMERIC,), "aggregate")
            if num_col is None:
                raise SQLGenerationError("No numeric column found across joined tables for aggregation.")
            g_alias, group_col = (None, None)
            if detected.wants_grouping:
                g_alias, group_col = best((ColumnKind.CATEGORICAL, ColumnKind.BOOLEAN), "group")

            select_expr = f'{agg_fn}(TRY_CAST({n_alias}."{num_col.name}" AS DOUBLE)) AS {agg_fn.lower()}_value'
            if group_col is not None:
                return (
                    f'SELECT {g_alias}."{group_col.name}" AS group_key, {select_expr} '
                    f'FROM {from_sql}{where_sql} '
                    f'GROUP BY {g_alias}."{group_col.name}" '
                    f'ORDER BY {agg_fn.lower()}_value DESC '
                    f'{"LIMIT " + str(limit) if limit is not None else ""};'
                )
            return f'SELECT {select_expr} FROM {from_sql}{where_sql};'

        if detected.intent == Intent.COUNT:
            g_alias, group_col = (None, None)
            if detected.wants_grouping:
                g_alias, group_col = best((ColumnKind.CATEGORICAL, ColumnKind.BOOLEAN), "group")
            if group_col is not None:
                return (
                    f'SELECT {g_alias}."{group_col.name}" AS group_key, COUNT(*) AS row_count '
                    f'FROM {from_sql}{where_sql} GROUP BY {g_alias}."{group_col.name}" ORDER BY row_count DESC;'
                )
            return f'SELECT COUNT(*) AS row_count FROM {from_sql}{where_sql};'

        # Default: informative, descriptive columns across the joined
        # tables (e.g. customer name, order date, product name, revenue)
        # instead of a raw `SELECT t0.*`, which tends to surface foreign
        # keys/IDs rather than business-meaningful fields.
        projection_cols = self._build_joined_select_columns(pool, detected)
        select_sql = ", ".join(f'{alias}."{c.name}"' for alias, c in projection_cols)
        return f'SELECT {select_sql} FROM {from_sql}{where_sql}{limit_clause};'

    def _resolve_join_filter(
        self, pool: List[Tuple[str, ColumnInfo]], f: ExtractedFilter
    ) -> Optional[str]:
        """
        Resolves a single filter against the pool of (alias, ColumnInfo)
        pairs spanning every table in a joined query, then renders an
        alias-qualified predicate. Mirrors `_resolve_filter_predicates`'
        column-kind routing and scoring, but qualifies the identifier
        with its table alias since a joined query has more than one
        table in scope.
        """
        if f.operator in ("gt", "gte", "lt", "lte", "between"):
            kinds = (ColumnKind.NUMERIC, ColumnKind.TEMPORAL)
        elif f.operator in ("contains", "startswith", "endswith"):
            kinds = (ColumnKind.TEXTUAL, ColumnKind.CATEGORICAL)
        else:
            kinds = (ColumnKind.CATEGORICAL, ColumnKind.TEXTUAL, ColumnKind.NUMERIC, ColumnKind.BOOLEAN)

        candidates = [(alias, c) for alias, c in pool if c.kind in kinds]
        if not candidates:
            return None

        query_tokens = expand_with_synonyms(f.context_tokens)
        phrase = " ".join(f.context_tokens) if f.context_tokens else ""

        scored: List[Tuple[float, str, ColumnInfo]] = []
        for alias, c in candidates:
            semantic = c.matches_query(query_tokens) if query_tokens else 0.0
            fuzzy = c.fuzzy_matches(phrase) if phrase else 0.0
            scored.append((max(semantic, fuzzy * 0.8), alias, c))
        scored.sort(key=lambda x: x[0], reverse=True)

        best_score, alias, col = scored[0]
        if best_score <= 0 and len(scored) > 1:
            # No confident match anywhere in the joined pool for this
            # filter's context — skip rather than guess.
            return None

        predicate = self._render_predicate(col, f)
        # `_render_predicate` emits a bare `"col"` / `TRY_CAST("col" ...)`
        # reference; qualify it with the owning table's alias so it's
        # unambiguous in a multi-table FROM/JOIN clause.
        return predicate.replace(f'"{col.name}"', f'{alias}."{col.name}"')


# =========================================================
# LLM-DRIVEN SQL GENERATION
# =========================================================

class LLMGenerationError(Exception):
    """Raised when the LLM cannot be reached or returns something that
    cannot be turned into a usable SQL string. Callers decide whether to
    fall back to the rule-based SQLBuilder or surface this directly."""


# Strips ```sql ... ``` / ``` ... ``` fences, and any leading prose the
# model may have added despite instructions not to.
_SQL_FENCE_RE = re.compile(r"```(?:sql)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_LEADING_SELECT_OR_WITH_RE = re.compile(r"(?is)\b(SELECT|WITH|INSERT|UPDATE|DELETE|MERGE)\b")


class LLMSQLGenerator:
    """
    Turns a natural-language query into SQL by prompting a local LLM
    (via Ollama's /api/generate endpoint, model + endpoint configurable
    through `AgentConfig.llm`) with a description of the discovered
    schema, rather than assembling SQL from fixed templates.

    The schema description is built entirely from `DiscoveredSchema`
    (table/column names, SQL types, column "kind", primary/candidate
    keys, discovered relationships, and — optionally — a few sample
    values per column) so no dataset-specific knowledge is hardcoded
    here either; only the *shape* of the prompt is fixed. Optionally,
    the description can be restricted to a caller-supplied subset of
    tables (`tables=`), which is how StructuredAgent keeps the prompt
    small for schemas with many tables (see `_select_prompt_tables`).
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        # ---- Connection pooling (spec #5) ----
        # A single http.client connection per (scheme, host, port) is
        # kept warm and reused across calls instead of opening a fresh
        # socket per request. Guarded by a lock since asyncio.to_thread
        # may invoke `_call_ollama` from different worker threads across
        # concurrent `run()` calls. On any I/O error the connection is
        # discarded and transparently reopened on the next attempt.
        self._conn_lock = threading.RLock()
        self._pooled_conn: Optional[Any] = None
        self._pooled_conn_key: Optional[Tuple[str, str, int]] = None

    # -----------------------------------------------------
    # Public entry point
    # -----------------------------------------------------
    
    async def generate(
        self,
        schema: DiscoveredSchema,
        query: str,
        row_limit: Optional[int],
        explicit_filters: Optional[List[ExtractedFilter]] = None,
        tables: Optional[Sequence[TableInfo]] = None,
        timeout_mode: str = "production",
        plan: Optional["QueryPlan"] = None,
    ) -> Tuple[str, Dict[str, Any]]:

        # ============================================================
        # STEP 1: BUILD PROMPT
        # ============================================================
        prompt = self._build_prompt(
            schema,
            query,
            row_limit,
            explicit_filters or [],
            tables,
            plan,
        )

        logger.debug(
            "llm_prompt_built: query=%r prompt_chars=%d preview=%r",
            query, len(prompt) if prompt else 0, (prompt[:500] if prompt else "EMPTY PROMPT"),
        )

        if not prompt or not prompt.strip():
            raise LLMGenerationError("Empty prompt generated for LLM SQL generation")

        # ============================================================
        # STEP 2: CALL LLM
        # ============================================================
        t0 = time.perf_counter()

        try:
            raw_text, call_debug = await asyncio.to_thread(
                self._call_ollama_with_retry,
                prompt,
                timeout_mode,
            )
        except Exception as e:
            raise LLMGenerationError(f"LLM call failed before response: {e}") from e

        elapsed_ms = (time.perf_counter() - t0) * 1000

        logger.debug(
            "llm_response_received: chars=%d preview=%r",
            len(raw_text) if raw_text else 0, repr(raw_text)[:500],
        )

        if not raw_text or not raw_text.strip():
            raise LLMGenerationError("Empty response from Ollama")

        # ============================================================
        # STEP 3: EXTRACT SQL
        # ============================================================
        sql = self._extract_sql(raw_text)

        logger.debug("sql_extracted: %r", sql)

        if not sql:
            raise LLMGenerationError(
                f"LLM returned no parseable SQL. Raw response: {raw_text[:500]!r}"
            )

        # ============================================================
        # STEP 4: SINGLE VALIDATION PIPELINE (IMPORTANT FIX)
        # ============================================================
        is_valid, error = self.validate_sql(sql, schema)

        if not is_valid:
            raise LLMGenerationError(f"SQL validation failed: {error}\nSQL: {sql}")

        # ============================================================
        # STEP 5: DEBUG INFO
        # ============================================================
        debug_info = {
            "llm_model": self.config.llm.model,
            "llm_latency_ms": round(elapsed_ms, 2),
            "raw_response_preview": raw_text[:500],
            "prompt_chars": len(prompt),
            "prompt_tokens_estimate": max(1, len(prompt) // 4),
            **call_debug,
        }

        return sql, debug_info

    # -----------------------------------------------------
    # Prompt construction
    # -----------------------------------------------------

    def _build_prompt(
        self,
        schema: DiscoveredSchema,
        query: str,
        row_limit: Optional[int],
        explicit_filters: List[ExtractedFilter],
        tables: Optional[Sequence[TableInfo]] = None,
        plan: Optional["QueryPlan"] = None,
    ) -> str:

        # ============================================================
        # STEP 1: VALIDATION (prevents empty prompt / silent failures)
        # ============================================================
        if not query or not query.strip():
            raise ValueError("Query is empty in _build_prompt")

        if schema is None:
            raise ValueError("Schema is None in _build_prompt")

        # ============================================================
        # STEP 2: BUILD COMPONENTS SAFELY
        # ============================================================
        # `_tables_for_prompt` narrows each table's *displayed* columns
        # down to the set the planner already resolved as relevant to
        # this query (`plan.relevant_columns`) and strips sample-value
        # listings from columns where they don't aid SQL generation
        # (anything that isn't categorical/textual/boolean). It does
        # NOT change which tables/columns were selected -- that
        # selection already happened upstream (SchemaLinker/
        # QueryPlanner); this only decides what subset of that existing
        # selection gets rendered into the prompt text. When no plan is
        # available, or the plan has nothing on record for a given
        # table, that table is passed through unchanged.
        prompt_tables = self._tables_for_prompt(schema, tables, plan)

        schema_block = self._describe_schema(schema, prompt_tables) or ""
        # `plan.joins.edges` -- when present -- is already the exact,
        # already-decided join path for this query (computed upstream
        # by SchemaLinker.find_join_path / QueryPlanner). Rendering
        # those edges directly is a strict subset of what
        # `_describe_relationships` would otherwise print (every
        # relationship between the prompt tables, including ones this
        # query doesn't use), so this cannot introduce a join the
        # planner didn't already decide on.
        if plan is not None and plan.joins.edges:
            relationships_block = self._format_relationship_edges(plan.joins.edges)
        else:
            relationships_block = self._describe_relationships(schema, prompt_tables) or ""
        filters_block = self._describe_explicit_filters(explicit_filters) or ""
        plan_block = self._describe_plan(plan)

        # ============================================================
        # STEP 3: DEBUG (optional but useful)
        # ============================================================
        logger.debug(
            "prompt_block_sizes: schema=%d relationships=%d filters=%d plan=%d",
            len(schema_block), len(relationships_block), len(filters_block), len(plan_block),
        )

        # ============================================================
        # STEP 4: FINAL PROMPT
        # ============================================================
        # Formatting only below: the double 80-char "====" border per
        # section has been collapsed to a single "## " header. Every
        # instruction, rule, and example is unchanged and in the same
        # order -- this only removes fixed decorative characters that
        # were repeated, unchanged, on every single request.
        prompt = f"""
    You are a senior SQL generation engine for DuckDB.

    ## 🚨 OUTPUT RULES (STRICT)
    - Output ONLY SQL
    - No explanations
    - No markdown
    - No <think> blocks
    - Must return ONE valid DuckDB SQL query
    - Never return empty output
    - End query with semicolon (;)

    ## ⚡ CORE RULES
    - Use ONLY schema tables/columns provided below
    - NEVER invent tables or columns
    - Always use double quotes: "table"."column"
    - NEVER use SELECT *
    - Select only required columns

    ## 📊 AGGREGATION RULES (CRITICAL)
    If query involves: "for each", "by", "per", "average", "sum", "count":

    YOU MUST:
    - Use GROUP BY for all non-aggregated columns
    - Each SELECT column must be either:
    (a) grouped column OR
    (b) aggregate function

    VALID:
    SELECT "brand", AVG("rating")
    FROM "metadata_data"
    GROUP BY "brand";

    INVALID:
    SELECT "brand", AVG("rating"), "price"
    FROM "metadata_data"
    GROUP BY "brand";

    ## 🧠 SEMANTIC MAPPING
    - show all X → DISTINCT X
    - for each X → GROUP BY X
    - average X by Y → GROUP BY Y + AVG(X)
    - top N → ORDER BY + LIMIT
    - highest/lowest → ORDER BY + LIMIT 1

    ## 🔗 JOIN RULES
    - Use joins ONLY if relationships are provided
    - Do NOT guess joins
    - Prefer single-table queries if possible

    ## 📌 QUERY PLAN (if provided)
    {plan_block}

    ## 📌 SCHEMA
    {schema_block}

    ## 🔗 RELATIONSHIPS
    {relationships_block}

    ## ⚠️ FILTERS
    {filters_block}

    ## ❓ QUESTION
    {query}

    SQL:
    """.strip()

        # ============================================================
        # STEP 5: SAFETY CHECK
        # ============================================================
        if len(prompt) < 50:
            raise ValueError(f"Prompt too small. Length={len(prompt)}")

        logger.debug("final_prompt_length=%d", len(prompt))

        return prompt

    # -----------------------------------------------------
    # Prompt-only display narrowing (NEW; does not alter selection)
    # -----------------------------------------------------
    #
    # Both helpers below exist solely to decide what `_describe_schema`/
    # `_describe_relationships` render. Neither one selects, ranks, or
    # filters tables/columns/joins on the merits -- that decision was
    # already made upstream by SchemaLinker/QueryPlanner and is passed
    # in via `tables` and `plan`. These helpers only take that existing
    # decision and produce a smaller *view* of it for the prompt.

    def _tables_for_prompt(
        self,
        schema: DiscoveredSchema,
        tables: Optional[Sequence[TableInfo]],
        plan: Optional["QueryPlan"],
    ) -> List[TableInfo]:
        table_list = list(tables) if tables is not None else list(schema.tables.values())
        if plan is None:
            return table_list

        narrowed: List[TableInfo] = []
        for table in table_list:
            relevant_names = plan.relevant_columns.get(table.name)
            if not relevant_names:
                # Nothing on record for this table in the plan (e.g. it
                # was added only by table-level relevance ranking, not
                # by the planner) -- show it exactly as before.
                narrowed.append(table)
                continue

            by_name = {c.name: c for c in table.columns}
            display_columns: List[ColumnInfo] = []
            for name in relevant_names:
                col = by_name.get(name)
                if col is None:
                    continue
                if col.kind in (ColumnKind.CATEGORICAL, ColumnKind.TEXTUAL, ColumnKind.BOOLEAN):
                    display_columns.append(col)
                else:
                    # Sample values only help the model pick an exact
                    # literal spelling for categorical/text/boolean
                    # filters -- they add no value on numeric/identifier
                    # columns, so drop them from the *displayed* copy
                    # only (the real column/profile is untouched).
                    display_columns.append(replace(col, profile=replace(col.profile, top_values=[])))

            narrowed.append(replace(table, columns=display_columns) if display_columns else table)
        return narrowed

    def _format_relationship_edges(self, edges: Sequence[Relationship]) -> str:
        if not edges:
            return "(none discovered)"
        return "\n".join(
            f'- "{rel.from_table}"."{rel.from_column}" -> '
            f'"{rel.to_table}"."{rel.to_column}" '
            f'(cardinality={rel.cardinality}, confidence={rel.confidence})'
            for rel in edges
        )

    # ============================================================
    # PLAN DESCRIBER
    # ============================================================
    def _describe_plan(self, plan: Optional["QueryPlan"]) -> str:
        if not plan:
            return ""

        try:
            return f"\nQUERY PLAN:\n{plan.to_prompt_block()}\n"
        except Exception:
            return ""

    def _describe_schema(self, schema: DiscoveredSchema, tables: Optional[Sequence[TableInfo]] = None) -> str:
        table_list = list(tables) if tables is not None else list(schema.tables.values())
        lines: List[str] = []
        for table in table_list:
            lines.append(f'- Table "{table.name}" (row_count={table.row_count}):')
            cols = table.columns[: self.config.llm_max_columns_per_table]
            for col in cols:
                descriptor = f'    - "{col.name}" ({col.sql_type}, kind={col.kind.value}'
                if col.name == table.primary_key:
                    descriptor += ", primary_key"
                elif col.is_candidate_key:
                    descriptor += ", candidate_key"
                descriptor += ")"
                if self.config.llm_include_sample_values and col.profile.top_values:
                    samples = ", ".join(repr(v) for v, _ in col.profile.top_values[:5])
                    descriptor += f" sample_values=[{samples}]"
                lines.append(descriptor)
            if len(table.columns) > len(cols):
                lines.append(f"    ... ({len(table.columns) - len(cols)} more columns omitted)")
        return "\n".join(lines) if lines else "(no tables discovered)"

    def _describe_relationships(self, schema: DiscoveredSchema, tables: Optional[Sequence[TableInfo]] = None) -> str:
        if tables is not None:
            names = {t.name for t in tables}
            rels = [r for r in schema.relationships if r.from_table in names and r.to_table in names]
        else:
            rels = schema.relationships
        if not rels:
            return "(none discovered)"
        lines = []
        for rel in rels:
            lines.append(
                f'- "{rel.from_table}"."{rel.from_column}" -> '
                f'"{rel.to_table}"."{rel.to_column}" '
                f'(cardinality={rel.cardinality}, confidence={rel.confidence})'
            )
        return "\n".join(lines)

    def _describe_explicit_filters(self, filters: List[ExtractedFilter]) -> str:
        if not filters:
            return ""
        lines = ["\nEXPLICIT FILTERS SUPPLIED BY THE CALLER (must be applied in the WHERE clause,"
                 " matched against whichever column best fits the given field context):"]
        for f in filters:
            context = " ".join(sorted(f.context_tokens)) or "(no field hint)"
            value = f.value if f.operator != "between" else f"{f.value} AND {f.value2}"
            lines.append(f'- field~"{context}" {f.operator} {value!r}')
        return "\n".join(lines) + "\n"

    # -----------------------------------------------------
    # Ollama call (blocking; run via asyncio.to_thread)
    # -----------------------------------------------------

    def _call_ollama_with_retry(self, prompt: str, timeout_mode: str = "production") -> Tuple[str, Dict[str, Any]]:
        """
        Wraps `_call_ollama` with a bounded retry loop (spec #5, #12):
        network errors and timeouts are retried up to
        `LLMConfig.max_retries` additional times with exponential
        backoff, rather than immediately failing the whole request.
        Never silently swallows the final failure -- if every attempt
        is exhausted, the last error is re-raised with a full attempt
        history attached for diagnostics.
        """
        cfg = self.config.llm
        timeout = cfg.effective_timeout(timeout_mode)
        attempts_made: List[Dict[str, Any]] = []
        last_error: Optional[Exception] = None

        for attempt_num in range(cfg.max_retries + 1):
            attempt_t0 = time.perf_counter()
            try:
                raw_text, call_debug = self._call_ollama(prompt, timeout)
                attempts_made.append({
                    "attempt": attempt_num + 1,
                    "ok": True,
                    "elapsed_ms": round((time.perf_counter() - attempt_t0) * 1000, 2),
                })
                call_debug["retry_attempts"] = attempt_num
                call_debug["attempt_history"] = attempts_made
                return raw_text, call_debug
            except LLMGenerationError as e:
                last_error = e
                attempts_made.append({
                    "attempt": attempt_num + 1,
                    "ok": False,
                    "error": str(e),
                    "elapsed_ms": round((time.perf_counter() - attempt_t0) * 1000, 2),
                })
                logger.warning(
                    "ollama_call_failed attempt=%d/%d error=%s",
                    attempt_num + 1, cfg.max_retries + 1, e,
                )
                if attempt_num < cfg.max_retries:
                    backoff = cfg.retry_backoff_seconds * (cfg.retry_backoff_multiplier ** attempt_num)
                    time.sleep(backoff)

        # Every attempt failed -- never fail silently; surface full history.
        raise LLMGenerationError(
            f"LLM ({cfg.model}) unreachable after {cfg.max_retries + 1} attempt(s): {last_error}. "
            f"History: {attempts_made}"
        ) from last_error

    def _get_pooled_connection(self, scheme: str, host: str, port: int):
        """Returns a warm http.client connection for (scheme, host, port),
        creating a new one if none is pooled yet or the pooled one is for
        a different endpoint. Caller must hold `self._conn_lock`."""
        import http.client
        key = (scheme, host, port)
        if self._pooled_conn is not None and self._pooled_conn_key == key:
            return self._pooled_conn
        if self._pooled_conn is not None:
            try:
                self._pooled_conn.close()
            except Exception:
                pass
        conn_cls = http.client.HTTPSConnection if scheme == "https" else http.client.HTTPConnection
        self._pooled_conn = conn_cls(host, port)
        self._pooled_conn_key = key
        return self._pooled_conn

    def _extract_sql_fallback(self, text: str) -> str:
        if not text:
            return ""

        match = re.search(r"(SELECT|WITH)\s.*", text, re.IGNORECASE | re.DOTALL)
        return match.group(0).strip() if match else ""

    def _call_ollama(self, prompt: str, timeout: float) -> Tuple[str, Dict[str, Any]]:
        """
        Stable Ollama caller (non-streaming).
        Fixes:
        - indentation bug
        - empty response handling
        - safer JSON parsing
        - consistent debug output
        - connection cleanup
        """

        from urllib.parse import urlsplit

        cfg = self.config.llm

        parsed_url = urlsplit(cfg.endpoint)
        scheme = parsed_url.scheme or "http"
        host = parsed_url.hostname or "localhost"
        port = parsed_url.port or (443 if scheme == "https" else 80)
        path = parsed_url.path or "/api/generate"

        payload = {
            "model": cfg.model,
            "prompt": prompt,
            "stream": False,  # IMPORTANT: keep false for stable parsing
            "options": {
                "temperature": cfg.temperature,
                "num_predict": cfg.max_tokens,
            },
        }

        data = json.dumps(payload).encode("utf-8")

        try:
            with self._conn_lock:
                conn = self._get_pooled_connection(scheme, host, port)

                conn.timeout = timeout
                if getattr(conn, "sock", None) is not None:
                    conn.sock.settimeout(timeout)

                conn.request(
                    "POST",
                    path,
                    body=data,
                    headers={"Content-Type": "application/json"},
                )

                resp = conn.getresponse()
                status = resp.status
                body_bytes = resp.read()

        except Exception as e:
            # 🔥 reset broken connection
            if self._pooled_conn is not None:
                try:
                    self._pooled_conn.close()
                except Exception:
                    pass
                self._pooled_conn = None
                self._pooled_conn_key = None

            raise LLMGenerationError(
                f"Could not reach Ollama endpoint {cfg.endpoint!r}: {e}"
            ) from e

        # ============================================================
        # Decode response
        # ============================================================
        body_text = body_bytes.decode("utf-8", errors="replace")

        logger.debug("ollama_raw_response: status=%s body=%r", status, body_text[:1000])

        # ============================================================
        # HTTP validation
        # ============================================================
        if status >= 400:
            raise LLMGenerationError(
                f"Ollama HTTP {status}: {body_text[:500]}"
            )

        if not body_text or not body_text.strip():
            raise LLMGenerationError(
                "❌ Empty response body from Ollama (model likely stalled or crashed)"
            )

        # ============================================================
        # JSON parsing
        # ============================================================
        try:
            data = json.loads(body_text)
        except json.JSONDecodeError as e:
            raise LLMGenerationError(
                f"Invalid JSON from Ollama: {e} | raw={body_text[:300]}"
            ) from e

        # ============================================================
        # RESPONSE EXTRACTION (SAFE)
        # ============================================================
        response_text = (
            data.get("response")
            or data.get("message", {}).get("content")
            or data.get("output")
            or data.get("thinking")   # 🔥 fallback for models that stream only "thinking"
        )

        if response_text is None:
            raise LLMGenerationError(
                f"Missing 'response' field: keys={list(data.keys())}"
            )

        response_text = response_text.strip()

        if not response_text:
            response_text = self._extract_sql_fallback(data.get("thinking", ""))
        # ============================================================
        # DEBUG INFO
        # ============================================================
        debug: Dict[str, Any] = {
            "model": cfg.model,
            "status": status,
            "response_preview": response_text[:300],
        }

        if "eval_count" in data or "prompt_eval_count" in data:
            debug["token_usage"] = {
                "prompt_tokens": data.get("prompt_eval_count"),
                "completion_tokens": data.get("eval_count"),
            }

        return response_text, debug

    @staticmethod
    def _extract_text_from_json_obj(parsed: Dict[str, Any]) -> Optional[str]:
        """
        Response-shape resilience (spec #15): Ollama's /api/generate
        returns {"response": "..."}; its OpenAI-compatible /api/chat
        (and some proxies) return {"message": {"content": "..."}}.
        Support both rather than assuming one fixed shape.
        """
        if "response" in parsed and parsed["response"] is not None:
            return parsed["response"]
        message = parsed.get("message")
        if isinstance(message, dict) and message.get("content") is not None:
            return message["content"]
        if "text" in parsed and parsed["text"] is not None:
            return parsed["text"]
        logger.debug("llm_response_shape_unrecognized: keys=%s", list(parsed.keys()))
        return None

    @classmethod
    def _parse_single_body(cls, body_text: str) -> Tuple[str, Optional[Dict[str, Any]]]:
        try:
            parsed = json.loads(body_text)
        except json.JSONDecodeError as e:
            raise LLMGenerationError(f"LLM response was not valid JSON: {e}") from e

        response_text = cls._extract_text_from_json_obj(parsed)
        if not response_text or not response_text.strip():
            raise LLMGenerationError(
                f"Empty response from model: {parsed}"
            )

        token_usage = None
        if "eval_count" in parsed or "prompt_eval_count" in parsed:
            token_usage = {
                "prompt_tokens": parsed.get("prompt_eval_count"),
                "completion_tokens": parsed.get("eval_count"),
            }
        return response_text, token_usage

    @classmethod
    def _parse_streaming_body(cls, body_text: str) -> Tuple[str, Optional[Dict[str, Any]]]:

        chunks: List[str] = []
        token_usage: Optional[Dict[str, Any]] = None

        lines = body_text.splitlines()

        if not lines:
            raise LLMGenerationError(
                f"❌ Empty streaming body received: {body_text[:200]!r}"
            )

        saw_valid_json = False

        for line in lines:
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # Fix 1: log instead of silently skipping
                logger.debug("skipping_invalid_json_line: %r", line[:200])
                continue

            saw_valid_json = True

            # ========================================================
            # 🔥 FIX 2: safer extraction (no silent loss)
            # ========================================================
            piece = cls._extract_text_from_json_obj(obj)

            if piece is None:
                # fallback to raw fields (VERY IMPORTANT for Ollama)
                piece = obj.get("response") or obj.get("message", {}).get("content")

            if piece:
                chunks.append(piece)

            # ========================================================
            # token usage detection (final chunk)
            # ========================================================
            if obj.get("done") is True:
                token_usage = {
                    "prompt_tokens": obj.get("prompt_eval_count"),
                    "completion_tokens": obj.get("eval_count"),
                }

        # ============================================================
        # 🔥 FIX 3: HARD FAILURE if NOTHING parsed
        # ============================================================
        if not saw_valid_json:
            raise LLMGenerationError(
                f"❌ No valid JSON lines in streaming response: {body_text[:300]!r}"
            )

        if not chunks:
            raise LLMGenerationError(
                f"❌ Streaming parsed but no text extracted: {body_text[:300]!r}"
            )

        return "".join(chunks), token_usage
    # -----------------------------------------------------
    # Response cleanup
    # -----------------------------------------------------
    def validate_sql(self, sql: str, schema: DiscoveredSchema) -> Tuple[bool, Optional[str]]:
        """
        Lightweight SQL validator used specifically on the raw LLM
        output before it's handed to the rest of the pipeline.

        FIX: the previous version only ever added *column* names to
        `valid_columns`, never table names -- so every syntactically
        correct query (which always quotes its own FROM/JOIN table,
        e.g. `FROM "metadata_data"`) was unconditionally rejected as
        referencing an "unknown column". That forced every LLM-
        generated query to fall back to the rule-based SQLBuilder even
        when the LLM's SQL was completely correct. Table names are now
        included in the set of valid quoted identifiers, and `AS
        "alias"` occurrences (which define a new name rather than
        reference an existing one) are exempted from the check
        entirely, since output aliases are not schema identifiers.

        Returns (is_valid, error_message).
        """

        if not sql or not sql.strip():
            return False, "Empty SQL"

        sql_lower = sql.lower()

        # 1. Must be SELECT/WITH
        if not (sql_lower.startswith("select") or sql_lower.startswith("with")):
            return False, "Only SELECT/WITH queries allowed"

        # 2. Extract valid table + column names from schema
        valid_columns: Set[str] = set()
        valid_tables: Set[str] = set()
        for table in schema.tables.values():
            valid_tables.add(table.name.lower())
            for col in table.columns:
                valid_columns.add(col.name.lower())

        # 3. Exempt output aliases (`AS "alias"`) -- these are names the
        #    query is DEFINING, not referencing against the schema.
        alias_spans = [m.span() for m in _ALIAS_AS_RE.finditer(sql)]

        def _is_alias_occurrence(pos: int) -> bool:
            return any(start <= pos < end for start, end in alias_spans)

        # 4. Extract referenced quoted identifiers (simple heuristic),
        #    skipping alias definitions, and check each against both
        #    valid_columns AND valid_tables.
        for match in re.finditer(r'"([^"]+)"', sql):
            if _is_alias_occurrence(match.start()):
                continue
            ident = match.group(1)
            if ident.lower() in valid_columns or ident.lower() in valid_tables:
                continue
            return False, f"Unknown identifier: {ident}"

        # 5. Basic safety check
        if "drop " in sql_lower or "delete " in sql_lower or "insert " in sql_lower or "update " in sql_lower:
            return False, "Unsafe SQL detected"

        return True, None
     
    def _extract_sql(self, raw_text: str) -> Optional[str]:
        """
        Thin wrapper around the centralized `SQLNormalizer` (spec #1,
        #2): markdown/prose stripping, statement-boundary detection,
        and multi-statement collapsing all happen in exactly one place
        now, instead of being reimplemented here with ad hoc regexes.
        Kept as a method with the same name/signature so any existing
        caller of `LLMSQLGenerator()._extract_sql(...)` is unaffected.
        """
        normalized = SQLNormalizer.extract_and_normalize(raw_text)
        if normalized.extra_statements_dropped:
            logger.warning(
                "sql_extraction: LLM output contained %d extra executable "
                "statement(s) beyond the first; only the first statement is "
                "used, per spec #1 (extract only the first executable "
                "statement).",
                normalized.extra_statements_dropped,
            )
        return normalized.sql


# =========================================================
# DEFENSIVE ROW-CAP FOR GENERATED SQL
# =========================================================
#
# The LLM prompt *asks* the model to add LIMIT, but that isn't
# enforced. This is a cheap, generic (schema-agnostic) safety net: any
# generated SQL that has no LIMIT clause and isn't obviously a bare
# scalar aggregate (SUM/AVG/... with no GROUP BY, which already returns
# exactly one row) is wrapped in an outer LIMIT so a wide/unbounded
# query can never balloon memory usage or execution time.
#
# Fix 1: the outer cap threshold is now configurable
# (`AgentConfig.max_result_rows`) and defaults to `None`, meaning "no
# defensive wrapping at all" -- generated SQL is executed exactly as
# produced. Passing a positive integer restores the previous wrapping
# behavior with that integer as the LIMIT (instead of always using a
# hardcoded 10).

_LIMIT_RE = re.compile(r"\bLIMIT\s+\d+", re.IGNORECASE)
_AGGREGATE_FN_RE = re.compile(r"\b(SUM|AVG|COUNT|MIN|MAX|MEDIAN|STDDEV|VARIANCE)\s*\(", re.IGNORECASE)
_GROUP_BY_RE = re.compile(r"\bGROUP\s+BY\b", re.IGNORECASE)


def _ensure_row_cap(sql: str, max_rows: Optional[int]) -> str:
    """
    Wrap SQL in a defensive LIMIT if needed.
    Avoids double wrapping and handles aggregates correctly.

    Fix 1: `max_rows=None` means "no cap" -- the SQL is returned
    completely unmodified (e.g. a `SELECT DISTINCT "brand" FROM
    "metadata_data";` query is executed exactly as generated, with no
    outer `LIMIT` added, so the client gets the full result set).
    Passing a positive integer preserves the previous defensive-wrap
    behavior, using that integer instead of a hardcoded 10.
    """

    if not sql or not sql.strip():
        return sql

    # ============================================================
    # Fix 1: None => unlimited results; skip wrapping entirely.
    # ============================================================
    if max_rows is None:
        return sql.strip()

    sql_clean = sql.strip()
    sql_upper = sql_clean.upper()

    # ============================================================
    # FIX 1: SAFE LIMIT DETECTION
    # ============================================================
    has_top_level_limit = bool(
        re.search(r"\bLIMIT\b", sql_upper)
    )

    if has_top_level_limit:
        return sql_clean

    # ============================================================
    # FIX 2: AGGREGATE + GROUP BY DETECTION
    # ============================================================
    has_aggregate = bool(_AGGREGATE_FN_RE.search(sql_clean))
    has_group_by = bool(_GROUP_BY_RE.search(sql_clean))

    select_part = sql_clean.lower().split("select", 1)[-1].split("from", 1)[0]

    is_scalar_aggregate = (
        has_aggregate
        and not has_group_by
        and len([c for c in select_part.split(",") if c.strip()]) <= 1
    )

    if is_scalar_aggregate:
        return sql_clean

    # ============================================================
    # FIX 3: PREVENT OVER-WRAPPING
    # ============================================================
    if sql_upper.startswith("SELECT") and sql_upper.count("FROM (") > 1:
        return sql_clean

    # ============================================================
    # SAFE WRAP
    # ============================================================
    trimmed = sql_clean.rstrip().rstrip(";")

    return f"SELECT * FROM ({trimmed}) AS _capped_result LIMIT {max_rows};"


# =========================================================
# DUCKDB POOL
# =========================================================

class DuckDBPool:
    """
    A single shared DuckDB connection, guarded by a re-entrant lock.

    This intentionally replaces the earlier design of one connection
    *per calling thread*. Thread-local connections were safe only as
    long as every DuckDB call happened on the same thread that
    constructed the agent. That stopped being true once blocking
    DuckDB calls started being off-loaded to `asyncio.to_thread()`
    (see `StructuredAgent.run()`), which can dispatch consecutive calls
    to *different* worker threads. A thread-local connection to an
    in-memory database (":memory:" -- now this class's default, and
    also HybridAgent's own default) is a **separate, isolated
    database per thread**: two calls landing on two different threads
    would silently see two different, mostly-empty databases, causing
    intermittent "table not found" errors that only reproduce under
    concurrency.

    A single connection protected by a lock is correct for both
    file-backed and in-memory databases. Because individual DuckDB
    queries are typically sub-millisecond to a few hundred
    milliseconds, lock contention is negligible next to the I/O this
    pool now off-loads to a worker thread anyway.
    """

    def __init__(self, db_path: str = "F:/PROJECT/Database/Duckdb/analytics.duckdb"):
        self._db_path = db_path
        self._lock = threading.RLock()
        self._conn: Optional["duckdb.DuckDBPyConnection"] = None

    @property
    def lock(self) -> threading.RLock:
        """Exposed so callers can hold the lock across a *sequence* of
        related blocking calls (e.g. execute + repair-and-retry) so the
        sequence is atomic with respect to other coroutines sharing this
        connection."""
        return self._lock

    def connection(self) -> "duckdb.DuckDBPyConnection":
        if self._conn is None:
            with self._lock:
                if self._conn is None:
                    self._conn = duckdb.connect(self._db_path)
        return self._conn

    def close_all(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None


# Backwards-compatible alias for the original private name.
_DuckDBPool = DuckDBPool


# =========================================================
# SQL EXECUTION
# =========================================================

def execute_sql(
    sql: str, pool: DuckDBPool, max_rows: int = 500
) -> Tuple[bool, List[Dict[str, Any]], List[str], Optional[str]]:
    """
    Executes `sql` against the pool's connection and returns
    (success, rows, columns, error). Uses `fetchmany(max_rows)` instead
    of `fetchall()` + slicing, so a query that (despite `_ensure_row_cap`)
    still returns far more than `max_rows` rows never has its full
    result set materialized in Python before being discarded.
    """
    try:
        conn = pool.connection()
        result = conn.execute(sql)
        cols = [d[0] for d in result.description] if result.description else []
        try:
            rows_raw = result.fetchmany(max_rows)
        except AttributeError:
            # Defensive fallback for older duckdb versions without
            # fetchmany() on the result object.
            rows_raw = result.fetchall()[:max_rows]

        rows = [dict(zip(cols, r)) for r in rows_raw]
        return True, rows, cols, None

    except Exception as e:
        return False, [], [], str(e)


# =========================================================
# RESULT CACHE
# =========================================================

class _ResultCache:
    """
    Small, dependency-free TTL + size-bounded cache for `AgentResult`,
    keyed on (query, filters, entities, needs_time_series,
    schema.version). Only successful results are cached (see
    `StructuredAgent.run`), and the key incorporates the schema's
    version number so any `seed_dataframe`/`seed_csv` call -- which
    invalidates the schema cache and bumps its version -- naturally
    invalidates every cache entry keyed on the old version too, without
    needing to eagerly scan/evict them.
    """

    def __init__(self, maxsize: int, ttl_seconds: float):
        self._maxsize = max(1, maxsize)
        self._ttl = max(0.0, ttl_seconds)
        self._store: "OrderedDict[str, Tuple[float, AgentResult]]" = OrderedDict()
        self._lock = threading.Lock()

     
    def make_key(
        query: str,
        filters: Optional[Dict[str, Any]],
        entities: Optional[Dict[str, Any]],
        needs_time_series: bool,
        schema_version: int,
    ) -> str:
        payload = json.dumps(
            {
                "q": query, "f": filters or {}, "e": entities or {},
                "t": bool(needs_time_series), "v": schema_version,
            },
            sort_keys=True, default=str,
        )
        return hashlib.md5(payload.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[AgentResult]:
        with self._lock:
            item = self._store.get(key)
            if item is None:
                return None
            ts, result = item
            if time.time() - ts > self._ttl:
                self._store.pop(key, None)
                return None
            self._store.move_to_end(key)
            return result

    def set(self, key: str, result: AgentResult) -> None:
        with self._lock:
            self._store[key] = (time.time(), result)
            self._store.move_to_end(key)
            while len(self._store) > self._maxsize:
                self._store.popitem(last=False)

    def invalidate_all(self) -> None:
        with self._lock:
            self._store.clear()


# =========================================================
# SQL ATTEMPT TRACKING (Changes 1/2/6/8/9)
# =========================================================

@dataclass
class _SQLAttempt:
    """
    Records the full outcome of ONE SQL-generation-and-execution
    strategy (either "llm" or "rules"), so `StructuredAgent.run()` can
    report rich per-attempt metadata (Change 6) and decide whether a
    fallback is warranted, without re-deriving any of this after the
    fact. Internal to this module -- not part of the public contract.
    """
    source: str  # "llm" | "rules"
    sql: Optional[str] = None
    tables_used: List[TableInfo] = field(default_factory=list)
    join_edges: List[Relationship] = field(default_factory=list)
    llm_debug: Optional[Dict[str, Any]] = None
    validation: Optional[ValidationResult] = None
    ok: bool = False
    rows: List[Dict[str, Any]] = field(default_factory=list)
    columns: List[str] = field(default_factory=list)
    error: Optional[str] = None
    final_sql: Optional[str] = None
    repair_attempts: int = 0
    # Coarse, stable failure category (Change 8), e.g. "llm_unreachable",
    # "llm_timeout", "no_sql_returned", "llm_invalid_response",
    # "validation_below_threshold", "execution_failed",
    # "rule_generation_failed". None when the attempt succeeded.
    failure_reason: Optional[str] = None
    # The QueryPlan (if any) that was built for and handed to this
    # attempt's generator -- exposed so `run()` can attach it to
    # `result.metadata["query_plan"]` for debugging/introspection.
    plan: Optional["QueryPlan"] = None


class _DebugTrace:
    """
    Accumulates the bracketed `[Stage] status ...` lines described in
    spec #17 ("Debug Mode"). A no-op (near-zero overhead) when
    `enabled=False` so normal (non-debug) requests pay nothing extra.
    """

    __slots__ = ("enabled", "_lines")

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self._lines: List[str] = []

    def add(self, line: str) -> None:
        if self.enabled:
            self._lines.append(line)

    def finalize(self) -> List[str]:
        return list(self._lines)


# =========================================================
# STRUCTURED AGENT
# =========================================================

class StructuredAgent:
    """
    Fully generic, schema-agnostic SQL analytics agent.

    Every decision — which table(s), which columns, whether to group,
    whether a join is needed, what time granularity to use — is derived
    at runtime from statistical profiling and semantic query
    understanding. No table, column, or domain name is ever assumed to
    exist. Generated SQL is validated before execution and, on
    execution failure, a bounded number of automatic repair attempts is
    made before the query is reported as failed.

    The agent auto-discovers schema on construction: any tables already
    present on the underlying DuckDB connection (e.g. a persistent
    `.duckdb` file containing `structured_data`, `metadata_data`, etc.)
    are picked up immediately via SHOW TABLES / DESCRIBE / information_
    schema — seed_dataframe()/seed_csv() remain available for adding
    *new* tables at runtime, but are no longer required to get started.

    Callers can supply filters two ways, and both are honored:
      - Implicitly, via natural language in `query` (e.g. "rating
        greater than 4").
      - Explicitly, via the `filters=`/`entities=` kwargs on `run()`
        (e.g. `filters={"rating": {"gte": 4}}` or
        `entities={"brand": "Apple"}`), which is useful when an upstream
        NLU/entity-extraction step has already identified the field and
        value and you don't want to rely on regex phrase-matching.
    Both sources are merged before SQL generation and resolved to real
    columns using the same semantic/fuzzy matching either way.

    SQL generation strategy (Changes 1/2): the LLM-driven
    `LLMSQLGenerator` is now the PRIMARY generation strategy, always
    attempted first. The rule-based, deterministic `SQLBuilder` is a
    true FALLBACK, invoked only after the LLM attempt fails outright,
    fails validation below `AgentConfig.min_sql_validation_confidence`,
    or fails execution (even after the existing bounded repair).

    Responsibility boundary: this class ONLY receives a query, generates
    SQL, validates it, executes it, and returns a normalized
    `AgentResult`. It performs no orchestration, no execution-path
    planning, no semantic retrieval, no answer generation, and no
    fusion -- all of that stays in HybridAgent, exactly as before.
    """

    def __init__(
        self,
        db_path: str = ":memory:",
        max_rows: int = 500,
        config: Optional[AgentConfig] = None,
    ):
        self.pool = DuckDBPool(db_path)
        self.max_rows = max_rows
        self.config = config or AgentConfig(max_row_limit=max_rows)

        self._schema_cache = SchemaCache(self.config)
        self._intent_detector = IntentDetector()
        self._linker = SchemaLinker(self.config)
        self._query_planner = QueryPlanner(self.config, self._linker)
        self._sql_builder = SQLBuilder(self.config, self._linker)
        self._validator = SQLValidator()
        self._repair_engine = SQLRepairEngine(self.config)
        self._llm_generator = LLMSQLGenerator(self.config)
        self._result_cache: Optional[_ResultCache] = (
            _ResultCache(self.config.result_cache_maxsize, self.config.result_cache_ttl_seconds)
            if self.config.enable_result_cache else None
        )

        # ---- Production metrics (spec #14) ----
        # Bounded rolling window of recent request latencies, used to
        # compute a running average execution time via
        # `get_metrics_summary()` without unbounded memory growth.
        self._latency_lock = threading.Lock()
        self._recent_latencies_ms: "OrderedDict[int, float]" = OrderedDict()
        self._latency_window = 500
        self._request_count = 0

        # Public connection handle (compatibility alias). `self.conn` is
        # the primary attribute; `self.connection` is an alias kept for
        # callers/integrations (HybridAgent, SemanticAgent, notebooks,
        # etc.) that expect that name. Both now point at the single
        # shared DuckDBPool connection (see DuckDBPool's docstring for
        # why thread-local connections were retired).
        self.conn = self.pool.connection()
        self.connection = self.conn

        # Eagerly discover schema so the agent is immediately queryable
        # against a pre-existing database without requiring a manual
        # seed_dataframe()/seed_csv() call first. If discovery fails for
        # any reason (e.g. a genuinely empty/new database), this is not
        # fatal — run() will retry discovery on first use. Constructors
        # can't be async, so this stays a direct (synchronous) call.
        try:
            schema = self._discover_schema_blocking()
            logger.info(
                "agent_init: auto-discovered %d table(s): %s",
                len(schema.tables), sorted(schema.tables.keys()),
            )
        except Exception:
            logger.exception("agent_init: eager schema discovery failed; will retry lazily on first query")
    # -----------------------------------------------------
    # Blocking helpers (always invoked via asyncio.to_thread from run())
    # -----------------------------------------------------

    def _discover_schema_blocking(self, force_refresh: bool = False) -> DiscoveredSchema:
        """Runs schema discovery (if the TTL cache is stale/empty) under
        the connection pool's lock. Kept as a single blocking unit so it
        can be off-loaded wholesale via `asyncio.to_thread` without
        blocking the event loop for its (potentially non-trivial)
        duration."""
        with self.pool.lock:
            if force_refresh:
                self._schema_cache.invalidate()
            return self._schema_cache.get(self.pool.connection())

    def _execute_with_repair_blocking(
        self, sql: str, schema: DiscoveredSchema, tables_used: List[TableInfo]
    ) -> Tuple[bool, List[Dict[str, Any]], List[str], Optional[str], str, int]:
        """
        Executes `sql` and, on a recoverable failure, attempts up to
        `AgentConfig.max_repair_attempts` automatic repair-and-retry
        cycles -- all under the connection pool's lock, entirely on a
        worker thread. Returns
        (ok, rows, columns, error, final_sql, repair_attempts).
        """
        with self.pool.lock:
            ok, rows, cols, err = execute_sql(sql, self.pool, self.max_rows)
            current_sql = sql
            repair_attempts = 0
            if not ok and self.config.enable_sql_auto_repair:
                while not ok and repair_attempts < self.config.max_repair_attempts:
                    repaired_sql = self._repair_engine.attempt_repair(current_sql, err or "", schema, tables_used)
                    if not repaired_sql or repaired_sql == current_sql:
                        break

                    # Fix (spec #10, #13): classify the proposed edit
                    # before applying it. The rule engine repairs
                    # STRUCTURE (renamed identifiers, added GROUP BY
                    # entries, alias qualification) -- it must never
                    # reinterpret business logic. If the diff between
                    # `current_sql` and `repaired_sql` structurally
                    # changes the query's shape (different clause
                    # keywords/aggregates/operators), that's a semantic
                    # edit and is rejected: the original SQL and error
                    # are kept, and repair stops here rather than
                    # silently executing a query with different
                    # meaning than what was validated/intended.
                    mutation_kind = classify_sql_mutation(current_sql, repaired_sql)
                    if mutation_kind == SQLMutationKind.SEMANTIC.value:
                        logger.warning(
                            "sql_repair_rejected: proposed repair changed SQL semantics "
                            "(mutation_kind=%s); keeping prior SQL/error instead of applying it. "
                            "current=%s proposed=%s",
                            mutation_kind, current_sql, repaired_sql,
                        )
                        break

                    repair_attempts += 1
                    logger.info(
                        "sql_repair_attempt=%d sql=%s mutation_kind=%s",
                        repair_attempts, repaired_sql, mutation_kind,
                    )
                    ok, rows, cols, err = execute_sql(repaired_sql, self.pool, self.max_rows)
                    current_sql = repaired_sql
        return ok, rows, cols, err, current_sql, repair_attempts

    # -----------------------------------------------------
    # SQL generation orchestration
    # -----------------------------------------------------

    def _select_prompt_tables(
        self, schema: DiscoveredSchema, detected: DetectedIntent
    ) -> List[TableInfo]:
        """
        Picks which tables to describe to the LLM. For small schemas,
        every table is included (unchanged behavior). For schemas with
        more tables than `AgentConfig.llm_max_tables_in_prompt`, only
        the top-ranked tables (via the same SchemaLinker relevance
        scoring used for deterministic generation) are included, plus
        any table literally named in the query -- keeping prompt size,
        latency, and cost from scaling with total schema size.
        """
        all_tables = list(schema.tables.values())
        if len(all_tables) <= self.config.llm_max_tables_in_prompt:
            return all_tables

        ranked = self._linker.rank_tables(schema, detected)
        top = [t for _, t in ranked[: self.config.llm_max_tables_in_prompt]]
        top_names = {t.name for t in top}
        lowered_query = detected.raw_query.lower()
        mentioned = [
            t for t in all_tables
            if t.name.lower() in lowered_query and t.name not in top_names
        ]
        return top + mentioned

     
    def _classify_llm_error(self, message: str) -> str:
        """
        Maps an `LLMGenerationError` message to a coarse, stable failure
        category (Change 8) for metadata/debugging. Classified by
        message content rather than exception subclassing, since
        `LLMSQLGenerator` intentionally raises a single
        `LLMGenerationError` type for every failure mode (network,
        timeout, malformed JSON, empty response, unparseable SQL) --
        see that class's docstring -- and is not being restructured
        here per the "preserve everything else" constraint.
        """
        m = message.lower()
        if "could not reach" in m or "urlerror" in m:
            return "llm_unreachable"
        if "timed out" in m or "timeout" in m:
            return "llm_timeout"
        if "not valid json" in m or "missing 'response' field" in m:
            return "llm_invalid_response"
        if "no parseable sql" in m:
            return "no_sql_returned"
        if "llm request failed" in m:
            return "llm_request_failed"
        return "llm_unknown_error"

     
    def _compute_overall_confidence(
        self, planner: float, schema: float, sql: float, repair: float, execution: float,
        *, fallback_used: bool = False, repair_attempts: int = 0,
        semantic_issue_count: int = 0, empty_result_unexpected: bool = False,
        wrong_table: bool = False,
    ) -> float:
        """
        Fix 7: confidence now reflects the WEAKEST stage rather than a
        blended weighted average. The previous formula
        (0.20*planner + 0.30*sql + 0.30*execution + 0.10*repair + 0.10*schema)
        let a strong score in one stage dilute a catastrophic failure in
        another -- e.g. planner=0.98 with sql=0/execution=0 still isn't
        forced to (near-)zero unless you check the arithmetic, and
        conversely a good SQL/execution score couldn't be trusted to
        mean "this SQL is right" once a fallback silently replaced it.

        The new score starts from `min()` of the per-stage confidences
        (repair confidence only counts when repair actually ran -- an
        untouched query shouldn't be penalized for a repair it never
        needed), then subtracts explicit penalties for problems a
        per-stage score alone can't see: the rule-based fallback firing,
        repair being required, the final SQL not matching what the
        QueryPlan asked for (missing metrics/grouping/etc.), an
        unexpectedly empty result, or the wrong table being queried.
        Clamped to [0.0, 1.0].
        """
        stage_scores = [planner, schema, sql, execution]
        if repair_attempts > 0:
            stage_scores.append(repair)
        base = min(stage_scores) if stage_scores else 0.0

        penalty = 0.0
        if fallback_used:
            penalty += 0.15
        if repair_attempts > 0:
            penalty += min(0.05 * repair_attempts, 0.15)
        if semantic_issue_count > 0:
            penalty += min(0.10 * semantic_issue_count, 0.30)
        if empty_result_unexpected:
            penalty += 0.20
        if wrong_table:
            penalty += 0.30

        return round(max(0.0, min(1.0, base - penalty)), 3)

    async def _attempt_llm_sql(
        self, schema: DiscoveredSchema, detected: DetectedIntent,
        explicit_filters: List[ExtractedFilter], limit: Optional[int],
        plan: Optional["QueryPlan"] = None,
    ) -> _SQLAttempt:
        """
        Change 1: LLM is the PRIMARY generation strategy. Generates SQL
        via the LLM, validates it, and -- only if validation confidence
        clears `min_sql_validation_confidence` (Change 9) -- executes it
        (with the existing bounded repair-and-retry). Never raises; all
        failure modes are captured on the returned `_SQLAttempt`.

        When `plan` is supplied (built once in `run()` via
        `QueryPlanner`), it's passed through to both the prompt builder
        (spec #11 -- the LLM translates the plan instead of
        re-reasoning about the question) and the validator (spec #14 --
        generated SQL is checked against exactly what the plan asked
        for).
        """
        attempt = _SQLAttempt(source="llm", plan=plan)
        if not self.config.use_llm_sql_generation:
            attempt.failure_reason = "llm_disabled"
            return attempt

        try:
            prompt_tables = self._select_prompt_tables(schema, detected)
            llm_sql, llm_debug = await self._llm_generator.generate(
                schema, detected.raw_query, limit, explicit_filters, tables=prompt_tables,
                timeout_mode=self.config.timeout_mode, plan=plan,
            )
        except LLMGenerationError as e:
            attempt.error = str(e)
            attempt.failure_reason = self._classify_llm_error(str(e))
            logger.warning("llm_sql_generation_failed reason=%s error=%s", attempt.failure_reason, e)
            return attempt

        llm_tables, llm_edges = self._resolve_llm_tables(schema, llm_sql)
        # Fix 1: use the dedicated result-cap config (`max_result_rows`,
        # default None = unlimited) rather than the TOP-N/default-row-
        # limit value `limit`, which was previously misused here and
        # silently truncated every LLM-generated query to 10 rows.
        attempt.sql = _ensure_row_cap(llm_sql, self.config.max_result_rows)
        attempt.tables_used = llm_tables
        attempt.join_edges = llm_edges
        attempt.llm_debug = llm_debug

        # ================================
        # SQL VALIDATION STEP
        # ================================
        # Fix 3/6: validation issues alone no longer discard otherwise-
        # correct LLM SQL. The static validator is a heuristic
        # (identifier spelling, plan-shape checks) and can produce false
        # positives (the alias-in-ORDER-BY case fixed above being one).
        # It still gates on *confidence* -- SQL the validator has very
        # low confidence in is skipped before spending a DB round-trip
        # -- but anything above that bar is executed. Genuine problems
        # (a truly unknown column, a missing GROUP BY entry, etc.) are
        # then caught by DuckDB's own execution error and handled by
        # `SQLRepairEngine` in `_execute_with_repair_blocking` below,
        # which is the correct place to fix them (it sees DuckDB's own,
        # authoritative error message rather than a static guess). Only
        # if execution still fails after repair does `run()` fall back
        # to the rule-based `SQLBuilder`.
        if self.config.validate_sql_before_execution:
            validation = self._validator.validate(schema, attempt.sql, attempt.tables_used, plan=plan)
            attempt.validation = validation
            if not validation.valid:
                logger.info("llm_sql_validation_issues (non-fatal, will still attempt execution+repair): %s", validation.issues)

            if validation.confidence < self.config.min_sql_validation_confidence:
                attempt.failure_reason = "validation_below_threshold"
                attempt.error = (
                    f"Validation confidence "
                    f"{validation.confidence:.2f} below threshold "
                    f"{self.config.min_sql_validation_confidence:.2f}"
                    + (f" (issues: {validation.issues})" if validation.issues else "")
                )
                return attempt

        ok, rows, cols, err, final_sql, repair_attempts = await asyncio.to_thread(
            self._execute_with_repair_blocking, attempt.sql, schema, attempt.tables_used
        )
        attempt.ok = ok
        attempt.rows = rows
        attempt.columns = cols
        attempt.error = err
        attempt.final_sql = final_sql
        attempt.repair_attempts = repair_attempts
        if not ok:
            attempt.failure_reason = "execution_failed"
        return attempt

    async def _attempt_rule_sql(
        self, schema: DiscoveredSchema, detected: DetectedIntent,
        explicit_filters: List[ExtractedFilter], limit: Optional[int],
        plan: Optional["QueryPlan"] = None,
    ) -> _SQLAttempt:
        """
        Change 2: the deterministic `SQLBuilder` is now a TRUE fallback,
        invoked only after the LLM attempt fails generation, validation,
        or execution. Generates SQL, validates it, and executes it (with
        the existing repair-and-retry) exactly once. Never raises: any
        unexpected exception from the deterministic builder itself
        (not just the documented `SQLGenerationError`) is now caught
        too, so a bug in SQLBuilder can never escape as an opaque,
        untyped error from deep inside `run()` -- it's captured on the
        `_SQLAttempt` like every other failure mode.

        Fix 3: when a `QueryPlan` is available, the rule engine now
        PREFERS `SQLBuilder.build_from_plan()` -- which reuses the
        already-resolved metrics/dimensions/filters/joins/sort/limit
        instead of re-parsing `detected.raw_query` from scratch. This
        is what stops the fallback from ever constructing SQL that
        contradicts a semantically-correct (but merely mis-validated or
        execution-failed) LLM attempt. `build()` (the original NL-
        parsing path) is used only when no plan is available, or the
        plan doesn't carry enough resolved information to build from
        (`SQLGenerationError` from `build_from_plan`) -- so a thin plan
        never produces *worse* SQL than the pre-planner behavior.
        """
        attempt = _SQLAttempt(source="rules", plan=plan)
        try:
            if plan is not None:
                try:
                    sql, tables_used, join_edges = self._sql_builder.build_from_plan(schema, plan)
                except SQLGenerationError as plan_err:
                    logger.info(
                        "rule_sql_plan_build_unusable (%s); falling back to NL-parsing SQLBuilder.build()",
                        plan_err,
                    )
                    sql, tables_used, join_edges = self._sql_builder.build(
                        schema, detected, extra_filters=explicit_filters
                    )
            else:
                sql, tables_used, join_edges = self._sql_builder.build(
                    schema, detected, extra_filters=explicit_filters
                )
        except SQLGenerationError as e:
            attempt.error = str(e)
            attempt.failure_reason = "rule_generation_failed"
            return attempt
        except Exception as e:
            # Defensive catch-all: an unexpected bug inside SQLBuilder
            # (or its collaborators) must not propagate past this point
            # as an opaque error -- it's recorded exactly like any other
            # rule-based generation failure, and the caller still gets a
            # well-formed AgentResult back.
            attempt.error = str(e)
            attempt.failure_reason = "rule_generation_failed"
            logger.exception("rule_based_sql_generation_unexpected_error")
            return attempt

        # Fix 1: use the dedicated result-cap config instead of the
        # TOP-N/default-row-limit value `limit` (see comment in
        # `_attempt_llm_sql` above for the full rationale).
        attempt.sql = _ensure_row_cap(sql, self.config.max_result_rows)
        attempt.tables_used = tables_used
        attempt.join_edges = join_edges

        if self.config.validate_sql_before_execution:
            validation = self._validator.validate(schema, attempt.sql, attempt.tables_used, plan=plan)
            attempt.validation = validation
            if not validation.valid:
                logger.info("rule_sql_validation_issues: %s", validation.issues)

        ok, rows, cols, err, final_sql, repair_attempts = await asyncio.to_thread(
            self._execute_with_repair_blocking, attempt.sql, schema, attempt.tables_used
        )
        attempt.ok = ok
        attempt.rows = rows
        attempt.columns = cols
        attempt.error = err
        attempt.final_sql = final_sql
        attempt.repair_attempts = repair_attempts
        if not ok:
            attempt.failure_reason = "execution_failed"
        return attempt

    # -----------------------------------------------------
    # Public API
    # -----------------------------------------------------

    async def run(
        self,
        query: str,
        needs_time_series: bool = False,
        filters: Optional[Dict[str, Any]] = None,
        entities: Optional[Dict[str, Any]] = None,
        schema: Optional[DiscoveredSchema] = None,
        metadata: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
        debug: Optional[bool] = None,
        **kwargs,
    ) -> AgentResult:
        """
        Receive query -> generate SQL -> validate SQL -> execute SQL ->
        normalize result -> return structured response. Nothing else.

        SQL generation strategy (Changes 1/2): the LLM is now the
        PRIMARY generation strategy, always attempted first. The
        rule-based `SQLBuilder` runs strictly as a fallback -- only
        after LLM generation fails outright, LLM SQL fails validation
        below `AgentConfig.min_sql_validation_confidence`, or LLM SQL
        fails execution (even after the existing bounded repair). Each
        strategy is attempted at most once; there are no retry loops.

        Args:
            query: natural-language question. May itself contain filter
                phrases ("... where rating is greater than 4") and/or
                grouping/time hints ("... by category", "monthly ...").
            needs_time_series: force trend/time-series intent regardless
                of how `query` is phrased.
            filters: optional explicit filters, e.g.
                {"rating": {"gte": 4}, "category": {"in": ["a", "b"]}}
                or the equality shorthand {"category": "Books"}.
                Merged with any filters parsed out of `query` itself.
            entities: optional explicit equality filters, e.g.
                {"brand": "Apple"} — a convenience shape for the common
                case of "this field must equal this exact value" that an
                upstream entity-extraction step might hand over.
            schema: optional pre-discovered `DiscoveredSchema`. When
                supplied, schema discovery/caching is skipped entirely
                for this call -- useful for a caller (e.g. an
                orchestrator) that already called `get_schema()` and
                wants to reuse it across several `run()` calls instead
                of paying for repeated TTL-based rediscovery.
            metadata: optional opaque caller metadata (e.g. planner
                hints), merged into `result.metadata["caller_metadata"]`
                without StructuredAgent interpreting it in any way.
            use_cache: whether to check/populate the in-process result
                cache for this call (default True). Set False to force
                fresh generation + execution regardless of caching.
            debug: when True, populates `result.metadata["debug_trace"]`
                with a full bracketed execution trace (spec #17) and
                also emits it via `logger.debug()`. Defaults to
                `AgentConfig.debug_mode` when not explicitly passed.
            **kwargs: reserved for future extension; unrecognized keys
                are accepted (and ignored) rather than raising, so
                callers passing extra metadata don't break.

        Always returns an `AgentResult` -- never raises for query-time
        failures (schema/generation/validation/execution errors are all
        captured into `result.error` with `result.success=False`).
        """
        start = time.perf_counter()
        result = AgentResult()
        caller_supplied_schema = schema is not None
        debug = self.config.debug_mode if debug is None else debug
        trace = _DebugTrace(enabled=debug)
        metrics = RequestMetrics()

        try:
            t_schema0 = time.perf_counter()
            if schema is None:
                schema = await asyncio.to_thread(self._discover_schema_blocking)
                if schema.is_empty():
                    # Tables may have been created directly on the
                    # underlying DuckDB connection since the last
                    # discovery — re-discover once before concluding
                    # there's really nothing to query.
                    schema = await asyncio.to_thread(self._discover_schema_blocking, True)
            metrics.schema_discovery_ms = round((time.perf_counter() - t_schema0) * 1000, 2)
            trace.add(f"[Schema Discovery] \u2713 {len(schema.tables)} table(s)")

            result.schema_confidence = _clamp01(schema.confidence_score())

            if schema.is_empty():
                structured_err = StructuredError(
                    stage="Schema Discovery",
                    reason="No tables found in the connected DuckDB database.",
                    category="schema_empty",
                    recommendation="Create a table directly on the connection, or call "
                                   "seed_dataframe()/seed_csv() to load one.",
                )
                result.success = False
                result.error = (
                    "No tables were found in the connected DuckDB database. "
                    "Create a table directly on the connection, or call "
                    "seed_dataframe()/seed_csv() to load one."
                )
                result.structured_error = structured_err.to_dict()
                result.metadata["debug_trace"] = trace.finalize()
                result.latency_ms = round((time.perf_counter() - start) * 1000, 2)
                self._record_latency(result.latency_ms)
                logger.warning("run: no tables found in database; query=%r", query)
                return result

            cache_key: Optional[str] = None
            if use_cache and self._result_cache is not None:
                cache_key = _ResultCache.make_key(query, filters, entities, needs_time_series, schema.version)
                cached = self._result_cache.get(cache_key)
                if cached is not None:
                    cache_hit = replace(cached, metadata={**cached.metadata, "cache_hit": True})
                    cache_hit.latency_ms = round((time.perf_counter() - start) * 1000, 2)
                    self._record_latency(cache_hit.latency_ms)
                    logger.info("run_complete: cache_hit=True query=%r", query)
                    return cache_hit

            t_intent0 = time.perf_counter()
            detected = self._intent_detector.detect(query, needs_time_series)
            metrics.intent_detection_ms = round((time.perf_counter() - t_intent0) * 1000, 2)
            result.planner_confidence = _clamp01(detected.confidence)
            logger.debug(
                "planner: intent=%s confidence=%.2f grouping=%s granularity=%s",
                detected.intent.value, detected.confidence, detected.wants_grouping, detected.granularity,
            )
            trace.add(f"[Intent Detection] \u2713 {detected.intent.value} (confidence={detected.confidence:.2f})")

            explicit_filters = self._sql_builder.filter_extractor.from_structured(filters, entities)
            # Fix 2: previously this was
            #   `min(detected.requested_n or self.config.default_row_limit, ...)`
            # which silently applied LIMIT 10 to EVERY query, including
            # ones like "show all brands" that never asked for a limit.
            # Now `limit` is None unless the user explicitly asked for a
            # bounded result ("top N" / "first N" / "highest N" / ...,
            # captured in `detected.requested_n`) -- in which case it's
            # still capped at `max_row_limit` as a safety ceiling.
            limit: Optional[int] = (
                min(detected.requested_n, self.config.max_row_limit)
                if detected.requested_n is not None
                else None
            )

            # ---- Semantic query planning (spec #2-#10) -----------------
            # Built once per query, when enabled, and handed to BOTH the
            # LLM generator (as an already-reasoned instruction block --
            # spec #11) and the SQL validator (spec #14). A failure here
            # is never fatal to the request: the pipeline simply
            # continues without a plan, exactly as it always could.
            t_plan0 = time.perf_counter()
            query_plan: Optional["QueryPlan"] = None
            if self.config.enable_query_planner:
                try:
                    query_plan = self._query_planner.plan(
                        schema, detected,
                        explicit_filters=explicit_filters,
                        row_limit=limit,
                    )
                    result.metadata["query_plan"] = query_plan.to_dict()

                    # Fix 2: propagate the QueryPlanner's own confidence
                    # score into `result.planner_confidence`. Previously
                    # this field was only ever set from the (unrelated,
                    # earlier) IntentDetector confidence a few lines
                    # above and never updated once the richer, plan-
                    # aware confidence was computed -- so the displayed
                    # `planner_confidence` stayed stale (often 0.0) even
                    # though `query_plan.overall_confidence` (e.g. 0.78)
                    # was already correctly stored in
                    # `result.metadata["query_plan"]`. This is the single
                    # place that value now flows into the top-level
                    # confidence field, and nothing else overwrites it
                    # afterward.
                    result.planner_confidence = _clamp01(query_plan.overall_confidence)

                    trace.add(
                        f"[Query Planner] \u2713 intent={query_plan.primary_intent} "
                        f"metrics={len(query_plan.metrics)} dims={len(query_plan.dimensions)} "
                        f"(confidence={query_plan.overall_confidence:.2f})"
                    )
                except Exception:
                    logger.exception("query_planner_failed; continuing without a plan")
                    query_plan = None
                    trace.add("[Query Planner] \u2717 Failed -- continuing without a plan")
                    # NOTE: on failure, result.planner_confidence
                    # intentionally keeps the IntentDetector-derived
                    # value set earlier -- previous behavior preserved.
            metrics.schema_linking_ms = round((time.perf_counter() - t_plan0) * 1000, 2)

            # ---- Changes 1/2: LLM primary, rules as true fallback -----
            t_llm0 = time.perf_counter()
            attempts: List[_SQLAttempt] = []
            llm_attempt = await self._attempt_llm_sql(schema, detected, explicit_filters, limit, plan=query_plan)
            attempts.append(llm_attempt)
            metrics.llm_latency_ms = round((time.perf_counter() - t_llm0) * 1000, 2)
            if llm_attempt.llm_debug:
                metrics.prompt_chars = llm_attempt.llm_debug.get("prompt_chars", 0)
                metrics.prompt_tokens_estimate = llm_attempt.llm_debug.get("prompt_tokens_estimate", 0)
                metrics.token_usage = llm_attempt.llm_debug.get("token_usage")
            if llm_attempt.validation:
                validator_mark = "\u2713 Passed" if llm_attempt.validation.valid else "\u2717 Failed"
                trace.add(
                    f"[Validator] {validator_mark} "
                    f"(confidence={llm_attempt.validation.confidence:.2f})"
                )
            llm_mark = "\u2713 Completed" if llm_attempt.sql else "\u2717 Failed"
            trace.add(
                f"[LLM] {llm_mark} "
                f"in {metrics.llm_latency_ms / 1000:.1f}s"
                + (f" -- {llm_attempt.error}" if llm_attempt.error else "")
            )

            final_attempt = llm_attempt
            fallback_trigger: Optional[str] = None
            rules_used = False

            if not llm_attempt.ok:
                fallback_trigger = llm_attempt.failure_reason or "llm_attempt_failed"
                logger.info(
                    "sql_pipeline: LLM attempt did not succeed (%s); falling back to rule-based SQLBuilder",
                    fallback_trigger,
                )
                trace.add(f"[Fallback] \u2192 rule-based SQLBuilder (trigger={fallback_trigger})")
                rule_attempt = await self._attempt_rule_sql(schema, detected, explicit_filters, limit, plan=query_plan)
                attempts.append(rule_attempt)
                rules_used = True
                final_attempt = rule_attempt if rule_attempt.sql is not None else llm_attempt
                rule_mark = "\u2713 Success" if rule_attempt.sql else "\u2717 Failed"
                trace.add(f"[Rule-based SQLBuilder] {rule_mark}")

            metrics.repair_ms = 0.0  # repair time is embedded in execution below; see final_attempt.repair_attempts
            metrics.sql_chars = len(final_attempt.sql) if final_attempt.sql else 0


            # ---- Change 6: rich SQL-attempt metadata -------------------
            result.metadata["sql_attempts"] = [
                {
                    "source": a.source,
                    "sql": a.sql,
                    "ok": a.ok,
                    "error": a.error,
                    "failure_reason": a.failure_reason,
                    "repair_attempts": a.repair_attempts,
                    "validation_confidence": a.validation.confidence if a.validation else None,
                }
                for a in attempts
            ]
            result.metadata["initial_sql_source"] = (
                attempts[0].source if attempts else None
            )

            result.metadata["final_sql_source"] = final_attempt.source

            result.metadata["fallback_trigger"] = fallback_trigger

            result.metadata["llm_failed"] = (
                llm_attempt.sql is None
                or llm_attempt.failure_reason is not None
            )

            result.metadata["rules_used"] = rules_used

            result.metadata["sql_repair_attempts"] = sum(
                a.repair_attempts for a in attempts
            )

            result.metadata["execution_retry_count"] = len([
                a for a in attempts if getattr(a, "execution_retried", False)
            ])

            if final_attempt.sql is None:
                structured_err = StructuredError(
                    stage="LLM SQL Generation" if not rules_used else "SQL Generation (LLM + rule-based)",
                    reason=final_attempt.failure_reason or "sql_generation_failed",
                    detail=final_attempt.error,
                    category=final_attempt.failure_reason or "sql_generation_failed",
                    recommendation="Retry generation, rephrase the question, or verify the schema exposes "
                                   "the tables/columns the question refers to.",
                )
                result.success = False
                result.error = final_attempt.error or "SQL generation failed for both LLM and rule-based strategies."
                result.structured_error = structured_err.to_dict()
                result.metadata["intent"] = detected.intent.value
                trace.add(f"[SQL Generation] \u2717 Failed -- {structured_err.reason}")
                result.metadata["debug_trace"] = trace.finalize()
                result.metadata["metrics"] = metrics.to_dict()
                result.latency_ms = round((time.perf_counter() - start) * 1000, 2)
                metrics.total_latency_ms = result.latency_ms
                self._record_latency(result.latency_ms)
                logger.warning("sql_generation_failed: %s", result.error)
                return result

            result.sql = final_attempt.final_sql or final_attempt.sql
            result.tables_used = [t.name for t in final_attempt.tables_used]
            result.join_path = [
                f"{r.from_table}.{r.from_column} -> {r.to_table}.{r.to_column}"
                for r in final_attempt.join_edges
            ]
            result.metadata["intent"] = detected.intent.value
            result.metadata["grouping_requested"] = detected.wants_grouping
            if detected.granularity:
                result.metadata["granularity"] = detected.granularity
            if explicit_filters:
                result.metadata["explicit_filters_applied"] = len(explicit_filters)
            if llm_attempt.llm_debug:
                result.metadata["llm"] = llm_attempt.llm_debug
            if metadata:
                result.metadata["caller_metadata"] = metadata
            if caller_supplied_schema:
                result.metadata["schema_source"] = "caller_supplied"
            if final_attempt.repair_attempts > 0:
                result.metadata["repair_attempts"] = final_attempt.repair_attempts

            known_col_names = {c for t in final_attempt.tables_used for c in t.column_names()}
            result.metadata["columns_used"] = sorted(
                {ident for ident in _QUOTED_IDENTIFIER_RE.findall(result.sql) if ident in known_col_names}
            )

            result.sql_confidence = _clamp01(
                final_attempt.validation.confidence if final_attempt.validation else 1.0
            )
            if final_attempt.validation and not final_attempt.validation.valid:
                result.metadata["validation_issues"] = final_attempt.validation.issues

            result.repair_confidence = _clamp01(round(max(0.2, 1.0 - 0.3 * final_attempt.repair_attempts), 3))

            # ---- SQL mutation tracking (spec #10, #15) -------------------
            # Compares the SQL this attempt started with (before any
            # automatic repair) against what was actually executed, and
            # records the classification alongside both versions. The
            # repair loop itself already refuses to APPLY a repair that
            # would classify as "semantic" (see
            # `_execute_with_repair_blocking`) -- this is the audit
            # trail of whatever mutation *was* ultimately applied.
            mutation_kind = classify_sql_mutation(final_attempt.sql, result.sql)
            result.metadata["sql_mutation"] = {
                "kind": mutation_kind,
                "original_sql": final_attempt.sql,
                "final_sql": result.sql,
                "repair_attempts": final_attempt.repair_attempts,
            }

            if final_attempt.ok:
                result.success = True
                result.rows = final_attempt.rows
                result.columns = final_attempt.columns
                result.row_count = len(final_attempt.rows)
                result.execution_confidence = _clamp01(1.0)
                trace.add(f"[DuckDB] \u2713 Executed -- Rows: {result.row_count}")
            else:
                error_category = classify_execution_error(final_attempt.error)
                structured_err = StructuredError(
                    stage="Execution",
                    reason=final_attempt.error or "Unknown execution error",
                    detail=final_attempt.error,
                    category=error_category,
                    recommendation="Inspect result.sql for the exact statement executed; if this recurs, "
                                   "consider lowering min_sql_validation_confidence or reviewing the schema.",
                )
                result.success = False
                result.error = final_attempt.error or "Execution failed."
                result.structured_error = structured_err.to_dict()
                result.metadata["execution_error_category"] = error_category
                result.columns = []
                result.execution_confidence = _clamp01(0.0)
                trace.add(f"[DuckDB] \u2717 Execution failed -- {final_attempt.error} (category={error_category})")
                logger.error(
                    "execution_failed: query=%r sql=%s error=%s category=%s",
                    query, result.sql, final_attempt.error, error_category,
                )

            # ---- Fix 7/8: semantic-mismatch signals ---------------------
            # Everything below is additive introspection used to (a) set
            # the new `result.status` tri-state and (b) penalize
            # `overall_confidence` for problems that "the query executed
            # without a DB error" alone doesn't capture.
            semantic_issues: List[str] = []
            metric_issues: List[str] = []
            if final_attempt.validation is not None:
                semantic_issues = [
                    issue for issue in final_attempt.validation.issues
                    if issue.startswith("Requested metric") or issue.startswith("Requested aggregation")
                    or issue.startswith("QueryPlan requested")
                ]
                metric_issues = [
                    issue for issue in semantic_issues
                    if issue.startswith("Requested metric") or issue.startswith("Requested aggregation")
                ]

            wrong_table = False
            if query_plan is not None and query_plan.relevant_tables and final_attempt.tables_used:
                used_names = {t.name for t in final_attempt.tables_used if not t.is_virtual}
                planned_names = set(query_plan.relevant_tables)
                if used_names and planned_names:
                    wrong_table = used_names.isdisjoint(planned_names)

            all_metrics_missing = bool(
                query_plan and query_plan.metrics and len(metric_issues) >= len(query_plan.metrics)
            )

            empty_result_unexpected = (
                final_attempt.ok and result.row_count == 0
                and query_plan is not None and not query_plan.aggregate_filters
            )

            # ---- Fix 8: tri-state semantic-quality status ----------------
            if not final_attempt.ok:
                result.status = "failed"
            elif wrong_table or all_metrics_missing:
                # Executed without a DB error, but queried the wrong
                # table or is missing every requested metric -- not
                # trustworthy as a correct answer.
                result.status = "failed"
            elif rules_used or final_attempt.repair_attempts > 0 or semantic_issues or empty_result_unexpected:
                result.status = "partial_success"
            else:
                result.status = "success"
            result.metadata["status"] = result.status
            if semantic_issues:
                result.metadata["semantic_issues"] = semantic_issues

            # ---- Fix 7: overall confidence -------------------------------
            overall = self._compute_overall_confidence(
                planner=result.planner_confidence,
                sql=result.sql_confidence,
                execution=result.execution_confidence,
                repair=result.repair_confidence,
                schema=result.schema_confidence,
                fallback_used=rules_used,
                repair_attempts=final_attempt.repair_attempts,
                semantic_issue_count=len(semantic_issues),
                empty_result_unexpected=empty_result_unexpected,
                wrong_table=wrong_table,
            )
            result.overall_confidence = overall
            result.metadata["overall_confidence"] = overall

            metrics.rows_returned = result.row_count
            result.metadata["metrics"] = metrics.to_dict()
            trace.add(
                f"[Overall Status] {result.status.upper()} "
                f"(overall_confidence={overall:.2f})"
            )
            result.metadata["debug_trace"] = trace.finalize()

            if use_cache and self._result_cache is not None and cache_key and result.success:
                self._result_cache.set(cache_key, result)

        except Exception as e:
            structured_err = StructuredError(
                stage="Pipeline",
                reason=type(e).__name__,
                detail=str(e),
                category="pipeline_error",
                recommendation="Check logs (logger.exception was called) for the full traceback; "
                               "this is an unhandled error outside the normal generation/validation/"
                               "execution failure paths.",
            )
            result.success = False
            result.error = str(e)
            result.structured_error = structured_err.to_dict()
            result.metadata["debug_trace"] = trace.finalize()
            logger.exception("run: unhandled error for query=%r", query)

        result.latency_ms = round((time.perf_counter() - start) * 1000, 2)
        metrics.total_latency_ms = result.latency_ms
        result.metadata.setdefault("metrics", metrics.to_dict())
        self._record_latency(result.latency_ms)
        logger.info(
            "run_complete: success=%s intent=%s sql_source=%s rows=%d latency_ms=%.1f "
            "planner_confidence=%.2f sql_confidence=%.2f execution_confidence=%.2f overall_confidence=%.2f",
            result.success, result.metadata.get("intent"), result.metadata.get("final_sql_source"),
            result.row_count, result.latency_ms,
            result.planner_confidence, result.sql_confidence, result.execution_confidence,
            result.overall_confidence,
        )
        if debug:
            for line in result.metadata.get("debug_trace", []):
                logger.debug(line)
        return result

     
    def _resolve_llm_tables(
    self,
    schema: DiscoveredSchema,
    sql: str
) -> Tuple[List[TableInfo], List[Relationship]]:
        """
        The LLM path doesn't go through SchemaLinker/find_join_path, so
        `tables_used`/`join_path` aren't known a priori. This recovers
        them by checking which known table names actually appear as
        quoted identifiers in the generated SQL (used for validation,
        the repair engine's ambiguous-column qualification, and result
        reporting), and which discovered relationships connect two
        referenced tables (used only for reporting `join_path`, not for
        altering the query itself).
        """
        referenced_names = {
            ident for ident in _QUOTED_IDENTIFIER_RE.findall(sql) if ident in schema.tables
        }
        # Bare (unquoted) table names after FROM/JOIN are also valid
        # DuckDB SQL and some models emit them unquoted.
        for name in schema.tables:
            if re.search(rf'\b(?:FROM|JOIN)\s+{re.escape(name)}\b', sql, re.IGNORECASE):
                referenced_names.add(name)

        tables_used = [schema.tables[n] for n in referenced_names]
        edges = [
            rel for rel in schema.relationships
            if rel.from_table in referenced_names and rel.to_table in referenced_names
        ]
        return tables_used, edges

    def get_schema(self, refresh: bool = False) -> DiscoveredSchema:
        """Expose the discovered schema (incl. relationships), e.g. for
        debugging, UI display, or passing into `run(schema=...)` to
        avoid repeated discovery across several calls."""
        return self._discover_schema_blocking(force_refresh=refresh)

    def get_schema_registry(self, refresh: bool = False) -> Dict[str, Any]:
        """
        Flat, JSON-serializable schema registry: table names, columns,
        SQL types, per-kind column buckets (numeric/text/categorical/
        boolean/temporal/identifier), row counts, and primary-key /
        candidate-key info. Convenience wrapper over
        DiscoveredSchema.as_registry_dict() for callers that don't want
        to deal with the dataclasses directly.
        """
        return self.get_schema(refresh=refresh).as_registry_dict()

    # -----------------------------------------------------
    # Production metrics (spec #14)
    # -----------------------------------------------------

    def _record_latency(self, latency_ms: float) -> None:
        """Appends one request's total latency to a bounded rolling
        window (oldest entries evicted once `_latency_window` is
        exceeded), used to compute a running average in
        `get_metrics_summary()`. O(1) amortized, thread-safe."""
        with self._latency_lock:
            self._request_count += 1
            self._recent_latencies_ms[self._request_count] = latency_ms
            while len(self._recent_latencies_ms) > self._latency_window:
                self._recent_latencies_ms.popitem(last=False)

    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Aggregate, process-lifetime metrics (spec #14): total requests
        served and the average total latency over the most recent
        `_latency_window` requests. Intended for periodic export to a
        monitoring system (logged/scraped by the caller) rather than
        per-request diagnostics -- see `AgentResult.metadata["metrics"]`
        for that.
        """
        with self._latency_lock:
            values = list(self._recent_latencies_ms.values())
            total_requests = self._request_count
        if not values:
            return {"total_requests": total_requests, "avg_latency_ms": 0.0, "window_size": 0}
        return {
            "total_requests": total_requests,
            "avg_latency_ms": round(sum(values) / len(values), 2),
            "min_latency_ms": round(min(values), 2),
            "max_latency_ms": round(max(values), 2),
            "window_size": len(values),
        }

    def seed_dataframe(self, table_name: str, df) -> None:
        with self.pool.lock:
            conn = self.pool.connection()
            conn.register(table_name, df)
        # New/changed data invalidates any cached schema, profiling, and
        # any previously cached AgentResults (which may reference the
        # now-stale schema version or now-outdated data).
        self._schema_cache.invalidate()
        if self._result_cache is not None:
            self._result_cache.invalidate_all()

    def seed_csv(self, table_name: str, path: str, **read_csv_kwargs: Any) -> None:
        """
        Register a CSV file on disk as a queryable table, analogous to
        seed_dataframe() but reading directly through DuckDB's CSV
        reader (avoids loading the file into a pandas DataFrame first).
        Any keyword arguments are forwarded to DuckDB's read_csv_auto
        (e.g. delim=',', header=True).
        """
        with self.pool.lock:
            conn = self.pool.connection()
            options = ", ".join(f"{k}={v!r}" for k, v in read_csv_kwargs.items())
            options_sql = f", {options}" if options else ""
            safe_path = str(path).replace("'", "''")
            conn.execute(
                f'CREATE OR REPLACE TABLE "{table_name}" AS '
                f"SELECT * FROM read_csv_auto('{safe_path}'{options_sql})"
            )
        self._schema_cache.invalidate()
        if self._result_cache is not None:
            self._result_cache.invalidate_all()

    def close(self) -> None:
        with self._llm_generator._conn_lock:
            if self._llm_generator._pooled_conn is not None:
                try:
                    self._llm_generator._pooled_conn.close()
                except Exception:
                    pass
                self._llm_generator._pooled_conn = None
                self._llm_generator._pooled_conn_key = None
        self.pool.close_all()