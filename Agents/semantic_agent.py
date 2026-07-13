"""
semantic_agent.py (UNIVERSAL + DYNAMIC + PRODUCTION-READY)
============================================================
A dataset-agnostic semantic retrieval engine built on:
    - Python 3.11+ / asyncio
    - httpx (async HTTP client)
    - sentence-transformers (embeddings)
    - Qdrant REST API (vector search)

Design goals
------------
- Works with ANY unstructured dataset (PDFs, DOCX, HTML, Markdown, wikis,
  research papers, policies, manuals, emails, legal/medical documents, ...)
  without requiring code changes.
- Automatic schema inference: text / id / ordering fields are detected from
  the payloads returned by Qdrant. Manual ``SchemaConfig`` values, when
  provided, always take precedence over inference.
- Automatic chunk detection: chunked datasets are reconstructed into full
  documents (ordered, de-duplicated); non-chunked datasets are returned as
  individual documents with no merging.
- Runtime search parameters (``top_k``, ``score_threshold``, ``ef``,
  ``with_vectors``) are NOT part of persistent configuration -- they are
  supplied per-call to :meth:`SemanticAgent.run`.
- Generic, composable Qdrant filter construction supporting exact match,
  multi-value ("any"), numeric/float/date ranges, and nested
  must/should/must_not boolean logic.
- Multi-stage retrieval pipeline (vector search -> metadata filtering ->
  multi-query fusion -> diversity reranking -> final ranking), adaptive
  retrieval depth, similarity-metric-aware score normalization, and
  retrieval confidence scoring -- all fully dataset-agnostic.

Public API (kept stable / backward compatible)
-----------------------------------------------
    - ``AgentResult``
    - ``QdrantConfig``
    - ``EmbedConfig``
    - ``SchemaConfig``
    - ``SemanticAgent.run(...)``
    - ``SemanticAgent.close()``

Additive public surface (new, does not affect the above)
----------------------------------------------------------
    - ``RetrievalConfig`` -- optional tuning knobs for adaptive retrieval,
      multi-query fusion, diversity reranking, and result caching.
    - ``LLMConfig`` / ``LLMClient`` -- optional LLM-assisted NLP layer
      (query classification, expansion, entity extraction, and intent
      understanding). Fully opt-in, fully gated, fully fallback-safe: if
      ``LLMConfig.enabled`` is ``False`` (the default) or the LLM call
      fails/times out/returns invalid output, the pipeline behaves exactly
      as it did before -- deterministic regex/rule-based logic only. See
      the "LLM-ASSISTED NLP LAYER" section below for details.
    - ``SemanticAgent.health()`` -- unchanged, kept for completeness.
    - ``SemanticAgent.invalidate_cache()`` -- clears embedding/result caches.
    - New *keyword-only* parameters on ``SemanticAgent.run`` (all default
      to values that preserve the spirit of the original behavior; see
      each parameter's docstring for exact semantics and how to fully
      restore the previous fixed-top-k behavior if needed), including
      ``compressor`` (optional context-compression coordination) and
      ``planner_confidence`` (optional upstream confidence signal used to
      widen adaptive retrieval depth). ``run()`` also silently absorbs
      unrecognized keyword arguments so orchestrators (e.g. ``HybridAgent``)
      can evolve their call signature without this method raising
      ``TypeError``.

Scope notes
-----------
A few items are intentionally NOT implemented here, on the principle that
faking them would be worse than omitting them:
    - "Synonym"/domain concept expansion is not backed by a hardcoded or
      generated dictionary (that would itself be a domain assumption).
      Instead, ``RetrievalConfig.query_expander`` is an extension point:
      supply a callable (e.g. backed by an LLM, WordNet, or a company
      thesaurus) and it will be used automatically.
    - Caching here is a single-process, in-memory TTL/LRU cache. It is NOT
      a distributed cache. For multi-process/multi-node deployments, wrap
      the same interface around Redis or similar at the application layer.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import re
import time
import copy
from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import httpx

try:
    from sentence_transformers import SentenceTransformer as _ST
    _ST_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    _ST_AVAILABLE = False


# ============================================================
# LOGGER
# ============================================================

logger = logging.getLogger(__name__)


# ============================================================
# MODULE-LEVEL DEFAULTS
# ============================================================
# Runtime search defaults. These intentionally live OUTSIDE of QdrantConfig
# because top_k / score_threshold / ef / with_vectors are per-query search
# parameters, not persistent system configuration.

DEFAULT_TOP_K: int = 10
DEFAULT_SCORE_THRESHOLD: float = 0.3
DEFAULT_HNSW_EF: int = 128
DEFAULT_WITH_VECTORS: bool = False

_MAX_EF: int = 512  # sane upper bound so adaptive scaling can't blow up latency

# Fallback schema values, used ONLY when both explicit SchemaConfig
# overrides AND automatic inference fail to identify a field.
FALLBACK_TEXT_FIELD: str = "chunk_text"
FALLBACK_ALT_TEXT_FIELD: str = "clean_chunk_text"
FALLBACK_ID_FIELD: str = "chunk_id"
FALLBACK_ORDER_FIELD: str = "chunk_index"

# Ordered candidate field names used for automatic schema inference.
# Order matters: earlier entries are preferred when multiple candidates
# are present in a payload.
CANDIDATE_TEXT_FIELDS: List[str] = [
    "chunk_text", "clean_chunk_text", "text", "content", "body",
    "page_content", "document", "description", "summary",
    "paragraph", "article", "markdown",
]

CANDIDATE_ID_FIELDS: List[str] = [
    "document_id", "doc_id", "chunk_id", "id", "uuid",
    "file_id", "source_id",
]

CANDIDATE_ORDER_FIELDS: List[str] = [
    "chunk_index", "order", "position", "page", "page_number",
    "paragraph_number", "sequence", "index",
]

# Fields that, when present, are combined (in this priority order) into a
# single searchable / displayable text block per chunk/document. This lets
# the agent surface richer text (titles, headings, summaries, ...) instead
# of assuming a single monolithic text field.
CANDIDATE_COMBINABLE_TEXT_FIELDS: List[str] = [
    "title", "heading", "summary", "description", "abstract",
    "body", "content", "text", "markdown",
]

# Fields commonly used to describe a chunk's position within its source
# document, for the human-readable "document_location" evidence field.
CANDIDATE_LOCATION_FIELDS: List[str] = ["page", "section", "location", "position"]

# Generic (non-domain) English stopwords, used only for building an
# additional lexical query *variant* for multi-query retrieval -- never for
# filtering, ranking, or any dataset-specific interpretation.
_GENERIC_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "with", "about", "as", "by",
    "and", "or", "but", "if", "so", "do", "does", "did", "please",
    "can", "could", "would", "should", "will", "shall", "me", "you",
    "i", "we", "it", "this", "that", "these", "those",
}

_LEADING_QUESTION_WORD_RE = re.compile(
    r"^(what|who|whom|when|where|which|why|how|is|are|was|were|does|do|did|can|could|should|would)\b\s*",
    re.IGNORECASE,
)


# ============================================================
# QUERY-TYPE CLASSIFICATION (heuristic, dataset-agnostic)
# ============================================================

_QUERY_TYPE_PATTERNS: Dict[str, "re.Pattern[str]"] = {
    "comparison": re.compile(
        r"\b(vs\.?|versus|compare[sd]?|comparison|difference between|better than|which is)\b",
        re.IGNORECASE,
    ),
    "summarization": re.compile(
        r"\b(summar(?:y|ize|ise)|tl;?dr|overview of|briefly explain|key points|main points)\b",
        re.IGNORECASE,
    ),
    "reasoning": re.compile(
        r"\b(if\s.+\sthen|what would happen|implications? of|infer|deduce|conclude|reasoning behind)\b",
        re.IGNORECASE,
    ),
    "analytical": re.compile(
        r"\b(analy[sz]e|analysis|trend|pattern|correlation|root cause|impact of|breakdown of)\b",
        re.IGNORECASE,
    ),
    "explanation": re.compile(
        r"\b(why|how does|how do|explain|what causes|reason for)\b",
        re.IGNORECASE,
    ),
}

# Per-query-type retrieval strategy adjustments. These are intentionally
# gentle multipliers/deltas so behavior degrades gracefully rather than
# swinging wildly for an unusual dataset or query style.
_QUERY_TYPE_STRATEGY: Dict[str, Dict[str, float]] = {
    "search":         {"top_k_multiplier": 0.7, "threshold_delta": 0.05,  "diversity_lambda": 0.3},
    "informational":  {"top_k_multiplier": 1.0, "threshold_delta": 0.0,   "diversity_lambda": 0.4},
    "comparison":     {"top_k_multiplier": 1.6, "threshold_delta": -0.05, "diversity_lambda": 0.6},
    "summarization":  {"top_k_multiplier": 2.0, "threshold_delta": -0.10, "diversity_lambda": 0.5},
    "explanation":    {"top_k_multiplier": 1.2, "threshold_delta": 0.0,   "diversity_lambda": 0.3},
    "reasoning":      {"top_k_multiplier": 1.5, "threshold_delta": -0.05, "diversity_lambda": 0.4},
    "analytical":     {"top_k_multiplier": 1.8, "threshold_delta": -0.05, "diversity_lambda": 0.5},
    # -- Additional generic retrieval intents, reachable only via the
    # optional LLM classifier (see QueryIntent / LLMClient.classify).
    # The regex classifier never emits these directly; they extend the
    # taxonomy for LLM-based classification without changing the
    # regex-only default behavior of classify_query_type().
    "retrieval":      {"top_k_multiplier": 1.0, "threshold_delta": 0.0,   "diversity_lambda": 0.4},
    "exploration":    {"top_k_multiplier": 1.7, "threshold_delta": -0.05, "diversity_lambda": 0.65},
    "filtering":      {"top_k_multiplier": 0.8, "threshold_delta": 0.05,  "diversity_lambda": 0.25},
    "aggregation":    {"top_k_multiplier": 1.9, "threshold_delta": -0.08, "diversity_lambda": 0.45},
    "unknown":        {"top_k_multiplier": 1.0, "threshold_delta": 0.0,   "diversity_lambda": 0.4},
}

# Query types for which the regex classifier already has reasonable
# lexical signal ("complex" intents where semantic reasoning tends to
# improve retrieval quality). Used as one of the LLM-gating signals --
# see ``_should_invoke_llm()``. Purely a gating heuristic, not a
# dataset-specific assumption.
_LLM_WORTHY_QUERY_TYPES = frozenset({
    "comparison", "summarization", "reasoning", "analytical", "explanation",
})

# Cheap lexical signal for "this query likely needs semantic reasoning",
# used only to decide *whether* to call the LLM -- never to decide the
# actual classification/expansion/entity output itself.
_LLM_TRIGGER_RE = re.compile(
    r"\b(compare|comparison|versus|vs\.?|explain why|explain how|summar(?:y|ize|ise)|"
    r"relationship between|relationships? among|analy[sz]e|analysis|reason about|"
    r"implications? of|root cause|trend|pattern|correlation|infer|deduce|"
    r"pros and cons|advantages and disadvantages|trade-?offs?)\b",
    re.IGNORECASE,
)


def classify_query_type(query: str) -> str:
    """Heuristically classify a query's retrieval intent.

    Purely lexical/pattern-based -- no domain dictionaries, no ML model,
    works identically regardless of dataset. Used to adjust retrieval
    depth, score threshold, and diversity weighting per query.

    Returns one of: "comparison", "summarization", "reasoning",
    "analytical", "explanation", "informational", "search".
    """
    q = (query or "").strip()
    if not q:
        return "search"

    for qtype in ("comparison", "summarization", "reasoning", "analytical", "explanation"):
        if _QUERY_TYPE_PATTERNS[qtype].search(q):
            return qtype

    if q.endswith("?") or _LEADING_QUESTION_WORD_RE.match(q):
        return "informational"

    if len(q.split()) <= 4:
        return "search"

    return "informational"


def _should_invoke_llm(query: str, base_query_type: str, rcfg: "RetrievalConfig") -> bool:
    """Cheap, inexpensive gate deciding whether a query is worth an LLM call.

    This is the core of requirement "LLM GATING": simple queries
    ("What is battery?", "Show reviews", "List products") must never reach
    the LLM. Only genuinely complex/semantic queries should. All checks
    here are O(1)/O(n) string operations -- no embeddings, no network
    calls -- so gating itself never becomes the bottleneck it's meant to
    avoid.

    A query is considered LLM-worthy if ANY of the following hold:
        - the cheap regex classifier already placed it in a "complex"
          category (comparison/summarization/reasoning/analytical/explanation),
        - it exceeds a configurable token-count threshold (long queries
          tend to bundle multiple sub-intents that benefit from semantic
          understanding), or
        - it contains an explicit reasoning/analysis trigger phrase.

    Short, simple, lookup-style queries fall through all three checks and
    are handled entirely by regex/rule-based logic, keeping the common
    case fast and LLM-free.
    """
    if not rcfg.enable_llm_augmentation:
        return False

    q = (query or "").strip()
    if not q:
        return False

    if base_query_type in _LLM_WORTHY_QUERY_TYPES:
        return True

    token_count = len(q.split())
    if token_count >= rcfg.llm_min_tokens_for_complexity:
        return True

    if _LLM_TRIGGER_RE.search(q):
        return True

    return False


def _compute_adaptive_top_k(
    requested_top_k: int,
    query: str,
    query_type: str,
    min_k: int,
    max_k: int,
    planner_confidence: Optional[float] = None,
) -> int:
    """Scale ``requested_top_k`` by query-type/complexity signals, bounded
    to ``[min_k, max_k]``. The caller's requested value remains the anchor
    -- this widens or narrows around it rather than ignoring it.

    ``planner_confidence``, if supplied, is an optional upstream signal
    (e.g. from a query planner/router) in ``[0, 1]``. Below 0.65 it widens
    the effective top-k slightly to compensate for planner uncertainty by
    casting a wider retrieval net; it never narrows top-k, since high
    upstream confidence isn't a valid reason to retrieve less evidence.
    """
    strategy = _QUERY_TYPE_STRATEGY.get(query_type, _QUERY_TYPE_STRATEGY["informational"])
    scaled = max(1, round(requested_top_k * strategy["top_k_multiplier"]))

    token_count = len(query.split())
    if token_count > 25:
        scaled = round(scaled * 1.2)

    if planner_confidence is not None and 0.0 <= planner_confidence < 0.65:
        scaled += 4

    return max(min_k, min(max_k, scaled))


# ============================================================
# SIMILARITY-METRIC-AWARE SCORE NORMALIZATION
# ============================================================

def normalize_score(raw_score: float, metric: str) -> float:
    """Normalize a Qdrant hit score to a bounded ``[0, 1]`` similarity.

    Qdrant's convention is that a higher ``score`` is always better,
    regardless of the collection's configured distance metric (it negates
    true distances internally). What differs per metric is the *range* and
    *shape* of that score, so each metric gets its own normalization:

        - cosine: raw score is already in [-1, 1] -> linear rescale.
        - dot / inner product: unbounded in both directions -> logistic
          squash centered at 0.
        - euclidean / manhattan: raw score is ``-distance`` (<= 0, closer
          to 0 is better) -> inverse-distance transform.

    Unknown metric names fall back to a clamped raw score rather than
    raising, so a misconfigured/unfamiliar metric degrades gracefully
    instead of breaking retrieval.
    """
    try:
        raw = float(raw_score)
    except (TypeError, ValueError):
        return 0.0

    m = (metric or "cosine").strip().lower()

    if m in ("cosine", "cos"):
        return max(0.0, min(1.0, (raw + 1.0) / 2.0))

    if m in ("dot", "dotproduct", "dot_product", "inner", "ip", "inner_product"):
        try:
            return 1.0 / (1.0 + math.exp(-raw))
        except OverflowError:
            return 1.0 if raw > 0 else 0.0

    if m in ("euclidean", "l2", "euclid", "manhattan", "l1"):
        distance = max(-raw, 0.0)
        return 1.0 / (1.0 + distance)

    return max(0.0, min(1.0, raw))


# ============================================================
# RESULT OBJECT
# ============================================================

@dataclass
class AgentResult:
    """Structured result returned by :meth:`SemanticAgent.run`.

    Attributes:
        success: Whether the query executed without raising an exception.
        query_type: Retrieval strategy used (currently always "semantic").
        chunks: Reconstructed documents / chunks, sorted by descending score.
        entities: Entities supplied for query expansion.
        filters: Filters supplied for this query (raw, pre-Qdrant-format).
        error: Human-readable error message, populated only on failure.
        latency_ms: Wall-clock latency of the full run() call, in ms.
        metadata: Diagnostic/monitoring metadata (raw hit counts, inferred
            schema, search parameters actually used, retrieval confidence,
            per-stage latency breakdown, etc.).
    """

    success: bool = False
    query_type: str = "semantic"
    chunks: List[Dict[str, Any]] = field(default_factory=list)
    entities: Dict[str, Any] = field(default_factory=dict)
    filters: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize this result to a plain ``dict``."""
        return asdict(self)


# ============================================================
# CONFIGURATION (persistent only -- no runtime search parameters)
# ============================================================

@dataclass
class QdrantConfig:
    """Persistent connection configuration for the Qdrant REST API.

    Note:
        Runtime search parameters (``top_k``, ``score_threshold``, ``ef``,
        ``with_vectors``) are intentionally NOT part of this class. Pass
        them as optional keyword arguments to :meth:`SemanticAgent.run`.
    """

    host: str = "http://localhost:6333"
    collection: str = "documents"
    timeout: float = 15.0
    max_retries: int = 3
    retry_backoff_base: float = 0.5
    api_key: Optional[str] = None
    max_connections: int = 100
    max_keepalive_connections: int = 20
    distance_metric: str = "cosine"
    """The distance metric configured on the Qdrant collection: one of
    "cosine", "dot", "euclidean", or "manhattan". Used only for score
    normalization (see :func:`normalize_score`) -- never sent to Qdrant,
    since the collection itself already enforces its configured metric."""


@dataclass
class EmbedConfig:
    """Persistent configuration for the embedding model."""

    model_name: str = "BAAI/bge-small-en-v1.5"
    device: Optional[str] = None
    """e.g. 'cpu', 'cuda', 'cuda:0', 'mps'. ``None`` lets
    sentence-transformers pick automatically."""
    batch_size: int = 32
    normalize_embeddings: bool = True
    enable_cache: bool = True
    """Cache embeddings by (model_name, text). Safe to leave on: embedding
    a given text with a given model is deterministic, so this can only
    reduce redundant work, never change results."""
    cache_size: int = 4096
    cache_ttl_seconds: float = 3600.0


@dataclass
class RetrievalConfig:
    """Optional tuning knobs for the multi-stage retrieval pipeline.

    Every field has a conservative default; existing callers that don't
    construct a ``RetrievalConfig`` at all get sensible out-of-the-box
    behavior.

    Attributes:
        adaptive_min_top_k / adaptive_max_top_k: Hard bounds for adaptive
            top-k scaling (see ``SemanticAgent.run(adaptive_top_k=...)``).
        max_query_variants: Maximum number of reformulated queries used
            when multi-query retrieval is enabled.
        rrf_k: Reciprocal Rank Fusion constant (higher = flatter weighting
            across ranks).
        dedup_similarity_threshold: Jaccard-shingle threshold above which
            two chunks are considered near-duplicates.
        enable_result_cache: Cache full :class:`AgentResult` objects by
            query+parameters. Off by default -- unlike embedding caching,
            this can serve stale data if the underlying collection
            changes, so it's an explicit opt-in.
        result_cache_size / result_cache_ttl_seconds: Result cache sizing.
        query_expander: Optional callable ``(query: str) -> List[str]``
            returning additional synonyms/related terms to fold into the
            expanded query. This is the extension point for domain-aware
            expansion (LLM, WordNet, a company thesaurus, ...) without
            this module itself embedding any domain knowledge. Runs in
            addition to (not instead of) the built-in LLM/regex expansion
            described below.
        enable_llm_augmentation: Master switch for the optional LLM-assisted
            NLP layer (classification refinement, expansion, entity
            extraction, intent understanding). Even when ``True``, the LLM
            is only actually invoked for queries that pass
            ``_should_invoke_llm`` (see "LLM GATING"), and only if a
            :class:`LLMClient` was actually configured/enabled on the
            :class:`SemanticAgent`. Defaults to ``True`` so LLM
            augmentation activates automatically once an ``LLMConfig`` is
            supplied, without requiring a second flag to be flipped.
        llm_min_tokens_for_complexity: Token-count threshold above which a
            query is considered complex enough to warrant an LLM call,
            even if the regex classifier didn't already flag it.
        llm_stage_timeout_seconds: Per-stage timeout for each individual
            LLM call (classification/expansion/entities/intent). A stage
            that exceeds this falls back to its deterministic counterpart
            rather than blocking retrieval.
        llm_max_concurrent_stages: Upper bound on how many LLM stages run
            concurrently for a single query (they are independent calls,
            fired together via ``asyncio.gather`` rather than serially).
        exhaustive_retrieval: Master switch for the retrieval *paradigm*.
            When ``True`` (the default), ``SemanticAgent.run()`` retrieves
            EVERY document above ``score_threshold`` -- paginating through
            Qdrant in ``batch_size`` pages until a genuine stopping
            condition is hit (see ``QdrantClient.search_paginated``) --
            rather than stopping at a fixed ``top_k``. ``top_k`` still
            seeds adaptive-threshold/``ef`` sizing, but no longer caps how
            many results come back. Set ``False`` (or pass
            ``exhaustive=False`` to a specific ``run()`` call) to restore
            the original single-page, top-K-bounded retrieval behavior
            exactly.
        batch_size: Page size for exhaustive retrieval -- how many hits
            are requested per Qdrant round-trip while paginating. This is
            a latency/memory trade-off, not a result-count limit: bigger
            pages mean fewer round-trips per query but a larger single
            response to parse; smaller pages mean tighter memory/latency
            per step but more round-trips for very large result sets. 200
            balances both well for typical chunk-sized (a few hundred
            words) payloads; tune based on payload size and network
            latency to your Qdrant deployment.
        max_search_depth: Hard safety cap on the number of pages fetched
            per query vector during exhaustive retrieval, regardless of
            how many matches remain above threshold. Prevents a
            misconfigured near-zero ``score_threshold`` against a huge
            collection from turning into an unbounded loop. With a
            sensibly chosen threshold this should never actually be the
            binding stopping condition.
        max_results_default: Default cap on total results returned by
            exhaustive retrieval, per query vector, used whenever a
            ``run()`` call doesn't pass its own ``max_results``. ``None``
            (default) means "no cap besides ``max_search_depth``" --
            genuinely retrieve everything relevant. Set to an integer
            (e.g. ``5000``) to bound cost/latency for very large corpora
            while still going well beyond a traditional top-K.
        diversity_max_candidates: If the fused candidate set exceeds this
            size, MMR diversity reranking (which is effectively
            ``O(top_n * n)``) is skipped in favor of a plain ``O(n log n)``
            score sort, to avoid quadratic blow-up when exhaustive
            retrieval returns thousands of documents. Below this size,
            MMR still runs as before.
        dedup_max_candidates: If the reconstructed document count exceeds
            this size, the near-duplicate (Jaccard-shingle) pass across
            documents -- itself ``O(n^2)`` -- is skipped; exact/point-id
            and normalized-text-signature de-duplication (both ``O(n)``,
            applied earlier and always-on) still remove exact and
            within-document duplicates regardless of this setting.
    """

    adaptive_min_top_k: int = 3
    adaptive_max_top_k: int = 50
    max_query_variants: int = 3
    rrf_k: int = 60
    dedup_similarity_threshold: float = 0.85
    enable_result_cache: bool = False
    result_cache_size: int = 256
    result_cache_ttl_seconds: float = 60.0
    query_expander: Optional[Callable[[str], List[str]]] = None

    enable_llm_augmentation: bool = True
    llm_min_tokens_for_complexity: int = 12
    llm_stage_timeout_seconds: float = 8.0
    llm_max_concurrent_stages: int = 4

    exhaustive_retrieval: bool = True
    batch_size: int = 200
    max_search_depth: int = 100
    max_results_default: Optional[int] = None
    diversity_max_candidates: int = 500
    dedup_max_candidates: int = 500


@dataclass
class SchemaConfig:
    """Dynamic, dataset-agnostic schema configuration.

    All override fields default to ``None``, meaning "let the agent infer
    this automatically from the actual payloads returned by Qdrant". Any
    field explicitly set here always takes precedence over inference.

    Attributes:
        text_field: Force a single field to be used as the document text.
        text_fields: Force an explicit, ordered list of fields to combine
            into the document text (takes precedence over ``text_field``).
        fallback_text_field: Secondary field to use if ``text_field`` is
            empty/missing on a given payload (kept for backward
            compatibility with the original single-field design).
        id_field: Force the field used to identify/group documents.
        order_field: Force the field used to order chunks within a document.
        is_chunked: Force chunked/non-chunked handling. ``None`` lets the
            agent auto-detect based on payload shape.
        candidate_text_fields: Candidate list used for single-field text
            inference. Override to customize inference for unusual schemas.
        candidate_id_fields: Candidate list used for id-field inference.
        candidate_order_fields: Candidate list used for order-field
            inference.
        candidate_combinable_fields: Candidate list used for multi-field
            text combination.
    """

    text_field: Optional[str] = None
    text_fields: Optional[List[str]] = None
    fallback_text_field: Optional[str] = None
    id_field: Optional[str] = None
    order_field: Optional[str] = None
    is_chunked: Optional[bool] = None

    candidate_text_fields: List[str] = field(
        default_factory=lambda: list(CANDIDATE_TEXT_FIELDS)
    )
    candidate_id_fields: List[str] = field(
        default_factory=lambda: list(CANDIDATE_ID_FIELDS)
    )
    candidate_order_fields: List[str] = field(
        default_factory=lambda: list(CANDIDATE_ORDER_FIELDS)
    )
    candidate_combinable_fields: List[str] = field(
        default_factory=lambda: list(CANDIDATE_COMBINABLE_TEXT_FIELDS)
    )


# ============================================================
# GENERIC TTL/LRU CACHE
# ============================================================

class TTLCache:
    """Small, dependency-free TTL + LRU cache.

    Safe for concurrent use from coroutines on the same event loop (guarded
    by an ``asyncio.Lock``). This is NOT a cross-thread or cross-process
    cache -- for that, front this same interface with Redis or similar.

    Eviction is size-bounded (LRU) and time-bounded (TTL); whichever fires
    first wins.
    """

    def __init__(self, max_size: int = 1024, ttl_seconds: float = 300.0):
        self.max_size = max(1, max_size)
        self.ttl_seconds = max(0.0, ttl_seconds)
        self._store: "OrderedDict[str, Tuple[float, Any]]" = OrderedDict()
        self._lock = asyncio.Lock()
        self.hits = 0
        self.misses = 0

    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self.misses += 1
                return None
            expires_at, value = entry
            if expires_at < time.monotonic():
                del self._store[key]
                self.misses += 1
                return None
            self._store.move_to_end(key)
            self.hits += 1
            return value

    async def set(self, key: str, value: Any) -> None:
        async with self._lock:
            self._store[key] = (time.monotonic() + self.ttl_seconds, value)
            self._store.move_to_end(key)
            while len(self._store) > self.max_size:
                self._store.popitem(last=False)

    async def clear(self) -> None:
        async with self._lock:
            self._store.clear()

    def stats(self) -> Dict[str, Any]:
        return {
            "size": len(self._store),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
        }


# ============================================================
# EMBEDDING ENGINE
# ============================================================

class EmbeddingEngine:
    """Lazy, thread-safe wrapper around a sentence-transformers model.

    The underlying model is loaded exactly once (on first use) and reused
    for all subsequent calls, regardless of how many concurrent coroutines
    request embeddings. Optionally caches embeddings by (model, text) to
    avoid redundant encode() calls for repeated queries.
    """

    def __init__(self, cfg: EmbedConfig):
        self.cfg = cfg
        self._model: Optional["_ST"] = None
        self._lock = asyncio.Lock()
        self._cache: Optional[TTLCache] = (
            TTLCache(max_size=cfg.cache_size, ttl_seconds=cfg.cache_ttl_seconds)
            if cfg.enable_cache else None
        )

    async def _load(self) -> None:
        """Load the embedding model exactly once, in a thread-safe manner."""
        if self._model is not None:
            return

        async with self._lock:
            # Re-check after acquiring the lock (another coroutine may have
            # already loaded the model while we were waiting).
            if self._model is not None:
                return

            if not _ST_AVAILABLE:
                raise RuntimeError(
                    "sentence-transformers is not installed. "
                    "Install it with: pip install sentence-transformers"
                )

            logger.info(
                "Loading embedding model '%s' (device=%s)",
                self.cfg.model_name, self.cfg.device or "auto",
            )

            loop = asyncio.get_running_loop()
            try:
                self._model = await loop.run_in_executor(
                    None,
                    lambda: _ST(self.cfg.model_name, device=self.cfg.device),
                )
            except Exception as exc:  # pragma: no cover - depends on env
                raise RuntimeError(
                    f"Failed to load embedding model "
                    f"'{self.cfg.model_name}': {exc}"
                ) from exc

    async def _encode_batch(self, texts: Sequence[str]) -> List[List[float]]:
        loop = asyncio.get_running_loop()

        def _encode() -> List[List[float]]:
            try:
                vectors = self._model.encode(  # type: ignore[union-attr]
                    list(texts),
                    batch_size=self.cfg.batch_size,
                    normalize_embeddings=self.cfg.normalize_embeddings,
                    show_progress_bar=False,
                )
                return vectors.tolist()
            except Exception as exc:  # pragma: no cover - depends on env
                raise RuntimeError(f"Embedding encode() failed: {exc}") from exc

        return await loop.run_in_executor(None, _encode)

    async def embed(self, texts: Sequence[str]) -> List[List[float]]:
        """Embed a list of texts, batching internally per ``EmbedConfig``.

        Args:
            texts: Non-empty sequence of strings to embed.

        Returns:
            A list of embedding vectors, one per input text, in the same
            order as the input.

        Raises:
            ValueError: If ``texts`` is empty.
            RuntimeError: If the model fails to load or encode.
        """
        if not texts:
            raise ValueError("EmbeddingEngine.embed() requires at least one text")

        await self._load()

        # Static/type safety: _load() guarantees self._model is populated
        # (or raises). This assertion makes that invariant explicit for
        # type checkers and catches any future regression early.
        assert self._model is not None

        if self._cache is None:
            return await self._encode_batch(texts)

        results: List[Optional[List[float]]] = [None] * len(texts)
        cache_keys = [f"{self.cfg.model_name}:{t}" for t in texts]
        to_encode: List[str] = []
        to_encode_idx: List[int] = []

        for i, key in enumerate(cache_keys):
            cached = await self._cache.get(key)
            if cached is not None:
                results[i] = cached
            else:
                to_encode.append(texts[i])
                to_encode_idx.append(i)

        if to_encode:
            encoded = await self._encode_batch(to_encode)
            for idx, vec in zip(to_encode_idx, encoded):
                results[idx] = vec
                await self._cache.set(cache_keys[idx], vec)

        return results  # type: ignore[return-value]


# ============================================================
# LLM-ASSISTED NLP LAYER (fully opt-in, fully gated, fully fallback-safe)
# ============================================================
#
# Everything in this section is additive. With no ``LLMConfig`` supplied
# (or ``LLMConfig.enabled=False``, the default), none of this code runs
# and SemanticAgent behaves exactly as before -- pure regex/rule-based
# classification and expansion, no network calls beyond Qdrant.
#
# Design summary (mirrors the module's improvement brief):
#   1. QUERY CLASSIFICATION: hybrid regex-first / LLM-refine, validated
#      into ``QueryIntent``, with automatic regex fallback on any failure.
#   2. QUERY EXPANSION: LLM-assisted synonym/related-concept/entity
#      generation, merged with (never replacing) the existing
#      regex-based expansion and the ``RetrievalConfig.query_expander``
#      hook.
#   3. ENTITY EXTRACTION: new stage, LLM-first with a regex fallback,
#      fully dataset-agnostic (no hardcoded entity vocabularies).
#   4. INTENT UNDERSTANDING: new stage inferring goal / answer type /
#      breadth / depth, used to gently nudge adaptive top-k, threshold,
#      diversity, and multi-query behavior.
#   5. LLM GATING: see ``_should_invoke_llm`` above -- simple queries
#      never reach this layer.
#   6. FALLBACK STRATEGY: every public method on ``LLMClient`` catches
#      its own failures (timeout, bad JSON, schema violation, connection
#      error, rate limit) and raises ``LLMStageError``; callers in
#      ``SemanticAgent.run`` catch that and silently fall back to the
#      deterministic implementation, per stage, independently.
#   7. PERFORMANCE: response caching (keyed by stage+model+prompt hash,
#      reusing the same ``TTLCache`` used for embeddings), stage timeouts,
#      and concurrent (not sequential) stage execution via
#      ``asyncio.gather``.
#   8. PROMPT ENGINEERING: one deterministic, reusable, JSON-only prompt
#      template per stage.
#   9. OUTPUT VALIDATION: every response is schema-validated; invalid
#      output triggers a single retry, then falls back.
#  10. GENERIC DESIGN: no dataset/column/industry/schema assumptions
#      anywhere in this section -- prompts only ever reference the raw
#      query text and (optionally) previously-extracted entities.


class QueryIntent(str, Enum):
    """Dataset-agnostic retrieval-intent taxonomy.

    A strict superset of the regex classifier's categories (see
    ``classify_query_type``); the extra members (``RETRIEVAL``,
    ``EXPLORATION``, ``FILTERING``, ``AGGREGATION``, ``UNKNOWN``) are
    reachable only through LLM classification. Subclassing ``str`` keeps
    values trivially JSON-serializable and comparable to plain strings.
    """

    COMPARISON = "comparison"
    SUMMARIZATION = "summarization"
    EXPLANATION = "explanation"
    REASONING = "reasoning"
    ANALYTICAL = "analytical"
    SEARCH = "search"
    INFORMATIONAL = "informational"
    RETRIEVAL = "retrieval"
    EXPLORATION = "exploration"
    FILTERING = "filtering"
    AGGREGATION = "aggregation"
    UNKNOWN = "unknown"

    @classmethod
    def coerce(cls, value: Any) -> "QueryIntent":
        """Best-effort coercion of arbitrary (e.g. LLM-produced) text into
        a valid member. Never raises -- unrecognized input maps to
        ``UNKNOWN`` so a slightly-off LLM label degrades gracefully
        instead of blowing up the pipeline."""
        if isinstance(value, QueryIntent):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError:
            return cls.UNKNOWN


class LLMStageError(RuntimeError):
    """Raised internally by any :class:`LLMClient` stage on failure
    (timeout, connection error, invalid/unvalidatable JSON, rate limit,
    etc). Always caught by the caller in ``SemanticAgent.run`` -- this
    exception should never propagate out of ``run()``."""


@dataclass
class LLMConfig:
    """Persistent configuration for the optional LLM-assisted NLP layer.

    Talks to any Ollama-compatible ``/api/generate`` endpoint by default
    (matching the project's existing local-LLM setup), but any
    OpenAI-compatible chat/completions server can be used by adjusting
    ``base_url``/``chat_path``/``request_style``.

    Attributes:
        enabled: Master on/off switch. ``False`` by default -- the agent
            is fully functional, regex-only, with no LLM configured at
            all. Set ``True`` (and configure ``base_url``/``model``) to
            activate the hybrid NLP layer.
        base_url: Base URL of the LLM server, e.g. ``"http://localhost:11434"``
            for a local Ollama instance.
        model: Model name as known to the server, e.g. ``"qwen3:14b"``.
        request_style: ``"ollama"`` (default) posts to
            ``{base_url}/api/generate`` with an Ollama-shaped body.
            ``"openai"`` posts to ``{base_url}/v1/chat/completions`` with
            an OpenAI-chat-shaped body. Both request JSON-only output.
        timeout: Per-request HTTP timeout, in seconds.
        max_retries: Retries for transient connection failures (not for
            validation failures -- those get exactly one retry via
            ``LLMClient._call_validated``, independent of this).
        retry_backoff_base: Exponential backoff base, in seconds.
        temperature: Sampling temperature. Kept low by default since every
            prompt in this layer asks for deterministic, structured JSON.
        max_tokens: Upper bound on generated tokens per call. Classification/
            entity/intent responses are small JSON objects, so this can
            stay modest to keep latency down.
        cache_size / cache_ttl_seconds: Response cache sizing (keyed by
            stage + model + prompt hash). A cache hit costs zero network
            round-trips.
    """

    enabled: bool = False
    base_url: str = "http://localhost:11434"
    model: str = "qwen3:14b"
    request_style: str = "ollama"  # "ollama" | "openai"
    timeout: float = 8.0
    max_retries: int = 2
    retry_backoff_base: float = 0.4
    temperature: float = 0.0
    max_tokens: int = 512
    cache_size: int = 2048
    cache_ttl_seconds: float = 1800.0


@dataclass
class LLMClassification:
    """Validated output of the LLM classification stage."""

    intent: QueryIntent
    confidence: float
    source: str  # "llm" | "regex"


@dataclass
class LLMExpansion:
    """Validated output of the LLM query-expansion stage."""

    synonyms: List[str] = field(default_factory=list)
    related_concepts: List[str] = field(default_factory=list)
    alternate_wordings: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    abbreviations: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    source: str = "regex"  # "llm" | "regex"

    def all_terms(self) -> List[str]:
        """Flatten every generated term into one de-duplicated list,
        preserving first-seen order."""
        merged = (
            self.synonyms + self.related_concepts + self.alternate_wordings
            + self.entities + self.abbreviations + self.keywords
        )
        return list(dict.fromkeys(t for t in merged if t and t.strip()))


@dataclass
class LLMEntities:
    """Validated output of the LLM entity-extraction stage. Every field is
    dataset-agnostic -- these are generic entity *categories*, not
    dataset-specific fields."""

    products: List[str] = field(default_factory=list)
    companies: List[str] = field(default_factory=list)
    people: List[str] = field(default_factory=list)
    dates: List[str] = field(default_factory=list)
    locations: List[str] = field(default_factory=list)
    metrics: List[str] = field(default_factory=list)
    technical_terms: List[str] = field(default_factory=list)
    concepts: List[str] = field(default_factory=list)
    acronyms: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    source: str = "regex"  # "llm" | "regex"

    def as_query_entities(self) -> Dict[str, Any]:
        """Flatten into the ``{key: value}`` shape ``build_query()``
        expects, skipping empty categories."""
        out: Dict[str, Any] = {}
        for cat in (
            "products", "companies", "people", "dates", "locations",
            "metrics", "technical_terms", "concepts", "acronyms", "categories",
        ):
            values = getattr(self, cat)
            if values:
                out[cat] = ", ".join(values)
        return out


@dataclass
class LLMIntentUnderstanding:
    """Validated output of the LLM intent-understanding stage. Purely
    advisory -- consumers apply gentle, bounded adjustments based on this,
    never a full override of caller-supplied parameters."""

    user_goal: str = ""
    expected_answer_type: str = ""
    retrieval_style: str = ""
    search_breadth: str = "normal"    # "narrow" | "normal" | "wide"
    search_depth: str = "normal"      # "shallow" | "normal" | "deep"
    required_context: str = "normal"  # "minimal" | "normal" | "extensive"
    expected_granularity: str = "normal"  # "fine" | "normal" | "coarse"
    source: str = "none"  # "llm" | "none"


# ---- Prompt templates (one per stage, deterministic, JSON-only) --------

_CLASSIFICATION_PROMPT_TEMPLATE = """You are a query-intent classifier for a retrieval system. \
Classify the user query into exactly one retrieval intent.

Allowed intents (choose exactly one): comparison, summarization, explanation, \
reasoning, analytical, search, informational, retrieval, exploration, \
filtering, aggregation, unknown.

Rules:
- Base your decision only on the query's linguistic structure and intent, never on any \
  specific dataset, domain, or subject matter.
- If uncertain, use "unknown".
- Respond with ONLY a single-line JSON object, no prose, no markdown fences, no explanation.
- JSON schema: {{"intent": "<one of the allowed intents>", "confidence": <float 0..1>}}

Query: {query}
JSON:"""

_EXPANSION_PROMPT_TEMPLATE = """You are a query-expansion assistant for a retrieval system. \
Given a user query, generate terms that would help retrieve relevant documents, \
regardless of dataset or domain.

Generate, where applicable:
- semantic synonyms
- related concepts
- alternate wordings/phrasings
- named entities mentioned in the query
- abbreviations or their expansions
- domain-independent keywords

Rules:
- Do not invent facts; only reformulate/expand the query itself.
- Keep every list short (at most 6 items each) and each item short (1-4 words).
- Respond with ONLY a single-line JSON object, no prose, no markdown fences.
- JSON schema: {{"synonyms": [...], "related_concepts": [...], "alternate_wordings": [...], \
"entities": [...], "abbreviations": [...], "keywords": [...]}}

Query: {query}
JSON:"""

_ENTITY_PROMPT_TEMPLATE = """You are an entity-extraction assistant. Extract entities from the \
user query into generic categories. This must work for any dataset or domain -- do not assume \
a specific subject matter.

Categories: products, companies, people, dates, locations, metrics, technical_terms, \
concepts, acronyms, categories.

Rules:
- Only extract entities actually present or clearly implied in the query text.
- Leave a category as an empty list if nothing applies.
- Respond with ONLY a single-line JSON object, no prose, no markdown fences.
- JSON schema: {{"products": [...], "companies": [...], "people": [...], "dates": [...], \
"locations": [...], "metrics": [...], "technical_terms": [...], "concepts": [...], \
"acronyms": [...], "categories": [...]}}

Query: {query}
JSON:"""

_INTENT_PROMPT_TEMPLATE = """You are a retrieval-strategy assistant. Analyze what the user \
actually wants from the following query so a retrieval system can tune itself.

Rules:
- Base your analysis only on the query text; do not assume any specific dataset or domain.
- search_breadth must be one of: narrow, normal, wide.
- search_depth must be one of: shallow, normal, deep.
- required_context must be one of: minimal, normal, extensive.
- expected_granularity must be one of: fine, normal, coarse.
- Respond with ONLY a single-line JSON object, no prose, no markdown fences.
- JSON schema: {{"user_goal": "<short phrase>", "expected_answer_type": "<short phrase>", \
"retrieval_style": "<short phrase>", "search_breadth": "narrow|normal|wide", \
"search_depth": "shallow|normal|deep", "required_context": "minimal|normal|extensive", \
"expected_granularity": "fine|normal|coarse"}}

Query: {query}
Preliminary query type (heuristic, may be refined): {query_type}
JSON:"""


# ---- Output validation (never trust raw LLM output) ---------------------

def _extract_json_object(text: str) -> Dict[str, Any]:
    """Best-effort extraction of a JSON object from raw LLM text.

    Handles the common failure modes of small/local models: markdown code
    fences, leading/trailing prose, or extra whitespace. Raises
    ``ValueError`` (never anything else) if no parseable JSON object can
    be found, so callers can uniformly treat this as a validation failure.
    """
    if not text:
        raise ValueError("empty LLM response")

    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned.strip(), flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned.strip()).strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"no JSON object found in LLM response: {text[:200]!r}")

    candidate = cleaned[start:end + 1]
    parsed = json.loads(candidate)
    if not isinstance(parsed, dict):
        raise ValueError("parsed JSON is not an object")
    return parsed


def _as_str_list(value: Any, max_items: int = 12) -> List[str]:
    """Coerce an arbitrary JSON value into a bounded list of clean
    strings. Never raises -- unusable input becomes an empty list."""
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return []
    out: List[str] = []
    for item in value:
        if isinstance(item, str):
            s = item.strip()
        else:
            s = str(item).strip()
        if s and s not in out:
            out.append(s)
        if len(out) >= max_items:
            break
    return out


def _validate_classification(parsed: Dict[str, Any]) -> LLMClassification:
    if "intent" not in parsed:
        raise ValueError("classification response missing 'intent'")
    intent = QueryIntent.coerce(parsed.get("intent"))
    try:
        confidence = float(parsed.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))
    return LLMClassification(intent=intent, confidence=confidence, source="llm")


def _validate_expansion(parsed: Dict[str, Any]) -> LLMExpansion:
    return LLMExpansion(
        synonyms=_as_str_list(parsed.get("synonyms"), 6),
        related_concepts=_as_str_list(parsed.get("related_concepts"), 6),
        alternate_wordings=_as_str_list(parsed.get("alternate_wordings"), 6),
        entities=_as_str_list(parsed.get("entities"), 6),
        abbreviations=_as_str_list(parsed.get("abbreviations"), 6),
        keywords=_as_str_list(parsed.get("keywords"), 6),
        source="llm",
    )


def _validate_entities(parsed: Dict[str, Any]) -> LLMEntities:
    return LLMEntities(
        products=_as_str_list(parsed.get("products")),
        companies=_as_str_list(parsed.get("companies")),
        people=_as_str_list(parsed.get("people")),
        dates=_as_str_list(parsed.get("dates")),
        locations=_as_str_list(parsed.get("locations")),
        metrics=_as_str_list(parsed.get("metrics")),
        technical_terms=_as_str_list(parsed.get("technical_terms")),
        concepts=_as_str_list(parsed.get("concepts")),
        acronyms=_as_str_list(parsed.get("acronyms")),
        categories=_as_str_list(parsed.get("categories")),
        source="llm",
    )


_ALLOWED_BREADTH = {"narrow", "normal", "wide"}
_ALLOWED_DEPTH = {"shallow", "normal", "deep"}
_ALLOWED_CONTEXT = {"minimal", "normal", "extensive"}
_ALLOWED_GRANULARITY = {"fine", "normal", "coarse"}


def _validate_intent_understanding(parsed: Dict[str, Any]) -> LLMIntentUnderstanding:
    def _pick(key: str, allowed: set, default: str) -> str:
        val = str(parsed.get(key, default) or default).strip().lower()
        return val if val in allowed else default

    return LLMIntentUnderstanding(
        user_goal=str(parsed.get("user_goal", "") or "").strip()[:200],
        expected_answer_type=str(parsed.get("expected_answer_type", "") or "").strip()[:100],
        retrieval_style=str(parsed.get("retrieval_style", "") or "").strip()[:100],
        search_breadth=_pick("search_breadth", _ALLOWED_BREADTH, "normal"),
        search_depth=_pick("search_depth", _ALLOWED_DEPTH, "normal"),
        required_context=_pick("required_context", _ALLOWED_CONTEXT, "normal"),
        expected_granularity=_pick("expected_granularity", _ALLOWED_GRANULARITY, "normal"),
        source="llm",
    )


# ---- Deterministic regex/rule-based fallbacks (used when the LLM is
# disabled, gated out, or fails at any stage) -----------------------------

_ACRONYM_RE = re.compile(r"\b[A-Z]{2,}\b")
_CAPITALIZED_PHRASE_RE = re.compile(r"\b[A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]{2,})*\b")
_DATE_LIKE_RE = re.compile(
    r"\b(\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|"
    r"(?:19|20)\d{2}|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2}?,?\s*\d{0,4})\b",
    re.IGNORECASE,
)
_METRIC_LIKE_RE = re.compile(
    r"\b(\d+(?:\.\d+)?\s?%|\$\s?\d+(?:[.,]\d+)?[kmb]?|\d+(?:\.\d+)?\s?(?:kg|lb|km|mi|gb|tb|mb|ms|sec|hrs?|hours?))\b",
    re.IGNORECASE,
)


def regex_expand_query(query: str) -> LLMExpansion:
    """Deterministic, dataset-agnostic query expansion used when the LLM
    is disabled, gated out for a simple query, or fails. Purely syntactic:
    no vocabulary, no domain dictionary -- only generic reformulations
    already used elsewhere in this module (declarative form, stopword-
    stripped keyword form) plus acronym/number-aware keyword pulls.
    """
    q = (query or "").strip()
    alternate = []
    declarative = _strip_leading_question_word(q)
    if declarative and declarative.lower() != q.lower():
        alternate.append(declarative)

    keywords = [t for t in re.findall(r"\w+", q.lower()) if t not in _GENERIC_STOPWORDS]
    acronyms = _as_str_list(_ACRONYM_RE.findall(q))

    return LLMExpansion(
        alternate_wordings=alternate,
        keywords=keywords[:8],
        abbreviations=acronyms,
        source="regex",
    )


def regex_extract_entities(query: str) -> LLMEntities:
    """Deterministic, dataset-agnostic entity extraction fallback. Uses
    only generic lexical patterns (capitalized phrases, ALLCAPS acronyms,
    date-like and metric-like tokens) -- no domain vocabulary."""
    q = query or ""
    capitalized = [
        m for m in _CAPITALIZED_PHRASE_RE.findall(q)
        if m.lower() not in _GENERIC_STOPWORDS
    ]
    return LLMEntities(
        concepts=_as_str_list(capitalized),
        dates=_as_str_list(_DATE_LIKE_RE.findall(q)),
        metrics=_as_str_list(_METRIC_LIKE_RE.findall(q)),
        acronyms=_as_str_list(_ACRONYM_RE.findall(q)),
        source="regex",
    )


# ---- LLM client -----------------------------------------------------------

class LLMClient:
    """Thin async client for the optional LLM-assisted NLP layer.

    Every public method (``classify``, ``expand``, ``extract_entities``,
    ``understand_intent``) is independently cached, timeout-bounded,
    retried once on validation failure, and raises only
    :class:`LLMStageError` on any unrecoverable failure -- callers always
    get either a validated dataclass or a single well-typed exception to
    catch, never a raw network/parsing error.
    """

    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.client = httpx.AsyncClient(timeout=cfg.timeout)
        self._cache: Optional[TTLCache] = (
            TTLCache(max_size=cfg.cache_size, ttl_seconds=cfg.cache_ttl_seconds)
            if cfg.enabled else None
        )

    async def close(self) -> None:
        await self.client.aclose()

    def _cache_key(self, stage: str, prompt: str) -> str:
        digest = hashlib.sha256(f"{self.cfg.model}:{prompt}".encode("utf-8")).hexdigest()
        return f"{stage}:{digest}"

    async def _raw_complete(self, prompt: str) -> str:
        """Single completion request against the configured server.
        Raises ``LLMStageError`` on any transport/HTTP failure."""
        try:
            if self.cfg.request_style == "openai":
                url = f"{self.cfg.base_url}/v1/chat/completions"
                body = {
                    "model": self.cfg.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": self.cfg.temperature,
                    "max_tokens": self.cfg.max_tokens,
                }
                response = await self.client.post(url, json=body)
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]

            # Default: Ollama-style /api/generate
            url = f"{self.cfg.base_url}/api/generate"
            body = {
                "model": self.cfg.model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": self.cfg.temperature,
                    "num_predict": self.cfg.max_tokens,
                },
            }
            response = await self.client.post(url, json=body)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "")

        except LLMStageError:
            raise
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise LLMStageError(f"LLM connection/timeout failure: {exc}") from exc
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status == 429:
                raise LLMStageError(f"LLM rate limited: {exc}") from exc
            raise LLMStageError(f"LLM request failed (status={status}): {exc}") from exc
        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            raise LLMStageError(f"LLM returned unparsable response: {exc}") from exc

    async def _complete_with_retries(self, prompt: str) -> str:
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                return await self._raw_complete(prompt)
            except LLMStageError as exc:
                last_exc = exc
                if attempt < self.cfg.max_retries:
                    await asyncio.sleep(self.cfg.retry_backoff_base * (2 ** (attempt - 1)))
        raise last_exc or LLMStageError("LLM completion failed with no exception recorded")

    async def _call_validated(
        self, stage: str, prompt: str, validator: Callable[[Dict[str, Any]], Any],
    ) -> Any:
        """Full stage pipeline: cache check -> timeout-bounded completion
        -> JSON extraction -> schema validation -> one retry on validation
        failure -> cache store. Raises ``LLMStageError`` if every attempt
        fails, so the caller can fall back deterministically."""
        if not self.cfg.enabled:
            raise LLMStageError("LLM layer is disabled (LLMConfig.enabled=False)")

        cache_key = self._cache_key(stage, prompt)
        if self._cache is not None:
            cached = await self._cache.get(cache_key)
            if cached is not None:
                return cached

        last_exc: Optional[Exception] = None
        for validation_attempt in range(2):  # one initial try + one retry
            try:
                raw = await asyncio.wait_for(
                    self._complete_with_retries(prompt),
                    timeout=self.cfg.timeout * self.cfg.max_retries + 1.0,
                )
                parsed = _extract_json_object(raw)
                result = validator(parsed)
                if self._cache is not None:
                    await self._cache.set(cache_key, result)
                return result
            except asyncio.TimeoutError as exc:
                last_exc = LLMStageError(f"LLM stage '{stage}' timed out")
            except LLMStageError as exc:
                last_exc = exc
            except (ValueError, json.JSONDecodeError) as exc:
                last_exc = LLMStageError(f"LLM stage '{stage}' produced invalid JSON: {exc}")
            except Exception as exc:  # pragma: no cover - defensive catch-all
                last_exc = LLMStageError(f"LLM stage '{stage}' failed unexpectedly: {exc}")

        logger.warning("LLM stage '%s' failed after retry; caller will fall back: %s", stage, last_exc)
        raise last_exc or LLMStageError(f"LLM stage '{stage}' failed")

    async def classify(self, query: str) -> LLMClassification:
        prompt = _CLASSIFICATION_PROMPT_TEMPLATE.format(query=query)
        return await self._call_validated("classify", prompt, _validate_classification)

    async def expand(self, query: str) -> LLMExpansion:
        prompt = _EXPANSION_PROMPT_TEMPLATE.format(query=query)
        return await self._call_validated("expand", prompt, _validate_expansion)

    async def extract_entities(self, query: str) -> LLMEntities:
        prompt = _ENTITY_PROMPT_TEMPLATE.format(query=query)
        return await self._call_validated("extract_entities", prompt, _validate_entities)

    async def understand_intent(self, query: str, query_type: str) -> LLMIntentUnderstanding:
        prompt = _INTENT_PROMPT_TEMPLATE.format(query=query, query_type=query_type)
        return await self._call_validated("understand_intent", prompt, _validate_intent_understanding)


# Gentle, bounded adjustment factors applied when LLM intent understanding
# is available. Deliberately small so a single LLM call can nudge but
# never dominate the deterministic strategy already computed from
# query-type heuristics.
_BREADTH_TOP_K_ADJUST = {"narrow": 0.85, "normal": 1.0, "wide": 1.25}
_DEPTH_VARIANT_ADJUST = {"shallow": 0, "normal": 0, "deep": 1}
_CONTEXT_THRESHOLD_ADJUST = {"minimal": 0.03, "normal": 0.0, "extensive": -0.03}


def _apply_intent_understanding(
    effective_top_k: int,
    effective_threshold: float,
    max_variants: int,
    understanding: LLMIntentUnderstanding,
    min_k: int,
    max_k: int,
) -> Tuple[int, float, int]:
    """Apply bounded, advisory adjustments from LLM intent understanding
    on top of the already-computed deterministic strategy. Never used as
    the sole source of truth -- purely a gentle nudge."""
    top_k = round(effective_top_k * _BREADTH_TOP_K_ADJUST.get(understanding.search_breadth, 1.0))
    top_k = max(min_k, min(max_k, top_k))

    threshold = effective_threshold + _CONTEXT_THRESHOLD_ADJUST.get(understanding.required_context, 0.0)
    threshold = max(0.0, min(1.0, threshold)) if 0.0 <= effective_threshold <= 1.0 else effective_threshold

    variants = max_variants + _DEPTH_VARIANT_ADJUST.get(understanding.search_depth, 0)
    variants = max(1, min(6, variants))

    return top_k, threshold, variants


# ============================================================
# QUERY BUILDER
# ============================================================

def build_query(
    query: str,
    entities: Optional[Dict[str, Any]] = None,
    boosters: Optional[List[str]] = None,
    synonyms: Optional[List[str]] = None,
    context_keywords: Optional[List[str]] = None,
) -> str:
    """Dynamically enrich a raw query with entities, boosters, synonyms and
    contextual keywords. Completely dataset-independent -- no domain logic.

    Args:
        query: The raw user query.
        entities: Arbitrary key/value pairs to inject (e.g. extracted named
            entities), rendered as ``"key:value"``.
        boosters: Free-form terms to boost relevance for.
        synonyms: Alternate phrasings/synonyms to widen recall.
        context_keywords: Additional contextual keywords (e.g. surrounding
            conversation topics).

    Returns:
        A single expanded query string suitable for embedding.
    """
    entities = entities or {}
    boosters = boosters or []
    synonyms = synonyms or []
    context_keywords = context_keywords or []

    parts: List[str] = []

    for key, value in entities.items():
        if value is None:
            continue
        parts.append(f"{key}:{value}")

    parts.extend(b for b in boosters if b)
    parts.extend(c for c in context_keywords if c)

    if synonyms:
        cleaned = [s for s in synonyms if s]
        if cleaned:
            parts.append("(also: " + ", ".join(cleaned) + ")")

    parts.append(query.strip())

    return " | ".join(p for p in parts if p)


def _strip_leading_question_word(query: str) -> str:
    """Turn e.g. "how does X work" into "X work" -- a more declarative
    phrasing that sometimes retrieves better as an embedding variant.
    Purely syntactic; no domain knowledge."""
    stripped = _LEADING_QUESTION_WORD_RE.sub("", query.strip()).strip(" ?")
    return stripped


def _strip_generic_stopwords(query: str) -> str:
    """Keyword-only variant of a query, using a short list of generic
    (non-domain) English stopwords. Used only to widen multi-query recall,
    never for filtering or ranking decisions."""
    tokens = [t for t in re.findall(r"\w+", query.lower()) if t not in _GENERIC_STOPWORDS]
    return " ".join(tokens)


def _generate_query_variants(
    query: str, expanded_query: str, max_variants: int,
    extra: Optional[List[str]] = None,
) -> List[str]:
    """Build a small set of distinct query reformulations for multi-query
    retrieval. Always includes the fully expanded query first."""
    candidates = [expanded_query]

    declarative = _strip_leading_question_word(query)
    if declarative and declarative.lower() != query.strip().lower():
        candidates.append(declarative)

    keyword_only = _strip_generic_stopwords(query)
    if keyword_only and keyword_only.lower() != query.strip().lower():
        candidates.append(keyword_only)

    if extra:
        candidates.extend(e for e in extra if e)

    seen: set = set()
    out: List[str] = []
    for c in candidates:
        key = c.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(c.strip())

    return out[:max(1, max_variants)]


# ============================================================
# GENERIC FILTER BUILDER
# ============================================================

_RANGE_KEYS = {"gt", "gte", "lt", "lte"}


def _build_condition(key: str, value: Any) -> Optional[Dict[str, Any]]:
    """Convert a single ``field: value`` pair into a Qdrant condition.

    Supports:
        - ``None`` -> skipped (returns ``None``)
        - scalar -> exact match
        - list/tuple/set -> match-any (multiple values)
        - ``{"any": [...]}`` -> explicit match-any
        - ``{"except": [...]}`` -> match-except
        - ``{"gte": ..., "lte": ..., "gt": ..., "lt": ...}`` -> range
          (works uniformly for numeric, floating-point, and ISO-8601 date
          values -- Qdrant's range filter accepts any of these).
    """
    if value is None:
        return None

    if isinstance(value, dict):
        keys = set(value.keys())

        if keys & _RANGE_KEYS:
            range_body = {k: v for k, v in value.items() if k in _RANGE_KEYS}
            return {"key": key, "range": range_body}

        if "any" in value:
            return {"key": key, "match": {"any": list(value["any"])}}

        if "except" in value:
            return {"key": key, "match": {"except": list(value["except"])}}

        raise ValueError(
            f"Unsupported filter specification for field '{key}': {value!r}. "
            "Expected range keys (gt/gte/lt/lte), 'any', or 'except'."
        )

    if isinstance(value, (list, tuple, set)):
        values = [v for v in value if v is not None]
        if not values:
            return None
        return {"key": key, "match": {"any": list(values)}}

    return {"key": key, "match": {"value": value}}


def _build_clause_list(clauses: Any) -> List[Dict[str, Any]]:
    """Build a flat list of Qdrant conditions from a list of clause specs.

    Each item may be:
        - a single-key ``{field: value}`` dict (most common case), or
        - a nested boolean filter dict (``must``/``should``/``must_not``),
          allowing arbitrarily nested filter logic, or
        - an already-built Qdrant condition dict (passed through as-is).
    """
    conditions: List[Dict[str, Any]] = []

    for clause in clauses:
        if not isinstance(clause, dict):
            continue

        # Already-built raw Qdrant condition (has "key" + match/range).
        if "key" in clause and ("match" in clause or "range" in clause):
            conditions.append(clause)
            continue

        # Nested boolean filter.
        if any(k in clause for k in ("must", "should", "must_not")):
            nested = build_filter(clause)
            if nested:
                conditions.append(nested)
            continue

        # Plain {field: value, field2: value2, ...} shorthand.
        for field_name, field_value in clause.items():
            condition = _build_condition(field_name, field_value)
            if condition:
                conditions.append(condition)

    return conditions


def build_filter(filters: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Convert an arbitrary, dataset-agnostic filter spec into a Qdrant
    filter payload.

    Two input shapes are supported and may be mixed:

    1. **Flat shorthand** -- every key is treated as a field to match::

        {"category": "policy", "year": {"gte": 2020}, "tags": ["hr", "legal"]}

    2. **Structured boolean form** -- explicit ``must`` / ``should`` /
       ``must_not`` lists, each containing flat shorthand dicts and/or
       nested structured filters::

        {
            "must": [{"status": "active"}],
            "should": [{"tags": ["hr", "legal"]}],
            "must_not": [{"archived": True}],
        }

    Args:
        filters: The filter specification, or ``None``/empty for "no filter".

    Returns:
        A Qdrant-compatible filter dict, or ``None`` if no usable
        conditions were produced.
    """
    if not filters:
        return None

    is_structured = any(k in filters for k in ("must", "should", "must_not"))

    result: Dict[str, List[Dict[str, Any]]] = {}

    if is_structured:
        for boolean_key in ("must", "should", "must_not"):
            raw = filters.get(boolean_key)
            if not raw:
                continue
            conditions = _build_clause_list(raw if isinstance(raw, list) else [raw])
            if conditions:
                result[boolean_key] = conditions
    else:
        conditions = _build_clause_list([filters])
        if conditions:
            result["must"] = conditions

    return result or None


# ============================================================
# SCHEMA INFERENCE
# ============================================================

class SchemaInferer:
    """Infers text / id / order fields directly from Qdrant payload shapes.

    Explicit :class:`SchemaConfig` overrides always take precedence.
    Automatic inference is used whenever an override is not supplied, and
    the module-level ``FALLBACK_*`` constants are used only if inference
    itself finds nothing usable.
    """

    @staticmethod
    def infer_id_field(
        payload_keys: set, candidates: Sequence[str], override: Optional[str]
    ) -> str:
        if override:
            return override
        for candidate in candidates:
            if candidate in payload_keys:
                return candidate
        return FALLBACK_ID_FIELD

    @staticmethod
    def infer_order_field(
        payload_keys: set, candidates: Sequence[str], override: Optional[str]
    ) -> Optional[str]:
        if override:
            return override
        for candidate in candidates:
            if candidate in payload_keys:
                return candidate
        return None

    @staticmethod
    def infer_text_fields(
        payload: Dict[str, Any],
        candidates: Sequence[str],
        override_single: Optional[str],
        override_multi: Optional[List[str]],
        fallback_single: Optional[str],
    ) -> List[str]:
        if override_multi:
            present = [f for f in override_multi if payload.get(f)]
            return present or list(override_multi)

        if override_single:
            fields = [override_single]
            if fallback_single:
                fields.append(fallback_single)
            return fields

        found = [
            c for c in candidates
            if isinstance(payload.get(c), str) and payload.get(c).strip()
        ]
        if found:
            return found

        # Nothing could be inferred from the payload at all. Prefer the
        # module-level primary fallback, then the caller-supplied
        # ``fallback_text_field`` (from SchemaConfig) if one was given,
        # and only fall back to the module-level secondary fallback when
        # no configurable fallback was supplied. This keeps the fallback
        # behavior configurable via SchemaConfig instead of silently
        # ignoring it.
        if fallback_single:
            return [FALLBACK_TEXT_FIELD, fallback_single]
        return [FALLBACK_TEXT_FIELD, FALLBACK_ALT_TEXT_FIELD]

    @staticmethod
    def detect_chunked(
        hits: List[Dict[str, Any]], id_field: str, order_field: Optional[str]
    ) -> bool:
        """A dataset is considered "chunked" if either:
            - an ordering field is present in the payloads, or
            - multiple hits share the same document id.
        Otherwise each hit is treated as a standalone document.
        """
        if order_field is not None:
            if any(order_field in h.get("payload", {}) for h in hits):
                return True

        seen_ids: Dict[Any, int] = {}
        for hit in hits:
            payload = hit.get("payload", {})
            doc_id = payload.get(id_field, hit.get("id"))
            seen_ids[doc_id] = seen_ids.get(doc_id, 0) + 1
            if seen_ids[doc_id] > 1:
                return True

        return False


def _extract_text(payload: Dict[str, Any], text_fields: List[str]) -> str:
    """Combine one or more payload fields into a single text block."""
    parts: List[str] = []
    seen: set = set()

    for field_name in text_fields:
        value = payload.get(field_name)
        if isinstance(value, str) and value.strip() and value not in seen:
            parts.append(value.strip())
            seen.add(value)

    return "\n\n".join(parts)


# ============================================================
# NEAR-DUPLICATE DETECTION
# ============================================================

_PUNCT_WS_RE = re.compile(r"[\s\W_]+", re.UNICODE)


def _normalize_for_dedup(text: str) -> str:
    """Lowercase + collapse all whitespace/punctuation, so "Hello, World!"
    and "hello   world" compare equal."""
    return _PUNCT_WS_RE.sub(" ", (text or "").strip().lower()).strip()


def _shingles(text: str, n: int = 5) -> set:
    tokens = text.split()
    if not tokens:
        return set()
    if len(tokens) < n:
        return {" ".join(tokens)}
    return {" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    union = len(a | b)
    return (len(a & b) / union) if union else 0.0


def is_near_duplicate(text_a: str, text_b: str, threshold: float = 0.85) -> bool:
    """True if two text blocks are exact, whitespace/punctuation-equivalent,
    or near-duplicate (shingled Jaccard similarity above ``threshold``)."""
    norm_a, norm_b = _normalize_for_dedup(text_a), _normalize_for_dedup(text_b)
    if not norm_a or not norm_b:
        return False
    if norm_a == norm_b:
        return True
    return _jaccard(_shingles(norm_a), _shingles(norm_b)) >= threshold


def _remove_near_duplicate_chunks(
    chunks: List[Dict[str, Any]], threshold: float
) -> List[Dict[str, Any]]:
    """Greedy near-duplicate removal across already-ranked chunks. O(n^2)
    on the (small, top-k-sized) final result set."""
    kept: List[Dict[str, Any]] = []
    for c in chunks:
        text = c.get("text", "")
        if any(is_near_duplicate(text, k.get("text", ""), threshold) for k in kept):
            continue
        kept.append(c)
    return kept


# ============================================================
# RANK FUSION (multi-query retrieval)
# ============================================================

def _reciprocal_rank_fusion_with_counts(
    result_lists: List[List[Dict[str, Any]]], k: int = 60
) -> Tuple[List[Dict[str, Any]], Dict[Any, int]]:
    """Fuse multiple ranked hit lists (one per query variant) via
    Reciprocal Rank Fusion, keyed by Qdrant point id.

    Returns the fused, score-sorted hit list plus a per-id count of how
    many distinct variant searches surfaced that point (used later for
    query-coverage / confidence calculations).
    """
    scores: Dict[Any, float] = {}
    counts: Dict[Any, int] = {}
    best_hit: Dict[Any, Dict[str, Any]] = {}

    for hits in result_lists:
        seen_in_this_list: set = set()
        for rank, hit in enumerate(hits, start=1):
            key = hit.get("id")
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            if key not in seen_in_this_list:
                counts[key] = counts.get(key, 0) + 1
                seen_in_this_list.add(key)
            if key not in best_hit or hit.get("score", 0) > best_hit[key].get("score", 0):
                best_hit[key] = hit

    fused: List[Dict[str, Any]] = []
    for key, rrf_score in scores.items():
        hit = dict(best_hit[key])
        hit["_rrf_score"] = rrf_score
        fused.append(hit)

    fused.sort(key=lambda h: h["_rrf_score"], reverse=True)
    return fused, counts


# ============================================================
# DIVERSITY (MMR) RERANKING
# ============================================================

def _cosine_sim(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a)) or 1e-12
    norm_b = math.sqrt(sum(y * y for y in b)) or 1e-12
    return dot / (norm_a * norm_b)


def mmr_rerank(
    candidates: List[Dict[str, Any]], top_n: int, lambda_mult: float = 0.5
) -> List[Dict[str, Any]]:
    """Maximal Marginal Relevance reranking.

    Each candidate must carry a ``"_relevance"`` score (typically the
    normalized similarity) and, ideally, a ``"vector"`` key. Candidates
    without a vector cannot have their diversity contribution computed and
    are appended, in relevance order, after the vector-bearing candidates
    have been selected -- this degrades gracefully rather than crashing
    when vectors weren't fetched.

    ``lambda_mult`` in [0, 1]: 1.0 = pure relevance (no diversity
    pressure), 0.0 = pure diversity (ignores relevance after the first
    pick).
    """
    if top_n <= 0 or not candidates:
        return []

    have_vec = [c for c in candidates if c.get("vector")]
    no_vec = [c for c in candidates if not c.get("vector")]

    selected: List[Dict[str, Any]] = []
    remaining = list(have_vec)

    while remaining and len(selected) < top_n:
        if not selected:
            best = max(remaining, key=lambda c: c.get("_relevance", 0.0))
        else:
            def _mmr_score(c: Dict[str, Any]) -> float:
                relevance = c.get("_relevance", 0.0)
                diversity_penalty = max(
                    _cosine_sim(c["vector"], s["vector"]) for s in selected
                )
                return lambda_mult * relevance - (1.0 - lambda_mult) * diversity_penalty

            best = max(remaining, key=_mmr_score)

        selected.append(best)
        remaining.remove(best)

    no_vec_sorted = sorted(no_vec, key=lambda c: c.get("_relevance", 0.0), reverse=True)
    return (selected + no_vec_sorted)[:top_n]


# ============================================================
# RETRIEVAL CONFIDENCE
# ============================================================

def compute_confidence(chunks: List[Dict[str, Any]], query_coverage: float = 1.0) -> float:
    """Composite retrieval-confidence score in [0, 1], blending:
        - the top result's normalized similarity (primary signal)
        - the average normalized similarity across returned results
        - score spread (tight clustering near the top score is a positive
          signal; a huge drop-off after the first result is not)
        - document agreement (evidence spread across multiple documents)
        - fraction of returned chunks corroborated by metadata
        - query coverage (for multi-query retrieval: how consistently
          results were found across query reformulations; 1.0 for
          single-query retrieval)
    """
    if not chunks:
        return 0.0

    scores = [c.get("normalized_score", c.get("score", 0.0)) for c in chunks]
    top = scores[0]
    avg = sum(scores) / len(scores)
    spread = 1.0 - (max(scores) - min(scores)) if len(scores) > 1 else 1.0
    doc_agreement = len({c.get("document_id") for c in chunks}) / max(len(chunks), 1)
    meta_completeness = sum(1 for c in chunks if c.get("metadata")) / max(len(chunks), 1)

    confidence = (
        0.40 * top
        + 0.20 * avg
        + 0.15 * max(0.0, min(1.0, spread))
        + 0.15 * meta_completeness
        + 0.10 * max(0.0, min(1.0, query_coverage))
    )
    # doc_agreement is informative but deliberately not weighted into the
    # headline number for single-document-corpus use cases (where low
    # agreement is expected and not actually a quality signal); it is
    # still surfaced separately in AgentResult.metadata.
    return round(max(0.0, min(1.0, confidence)), 4)


def _query_coverage(
    chunks: List[Dict[str, Any]], variant_counts: Dict[Any, int], num_variants: int
) -> float:
    if num_variants <= 1 or not chunks:
        return 1.0
    total = sum(variant_counts.get(c.get("chunk_id"), 1) for c in chunks)
    return round(min(1.0, (total / len(chunks)) / num_variants), 4)


# ============================================================
# QDRANT CLIENT
# ============================================================

class QdrantSearchError(RuntimeError):
    """Raised when a Qdrant search request fails after all retries."""


class QdrantClient:
    """Thin, resilient async wrapper around the Qdrant REST search API.

    Provides connection reuse, timeout handling, exponential-backoff retry
    on transient failures, batch search with graceful fallback, and
    structured logging.

    API compatibility:
        Newer Qdrant versions expose search via ``POST
        /collections/{collection}/points/query`` with the vector supplied
        under the ``"query"`` key. Older versions expose the legacy
        ``POST /collections/{collection}/points/search`` endpoint with the
        vector supplied under the ``"vector"`` key. This client prefers the
        newer ``points/query`` endpoint and transparently falls back to the
        legacy ``points/search`` endpoint if the server responds with
        ``404`` (i.e. the newer endpoint doesn't exist on that server).
        Once a working endpoint is discovered, it is cached on the
        instance so subsequent calls skip the detection step. The same
        detect-and-cache pattern applies to the batch endpoints used by
        :meth:`search_batch`.
    """

    def __init__(self, cfg: QdrantConfig):
        self.cfg = cfg
        headers = {"api-key": cfg.api_key} if cfg.api_key else {}
        limits = httpx.Limits(
            max_connections=cfg.max_connections,
            max_keepalive_connections=cfg.max_keepalive_connections,
        )
        self.client = httpx.AsyncClient(
            timeout=cfg.timeout, headers=headers, limits=limits
        )
        # None = not yet detected (try new API first, fall back to legacy).
        # True = newer "points/query" API confirmed working.
        # False = legacy "points/search" API confirmed working.
        self._use_query_api: Optional[bool] = None
        # Same tri-state pattern for the batch search endpoint.
        self._use_batch_api: Optional[bool] = None

    def _build_request(
        self,
        use_query_api: bool,
        vector: List[float],
        qfilter: Optional[Dict[str, Any]],
        top_k: int,
        score_threshold: float,
        ef: int,
        with_vector: bool,
        offset: int = 0,
    ) -> Tuple[str, Dict[str, Any]]:
        """Build the (url, body) pair for either the new or legacy API.

        ``offset`` (default 0) skips the first N matches for this vector,
        enabling page-by-page pagination for exhaustive retrieval (see
        :meth:`search_paginated`) without changing single-page callers,
        which simply never set it.
        """
        body: Dict[str, Any] = {
            "limit": top_k,
            "score_threshold": score_threshold,
            "with_payload": True,
            "with_vector": with_vector,
            "params": {"hnsw_ef": ef},
        }
        if offset:
            body["offset"] = offset
        if qfilter:
            body["filter"] = qfilter

        if use_query_api:
            body["query"] = vector
            endpoint = "points/query"
        else:
            body["vector"] = vector
            endpoint = "points/search"

        url = f"{self.cfg.host}/collections/{self.cfg.collection}/{endpoint}"
        return url, body

    @staticmethod
    def _parse_hits(data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Normalize the response body across both API shapes.

        The legacy ``points/search`` endpoint returns ``{"result": [...]}``.
        The newer ``points/query`` endpoint returns
        ``{"result": {"points": [...]}}``.
        """
        result = data.get("result")
        if isinstance(result, dict):
            return result.get("points", [])
        return result or []

    async def search(
        self,
        vector: List[float],
        qfilter: Optional[Dict[str, Any]],
        top_k: int,
        score_threshold: float,
        ef: int,
        with_vector: bool = False,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Execute a dense vector search against the configured collection.

        Args:
            vector: The query embedding.
            qfilter: Optional pre-built Qdrant filter (see :func:`build_filter`).
            top_k: Maximum number of results to return in this single page.
            score_threshold: Minimum similarity score to include a result.
            ef: HNSW ``ef`` search parameter (higher = more accurate/slower).
            with_vector: Whether to include point vectors in the response.
            offset: Number of matches to skip before this page starts.
                Used internally by :meth:`search_paginated` for exhaustive,
                page-by-page retrieval; single-page callers simply omit it.

        Returns:
            The raw list of hits from Qdrant's ``result`` field.

        Raises:
            QdrantSearchError: If the request fails after exhausting retries.
        """
        if not vector:
            raise QdrantSearchError("Cannot search with an empty query vector")

        # Determine which API mode(s) to attempt this call.
        if self._use_query_api is True:
            modes = [True]
        elif self._use_query_api is False:
            modes = [False]
        else:
            modes = [True, False]  # auto-detect: try new API, fall back to legacy

        last_exc: Optional[Exception] = None

        for attempt in range(1, self.cfg.max_retries + 1):
            for mode in modes:
                url, body = self._build_request(
                    mode, vector, qfilter, top_k, score_threshold, ef, with_vector, offset
                )
                try:
                    response = await self.client.post(url, json=body)

                    # Newer endpoint not present on this Qdrant version --
                    # fall through to the legacy endpoint without consuming
                    # a retry attempt.
                    if response.status_code == 404 and mode is True and len(modes) > 1:
                        continue

                    if response.status_code == 404:
                        raise QdrantSearchError(
                            f"Qdrant collection '{self.cfg.collection}' or search "
                            f"endpoint not found (404)."
                        )

                    if response.status_code >= 500:
                        raise httpx.HTTPStatusError(
                            f"Qdrant server error: {response.status_code}",
                            request=response.request,
                            response=response,
                        )

                    response.raise_for_status()
                    data = response.json()
                    self._use_query_api = mode
                    return self._parse_hits(data)

                except QdrantSearchError:
                    raise

                except httpx.HTTPStatusError as exc:
                    # 4xx errors are not transient -- fail fast, no retry.
                    if exc.response is not None and exc.response.status_code < 500:
                        logger.error(
                            "Qdrant rejected the search request (status=%s): %s",
                            exc.response.status_code, exc.response.text,
                        )
                        raise QdrantSearchError(
                            f"Qdrant search request rejected "
                            f"(status={exc.response.status_code}): "
                            f"{exc.response.text}"
                        ) from exc
                    last_exc = exc

                except (httpx.TimeoutException, httpx.TransportError) as exc:
                    last_exc = exc

                except (ValueError, KeyError) as exc:
                    # Corrupted/unexpected JSON payload from the server.
                    last_exc = exc

            if attempt < self.cfg.max_retries:
                backoff = self.cfg.retry_backoff_base * (2 ** (attempt - 1))
                logger.warning(
                    "Qdrant search attempt %d/%d failed (%s); retrying in %.2fs",
                    attempt, self.cfg.max_retries, last_exc, backoff,
                )
                await asyncio.sleep(backoff)

        logger.error(
            "Qdrant search failed after %d attempts: %s",
            self.cfg.max_retries, last_exc,
        )
        raise QdrantSearchError(
            f"Qdrant search failed after {self.cfg.max_retries} attempts: {last_exc}"
        ) from last_exc

    async def search_batch(
        self,
        vectors: List[List[float]],
        qfilter: Optional[Dict[str, Any]],
        top_k: int,
        score_threshold: float,
        ef: int,
        with_vector: bool = False,
    ) -> List[List[Dict[str, Any]]]:
        """Execute multiple vector searches in as few round-trips as
        possible.

        Prefers Qdrant's native batch-search endpoint (one HTTP call for
        all query variants); if that endpoint isn't available on this
        server (404) or its response can't be parsed as expected, falls
        back to concurrent individual :meth:`search` calls via
        ``asyncio.gather`` -- still one connection pool, still concurrent,
        just more round-trips.
        """
        if not vectors:
            return []
        if len(vectors) == 1:
            return [await self.search(vectors[0], qfilter, top_k, score_threshold, ef, with_vector)]

        if self._use_batch_api is False:
            return await self._search_batch_fallback(vectors, qfilter, top_k, score_threshold, ef, with_vector)

        use_query_api = self._use_query_api if self._use_query_api is not None else True
        endpoint = "points/query/batch" if use_query_api else "points/search/batch"
        url = f"{self.cfg.host}/collections/{self.cfg.collection}/{endpoint}"

        searches = [
            self._build_request(use_query_api, v, qfilter, top_k, score_threshold, ef, with_vector)[1]
            for v in vectors
        ]

        try:
            response = await self.client.post(url, json={"searches": searches})

            if response.status_code == 404:
                self._use_batch_api = False
                logger.info("Qdrant batch search endpoint not available; falling back to per-query search")
                return await self._search_batch_fallback(vectors, qfilter, top_k, score_threshold, ef, with_vector)

            response.raise_for_status()
            data = response.json()
            raw_result = data.get("result", [])

            parsed: List[List[Dict[str, Any]]] = []
            for item in raw_result:
                if isinstance(item, dict):
                    parsed.append(item.get("points", []))
                elif isinstance(item, list):
                    parsed.append(item)
                else:
                    parsed.append([])

            if len(parsed) != len(vectors):
                raise ValueError(
                    f"Qdrant batch response length ({len(parsed)}) did not match "
                    f"request length ({len(vectors)})"
                )

            self._use_batch_api = True
            return parsed

        except QdrantSearchError:
            raise
        except Exception as exc:
            logger.warning(
                "Qdrant batch search failed (%s); falling back to concurrent per-query search", exc
            )
            return await self._search_batch_fallback(vectors, qfilter, top_k, score_threshold, ef, with_vector)

    async def _search_batch_fallback(
        self,
        vectors: List[List[float]],
        qfilter: Optional[Dict[str, Any]],
        top_k: int,
        score_threshold: float,
        ef: int,
        with_vector: bool,
    ) -> List[List[Dict[str, Any]]]:
        tasks = [self.search(v, qfilter, top_k, score_threshold, ef, with_vector) for v in vectors]
        return list(await asyncio.gather(*tasks))

    # -- Exhaustive (complete) semantic retrieval ---------------------
    #
    # Unlike `search()`/`search_batch()` (a single page, up to `top_k`
    # hits), `search_paginated()` keeps paging through a single query
    # vector's results -- via Qdrant's `offset` parameter -- until one of
    # several well-defined stopping conditions is met (see docstring). It
    # is an async *generator*: pages are yielded one at a time rather than
    # accumulated internally, so a caller that only needs to stream/count
    # results never holds the full result set in memory at once.

    async def search_paginated(
        self,
        vector: List[float],
        qfilter: Optional[Dict[str, Any]],
        score_threshold: float,
        ef: int,
        with_vector: bool = False,
        batch_size: int = 200,
        max_results: Optional[int] = None,
        max_search_depth: int = 100,
    ):
        """Page through ALL matches for a single query vector above
        ``score_threshold``, yielding one batch (page) at a time.

        This is the core of "complete semantic retrieval" (as opposed to
        top-K retrieval): instead of a single bounded call, it repeatedly
        calls :meth:`search` with an increasing ``offset``, stopping only
        when there is genuinely nothing more to retrieve -- not when an
        arbitrary K has been reached.

        Stopping conditions (whichever fires first):
            - Qdrant returns an empty page (no more matches at all).
            - Qdrant returns fewer hits than requested for a page (there
              is a `score_threshold` server-side, so a partial page means
              every remaining match already fell below it).
            - ``max_results`` (if not ``None``) has been reached -- an
              explicit safety/cost cap, not the primary stopping signal.
            - ``max_search_depth`` pages have been fetched -- a hard
              safety net against runaway loops (e.g. a misconfigured
              threshold of 0.0 against a huge collection).

        Args:
            vector: The query embedding.
            qfilter: Optional pre-built Qdrant filter.
            score_threshold: Minimum similarity score -- this is now the
                *primary* retrieval boundary, not ``batch_size``.
            ef: HNSW ``ef`` search parameter.
            with_vector: Whether to include point vectors in each hit.
            batch_size: Page size per Qdrant round-trip. Chosen as a
                latency/memory trade-off -- large enough to keep the
                number of round-trips low, small enough that a single
                page is cheap to embed-adjacent-process and never risks a
                single oversized response. 200 is a reasonable default
                for typical chunk-sized payloads; tune per deployment.
            max_results: Optional hard cap on total hits returned across
                all pages for this vector. ``None`` (default) means
                "retrieve everything above threshold" with no cap besides
                ``max_search_depth``.
            max_search_depth: Maximum number of pages to fetch for this
                vector, regardless of how many matches remain. Purely a
                safety net -- with a well-chosen ``score_threshold`` this
                should never be the actual stopping condition in practice.

        Yields:
            Successive pages (lists of raw Qdrant hits), each already
            deduplicated against previously-seen point ids for this
            vector (guards against the rare case of unstable ordering
            returning the same point across two adjacent pages).
        """
        if not vector:
            raise QdrantSearchError("Cannot search with an empty query vector")

        offset = 0
        depth = 0
        total_yielded = 0
        seen_ids: set = set()

        while True:
            depth += 1
            if depth > max(1, max_search_depth):
                logger.debug(
                    "search_paginated: max_search_depth=%d reached at offset=%d; stopping",
                    max_search_depth, offset,
                )
                break

            page_limit = batch_size
            if max_results is not None:
                remaining = max_results - total_yielded
                if remaining <= 0:
                    break
                page_limit = min(batch_size, remaining)

            page = await self.search(
                vector, qfilter, page_limit, score_threshold, ef, with_vector, offset=offset,
            )
            if not page:
                break

            # Defensive de-dup: guards against the rare case where
            # approximate-index ordering shifts between calls and the
            # same point resurfaces on two adjacent pages.
            fresh = [h for h in page if h.get("id") not in seen_ids]
            for h in fresh:
                seen_ids.add(h.get("id"))

            if fresh:
                yield fresh
                total_yielded += len(fresh)

            # A short page (fewer hits than requested) means Qdrant has
            # nothing more above `score_threshold` left to give us --
            # this, not a fixed K, is the real stopping signal.
            if len(page) < page_limit:
                break

            offset += len(page)

    async def search_exhaustive_batch(
        self,
        vectors: List[List[float]],
        qfilter: Optional[Dict[str, Any]],
        score_threshold: float,
        ef: int,
        with_vector: bool = False,
        batch_size: int = 200,
        max_results: Optional[int] = None,
        max_search_depth: int = 100,
    ) -> List[List[Dict[str, Any]]]:
        """Exhaustively retrieve every match (above ``score_threshold``)
        for each of several query vectors (e.g. multi-query variants),
        concurrently.

        Each vector's pagination runs independently (pagination is
        inherently sequential per-vector, since each page's offset
        depends on the previous one), but all vectors are paginated
        *concurrently* with each other via ``asyncio.gather`` -- so N
        query variants cost roughly the same wall-clock time as the
        slowest single variant's full pagination, not N times that.

        Returns:
            One fully-materialized, per-vector hit list (already
            deduplicated within that vector's pagination), in the same
            order as ``vectors`` -- ready for the existing Reciprocal
            Rank Fusion stage, which itself deduplicates *across* vectors
            by point id.
        """
        if not vectors:
            return []

        async def _collect_one(vector: List[float]) -> List[Dict[str, Any]]:
            collected: List[Dict[str, Any]] = []
            async for page in self.search_paginated(
                vector, qfilter, score_threshold, ef, with_vector,
                batch_size, max_results, max_search_depth,
            ):
                collected.extend(page)
            return collected

        return list(await asyncio.gather(*(_collect_one(v) for v in vectors)))

    async def health(self) -> bool:
        """Lightweight health check against the configured collection.

        Pings the collection endpoint to verify Qdrant is reachable and
        the configured collection exists. Never raises -- any failure
        (connection error, timeout, non-2xx status, etc.) is reported as
        ``False``.

        Returns:
            ``True`` if Qdrant responded successfully, ``False`` otherwise.
        """
        try:
            url = f"{self.cfg.host}/collections/{self.cfg.collection}"
            response = await self.client.get(url)
            return response.status_code < 400
        except Exception:  # pragma: no cover - defensive, never raises
            return False

    async def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        await self.client.aclose()


# ============================================================
# UNIVERSAL RECONSTRUCTION
# ============================================================

def reconstruct(
    hits: List[Dict[str, Any]],
    schema: SchemaConfig,
    embedding_model: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Reconstruct documents from raw Qdrant hits, dataset-agnostically.

    Automatically:
        - infers text / id / order fields (unless overridden in ``schema``),
        - detects whether the dataset is chunked,
        - if chunked: groups hits by document id, orders chunks, removes
          duplicate/near-duplicate chunks, and merges their text,
        - if not chunked: returns each hit as a standalone document,
        - preserves all payload fields as ``metadata``,
        - attaches evidence-quality fields (normalized score, rerank
          score, retrieval confidence, chunk id, document location,
          embedding model) to every returned document.

    Args:
        hits: Raw hits as returned by :meth:`QdrantClient.search` (ideally
            already carrying ``normalized_score`` / ``_relevance``, set by
            the caller after score normalization and reranking).
        schema: Schema configuration (overrides + candidate lists).
        embedding_model: Name of the embedding model used for this query,
            surfaced on each returned document for traceability.

    Returns:
        A list of document dicts with keys ``document_id``, ``text``,
        ``score``, ``normalized_score``, ``rerank_score``,
        ``retrieval_confidence``, ``chunk_count``, ``chunk_id``,
        ``document_location``, ``embedding_model``, and ``metadata``,
        sorted by descending score.
    """
    if not hits:
        return []

    payload_keys: set = set()
    for hit in hits[:20]:
        payload_keys.update(hit.get("payload", {}).keys())

    id_field = SchemaInferer.infer_id_field(
        payload_keys, schema.candidate_id_fields, schema.id_field
    )
    order_field = SchemaInferer.infer_order_field(
        payload_keys, schema.candidate_order_fields, schema.order_field
    )

    is_chunked = schema.is_chunked
    if is_chunked is None:
        is_chunked = SchemaInferer.detect_chunked(hits, id_field, order_field)

    if not is_chunked:
        return _reconstruct_flat(hits, schema, id_field, embedding_model)

    return _reconstruct_chunked(hits, schema, id_field, order_field, embedding_model)


def _resolve_text_fields(payload: Dict[str, Any], schema: SchemaConfig) -> List[str]:
    return SchemaInferer.infer_text_fields(
        payload,
        schema.candidate_combinable_fields,
        schema.text_field,
        schema.text_fields,
        schema.fallback_text_field,
    )


def _resolve_location(payload: Dict[str, Any], order_field: Optional[str] = None) -> Any:
    for f in ([order_field] if order_field else []) + CANDIDATE_LOCATION_FIELDS:
        if f and payload.get(f) is not None:
            return payload.get(f)
    return None


def _reconstruct_flat(
    hits: List[Dict[str, Any]],
    schema: SchemaConfig,
    id_field: str,
    embedding_model: Optional[str],
) -> List[Dict[str, Any]]:
    """Handle non-chunked datasets: one hit == one document, no merging."""
    output: List[Dict[str, Any]] = []

    for hit in hits:
        payload = hit.get("payload", {})
        text_fields = _resolve_text_fields(payload, schema)
        text = _extract_text(payload, text_fields)

        raw_score = hit.get("score", 0)
        normalized = hit.get("normalized_score", raw_score)
        rerank = hit.get("_relevance", normalized)

        output.append({
            "document_id": payload.get(id_field, hit.get("id")),
            "text": text,
            "score": round(raw_score, 4),
            "normalized_score": round(normalized, 4),
            "rerank_score": round(rerank, 4),
            "retrieval_confidence": round(0.6 * normalized + 0.4 * rerank, 4),
            "chunk_count": 1,
            "chunk_id": hit.get("id"),
            "document_location": _resolve_location(payload),
            "embedding_model": embedding_model,
            "metadata": dict(payload),
        })

    return sorted(output, key=lambda x: x["score"], reverse=True)


def _reconstruct_chunked(
    hits: List[Dict[str, Any]],
    schema: SchemaConfig,
    id_field: str,
    order_field: Optional[str],
    embedding_model: Optional[str],
) -> List[Dict[str, Any]]:
    """Handle chunked datasets: group, order, de-duplicate, and merge."""
    groups: Dict[Any, List[Dict[str, Any]]] = {}

    for hit in hits:
        payload = hit.get("payload", {})
        doc_id = payload.get(id_field, hit.get("id"))
        groups.setdefault(doc_id, []).append(hit)

    output: List[Dict[str, Any]] = []

    for doc_id, group in groups.items():
        group.sort(
            key=lambda h: h.get("payload", {}).get(order_field, 0) if order_field else 0
        )

        deduped: List[Dict[str, Any]] = []
        seen_signatures: set = set()

        for item in group:
            payload = item.get("payload", {})
            order_value = payload.get(order_field) if order_field else None
            text_fields = _resolve_text_fields(payload, schema)
            text = _extract_text(payload, text_fields)

            # Normalized text signature catches whitespace/punctuation-only
            # differences that a raw-text comparison would miss.
            signature = order_value if order_value is not None else _normalize_for_dedup(text)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            deduped.append(item)

        if not deduped:
            continue

        best_score = max(item.get("score", 0) for item in deduped)
        best_hit = max(
            deduped,
            key=lambda h: h.get("normalized_score", h.get("score", 0)),
        )

        # Build merged text with ordered, normalization-aware de-duplication:
        # near-identical text blocks (e.g. two chunks that happen to carry
        # the same content modulo whitespace/punctuation) are merged only
        # once, preserving first-seen order, to avoid bloated/inflated
        # merged output.
        merged_text_parts: List[str] = []
        seen_text_blocks: set = set()
        locations: List[Any] = []
        for item in deduped:
            payload = item.get("payload", {})
            text_fields = _resolve_text_fields(payload, schema)
            text = _extract_text(payload, text_fields)
            norm_text = _normalize_for_dedup(text)
            if text and norm_text not in seen_text_blocks:
                merged_text_parts.append(text)
                seen_text_blocks.add(norm_text)
            loc = _resolve_location(payload, order_field)
            if loc is not None:
                locations.append(loc)

        merged_text = " ... ".join(merged_text_parts)
        metadata = dict(deduped[0].get("payload", {}))

        normalized = best_hit.get("normalized_score", best_score)
        rerank = best_hit.get("_relevance", normalized)

        try:
            location_summary: Any = sorted(set(locations)) if locations else None
        except TypeError:
            # Mixed/unorderable location value types -- fall back to
            # insertion order rather than crashing on sort().
            location_summary = list(dict.fromkeys(locations)) if locations else None

        output.append({
            "document_id": doc_id,
            "text": merged_text,
            "score": round(best_score, 4),
            "normalized_score": round(normalized, 4),
            "rerank_score": round(rerank, 4),
            "retrieval_confidence": round(0.6 * normalized + 0.4 * rerank, 4),
            "chunk_count": len(deduped),
            "chunk_id": best_hit.get("id"),
            "document_location": location_summary,
            "embedding_model": embedding_model,
            "metadata": metadata,
        })

    return sorted(output, key=lambda x: x["score"], reverse=True)

from typing import Any, Dict, List

def _dedup_hits_by_id(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove duplicate retrieved hits based on their unique id.
    Keeps the highest-scoring occurrence of each id.
    """

    if not hits:
        return []

    dedup = {}

    for hit in hits:

        hit_id = hit.get("id")

        if hit_id is None:
            hit_id = hit.get("chunk_id")

        if hit_id is None:
            hit_id = hit.get("record_id")

        if hit_id is None:
            continue

        score = hit.get("score", 0.0)

        if hit_id not in dedup:
            dedup[hit_id] = hit
        elif score > dedup[hit_id].get("score", 0.0):
            dedup[hit_id] = hit

    return list(dedup.values())
# ============================================================
# UNIVERSAL SEMANTIC AGENT
# ============================================================

class SemanticAgent:
    """Dataset-agnostic semantic retrieval agent.

    Orchestrates query classification, expansion, embedding, multi-query
    vector search, rank fusion, diversity reranking, schema-driven document
    reconstruction, and confidence scoring. The retrieval pipeline is
    identical regardless of the underlying dataset -- only configuration
    (and, when necessary, ``SchemaConfig`` overrides) changes.

    Pipeline stages (see ``AgentResult.metadata["stage_latencies_ms"]``
    for a per-call latency breakdown of each):
        1. Query classification (hybrid: cheap regex first; for queries
           that pass the LLM gate -- see ``_should_invoke_llm`` -- an
           optional LLM refines this into a broader intent taxonomy,
           with automatic regex fallback on any LLM failure)
        1b. Entity extraction (new; LLM-first with a dataset-agnostic
            regex fallback; caller-supplied ``entities`` always win)
        2. Query expansion (+ optional multi-query variant generation;
           deterministic regex expansion always runs, LLM-assisted
           expansion merges on top of it when the query was gated in)
        2b. Intent understanding (new; LLM-only, purely advisory -- gently
            nudges adaptive top-k / threshold / variant count when
            available, otherwise this sub-stage is simply skipped)
        3. Embedding (batched, cached)
        4. Vector search (batched across variants when multi-query is on)
        5. Rank fusion (Reciprocal Rank Fusion across variants)
        6. Diversity reranking (MMR)
        7. Document reconstruction (schema-aware chunk merging + evidence
           field attachment)

    The LLM-assisted sub-stages (1's refinement, 1b, 2's LLM half, 2b) are
    entirely opt-in: pass an ``LLMConfig(enabled=True, ...)`` as
    ``llm_cfg`` to activate them. Without one, every stage above runs
    exactly as it did before -- deterministic, regex/rule-based, no
    network calls beyond Qdrant.

    Example:
        >>> agent = SemanticAgent()
        >>> result = await agent.run("refund policy", top_k=5)
        >>> await agent.close()
    """

    def __init__(
        self,
        qdrant_cfg: Optional[QdrantConfig] = None,
        embed_cfg: Optional[EmbedConfig] = None,
        schema_cfg: Optional[SchemaConfig] = None,
        retrieval_cfg: Optional[RetrievalConfig] = None,
        llm_cfg: Optional[LLMConfig] = None,
    ):
        self.qcfg = qdrant_cfg or QdrantConfig()
        self.ecfg = embed_cfg or EmbedConfig()
        self.scfg = schema_cfg or SchemaConfig()
        self.rcfg = retrieval_cfg or RetrievalConfig()
        self.lcfg = llm_cfg or LLMConfig()

        self.embedder = EmbeddingEngine(self.ecfg)
        self.qdrant = QdrantClient(self.qcfg)

        # The LLM layer is fully optional. `self.llm` stays `None` unless
        # explicitly enabled, so every code path that touches it must
        # already be prepared to fall back (and does -- see `run()`).
        self.llm: Optional[LLMClient] = LLMClient(self.lcfg) if self.lcfg.enabled else None

        self._result_cache: Optional[TTLCache] = (
            TTLCache(max_size=self.rcfg.result_cache_size, ttl_seconds=self.rcfg.result_cache_ttl_seconds)
            if self.rcfg.enable_result_cache else None
        )

    async def run(
        self,
        query: str,
        entities: Optional[Dict[str, Any]] = None,
        filters: Optional[Dict[str, Any]] = None,
        boosters: Optional[List[str]] = None,
        *,
        synonyms: Optional[List[str]] = None,
        context_keywords: Optional[List[str]] = None,
        top_k: int = DEFAULT_TOP_K,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
        ef: int = DEFAULT_HNSW_EF,
        with_vectors: bool = DEFAULT_WITH_VECTORS,
        adaptive_top_k: bool = True,
        enable_multi_query: bool = False,
        diversity: bool = True,
        query_type: Optional[str] = None,
        compressor: Optional[Any] = None,
        planner_confidence: Optional[float] = None,
        exhaustive: Optional[bool] = None,
        max_results: Optional[int] = None,
        batch_size: Optional[int] = None,
        **kwargs: Any,
    ) -> AgentResult:
        """Execute a full semantic retrieval query.

        By default this performs COMPLETE semantic retrieval: every
        document scoring at or above the effective ``score_threshold`` is
        returned, not merely the best ``top_k``. Internally this pages
        through Qdrant (see ``RetrievalConfig.batch_size``) until a real
        stopping condition is hit -- an empty/partial page, ``max_results``,
        or the ``max_search_depth`` safety net -- rather than stopping at a
        fixed count. Pass ``exhaustive=False`` (or set
        ``RetrievalConfig.exhaustive_retrieval=False``) to fall back to the
        original single-page, ``top_k``-bounded behavior exactly.

        Args:
            query: The raw natural-language query.
            entities: Optional key/value entities for query expansion.
            filters: Optional dataset-agnostic filter spec
                (see :func:`build_filter`).
            boosters: Optional free-form relevance-boosting terms.
            synonyms: Optional synonyms/alternate phrasings to widen recall.
            context_keywords: Optional contextual keywords.
            top_k: In exhaustive mode (the default), this only seeds
                adaptive ``ef``/threshold sizing and has no effect on how
                many results are returned. In non-exhaustive mode
                (``exhaustive=False``), this remains the anchor for
                adaptive scaling exactly as before.
            score_threshold: Minimum similarity score (runtime search
                parameter; applied on the raw Qdrant score scale). In
                exhaustive mode this is the PRIMARY retrieval boundary:
                every match at or above it is returned.
            ef: HNSW search parameter (runtime search parameter). Adaptive
                retrieval will raise this if needed to stay >= the
                effective top-k, capped at a safety ceiling.
            with_vectors: Whether to return raw vectors on each hit's
                metadata handling. Note: internally, vectors may still be
                *fetched* (but not returned) when ``diversity=True``, since
                MMR reranking requires them.
            adaptive_top_k: If ``True`` (default), the effective retrieval
                depth is scaled around ``top_k`` based on query type and
                complexity, then bounded by
                ``RetrievalConfig.adaptive_min_top_k/max_top_k``. Set to
                ``False`` to retrieve exactly ``top_k`` results, matching
                the original fixed-count behavior.
            enable_multi_query: If ``True``, generates additional lexical
                query reformulations, searches each, and fuses results via
                Reciprocal Rank Fusion. Off by default to preserve the
                original single-query call pattern unless explicitly
                requested.
            diversity: If ``True`` (default), applies Maximal Marginal
                Relevance reranking so results aren't dominated by
                near-identical chunks from a single document.
            query_type: Force a query-type classification instead of
                auto-detecting one (see :func:`classify_query_type`).
                Also disables the optional LLM classification refinement
                for this call, since an explicit override should not be
                second-guessed.
            compressor: Optional context-compression coordinator exposing
                a ``compress(chunks: List[Dict]) -> List[Dict]`` method
                (e.g. a ``ContextCompressor``). If provided and more than
                4 chunks were reconstructed, it is invoked after
                reconstruction; its output replaces ``result.chunks`` only
                if it returned no *more* chunks than it received (a
                compressor that unexpectedly grows the chunk list is
                treated as a no-op rather than trusted blindly). Any
                exception raised by the compressor is logged and
                swallowed -- retrieval never fails because compression did.
            planner_confidence: Optional upstream confidence signal (e.g.
                from a query planner/router) in ``[0, 1]``. When supplied
                and below 0.65, adaptive top-k is nudged wider to
                compensate for planner uncertainty by retrieving more
                candidates than the query-type heuristic alone would.
            exhaustive: Override ``RetrievalConfig.exhaustive_retrieval``
                for this call. ``None`` (default) defers to the agent's
                configured default (exhaustive by default). ``True``
                forces complete retrieval; ``False`` forces the original
                single-page, ``top_k``-bounded behavior.
            max_results: In exhaustive mode, an optional hard cap on total
                results (per query vector, pre-fusion). ``None`` (default)
                defers to ``RetrievalConfig.max_results_default`` (which
                itself defaults to ``None`` -- no cap besides the
                ``max_search_depth`` safety net). Ignored in non-exhaustive
                mode, where ``top_k`` already bounds the result count.
            batch_size: Page size for exhaustive retrieval. ``None``
                (default) defers to ``RetrievalConfig.batch_size``.
                Ignored in non-exhaustive mode.
            **kwargs: Silently absorbed. Lets orchestrators (e.g.
                ``HybridAgent``) evolve their call signature without this
                method raising ``TypeError`` on unrecognized keywords.
                Unexpected keys are logged at debug level for visibility.

        Returns:
            An :class:`AgentResult` describing success/failure, the
            reconstructed documents, and diagnostic metadata. Exceptions
            are captured and reported via ``result.error`` rather than
            propagated, so this method never raises for query-time
            failures.
        """
        start = time.perf_counter()

        entities = entities or {}
        filters = filters or {}
        boosters = boosters or []

        use_exhaustive = self.rcfg.exhaustive_retrieval if exhaustive is None else exhaustive
        effective_batch_size = batch_size if batch_size is not None else self.rcfg.batch_size
        effective_max_results = max_results if max_results is not None else self.rcfg.max_results_default

        if kwargs:
            logger.debug("SemanticAgent.run() received unrecognized kwargs: %s", list(kwargs.keys()))

        result = AgentResult(query_type="semantic", entities=entities, filters=filters)
        stage_latencies: Dict[str, float] = {}

        def _mark(name: str, t0: float) -> None:
            stage_latencies[name] = round((time.perf_counter() - t0) * 1000, 2)

        cache_key: Optional[str] = None

        try:
            if not query or not query.strip():
                raise ValueError("Query cannot be empty")

            if self._result_cache is not None:
                cache_key = self._make_cache_key(
                    query, entities, filters, boosters, synonyms, context_keywords,
                    top_k, score_threshold, ef, with_vectors, adaptive_top_k,
                    enable_multi_query, diversity, query_type,
                    use_exhaustive, effective_batch_size, effective_max_results,
                )
                cached = await self._result_cache.get(cache_key)
                if cached is not None:
                    logger.debug("Result cache hit for query=%r", query)
                    # Return a deep copy so callers can freely mutate the
                    # result (e.g. in-place compression of `.chunks`)
                    # without corrupting what's stored in the cache for
                    # subsequent callers with the same cache key.
                    return copy.deepcopy(cached)

            # -- Stage 1: query classification (hybrid regex / LLM) -----
            t0 = time.perf_counter()
            qtype_regex = classify_query_type(query)
            qtype = query_type or qtype_regex
            llm_classification_source = "regex"
            llm_classification_confidence: Optional[float] = None

            # -- LLM gating: decide ONCE whether this query is worth any
            # LLM call at all. Simple/short/lookup-style queries never
            # reach the LLM (see _should_invoke_llm docstring).
            use_llm = (
                query_type is None  # explicit caller override always wins outright
                and self.llm is not None
                and _should_invoke_llm(query, qtype_regex, self.rcfg)
            )

            llm_expansion = LLMExpansion(source="none")
            llm_entities = LLMEntities(source="none")
            llm_understanding = LLMIntentUnderstanding(source="none")

            if use_llm:
                # Fire the independent LLM stages concurrently (not
                # sequentially) to minimize added latency; each has its
                # own timeout/retry/validation and fails independently.
                coros = [
                    self.llm.classify(query),
                    self.llm.expand(query),
                    self.llm.extract_entities(query),
                    self.llm.understand_intent(query, qtype_regex),
                ]
                gathered = await asyncio.gather(*coros, return_exceptions=True)
                classify_res, expand_res, entities_res, intent_res = gathered

                if isinstance(classify_res, LLMClassification):
                    if classify_res.intent != QueryIntent.UNKNOWN:
                        qtype = classify_res.intent.value
                    llm_classification_source = "llm"
                    llm_classification_confidence = classify_res.confidence
                else:
                    logger.debug("LLM classification fell back to regex: %s", classify_res)

                if isinstance(expand_res, LLMExpansion):
                    llm_expansion = expand_res
                else:
                    logger.debug("LLM expansion fell back to regex: %s", expand_res)

                if isinstance(entities_res, LLMEntities):
                    llm_entities = entities_res
                else:
                    logger.debug("LLM entity extraction fell back to regex: %s", entities_res)

                if isinstance(intent_res, LLMIntentUnderstanding):
                    llm_understanding = intent_res
                else:
                    logger.debug("LLM intent understanding unavailable: %s", intent_res)

            strategy = _QUERY_TYPE_STRATEGY.get(qtype, _QUERY_TYPE_STRATEGY["informational"])
            _mark("classify_query", t0)

            effective_top_k = top_k
            if adaptive_top_k:
                effective_top_k = _compute_adaptive_top_k(
                    top_k, query, qtype,
                    self.rcfg.adaptive_min_top_k, self.rcfg.adaptive_max_top_k,
                    planner_confidence,
                )
            effective_ef = min(_MAX_EF, max(ef, effective_top_k * 2))
            effective_threshold = (
                max(0.0, min(1.0, score_threshold + strategy["threshold_delta"]))
                if 0.0 <= score_threshold <= 1.0 else score_threshold
            )

            effective_max_variants = self.rcfg.max_query_variants
            if llm_understanding.source == "llm":
                effective_top_k, effective_threshold, effective_max_variants = _apply_intent_understanding(
                    effective_top_k, effective_threshold, effective_max_variants,
                    llm_understanding, self.rcfg.adaptive_min_top_k, self.rcfg.adaptive_max_top_k,
                )
                effective_ef = min(_MAX_EF, max(effective_ef, effective_top_k * 2))

            # -- New stage: entity extraction ----------------------------
            # Explicit caller-supplied `entities` always take precedence;
            # auto-extracted entities only fill in categories the caller
            # didn't already provide.
            t0 = time.perf_counter()
            if llm_entities.source != "llm":
                llm_entities = regex_extract_entities(query)
            auto_entities = llm_entities.as_query_entities()
            merged_entities = {**auto_entities, **entities}
            _mark("entity_extraction", t0)

            # -- Stage 2: query expansion (+ optional multi-query) ------
            t0 = time.perf_counter()
            active_synonyms = list(synonyms or [])

            # Deterministic base expansion always runs (cheap, no I/O);
            # the LLM's output -- when available -- is merged on top of
            # it rather than replacing it.
            base_expansion = regex_expand_query(query)
            active_synonyms = list(dict.fromkeys(active_synonyms + base_expansion.all_terms()))
            if llm_expansion.source == "llm":
                active_synonyms = list(dict.fromkeys(active_synonyms + llm_expansion.all_terms()))

            if self.rcfg.query_expander is not None:
                try:
                    extra = self.rcfg.query_expander(query) or []
                    active_synonyms = list(dict.fromkeys(active_synonyms + list(extra)))
                except Exception:
                    logger.warning("query_expander hook raised; continuing without it", exc_info=True)

            expanded_query = build_query(
                query=query, entities=merged_entities, boosters=boosters,
                synonyms=active_synonyms, context_keywords=context_keywords,
            )

            query_variants = [expanded_query]
            if enable_multi_query:
                query_variants = _generate_query_variants(
                    query, expanded_query, effective_max_variants,
                )
            _mark("query_expansion", t0)
            logger.debug("Query variants: %s", query_variants)

            # -- Stage 3: embedding ---------------------------------------
            t0 = time.perf_counter()
            vectors = await self.embedder.embed(query_variants)
            if not vectors or not vectors[0]:
                raise RuntimeError("Embedding produced no usable vector for the query")
            _mark("embedding", t0)

            # -- Stage 4: vector search (+ metadata filtering) -----------
            #
            # Two retrieval paradigms, selected by `use_exhaustive`:
            #   - exhaustive (default): page through Qdrant per query
            #     vector until a real stopping condition is hit (empty/
            #     partial page, max_results, or max_search_depth), so
            #     EVERY match above `effective_threshold` comes back --
            #     not just the best `effective_top_k`.
            #   - legacy top-k (`exhaustive=False`): a single bounded page
            #     per query vector, exactly as before.
            qfilter = build_filter(filters)
            fetch_vectors = with_vectors or diversity

            t0 = time.perf_counter()
            if use_exhaustive:
                hit_lists = await self.qdrant.search_exhaustive_batch(
                    vectors=vectors, qfilter=qfilter, score_threshold=effective_threshold,
                    ef=effective_ef, with_vector=fetch_vectors,
                    batch_size=effective_batch_size, max_results=effective_max_results,
                    max_search_depth=self.rcfg.max_search_depth,
                )
            else:
                hit_lists = await self.qdrant.search_batch(
                    vectors=vectors, qfilter=qfilter, top_k=effective_top_k,
                    score_threshold=effective_threshold, ef=effective_ef, with_vector=fetch_vectors,
                )
            _mark("vector_search", t0)
            raw_hit_count = sum(len(hl) for hl in hit_lists)

            # -- Stage 5: rank fusion across query variants ---------------
            # Reciprocal Rank Fusion already deduplicates by Qdrant point
            # id across variants (see `_reciprocal_rank_fusion_with_counts`),
            # so this doubles as the cross-batch/cross-variant merge step
            # requirement 9/11 ask for -- O(n) in total hits, no O(n^2)
            # pairwise comparison needed for exact-duplicate removal.
            t0 = time.perf_counter()
            if len(hit_lists) > 1:
                fused_hits, variant_counts = _reciprocal_rank_fusion_with_counts(
                    hit_lists, k=self.rcfg.rrf_k
                )
                num_variants = len(hit_lists)
            else:
                fused_hits = _dedup_hits_by_id(hit_lists[0]) if hit_lists else []
                variant_counts = {h.get("id"): 1 for h in fused_hits}
                num_variants = 1
            _mark("fusion", t0)

            metric = self.qcfg.distance_metric
            for h in fused_hits:
                h["normalized_score"] = normalize_score(h.get("score", 0.0), metric)
                h["_relevance"] = h["normalized_score"]

            # -- Stage 6: diversity (MMR) reranking / final ranking -------
            t0 = time.perf_counter()
            if use_exhaustive:
                # Complete retrieval: keep everything (optionally capped
                # by `effective_max_results`) rather than truncating to
                # `top_k` -- the whole point is NOT to lose relevant
                # results here.
                target_n = len(fused_hits)
                if effective_max_results is not None:
                    target_n = min(target_n, effective_max_results)
            else:
                target_n = min(effective_top_k, len(fused_hits))

            if diversity and len(fused_hits) > 1:
                if len(fused_hits) <= self.rcfg.diversity_max_candidates:
                    reranked = mmr_rerank(fused_hits, top_n=target_n, lambda_mult=strategy["diversity_lambda"])
                else:
                    # MMR is O(top_n * n); for large exhaustive result
                    # sets this degrades toward O(n^2). Beyond the
                    # configured cap, fall back to a plain O(n log n)
                    # score sort so exhaustive retrieval never gets
                    # quadratically slow -- diversity reranking remains
                    # fully intact below the cap.
                    logger.debug(
                        "Skipping MMR for %d candidates (> diversity_max_candidates=%d); "
                        "sorting by score instead",
                        len(fused_hits), self.rcfg.diversity_max_candidates,
                    )
                    reranked = sorted(fused_hits, key=lambda h: h["normalized_score"], reverse=True)[:target_n]
            else:
                reranked = sorted(fused_hits, key=lambda h: h["normalized_score"], reverse=True)[:target_n]
            _mark("diversity_rerank", t0)

            if not with_vectors:
                for h in reranked:
                    h.pop("vector", None)

            # -- Stage 7: document reconstruction --------------------------
            # Operates on the full merged/deduplicated hit set regardless
            # of which batch or query variant each chunk originally came
            # from, so chunks of the same document collected across
            # different pages/variants still reconstruct into one document
            # (requirement 10) -- no change needed here beyond receiving
            # the complete `reranked` set from stages 4-6 above.
            t0 = time.perf_counter()
            chunks = reconstruct(hits=reranked, schema=self.scfg, embedding_model=self.ecfg.model_name)
            near_dup_removal_applied = False
            if enable_multi_query and len(chunks) > 1:
                if len(chunks) <= self.rcfg.dedup_max_candidates:
                    chunks = _remove_near_duplicate_chunks(chunks, self.rcfg.dedup_similarity_threshold)
                    near_dup_removal_applied = True
                else:
                    # Cross-document near-duplicate detection is O(n^2)
                    # (all-pairs Jaccard shingle comparison). Beyond the
                    # configured cap, skip it to avoid quadratic cost on
                    # large exhaustive result sets -- exact/point-id and
                    # within-document text-signature de-duplication
                    # (both O(n), applied earlier in fusion/reconstruction)
                    # still remove exact duplicates regardless.
                    logger.debug(
                        "Skipping cross-document near-duplicate pass for %d documents "
                        "(> dedup_max_candidates=%d)",
                        len(chunks), self.rcfg.dedup_max_candidates,
                    )
            _mark("reconstruction", t0)

            coverage = _query_coverage(chunks, variant_counts, num_variants)
            confidence = compute_confidence(chunks, query_coverage=coverage)

            # -- Stage 8 (optional): context compression coordination -----
            # Purely coordinative: this module has no opinion on *how*
            # compression works, only on invoking it safely. A compressor
            # that raises, times out, or unexpectedly returns more chunks
            # than it received is treated as a no-op rather than trusted.
            compression_applied = False
            if compressor is not None and len(chunks) > 4:
                t0 = time.perf_counter()
                try:
                    compressed = compressor.compress(chunks)
                    if (
                        compressed is not None
                        and isinstance(compressed, list)
                        and 0 < len(compressed) <= len(chunks)
                    ):
                        chunks = compressed
                        compression_applied = True
                        _mark("compression", t0)
                    else:
                        logger.warning(
                            "Compressor returned an unusable result "
                            "(type=%s, len=%s); keeping uncompressed chunks",
                            type(compressed).__name__,
                            len(compressed) if isinstance(compressed, list) else "n/a",
                        )
                except Exception:
                    logger.warning(
                        "Context compression failed; continuing with "
                        "uncompressed chunks",
                        exc_info=True,
                    )

            result.success = True
            result.chunks = chunks
            result.metadata.update({
                "raw_hits": sum(len(hl) for hl in hit_lists),
                "fused_hits": len(fused_hits),
                "documents": len(chunks),
                "expanded_query": expanded_query,
                "query_type": qtype,
                "query_variants_used": len(query_variants),
                "retrieval_confidence": confidence,
                "query_coverage": coverage,
                "compression_applied": compression_applied,
                "search_params": {
                    "requested_top_k": top_k,
                    "effective_top_k": effective_top_k,
                    "score_threshold": effective_threshold,
                    "ef": effective_ef,
                    "with_vectors": with_vectors,
                    "adaptive_top_k": adaptive_top_k,
                    "multi_query": enable_multi_query,
                    "diversity": diversity,
                    "distance_metric": metric,
                },
                "stage_latencies_ms": stage_latencies,
                "embedding_cache": self.embedder._cache.stats() if self.embedder._cache else None,
                "query_type_regex": qtype_regex,
                "llm_augmentation": {
                    "gated_in": use_llm,
                    "classification_source": llm_classification_source,
                    "classification_confidence": llm_classification_confidence,
                    "expansion_source": llm_expansion.source,
                    "entities_source": llm_entities.source,
                    "intent_understanding_source": llm_understanding.source,
                    "auto_extracted_entities": auto_entities,
                },
            })

        except Exception as exc:
            logger.exception("SemanticAgent.run() failed for query=%r", query)
            result.error = str(exc)

        result.latency_ms = round((time.perf_counter() - start) * 1000, 2)

        if self._result_cache is not None and cache_key and result.success:
            # Store a deep copy so any later in-place mutation of the
            # returned `result` (by this caller) can't silently corrupt
            # the cached entry for future callers.
            await self._result_cache.set(cache_key, copy.deepcopy(result))

        return result

    @staticmethod
    def _make_cache_key(*parts: Any) -> str:
        payload = json.dumps(parts, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    async def health(self) -> bool:
        """Lightweight readiness check for the underlying Qdrant backend.

        Delegates to :meth:`QdrantClient.health`. Never raises.

        Returns:
            ``True`` if Qdrant is reachable and healthy, ``False`` otherwise.
        """
        return await self.qdrant.health()

    async def invalidate_cache(self) -> None:
        """Clear the embedding cache and, if enabled, the result cache.

        Use this after re-indexing a collection if ``RetrievalConfig.
        enable_result_cache`` is turned on, since stale cached results
        would otherwise persist until their TTL expires.
        """
        if self.embedder._cache is not None:
            await self.embedder._cache.clear()
        if self._result_cache is not None:
            await self._result_cache.clear()

    async def close(self) -> None:
        """Release all held resources (HTTP connections)."""
        await self.qdrant.close()
        if self.llm is not None:
            await self.llm.close()