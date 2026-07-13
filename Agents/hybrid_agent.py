"""
hybrid_agent.py
============================================================
Enterprise Hybrid Orchestration Agent

Single responsibility: receive a user query, decide which existing
agent(s) should answer it, run them, merge/rank/limit their evidence,
hand that evidence to the existing AnswerGenerator, and return one
assembled response.

This module performs NO SQL generation, NO SQL validation/repair, NO
embedding, NO vector search, NO reranking, and NO LLM-based natural
language answer generation. Every one of those responsibilities
already belongs to, respectively, ``StructuredAgent`` (structured_agent.py),
``SemanticAgent`` (semantic_agent.py), and ``AnswerGenerator``
(answer_generator.py). HybridAgent is strictly an orchestrator that
consumes the outputs those three already produce.

Pipeline
--------
    1. QueryAnalyzer   -- detect query characteristics (rule-based).
    2. Router          -- decide STRUCTURED_ONLY / SEMANTIC_ONLY /
                           HYBRID_PARALLEL / HYBRID_SEQUENTIAL.
    3. Execution        -- run the chosen agent(s) concurrently via
                           their existing async interfaces; retry only
                           the failed branch, reusing information
                           discovered by the successful branch.
    4. ContextMerger    -- normalize + deduplicate + link evidence from
                           both branches.
    5. ContextRanker    -- rank evidence using confidences the agents
                           already computed (never recomputed here).
    6. AnswerGenerator  -- the ONLY component that produces the final
                           natural-language answer.
    7. Assembly         -- combine everything into one ``HybridResult``,
                           with confidence aggregation, metrics and logs.

No component's public interface is modified, wrapped with duck-typed
guesswork, or reimplemented. Every field read from ``StructuredAgent
.AgentResult``, ``SemanticAgent.AgentResult``, and ``AnswerGenerator
.AnswerResult`` below matches those dataclasses exactly as provided.
============================================================
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ------------------------------------------------------------------
# Source-of-truth imports. Aliased to avoid name collisions between
# the three modules (all three define an ``AgentResult``/config-style
# dataclass with overlapping names).
# ------------------------------------------------------------------
from Agents.structured_agent import (
    AgentConfig as StructuredAgentConfig,
    AgentResult as StructuredAgentResult,
    DiscoveredSchema,
    StructuredAgent,
)
from Agents.semantic_agent import (
    AgentResult as SemanticAgentResult,
    EmbedConfig,
    LLMConfig as SemanticLLMConfig,
    QdrantConfig,
    RetrievalConfig,
    SchemaConfig as SemanticSchemaConfig,
    SemanticAgent,
)
from LLM.answer_generator import (
    AnswerGenerator,
    AnswerResult,
    AnswerType,
)

logger = logging.getLogger("hybrid_agent")
if not logger.handlers:
    logger.addHandler(logging.NullHandler())


def _clamp01(value: Optional[float]) -> float:
    """Clamp an arbitrary numeric value into [0.0, 1.0]. Never raises;
    non-numeric/None/NaN input is treated as 0.0. This is trivial
    orchestration-level defensive plumbing, not a reimplementation of
    any agent's confidence computation."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    if v != v:  # NaN
        return 0.0
    return max(0.0, min(1.0, v))


# =========================================================
# CONFIGURATION
# =========================================================

@dataclass
class HybridAgentConfig:
    """Every tunable orchestration parameter lives here. No magic
    numbers are embedded in the routing, retry, merging, ranking, or
    aggregation logic below -- all of it reads from this object."""

    # ---- Timeouts (per branch, per hybrid run) ----
    structured_timeout_seconds: float = 25.0
    semantic_timeout_seconds: float = 20.0
    answer_timeout_seconds: float = 25.0

    # ---- Retry strategy ----
    enable_retry: bool = True
    max_branch_retries: int = 1

    # ---- Execution ----
    enable_parallel_execution: bool = True

    # ---- Routing thresholds ----
    # A branch-exclusive score (structured or semantic) at or above
    # this fraction of the total signal routes to that branch alone.
    single_branch_dominance_ratio: float = 0.75
    # Minimum routing confidence below which HybridAgent logs a
    # low-confidence-routing warning (informational only; does not
    # change the decision).
    min_routing_confidence_warning: float = 0.4

    # ---- Confidence aggregation weights ----
    # Applied only across the components that actually ran; renormalized
    # over the present subset so a skipped branch doesn't silently pull
    # the aggregate toward zero.
    weight_structured_confidence: float = 0.35
    weight_semantic_confidence: float = 0.30
    weight_answer_confidence: float = 0.35

    # ---- Context assembly ----
    max_context_items: int = 12
    weight_confidence_in_ranking: float = 0.55
    weight_richness_in_ranking: float = 0.30
    weight_freshness_in_ranking: float = 0.05
    linked_evidence_bonus: float = 0.10

    # ---- Conversation memory ----
    max_remembered_entities: int = 25
    max_history_turns: int = 20

    # ---- Feature flags ----
    enable_metrics: bool = True
    enable_schema_reuse: bool = True

    # ---- Fallback (used ONLY if AnswerGenerator is unavailable or errors) ----
    fallback_template: str = (
        "I found {count} relevant piece(s) of information but could not "
        "reach the answer-generation service. Top evidence: {preview}"
    )


# =========================================================
# QUERY ANALYSIS
# =========================================================

@dataclass
class QueryAnalysis:
    """Structured, rule-based characterization of a single query.
    Purely additive orchestration signal -- never used to generate SQL,
    embeddings, or answers directly."""

    raw_query: str
    normalized_query: str
    has_aggregation: bool = False
    has_filtering: bool = False
    has_ranking: bool = False
    has_temporal: bool = False
    has_comparison: bool = False
    has_recommendation: bool = False
    has_explanation: bool = False
    has_sentiment: bool = False
    has_complaint: bool = False
    is_follow_up: bool = False
    has_conversational_reference: bool = False
    matched_signals: List[str] = field(default_factory=list)


class QueryAnalyzer:
    """Dataset-agnostic, dependency-free rule-based query classifier.
    Uses word/phrase pattern lists analogous in spirit to
    ``AgentConfig.derived_metric_glossary`` in structured_agent.py --
    generic linguistic scaffolding, never a dataset-specific assumption.
    """

    _AGGREGATION = re.compile(
        r"\b(average|avg|sum|total|count|how many|how much|median|"
        r"mean|percentage|percent|proportion)\b", re.IGNORECASE,
    )
    _FILTERING = re.compile(
        r"\b(where|with a|greater than|less than|at least|at most|"
        r"only|excluding|filter(?:ed)? by|between)\b", re.IGNORECASE,
    )
    _RANKING = re.compile(
        r"\b(top|best|worst|highest|lowest|most|least|rank(?:ed|ing)?|"
        r"leading)\b", re.IGNORECASE,
    )
    _TEMPORAL = re.compile(
        r"\b(today|yesterday|last (?:week|month|year|quarter)|this "
        r"(?:week|month|year|quarter)|since|trend|over time|\d{4}|"
        r"monthly|weekly|daily|yearly|quarterly)\b", re.IGNORECASE,
    )
    _COMPARISON = re.compile(
        r"\b(compare|comparison|versus|vs\.?|difference between|"
        r"better than|worse than)\b", re.IGNORECASE,
    )
    _RECOMMENDATION = re.compile(
        r"\b(recommend|suggest|should i|what (?:should|would)|advice|"
        r"which (?:one|option) is)\b", re.IGNORECASE,
    )
    _EXPLANATION = re.compile(
        r"\b(why|explain|reason|how does|how do|what causes|because of "
        r"what)\b", re.IGNORECASE,
    )
    _SENTIMENT = re.compile(
        r"\b(feel|feeling|opinion|love|hate|like|dislike|great|"
        r"terrible|amazing|awful|satisfied|happy|unhappy)\b",
        re.IGNORECASE,
    )
    _COMPLAINT = re.compile(
        r"\b(complain(?:t|ing)?|issue|problem|broken|defective|"
        r"disappointed|refund|not working|doesn'?t work|faulty)\b",
        re.IGNORECASE,
    )
    _FOLLOW_UP_START = re.compile(
        r"^\s*(and|also|what about|how about|additionally|furthermore)\b",
        re.IGNORECASE,
    )
    _CONVERSATIONAL_REFERENCE = re.compile(
        r"\b(it|that|those|these|the same|the previous one|the last "
        r"one|them)\b", re.IGNORECASE,
    )

    def analyze(
        self,
        query: str,
        conversation_has_history: bool = False,
    ) -> QueryAnalysis:
        normalized = " ".join(query.strip().split())
        analysis = QueryAnalysis(raw_query=query, normalized_query=normalized)

        checks: Sequence[Tuple[str, re.Pattern, str]] = (
            ("has_aggregation", self._AGGREGATION, "aggregation"),
            ("has_filtering", self._FILTERING, "filtering"),
            ("has_ranking", self._RANKING, "ranking"),
            ("has_temporal", self._TEMPORAL, "temporal"),
            ("has_comparison", self._COMPARISON, "comparison"),
            ("has_recommendation", self._RECOMMENDATION, "recommendation"),
            ("has_explanation", self._EXPLANATION, "explanation"),
            ("has_sentiment", self._SENTIMENT, "sentiment"),
            ("has_complaint", self._COMPLAINT, "complaint"),
        )
        for attr, pattern, label in checks:
            if pattern.search(normalized):
                setattr(analysis, attr, True)
                analysis.matched_signals.append(label)

        if conversation_has_history:
            if self._FOLLOW_UP_START.search(normalized):
                analysis.is_follow_up = True
                analysis.matched_signals.append("follow_up")
            if self._CONVERSATIONAL_REFERENCE.search(normalized):
                analysis.has_conversational_reference = True
                analysis.matched_signals.append("conversational_reference")

        return analysis


# =========================================================
# ROUTING
# =========================================================

class RoutingMode(str, Enum):
    STRUCTURED_ONLY = "structured_only"
    SEMANTIC_ONLY = "semantic_only"
    HYBRID_PARALLEL = "hybrid_parallel"
    HYBRID_SEQUENTIAL = "hybrid_sequential"


@dataclass
class RoutingDecision:
    mode: RoutingMode
    confidence: float
    reason: str
    execution_strategy: str
    structured_signal: float = 0.0
    semantic_signal: float = 0.0


class Router:
    """Combines rule-based scoring over ``QueryAnalysis`` with a weak
    prior from the conversation's prior routing mode. Never touches
    SQL, embeddings, or answers -- purely a mode decision."""

    def __init__(self, config: HybridAgentConfig):
        self.config = config

    def route(
        self,
        analysis: QueryAnalysis,
        conversation: "ConversationContext",
    ) -> RoutingDecision:
        structured_score = 0.0
        semantic_score = 0.0

        if analysis.has_aggregation:
            structured_score += 1.0
        if analysis.has_filtering:
            structured_score += 0.8
        if analysis.has_ranking:
            structured_score += 0.8
        if analysis.has_temporal:
            structured_score += 0.6
        if analysis.has_comparison:
            structured_score += 0.5
            semantic_score += 0.3
        if analysis.has_recommendation:
            semantic_score += 1.0
        if analysis.has_explanation:
            semantic_score += 0.9
        if analysis.has_sentiment:
            semantic_score += 0.8
        if analysis.has_complaint:
            semantic_score += 0.7
            structured_score += 0.2

        if analysis.is_follow_up or analysis.has_conversational_reference:
            if conversation.last_routing_mode == RoutingMode.STRUCTURED_ONLY.value:
                structured_score += 0.3
            elif conversation.last_routing_mode == RoutingMode.SEMANTIC_ONLY.value:
                semantic_score += 0.3

        total = structured_score + semantic_score
        if total <= 0.0:
            decision = RoutingDecision(
                mode=RoutingMode.HYBRID_PARALLEL,
                confidence=0.3,
                reason="No strong routing signal detected in the query; "
                       "defaulting to hybrid parallel execution to cover both "
                       "retrieval strategies.",
                execution_strategy="parallel",
                structured_signal=0.0,
                semantic_signal=0.0,
            )
            return decision

        norm_structured = structured_score / total
        norm_semantic = semantic_score / total
        dominance = self.config.single_branch_dominance_ratio

        if norm_structured >= dominance:
            mode = RoutingMode.STRUCTURED_ONLY
            confidence = norm_structured
            reason = (
                f"Structured signals dominate ({', '.join(s for s in analysis.matched_signals if s in ('aggregation', 'filtering', 'ranking', 'temporal'))}); "
                "routing to StructuredAgent only."
            )
            strategy = "single_branch"
        elif norm_semantic >= dominance:
            mode = RoutingMode.SEMANTIC_ONLY
            confidence = norm_semantic
            reason = (
                f"Semantic signals dominate ({', '.join(s for s in analysis.matched_signals if s in ('recommendation', 'explanation', 'sentiment', 'complaint'))}); "
                "routing to SemanticAgent only."
            )
            strategy = "single_branch"
        elif (analysis.is_follow_up or analysis.has_conversational_reference) and conversation.turns > 0:
            mode = RoutingMode.HYBRID_SEQUENTIAL
            confidence = _clamp01(0.5 + 0.5 * (1 - abs(norm_structured - norm_semantic)))
            reason = (
                "Follow-up/conversational reference detected with mixed "
                "signal; running branches sequentially so entities discovered "
                "by the first branch can inform the second."
            )
            strategy = "sequential_with_info_sharing"
        else:
            mode = RoutingMode.HYBRID_PARALLEL
            confidence = _clamp01(0.5 + 0.5 * (1 - abs(norm_structured - norm_semantic)))
            reason = (
                "Mixed structured/semantic signal with no single-branch "
                "dominance; running both branches concurrently."
            )
            strategy = "parallel"

        if confidence < self.config.min_routing_confidence_warning:
            logger.info(
                "Router: low routing confidence (%.2f) for query=%r -- mode=%s",
                confidence, analysis.raw_query, mode.value,
            )

        return RoutingDecision(
            mode=mode,
            confidence=round(_clamp01(confidence), 4),
            reason=reason,
            execution_strategy=strategy,
            structured_signal=round(norm_structured, 4),
            semantic_signal=round(norm_semantic, 4),
        )


# =========================================================
# CONVERSATION STATE
# =========================================================

@dataclass
class ConversationContext:
    """Per-conversation memory. Deliberately holds only lightweight,
    JSON-serializable state -- never full agent results -- so it stays
    cheap to keep around across many turns."""

    conversation_id: str = "default"
    turns: int = 0
    entities: Dict[str, Any] = field(default_factory=dict)
    filters: Dict[str, Any] = field(default_factory=dict)
    schema: Optional[DiscoveredSchema] = None
    last_routing_mode: Optional[str] = None
    history: List[Dict[str, Any]] = field(default_factory=list)

    def remember_turn(self, query: str, mode: str, overall_confidence: float, max_turns: int) -> None:
        self.turns += 1
        self.last_routing_mode = mode
        self.history.append({
            "turn": self.turns,
            "query": query,
            "mode": mode,
            "overall_confidence": overall_confidence,
        })
        if len(self.history) > max_turns:
            self.history = self.history[-max_turns:]

    def merge_entities(self, new_entities: Dict[str, Any], max_entities: int) -> None:
        if not new_entities:
            return
        self.entities.update(new_entities)
        if len(self.entities) > max_entities:
            # Keep the most recently added entries (dict preserves
            # insertion order in Python 3.7+; oldest keys are dropped).
            overflow = len(self.entities) - max_entities
            for key in list(self.entities.keys())[:overflow]:
                del self.entities[key]

    def merge_filters(self, new_filters: Dict[str, Any]) -> None:
        if new_filters:
            self.filters.update(new_filters)


# =========================================================
# EVIDENCE MERGING / LINKING / RANKING
# =========================================================

# Generic, dataset-agnostic identifier field candidates used only for
# evidence linking between branches -- never assumed to exist, never
# hardcoded into query logic.
_COMMON_IDENTIFIER_FIELDS: Tuple[str, ...] = (
    "product_id", "asin", "movie_id", "sku", "document_id",
    "id", "title", "name",
)


def _extract_identifiers(record: Dict[str, Any]) -> Dict[str, Any]:
    """Scan a record (and its nested ``metadata`` dict, if any) for
    generic identifier-like fields, for evidence linking purposes only."""
    found: Dict[str, Any] = {}
    sources = [record]
    nested_metadata = record.get("metadata")
    if isinstance(nested_metadata, dict):
        sources.append(nested_metadata)
    for src in sources:
        for key in _COMMON_IDENTIFIER_FIELDS:
            if key in src and src[key] not in (None, ""):
                found.setdefault(key, src[key])
    return found


def _entities_from_records(
    records: Sequence[Dict[str, Any]],
    limit_records: int = 8,
) -> Dict[str, Any]:
    """Derive a generic entity dict (field -> value or list of values)
    from a handful of evidence records, for cross-branch info sharing
    on retry. Only non-numeric, non-boolean scalar fields are kept,
    since those behave like identifiers/categories rather than metrics."""
    collected: Dict[str, List[Any]] = defaultdict(list)
    for record in list(records)[:limit_records]:
        candidates = dict(record)
        nested_metadata = candidates.get("metadata")
        if isinstance(nested_metadata, dict):
            candidates.update(nested_metadata)
        for key, value in candidates.items():
            if key in ("metadata", "embedding", "vector"):
                continue
            if isinstance(value, (int, float, bool)) or value in (None, "", [], {}):
                continue
            if not isinstance(value, (str, list)):
                continue
            if value not in collected[key]:
                collected[key].append(value)
    return {
        key: (values[0] if len(values) == 1 else values)
        for key, values in collected.items()
    }


@dataclass
class EvidenceRecord:
    """One normalized piece of evidence, from either branch, ready for
    merging/ranking/limiting. ``record`` is left exactly as the source
    agent produced it (a structured row or a reconstructed semantic
    chunk) -- HybridAgent never rewrites agent output, only wraps it."""

    source: str  # "structured" | "semantic"
    record: Dict[str, Any]
    confidence: float
    provenance: Dict[str, Any] = field(default_factory=dict)
    identifiers: Dict[str, Any] = field(default_factory=dict)
    linked_sources: List[str] = field(default_factory=lambda: [])
    richness_score: float = 0.0

    def content_fingerprint(self) -> str:
        """Cheap structural fingerprint used only for exact-duplicate
        detection (not semantic dedup, which already happened inside
        SemanticAgent's reconstruction/rerank stage)."""
        keys = sorted(k for k in self.record.keys() if k not in ("metadata",))
        return "|".join(f"{k}={self.record.get(k)!r}" for k in keys)


class ContextMerger:
    """Normalizes, deduplicates, and links evidence produced by
    StructuredAgent and SemanticAgent. Never recomputes retrieval,
    SQL, or confidence -- only reorganizes what already exists."""

    def __init__(self, config: HybridAgentConfig):
        self.config = config

    def merge(
        self,
        structured_result: Optional[StructuredAgentResult],
        semantic_result: Optional[SemanticAgentResult],
    ) -> List[EvidenceRecord]:
        records: List[EvidenceRecord] = []

        if structured_result is not None and structured_result.success:
            for row in structured_result.rows:
                records.append(EvidenceRecord(
                    source="structured",
                    record=row,
                    confidence=_clamp01(structured_result.overall_confidence),
                    provenance={
                        "sql": structured_result.sql,
                        "tables_used": list(structured_result.tables_used),
                        "status": structured_result.status,
                    },
                    identifiers=_extract_identifiers(row),
                ))

        if semantic_result is not None and semantic_result.success:
            for chunk in semantic_result.chunks:
                records.append(EvidenceRecord(
                    source="semantic",
                    record=chunk,
                    confidence=_clamp01(chunk.get("retrieval_confidence", 0.0)),
                    provenance={
                        "document_location": chunk.get("document_location"),
                        "embedding_model": chunk.get("embedding_model"),
                        "chunk_count": chunk.get("chunk_count"),
                    },
                    identifiers=_extract_identifiers(chunk),
                ))

        records = self._dedupe(records)
        records = self._link(records)
        return records

    @staticmethod
    def _dedupe(records: List[EvidenceRecord]) -> List[EvidenceRecord]:
        """Removes exact structural duplicates, preferring the richer
        record (more non-null fields) when a duplicate is found."""
        best_by_fingerprint: Dict[Tuple[str, str], EvidenceRecord] = {}
        for rec in records:
            key = (rec.source, rec.content_fingerprint())
            existing = best_by_fingerprint.get(key)
            if existing is None:
                best_by_fingerprint[key] = rec
                continue
            if _non_null_count(rec.record) > _non_null_count(existing.record):
                best_by_fingerprint[key] = rec
        return list(best_by_fingerprint.values())

    @staticmethod
    def _link(records: List[EvidenceRecord]) -> List[EvidenceRecord]:
        """Links structured and semantic records that share a common
        identifier value, without any dataset-specific assumption
        about which identifier field applies."""
        by_identifier_value: Dict[Tuple[str, Any], List[EvidenceRecord]] = defaultdict(list)
        for rec in records:
            for field_name, value in rec.identifiers.items():
                if isinstance(value, (str, int, float)):
                    by_identifier_value[(field_name, value)].append(rec)

        for group in by_identifier_value.values():
            sources_in_group = {r.source for r in group}
            if len(sources_in_group) > 1:
                for rec in group:
                    for other_source in sources_in_group:
                        if other_source not in rec.linked_sources:
                            rec.linked_sources.append(other_source)
        return records


def _non_null_count(record: Dict[str, Any]) -> int:
    return sum(1 for v in record.values() if v not in (None, "", [], {}))


class ContextRanker:
    """Ranks evidence using confidences the agents already computed,
    plus orchestration-only signals (richness, linkage, freshness) that
    don't exist anywhere upstream. Never recomputes retrieval or SQL
    confidence."""

    _FRESHNESS_FIELD_CANDIDATES: Tuple[str, ...] = (
        "created_at", "updated_at", "timestamp", "date", "review_date",
    )

    def __init__(self, config: HybridAgentConfig):
        self.config = config

    def rank(self, records: List[EvidenceRecord]) -> List[EvidenceRecord]:
        for rec in records:
            rec.richness_score = self._richness(rec.record)
        scored = sorted(records, key=self._score, reverse=True)
        return scored

    def _score(self, rec: EvidenceRecord) -> float:
        linkage_bonus = self.config.linked_evidence_bonus if len(rec.linked_sources) > 1 else 0.0
        freshness = self._freshness_signal(rec.record)
        return (
            self.config.weight_confidence_in_ranking * rec.confidence
            + self.config.weight_richness_in_ranking * rec.richness_score
            + self.config.weight_freshness_in_ranking * freshness
            + linkage_bonus
        )

    @staticmethod
    def _richness(record: Dict[str, Any]) -> float:
        if not record:
            return 0.0
        return _non_null_count(record) / max(1, len(record))

    def _freshness_signal(self, record: Dict[str, Any]) -> float:
        for field_name in self._FRESHNESS_FIELD_CANDIDATES:
            if record.get(field_name):
                return 1.0
            metadata = record.get("metadata")
            if isinstance(metadata, dict) and metadata.get(field_name):
                return 1.0
        return 0.0

    def limit(self, records: List[EvidenceRecord]) -> List[EvidenceRecord]:
        return records[: self.config.max_context_items]


# =========================================================
# METRICS
# =========================================================

class MetricsCollector:
    """Process-lifetime orchestration metrics. Complements (never
    duplicates) each agent's own metrics -- e.g.
    ``StructuredAgent.get_metrics_summary()`` -- which HybridAgent
    surfaces separately via :meth:`HybridAgent.get_metrics`."""

    def __init__(self) -> None:
        self.total_runs = 0
        self.routing_counts: Dict[str, int] = defaultdict(int)
        self.retry_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.total_latency_ms = 0.0

    def record(self, mode: RoutingMode, retried: bool, success: bool, latency_ms: float) -> None:
        self.total_runs += 1
        self.routing_counts[mode.value] += 1
        if retried:
            self.retry_count += 1
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
        self.total_latency_ms += latency_ms

    def summary(self) -> Dict[str, Any]:
        avg_latency = (self.total_latency_ms / self.total_runs) if self.total_runs else 0.0
        success_rate = (self.success_count / self.total_runs) if self.total_runs else 0.0
        return {
            "total_runs": self.total_runs,
            "routing_counts": dict(self.routing_counts),
            "retry_count": self.retry_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": round(success_rate, 4),
            "avg_latency_ms": round(avg_latency, 2),
        }


# =========================================================
# RESULT
# =========================================================

@dataclass
class HybridResult:
    """Final assembled response of one :meth:`HybridAgent.run` call."""

    query: str
    answer: str
    answer_type: str
    status: str

    routing: RoutingDecision
    structured_result: Optional[StructuredAgentResult]
    semantic_result: Optional[SemanticAgentResult]
    answer_result: Optional[AnswerResult]

    evidence: List[Dict[str, Any]] = field(default_factory=list)
    sources: List[Dict[str, Any]] = field(default_factory=list)

    structured_confidence: float = 0.0
    semantic_confidence: float = 0.0
    answer_confidence: float = 0.0
    overall_confidence: float = 0.0

    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    retries: List[Dict[str, Any]] = field(default_factory=list)

    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "answer_type": self.answer_type,
            "status": self.status,
            "routing": asdict(self.routing),
            "structured_result": self.structured_result.to_dict() if self.structured_result else None,
            "semantic_result": self.semantic_result.to_dict() if self.semantic_result else None,
            "answer_result": self.answer_result.to_dict() if self.answer_result else None,
            "evidence": self.evidence,
            "sources": self.sources,
            "structured_confidence": self.structured_confidence,
            "semantic_confidence": self.semantic_confidence,
            "answer_confidence": self.answer_confidence,
            "overall_confidence": self.overall_confidence,
            "errors": self.errors,
            "warnings": self.warnings,
            "retries": self.retries,
            "latency_ms": self.latency_ms,
            "metadata": self.metadata,
        }


# =========================================================
# HYBRID AGENT
# =========================================================

class HybridAgent:
    """Orchestrates ``StructuredAgent``, ``SemanticAgent``, and
    ``AnswerGenerator`` behind one ``run()`` call. Depends on all
    three via constructor injection -- it never constructs its own SQL,
    embeddings, or answers.
    """

    def __init__(
        self,
        structured_agent: StructuredAgent,
        semantic_agent: SemanticAgent,
        answer_generator: AnswerGenerator,
        config: Optional[HybridAgentConfig] = None,
    ) -> None:
        self.structured_agent = structured_agent
        self.semantic_agent = semantic_agent
        self.answer_generator = answer_generator
        self.config = config or HybridAgentConfig()

        self._analyzer = QueryAnalyzer()
        self._router = Router(self.config)
        self._merger = ContextMerger(self.config)
        self._ranker = ContextRanker(self.config)
        self._metrics = MetricsCollector()

        self._conversations: Dict[str, ConversationContext] = {}
        self._conversation_lock = asyncio.Lock()

    # -----------------------------------------------------
    # Public API
    # -----------------------------------------------------

    async def run(
        self,
        query: str,
        conversation_id: str = "default",
        answer_type_override: Optional[AnswerType] = None,
        debug: bool = False,
    ) -> HybridResult:
        """Answer one query, maintaining conversation state under
        ``conversation_id`` across calls.

        Never raises for query-time failures: every branch failure,
        timeout, or AnswerGenerator error degrades gracefully into a
        best-effort ``HybridResult`` instead of propagating.
        """
        start = time.perf_counter()
        errors: List[str] = []
        warnings: List[str] = []
        retries: List[Dict[str, Any]] = []

        conversation = await self._get_conversation(conversation_id)

        analysis = self._analyzer.analyze(query, conversation_has_history=conversation.turns > 0)
        routing = self._router.route(analysis, conversation)
        logger.info(
            "HybridAgent.run: query=%r conversation=%s mode=%s routing_confidence=%.2f reason=%s",
            query, conversation_id, routing.mode.value, routing.confidence, routing.reason,
        )

        if self.config.enable_schema_reuse and routing.mode in (
            RoutingMode.STRUCTURED_ONLY, RoutingMode.HYBRID_PARALLEL, RoutingMode.HYBRID_SEQUENTIAL,
        ) and conversation.schema is None:
            try:
                conversation.schema = await asyncio.to_thread(self.structured_agent.get_schema)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("HybridAgent: schema pre-fetch failed: %s", exc)

        structured_result: Optional[StructuredAgentResult] = None
        semantic_result: Optional[SemanticAgentResult] = None

        try:
            structured_result, semantic_result, branch_errors, branch_retries = await self._execute(
                query=query, routing=routing, conversation=conversation,
            )
            errors.extend(branch_errors)
            retries.extend(branch_retries)
        except Exception as exc:  # pragma: no cover - defensive, execution already self-guards
            logger.exception("HybridAgent: unhandled execution error for query=%r", query)
            errors.append(f"execution_error: {exc}")

        # ---- Context merge / rank / limit ----
        evidence_records: List[EvidenceRecord] = []
        try:
            evidence_records = self._merger.merge(structured_result, semantic_result)
            evidence_records = self._ranker.rank(evidence_records)
            evidence_records = self._ranker.limit(evidence_records)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("HybridAgent: context merge/rank failed")
            errors.append(f"context_merge_error: {exc}")

        context_payload = [rec.record for rec in evidence_records]
        if not context_payload:
            warnings.append("No evidence survived merging/ranking; answer will be ungrounded.")

        # ---- Answer generation (the ONLY place answers are produced) ----
        answer_type = answer_type_override or self._answer_type_for_mode(routing.mode)
        answer_result: Optional[AnswerResult] = None
        try:
            answer_result = await asyncio.wait_for(
                asyncio.to_thread(
                    self.answer_generator.generate, query, context_payload, answer_type,
                ),
                timeout=self.config.answer_timeout_seconds,
            )
        except asyncio.TimeoutError:
            errors.append("answer_generator_timeout")
            logger.warning("HybridAgent: AnswerGenerator timed out for query=%r", query)
        except Exception as exc:
            errors.append(f"answer_generator_error: {exc}")
            logger.exception("HybridAgent: AnswerGenerator raised for query=%r", query)

        answer_text, answer_status, answer_confidence, sources = self._resolve_answer(
            answer_result, evidence_records,
        )
        if answer_result is not None and not answer_result.success:
            warnings.extend(answer_result.warnings)
            if answer_result.error:
                errors.append(f"answer_generator_reported_error: {answer_result.error}")

        # ---- Confidence aggregation (never recomputed from raw signals) ----
        structured_confidence = (
            _clamp01(structured_result.overall_confidence)
            if structured_result is not None and structured_result.success else 0.0
        )
        semantic_confidence = (
            _clamp01(semantic_result.metadata.get("retrieval_confidence", 0.0))
            if semantic_result is not None and semantic_result.success else 0.0
        )
        overall_confidence = self._aggregate_confidence(
            structured_result=structured_result,
            semantic_result=semantic_result,
            answer_confidence=answer_confidence,
        )

        overall_status = "success" if (answer_result is not None and answer_result.success) else (
            "partial" if context_payload else "failed"
        )

        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        # ---- Conversation memory update ----
        discovered_entities: Dict[str, Any] = {}
        if structured_result is not None and structured_result.success:
            discovered_entities.update(_entities_from_records(structured_result.rows))
        if semantic_result is not None and semantic_result.success:
            discovered_entities.update(_entities_from_records(semantic_result.chunks))
        conversation.merge_entities(discovered_entities, self.config.max_remembered_entities)
        conversation.remember_turn(query, routing.mode.value, overall_confidence, self.config.max_history_turns)

        if self.config.enable_metrics:
            self._metrics.record(
                mode=routing.mode,
                retried=bool(retries),
                success=(overall_status == "success"),
                latency_ms=latency_ms,
            )

        logger.info(
            "HybridAgent.run complete: query=%r mode=%s status=%s overall_confidence=%.2f "
            "latency_ms=%.1f retries=%d errors=%d",
            query, routing.mode.value, overall_status, overall_confidence,
            latency_ms, len(retries), len(errors),
        )

        metadata: Dict[str, Any] = {
            "query_analysis": asdict(analysis),
            "evidence_count": len(evidence_records),
            "structured_row_count": structured_result.row_count if structured_result else 0,
            "semantic_document_count": len(semantic_result.chunks) if semantic_result else 0,
        }
        if debug:
            metadata["structured_metadata"] = structured_result.metadata if structured_result else None
            metadata["semantic_metadata"] = semantic_result.metadata if semantic_result else None
            metadata["answer_metadata"] = answer_result.metadata if answer_result else None

        return HybridResult(
            query=query,
            answer=answer_text,
            answer_type=answer_type.value,
            status=overall_status,
            routing=routing,
            structured_result=structured_result,
            semantic_result=semantic_result,
            answer_result=answer_result,
            evidence=context_payload,
            sources=sources,
            structured_confidence=round(structured_confidence, 4),
            semantic_confidence=round(semantic_confidence, 4),
            answer_confidence=round(answer_confidence, 4),
            overall_confidence=round(overall_confidence, 4),
            errors=errors,
            warnings=warnings,
            retries=retries,
            latency_ms=latency_ms,
            metadata=metadata,
        )

    def get_metrics(self) -> Dict[str, Any]:
        """Orchestration-level metrics plus each agent's own metrics
        summary (surfaced, not recomputed)."""
        return {
            "hybrid": self._metrics.summary(),
            "structured_agent": self.structured_agent.get_metrics_summary(),
        }

    async def close(self) -> None:
        """Releases resources held by the injected agents. HybridAgent
        owns no network/DB resources of its own."""
        try:
            await self.semantic_agent.close()
        except Exception:  # pragma: no cover - defensive
            logger.warning("HybridAgent: error closing SemanticAgent", exc_info=True)
        try:
            self.structured_agent.close()
        except Exception:  # pragma: no cover - defensive
            logger.warning("HybridAgent: error closing StructuredAgent", exc_info=True)

    # -----------------------------------------------------
    # Execution planning
    # -----------------------------------------------------

    async def _execute(
        self,
        query: str,
        routing: RoutingDecision,
        conversation: ConversationContext,
    ) -> Tuple[Optional[StructuredAgentResult], Optional[SemanticAgentResult], List[str], List[Dict[str, Any]]]:
        errors: List[str] = []
        retries: List[Dict[str, Any]] = []
        structured_result: Optional[StructuredAgentResult] = None
        semantic_result: Optional[SemanticAgentResult] = None

        if routing.mode is RoutingMode.STRUCTURED_ONLY:
            structured_result, err = await self._run_structured(query, conversation)
            if err:
                errors.append(err)

        elif routing.mode is RoutingMode.SEMANTIC_ONLY:
            semantic_result, err = await self._run_semantic(query, conversation)
            if err:
                errors.append(err)

        elif routing.mode is RoutingMode.HYBRID_SEQUENTIAL:
            structured_result, err_s = await self._run_structured(query, conversation)
            if err_s:
                errors.append(err_s)

            extra_entities: Dict[str, Any] = {}
            if structured_result is not None and structured_result.success:
                extra_entities = _entities_from_records(structured_result.rows)

            semantic_result, err_sem = await self._run_semantic(
                query, conversation, extra_entities=extra_entities,
            )
            if err_sem:
                errors.append(err_sem)

        else:  # HYBRID_PARALLEL
            structured_task = asyncio.create_task(self._run_structured(query, conversation))
            semantic_task = asyncio.create_task(self._run_semantic(query, conversation))
            (structured_result, err_s), (semantic_result, err_sem) = await asyncio.gather(
                structured_task, semantic_task,
            )
            if err_s:
                errors.append(err_s)
            if err_sem:
                errors.append(err_sem)

        # ---- Bounded retry of the failed branch only, reusing info
        # discovered by the successful branch. ----
        if self.config.enable_retry and self.config.max_branch_retries > 0:
            structured_failed = structured_result is None or not structured_result.success
            semantic_failed = semantic_result is None or not semantic_result.success
            structured_attempted = routing.mode in (
                RoutingMode.STRUCTURED_ONLY, RoutingMode.HYBRID_PARALLEL, RoutingMode.HYBRID_SEQUENTIAL,
            )
            semantic_attempted = routing.mode in (
                RoutingMode.SEMANTIC_ONLY, RoutingMode.HYBRID_PARALLEL, RoutingMode.HYBRID_SEQUENTIAL,
            )

            if structured_attempted and structured_failed and semantic_result is not None and semantic_result.success:
                discovered = _entities_from_records(semantic_result.chunks)
                if discovered:
                    retried_result, err = await self._run_structured(
                        query, conversation, extra_entities=discovered,
                    )
                    retries.append({
                        "branch": "structured",
                        "reused_from": "semantic",
                        "entities_reused": list(discovered.keys()),
                        "success": bool(retried_result and retried_result.success),
                    })
                    if retried_result is not None and retried_result.success:
                        structured_result = retried_result
                    elif err:
                        errors.append(err)

            if semantic_attempted and semantic_failed and structured_result is not None and structured_result.success:
                discovered = _entities_from_records(structured_result.rows)
                if discovered:
                    retried_result, err = await self._run_semantic(
                        query, conversation, extra_entities=discovered,
                    )
                    retries.append({
                        "branch": "semantic",
                        "reused_from": "structured",
                        "entities_reused": list(discovered.keys()),
                        "success": bool(retried_result and retried_result.success),
                    })
                    if retried_result is not None and retried_result.success:
                        semantic_result = retried_result
                    elif err:
                        errors.append(err)

        return structured_result, semantic_result, errors, retries

    async def _run_structured(
        self,
        query: str,
        conversation: ConversationContext,
        extra_entities: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[StructuredAgentResult], Optional[str]]:
        entities = dict(conversation.entities)
        if extra_entities:
            entities.update(extra_entities)
        try:
            result = await asyncio.wait_for(
                self.structured_agent.run(
                    query=query,
                    filters=conversation.filters or None,
                    entities=entities or None,
                    schema=conversation.schema,
                    metadata={"conversation_id": conversation.conversation_id},
                ),
                timeout=self.config.structured_timeout_seconds,
            )
            return result, (None if result.success else f"structured_branch: {result.error}")
        except asyncio.TimeoutError:
            logger.warning("HybridAgent: StructuredAgent timed out for query=%r", query)
            return None, "structured_branch_timeout"
        except Exception as exc:  # pragma: no cover - StructuredAgent.run already self-guards
            logger.exception("HybridAgent: StructuredAgent raised for query=%r", query)
            return None, f"structured_branch_exception: {exc}"

    async def _run_semantic(
        self,
        query: str,
        conversation: ConversationContext,
        extra_entities: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[SemanticAgentResult], Optional[str]]:
        entities = dict(conversation.entities)
        if extra_entities:
            entities.update(extra_entities)
        try:
            result = await asyncio.wait_for(
                self.semantic_agent.run(
                    query=query,
                    entities=entities or None,
                    filters=conversation.filters or None,
                ),
                timeout=self.config.semantic_timeout_seconds,
            )
            return result, (None if result.success else f"semantic_branch: {result.error}")
        except asyncio.TimeoutError:
            logger.warning("HybridAgent: SemanticAgent timed out for query=%r", query)
            return None, "semantic_branch_timeout"
        except Exception as exc:  # pragma: no cover - SemanticAgent.run already self-guards
            logger.exception("HybridAgent: SemanticAgent raised for query=%r", query)
            return None, f"semantic_branch_exception: {exc}"

    # -----------------------------------------------------
    # Helpers
    # -----------------------------------------------------

    async def _get_conversation(self, conversation_id: str) -> ConversationContext:
        async with self._conversation_lock:
            conversation = self._conversations.get(conversation_id)
            if conversation is None:
                conversation = ConversationContext(conversation_id=conversation_id)
                self._conversations[conversation_id] = conversation
            return conversation

    @staticmethod
    def _answer_type_for_mode(mode: RoutingMode) -> AnswerType:
        if mode is RoutingMode.STRUCTURED_ONLY:
            return AnswerType.STRUCTURED
        if mode is RoutingMode.SEMANTIC_ONLY:
            return AnswerType.SEMANTIC
        return AnswerType.HYBRID

    def _aggregate_confidence(
        self,
        structured_result: Optional[StructuredAgentResult],
        semantic_result: Optional[SemanticAgentResult],
        answer_confidence: float,
    ) -> float:
        """Weighted aggregation of confidences the agents already
        computed, renormalized over the components actually present so
        a skipped/failed branch doesn't silently deflate the score."""
        components: List[Tuple[float, float]] = []  # (weight, value)

        if structured_result is not None and structured_result.success:
            components.append((
                self.config.weight_structured_confidence,
                _clamp01(structured_result.overall_confidence),
            ))
        if semantic_result is not None and semantic_result.success:
            components.append((
                self.config.weight_semantic_confidence,
                _clamp01(semantic_result.metadata.get("retrieval_confidence", 0.0)),
            ))
        components.append((self.config.weight_answer_confidence, _clamp01(answer_confidence)))

        total_weight = sum(w for w, _ in components)
        if total_weight <= 0.0:
            return 0.0
        return _clamp01(sum(w * v for w, v in components) / total_weight)

    def _resolve_answer(
        self,
        answer_result: Optional[AnswerResult],
        evidence_records: List[EvidenceRecord],
    ) -> Tuple[str, str, float, List[Dict[str, Any]]]:
        """Returns (answer_text, status, confidence, sources). Falls
        back to a minimal template ONLY if AnswerGenerator is
        unavailable or returned an error -- never generates a detailed
        answer itself."""
        if answer_result is not None and answer_result.success:
            return (
                answer_result.answer,
                answer_result.status,
                answer_result.confidence,
                answer_result.sources,
            )

        if not evidence_records:
            return (
                "I don't have enough information to answer that.",
                "failed",
                0.0,
                [],
            )

        preview_parts = []
        for rec in evidence_records[:3]:
            text = rec.record.get("text") or rec.record.get("content")
            if text:
                preview_parts.append(str(text)[:160])
            else:
                preview_parts.append(str({k: v for k, v in list(rec.record.items())[:4]}))
        preview = "; ".join(preview_parts)

        fallback_text = self.config.fallback_template.format(
            count=len(evidence_records), preview=preview,
        )
        fallback_sources = [
            {"source": rec.source, **rec.identifiers} for rec in evidence_records[:5]
        ]
        return (fallback_text, "partial", 0.2, fallback_sources)


# =========================================================
# CONVENIENCE FACTORY
# =========================================================

def build_default_hybrid_agent(
    db_path: str,
    qdrant_cfg: Optional[QdrantConfig] = None,
    embed_cfg: Optional[EmbedConfig] = None,
    semantic_schema_cfg: Optional[SemanticSchemaConfig] = None,
    retrieval_cfg: Optional[RetrievalConfig] = None,
    semantic_llm_cfg: Optional[SemanticLLMConfig] = None,
    structured_config: Optional[StructuredAgentConfig] = None,
    hybrid_config: Optional[HybridAgentConfig] = None,
) -> HybridAgent:
    """Convenience constructor wiring up default instances of all
    three underlying agents plus ``HybridAgent`` itself. Entirely
    optional -- callers with already-constructed agent instances
    should just call ``HybridAgent(...)`` directly."""
    structured_agent = StructuredAgent(db_path=db_path, config=structured_config)
    semantic_agent = SemanticAgent(
        qdrant_cfg=qdrant_cfg,
        embed_cfg=embed_cfg,
        schema_cfg=semantic_schema_cfg,
        retrieval_cfg=retrieval_cfg,
        llm_cfg=semantic_llm_cfg,
    )
    answer_generator = AnswerGenerator()
    return HybridAgent(
        structured_agent=structured_agent,
        semantic_agent=semantic_agent,
        answer_generator=answer_generator,
        config=hybrid_config,
    )


# =========================================================
# CLI ENTRY POINT
# =========================================================

def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="hybrid_agent",
        description=(
            "Run the HybridAgent orchestrator against a DuckDB database "
            "(StructuredAgent) and a Qdrant collection (SemanticAgent)."
        ),
    )
    parser.add_argument(
        "query", nargs="?", default=None,
        help="Natural-language query to answer. If omitted, starts an "
             "interactive REPL instead.",
    )
    parser.add_argument(
        "--db-path", default="F:/PROJECT/Database/Duckdb/analytics.duckdb",
        help="Path to the DuckDB database file used by StructuredAgent.",
    )
    parser.add_argument(
        "--qdrant-host", default="http://localhost:6333",
        help="Qdrant REST endpoint used by SemanticAgent.",
    )
    parser.add_argument(
        "--qdrant-collection", default="documents",
        help="Qdrant collection name used by SemanticAgent.",
    )
    parser.add_argument(
        "--conversation-id", default="cli",
        help="Conversation identifier, used to keep state across turns "
             "in interactive mode.",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Include per-agent raw metadata in the printed result.",
    )
    parser.add_argument(
        "--json", action="store_true", dest="as_json",
        help="Print the full HybridResult as JSON instead of a short "
             "human-readable summary.",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity for hybrid_agent/structured_agent/"
             "semantic_agent loggers.",
    )
    return parser.parse_args(argv)


def _print_result(result: HybridResult, as_json: bool) -> None:
    if as_json:
        print(json.dumps(result.to_dict(), indent=2, default=str))
        return

    print(f"\n[{result.status.upper()}] ({result.routing.mode.value}, "
          f"overall_confidence={result.overall_confidence:.2f})")
    print(result.answer)
    if result.warnings:
        print(f"\nWarnings: {'; '.join(result.warnings)}")
    if result.errors:
        print(f"Errors: {'; '.join(result.errors)}")
    print(f"(latency={result.latency_ms:.1f}ms, evidence_items={len(result.evidence)})")


async def _run_single_query(agent: HybridAgent, args: argparse.Namespace) -> int:
    try:
        result = await agent.run(
            query=args.query,
            conversation_id=args.conversation_id,
            debug=args.debug,
        )
        _print_result(result, args.as_json)
        return 0 if result.status != "failed" else 1
    finally:
        await agent.close()


async def _run_interactive(agent: HybridAgent, args: argparse.Namespace) -> int:
    print("HybridAgent interactive session. Type a query, or 'exit'/'quit' to stop.")
    try:
        while True:
            try:
                query = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not query:
                continue
            if query.lower() in ("exit", "quit"):
                break
            result = await agent.run(
                query=query,
                conversation_id=args.conversation_id,
                debug=args.debug,
            )
            _print_result(result, args.as_json)
        return 0
    finally:
        await agent.close()


async def _amain(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    agent = build_default_hybrid_agent(
        db_path=args.db_path,
        qdrant_cfg=QdrantConfig(host=args.qdrant_host, collection=args.qdrant_collection),
    )

    if args.query:
        return await _run_single_query(agent, args)
    return await _run_interactive(agent, args)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Synchronous CLI entry point (installed/invoked as
    ``python hybrid_agent.py [query] [options]``)."""
    try:
        return asyncio.run(_amain(argv))
    except KeyboardInterrupt:  # pragma: no cover - interactive convenience
        print("\nInterrupted.")
        return 130


if __name__ == "__main__":
    sys.exit(main())