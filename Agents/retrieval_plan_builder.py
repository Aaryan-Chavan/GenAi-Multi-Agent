from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "RetrievalPlanBuilder",
    "RetrievalPlan",
    "IntentResult",
    "SQLTask",
    "VectorTask",
    "SchemaRegistry",
    "StaticSchemaRegistry",
    "ColumnMeta",
    "TableMeta",
    "RelationshipMeta",
    "VectorCollectionMeta",
    "ColumnRole",
    "PlannerConfig",
]


# ============================================================
# LINGUISTIC UTILITIES (no dataset knowledge — just English)
# ============================================================

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower())


def split_identifier(name: str) -> List[str]:
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name)
    spaced = spaced.replace("_", " ").replace("-", " ")
    return tokenize(spaced)


# NOTE: these are generic English synonym groups used to expand query /
# column tokens for overlap scoring. They are NOT a dataset-specific
# keyword table — nothing here references a specific table or column
# name, and no code path branches on membership in a particular group.
_SYNONYM_GROUPS: List[Set[str]] = [
    {"count", "total", "number", "quantity", "volume"},
    {"average", "avg", "mean", "typical"},
    {"sum", "aggregate", "combined"},
    {"top", "best", "highest", "largest", "greatest", "most"},
    {"bottom", "worst", "lowest", "smallest", "least"},
    {"distinct", "unique", "different"},
    {"trend", "trends", "overtime", "history", "historical", "growth", "change"},
    {"percentage", "percent", "proportion", "ratio", "share"},
    {"compare", "comparison", "versus", "vs"},
    {"negative", "critical", "poor", "bad", "urgent", "problem", "issue"},
    {"opinion", "feedback", "sentiment", "experience", "review"},
    {"date", "time", "period", "when"},
    {"category", "type", "class", "group", "segment"},
    {"identifier", "id", "key", "code", "reference"},
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


def overlap_score(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = a & b
    if not inter:
        return 0.0
    return len(inter) / min(len(a), len(b))


_GRANULARITY_HINTS: Dict[str, str] = {
    "hour": "hour", "hourly": "hour",
    "day": "day", "daily": "day",
    "week": "week", "weekly": "week",
    "month": "month", "monthly": "month",
    "quarter": "quarter", "quarterly": "quarter",
    "year": "year", "yearly": "year", "annually": "year", "annual": "year",
}

_N_PATTERN = re.compile(r"\btop\s+(\d+)\b|\bfirst\s+(\d+)\b|\bbottom\s+(\d+)\b|\blast\s+(\d+)\b")


# ============================================================
# SCHEMA DATA CONTRACTS  (what a SchemaRegistry must expose)
# ============================================================

class ColumnRole(str, Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    TEMPORAL = "temporal"
    BOOLEAN = "boolean"
    IDENTIFIER = "identifier"
    TEXTUAL = "textual"
    JSON = "json"
    EMBEDDING = "embedding"
    VECTOR_REFERENCE = "vector_reference"  # column linking a row to an
    # external vector-store entry (e.g. a foreign key into a Qdrant
    # payload). Distinct from EMBEDDING, which is a column that *is*
    # the raw vector. Added per Schema Analyzer contract.
    UNKNOWN = "unknown"


class FilterOperator(str, Enum):
    EQ = "eq"
    NEQ = "neq"
    IN = "in"
    NOT_IN = "not_in"
    LIKE = "like"
    ILIKE = "ilike"
    BETWEEN = "between"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    IS_NULL = "is_null"
    IS_NOT_NULL = "is_not_null"
    ARRAY_CONTAINS = "array_contains"


class AggregationOp(str, Enum):
    COUNT = "COUNT"
    SUM = "SUM"
    AVG = "AVG"
    MEDIAN = "MEDIAN"
    MODE = "MODE"
    MIN = "MIN"
    MAX = "MAX"
    STDDEV = "STDDEV"
    VARIANCE = "VARIANCE"
    DISTINCT = "DISTINCT"
    TOP_N = "TOP_N"
    BOTTOM_N = "BOTTOM_N"
    RANK = "RANK"
    PERCENTILE = "PERCENTILE"
    MOVING_AVERAGE = "MOVING_AVERAGE"
    RUNNING_TOTAL = "RUNNING_TOTAL"
    DISTRIBUTION = "DISTRIBUTION"
    COMPARE = "COMPARE"
    TIME_SERIES = "TIME_SERIES"


@dataclass(frozen=True)
class ColumnMeta:
    name: str
    role: ColumnRole
    searchable: bool = False
    filterable: bool = True
    sortable: bool = True
    aggregatable: bool = False
    semantic_tokens: frozenset = field(default_factory=frozenset)

    def tokens(self) -> Set[str]:
        if self.semantic_tokens:
            return set(self.semantic_tokens)
        return expand_with_synonyms(split_identifier(self.name))


@dataclass(frozen=True)
class TableMeta:
    name: str
    columns: Tuple[ColumnMeta, ...]
    row_count: Optional[int] = None
    primary_key: Optional[str] = None

    def column(self, name: str) -> Optional[ColumnMeta]:
        return next((c for c in self.columns if c.name == name), None)

    def columns_with_role(self, *roles: ColumnRole) -> List[ColumnMeta]:
        return [c for c in self.columns if c.role in roles]

    def semantic_tokens(self) -> Set[str]:
        return expand_with_synonyms(split_identifier(self.name))


@dataclass(frozen=True)
class RelationshipMeta:
    from_table: str
    from_column: str
    to_table: str
    to_column: str
    cardinality: str = "many_to_one"
    confidence: float = 1.0


@dataclass(frozen=True)
class VectorCollectionMeta:
    name: str
    payload_fields: Tuple[ColumnMeta, ...] = ()
    linked_table: Optional[str] = None
    linked_column: Optional[str] = None
    approx_size: Optional[int] = None


class SchemaRegistry(ABC):
    """
    The only source of schema truth the planner is allowed to consult.
    Any backend (DuckDB, Postgres, Snowflake, Qdrant, ...) plugs in by
    implementing this interface — the planner never inspects a live
    connection itself.

    IMPORTANT: the method name here is `get_vector_collections()`. Some
    external specs refer to this capability as `list_vector_collections`.
    Whatever your Schema Analyzer implementation calls it, make sure it
    matches THIS interface exactly (or add a one-line alias in your
    concrete registry) — RetrievalPlanBuilder only ever calls the names
    declared below.
    """

    @abstractmethod
    def list_tables(self) -> List[str]: ...

    @abstractmethod
    def get_table(self, name: str) -> Optional[TableMeta]: ...

    @abstractmethod
    def get_relationships(self) -> List[RelationshipMeta]: ...

    @abstractmethod
    def get_vector_collections(self) -> List[VectorCollectionMeta]: ...

    @property
    @abstractmethod
    def version(self) -> str:
        """Opaque token that changes whenever the underlying schema
        changes — used to invalidate the planner's caches automatically
        on schema evolution."""
        ...


class StaticSchemaRegistry(SchemaRegistry):
    """
    Simple, dependency-free SchemaRegistry backed by plain data. This is
    the primary way new datasets are supported: construct one of these
    (typically from a live schema-discovery pass elsewhere in the
    pipeline) and hand it to RetrievalPlanBuilder — no planner code
    changes needed.
    """

    def __init__(
        self,
        tables: Sequence[TableMeta],
        relationships: Sequence[RelationshipMeta] = (),
        vector_collections: Sequence[VectorCollectionMeta] = (),
        version: str = "1",
    ):
        self._tables = {t.name: t for t in tables}
        self._relationships = list(relationships)
        self._vector_collections = list(vector_collections)
        self._version = version

    def list_tables(self) -> List[str]:
        return list(self._tables.keys())

    def get_table(self, name: str) -> Optional[TableMeta]:
        return self._tables.get(name)

    def get_relationships(self) -> List[RelationshipMeta]:
        return list(self._relationships)

    def get_vector_collections(self) -> List[VectorCollectionMeta]:
        return list(self._vector_collections)

    @property
    def version(self) -> str:
        return self._version


# ============================================================
# CONFIGURATION  (every threshold/weight lives here)
# ============================================================

@dataclass
class PlannerConfig:
    top_k_by_complexity: Dict[str, int] = field(
        default_factory=lambda: {"simple": 5, "medium": 10, "complex": 20}
    )
    min_table_score: float = 0.05
    min_entity_match_score: float = 0.15
    min_vector_collection_score: float = 0.0  # always pick the best available
    max_join_hops: int = 4

    complexity_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "multi_entity": 1.0,
            "time_series": 1.0,
            "focus_mode": 1.0,
            "hybrid_branch": 1.0,
            "multi_intent": 1.0,
        }
    )
    complexity_simple_max: float = 1.0
    complexity_medium_max: float = 3.0

    # Approximate relative cost weights used only to break ties between
    # otherwise-equal plans (e.g. whether to run sql+vector sequentially
    # or in parallel). Not a real query optimizer — a lightweight,
    # explainable heuristic.
    cost_per_join: float = 1.0
    cost_per_aggregation: float = 0.5
    cost_per_filter: float = 0.2
    cost_per_vector_row_log: float = 0.3


DEFAULT_CONFIG = PlannerConfig()


# ============================================================
# SCHEMA CONTEXT CACHE
# ============================================================

class SchemaContextCache:
    """
    Caches expensive, registry-derived views (join graph, per-table token
    index) and transparently invalidates whenever `registry.version`
    changes — this is how the planner "automatically adapts" to schema
    evolution without any code change.
    """

    def __init__(self, registry: SchemaRegistry):
        self.registry = registry
        self._version: Optional[str] = None
        self._tables: Dict[str, TableMeta] = {}
        self._relationships: List[RelationshipMeta] = []
        self._join_graph: Dict[str, List[RelationshipMeta]] = {}
        self._table_tokens: Dict[str, Set[str]] = {}
        self._refresh_if_needed()

    def _refresh_if_needed(self) -> None:
        current_version = getattr(self.registry, "version", "unknown")
        if current_version == self._version:
            return

        table_names = self.registry.list_tables() or []
        tables = {
            name: self.registry.get_table(name)
            for name in table_names
        }
        self._tables = {k: v for k, v in tables.items() if v is not None}

        relationships = self.registry.get_relationships()
        if relationships is None:
            relationships = []
        self._relationships = list(relationships)

        # Build the (bidirectional) join graph. Every table gets an
        # entry — even with zero relationships — so downstream code can
        # always safely do `join_graph.get(table_name, [])` and get `[]`
        # rather than a KeyError. Relationships are OPTIONAL: a dataset
        # with no FKs at all (single table, or a Schema Analyzer that
        # simply found none) must still produce a valid, empty graph.
        graph: Dict[str, List[RelationshipMeta]] = {name: [] for name in self._tables}

        for rel in self._relationships:
            # Read relationship attributes defensively — tolerate a
            # RelationshipMeta-like object that's missing a field rather
            # than crashing the whole cache refresh over one bad edge.
            from_table = getattr(rel, "from_table", None)
            to_table = getattr(rel, "to_table", None)
            from_column = getattr(rel, "from_column", None)
            to_column = getattr(rel, "to_column", None)
            cardinality = getattr(rel, "cardinality", None)
            confidence = getattr(rel, "confidence", 1.0)

            if not from_table or not to_table:
                # Invalid/incomplete relationship — skip it, don't
                # fabricate a join and don't crash the whole cache.
                continue

            graph.setdefault(from_table, []).append(rel)

            reverse = RelationshipMeta(
                from_table=to_table,
                to_table=from_table,
                from_column=to_column,
                to_column=from_column,
                cardinality=cardinality,
                confidence=confidence,
            )
            graph.setdefault(to_table, []).append(reverse)

        # IMPORTANT: these two must run once per refresh, OUTSIDE the
        # relationship loop above. Nesting them inside the loop means
        # they silently never execute for any schema with zero
        # relationships (single-table datasets, or any Schema Analyzer
        # output where no FKs were detected) — join_graph and
        # table_tokens would stay stuck at their empty __init__
        # defaults forever, breaking table selection and entity
        # resolution for that entire class of dataset.
        self._join_graph = graph
        self._table_tokens = {
            name: table.semantic_tokens() | {
                tok for c in table.columns for tok in c.tokens()
            }
            for name, table in self._tables.items()
        }

        self._version = current_version
        logger.debug("SchemaContextCache refreshed (version=%s, tables=%d, relationships=%d)",
                     current_version, len(self._tables), len(self._relationships))

    @property
    def tables(self) -> Dict[str, TableMeta]:
        self._refresh_if_needed()
        return self._tables

    @property
    def relationships(self) -> List[RelationshipMeta]:
        self._refresh_if_needed()
        return self._relationships

    @property
    def join_graph(self) -> Dict[str, List[RelationshipMeta]]:
        self._refresh_if_needed()
        return self._join_graph

    def table_tokens(self, table_name: str) -> Set[str]:
        self._refresh_if_needed()
        return self._table_tokens.get(table_name, set())


# ============================================================
# ENTITY RESOLUTION
# ============================================================

@dataclass
class ResolvedEntity:
    entity_key: str
    entity_value: Any
    table: str
    column: ColumnMeta
    confidence: float


class EntityResolver:
    """
    Maps arbitrary extracted entities (whatever keys upstream NER/entity
    extraction produced — anything at all) to concrete schema columns,
    purely via semantic token overlap. The planner never special-cases
    any entity key.
    """

    def __init__(self, cache: SchemaContextCache, config: PlannerConfig):
        self.cache = cache
        self.config = config

    def resolve(self, entities: Dict[str, Any]) -> List[ResolvedEntity]:
        resolved: List[ResolvedEntity] = []
        for key, value in (entities or {}).items():
            key_tokens = expand_with_synonyms(split_identifier(str(key)))
            best: Optional[Tuple[float, str, ColumnMeta]] = None

            for table_name, table in self.cache.tables.items():
                for col in table.columns:
                    score = overlap_score(key_tokens, col.tokens())
                    if best is None or score > best[0]:
                        best = (score, table_name, col)

            if best and best[0] >= self.config.min_entity_match_score:
                resolved.append(
                    ResolvedEntity(key, value, best[1], best[2], round(best[0], 3))
                )
            else:
                logger.debug("EntityResolver: no confident column match for entity '%s'", key)

        return resolved


# ============================================================
# TABLE SELECTION
# ============================================================

@dataclass
class TableSelection:
    tables: List[str]
    reasons: Dict[str, str]


class TableSelectionEngine:
    """
    Scores every table by semantic relevance to the query + resolved
    entities + join connectivity + capability requirements (e.g. "needs
    a temporal column"), replacing keyword-to-table lookup tables.
    """

    def __init__(self, cache: SchemaContextCache, config: PlannerConfig):
        self.cache = cache
        self.config = config

    def select(
        self,
        query_tokens: Set[str],
        resolved_entities: List[ResolvedEntity],
        needs_temporal: bool,
    ) -> TableSelection:
        tables = self.cache.tables
        if not tables:
            return TableSelection([], {})

        entity_tables = {e.table for e in resolved_entities}
        scores: Dict[str, float] = {}
        reasons: Dict[str, str] = {}

        max_connectivity = max(
            (len(v) for v in self.cache.join_graph.values()), default=1
        ) or 1

        for name, table in tables.items():
            table_tokens = self.cache.table_tokens(name)
            name_score = overlap_score(query_tokens, table_tokens)
            entity_bonus = 0.5 if name in entity_tables else 0.0
            connectivity = len(self.cache.join_graph.get(name, [])) / max_connectivity
            temporal_bonus = (
                0.3 if needs_temporal and table.columns_with_role(ColumnRole.TEMPORAL) else 0.0
            )

            score = (
                name_score * 0.4
                + entity_bonus * 0.35
                + connectivity * 0.1
                + temporal_bonus * 0.15
            )
            scores[name] = score

            reason_bits = []
            if name_score > 0:
                reason_bits.append(f"token_overlap={name_score:.2f}")
            if entity_bonus:
                reason_bits.append("matched_resolved_entity")
            if temporal_bonus:
                reason_bits.append("has_temporal_column")
            reasons[name] = ", ".join(reason_bits) or "fallback"

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        selected = [name for name, score in ranked if score >= self.config.min_table_score]

        if not selected:
            # Nothing cleared the bar — degrade gracefully rather than
            # returning nothing: take the single best-connected table.
            best = max(tables, key=lambda n: len(self.cache.join_graph.get(n, [])))
            selected = [best]
            reasons[best] = reasons.get(best, "") + " (fallback: no table cleared relevance threshold)"

        return TableSelection(selected, reasons)


# ============================================================
# JOIN PLANNING
# ============================================================

@dataclass
class JoinPlan:
    edges: List[RelationshipMeta]
    reachable: bool


class JoinPlanner:
    """BFS shortest-path join planner over the registry's relationship graph.
    Works identically whether the graph has zero edges (single table /
    no relationships) or many — no special-casing required."""

    def __init__(self, cache: SchemaContextCache, config: PlannerConfig):
        self.cache = cache
        self.config = config

    def plan(self, tables: List[str]) -> JoinPlan:
        if len(tables) <= 1:
            return JoinPlan([], True)

        edges: List[RelationshipMeta] = []
        connected = {tables[0]}
        remaining = set(tables[1:])

        # Greedily connect each additional table via its shortest path
        # to the already-connected set — handles star, snowflake, and
        # bridge-table topologies without assuming any particular shape.
        while remaining:
            progressed = False
            for target in list(remaining):
                path = self._shortest_path(connected, target)
                if path is not None:
                    edges.extend(path)
                    connected.add(target)
                    remaining.discard(target)
                    progressed = True
            if not progressed:
                break  # unreachable tables — reported via `reachable`

        return JoinPlan(edges, reachable=not remaining)

    def _shortest_path(
        self, sources: Set[str], target: str
    ) -> Optional[List[RelationshipMeta]]:
        visited = set(sources)
        queue = deque([(s, []) for s in sources])
        hops = 0

        while queue and hops <= self.config.max_join_hops:
            for _ in range(len(queue)):
                node, path = queue.popleft()
                if node == target:
                    return path
                for rel in self.cache.join_graph.get(node, []):
                    if rel.to_table not in visited:
                        visited.add(rel.to_table)
                        queue.append((rel.to_table, path + [rel]))
            hops += 1

        return None


# ============================================================
# FILTER PLANNING
# ============================================================

_ROLE_OPERATORS: Dict[ColumnRole, List[FilterOperator]] = {
    ColumnRole.NUMERIC: [FilterOperator.EQ, FilterOperator.BETWEEN, FilterOperator.GTE, FilterOperator.LTE],
    ColumnRole.CATEGORICAL: [FilterOperator.EQ, FilterOperator.IN, FilterOperator.NOT_IN],
    ColumnRole.TEMPORAL: [FilterOperator.BETWEEN, FilterOperator.GTE, FilterOperator.LTE],
    ColumnRole.BOOLEAN: [FilterOperator.EQ],
    ColumnRole.TEXTUAL: [FilterOperator.LIKE, FilterOperator.ILIKE],
    ColumnRole.JSON: [FilterOperator.ARRAY_CONTAINS],
    ColumnRole.IDENTIFIER: [FilterOperator.EQ, FilterOperator.IN],
    ColumnRole.VECTOR_REFERENCE: [FilterOperator.EQ, FilterOperator.IN],
}


@dataclass
class PlannedFilter:
    table: str
    column: str
    operator: FilterOperator
    value: Any


class FilterPlanner:
    """
    Builds filters purely from resolved (entity -> column) mappings and
    each column's declared role/operators — never from a hardcoded
    column name. The role, not the name, decides which operator is
    valid for a given value shape.
    """

    def build(self, resolved_entities: List[ResolvedEntity]) -> List[PlannedFilter]:
        filters: List[PlannedFilter] = []

        for entity in resolved_entities:
            role = entity.column.role
            allowed_ops = _ROLE_OPERATORS.get(role, [FilterOperator.EQ])
            operator, value = self._infer_operator_and_value(entity.entity_value, role, allowed_ops)
            if operator is None:
                continue
            filters.append(PlannedFilter(entity.table, entity.column.name, operator, value))

        return filters

    def _infer_operator_and_value(
        self, value: Any, role: ColumnRole, allowed_ops: List[FilterOperator]
    ) -> Tuple[Optional[FilterOperator], Any]:
        if value is None:
            return (FilterOperator.IS_NULL if FilterOperator.IS_NULL in allowed_ops else None), None

        if isinstance(value, dict):
            # Range-style entity, e.g. {"gte": 10}, {"start": ..., "end": ...},
            # {"year": 2023} — normalized into a BETWEEN/GTE/LTE filter.
            if "start" in value or "end" in value:
                return FilterOperator.BETWEEN, (value.get("start"), value.get("end"))
            if "gte" in value or "lte" in value or "gt" in value or "lt" in value:
                if "gte" in value:
                    return FilterOperator.GTE, value["gte"]
                if "gt" in value:
                    return FilterOperator.GT, value["gt"]
                if "lte" in value:
                    return FilterOperator.LTE, value["lte"]
                return FilterOperator.LT, value["lt"]
            if "year" in value and role == ColumnRole.TEMPORAL:
                y = value["year"]
                return FilterOperator.BETWEEN, (f"{y}-01-01", f"{y}-12-31")
            return None, None

        if isinstance(value, (list, tuple, set)):
            op = FilterOperator.IN if FilterOperator.IN in allowed_ops else FilterOperator.EQ
            return op, list(value)

        if isinstance(value, (int, float)) and FilterOperator.EQ in allowed_ops:
            return FilterOperator.EQ, value

        # Plain scalar (typically string) — prefer exact match for
        # categorical/identifier columns, fuzzy match for free text.
        if role in (ColumnRole.CATEGORICAL, ColumnRole.IDENTIFIER, ColumnRole.BOOLEAN, ColumnRole.VECTOR_REFERENCE):
            return FilterOperator.EQ, value
        if role == ColumnRole.TEXTUAL:
            return FilterOperator.ILIKE, f"%{value}%"
        return FilterOperator.EQ, value


# ============================================================
# AGGREGATION PLANNING
# ============================================================

_AGGREGATION_SYNONYMS: Dict[AggregationOp, Set[str]] = {
    AggregationOp.COUNT: {"count", "how", "many", "number"},
    AggregationOp.SUM: {"total", "sum", "aggregate", "combined"},
    AggregationOp.AVG: {"average", "avg", "mean", "typical"},
    AggregationOp.MEDIAN: {"median", "middle"},
    AggregationOp.MODE: {"mode", "common", "frequent", "popular"},
    AggregationOp.MIN: {"minimum", "min", "smallest", "lowest"},
    AggregationOp.MAX: {"maximum", "max", "highest", "largest", "peak"},
    AggregationOp.STDDEV: {"stddev", "deviation", "spread"},
    AggregationOp.VARIANCE: {"variance"},
    AggregationOp.DISTINCT: {"distinct", "unique", "different"},
    AggregationOp.TOP_N: {"top", "best", "highest", "largest", "greatest", "most"},
    AggregationOp.BOTTOM_N: {"bottom", "worst", "lowest", "smallest", "least"},
    AggregationOp.RANK: {"rank", "ranking", "position"},
    AggregationOp.PERCENTILE: {"percentile", "quartile", "decile"},
    AggregationOp.MOVING_AVERAGE: {"moving", "rolling"},
    AggregationOp.RUNNING_TOTAL: {"running", "cumulative"},
    AggregationOp.DISTRIBUTION: {"distribution", "breakdown", "percentage", "percent", "share"},
    AggregationOp.COMPARE: {"compare", "comparison", "versus", "vs"},
    AggregationOp.TIME_SERIES: {"trend", "trends", "overtime", "series", "history", "growth"},
}

_GROUPING_HINTS = {"by", "per", "each", "group", "grouped", "across", "breakdown", "segment"}


@dataclass
class AggregationPlan:
    operations: List[Tuple[AggregationOp, float]]  # (op, confidence)
    wants_grouping: bool
    requested_n: Optional[int]


class AggregationPlanner:
    """Scores aggregation intents via synonym overlap instead of a fixed keyword map."""

    def plan(self, query_tokens: Set[str], raw_query: str) -> AggregationPlan:
        expanded = expand_with_synonyms(query_tokens)
        scored: List[Tuple[AggregationOp, float]] = []

        for op, synonyms in _AGGREGATION_SYNONYMS.items():
            score = overlap_score(expanded, synonyms)
            if query_tokens & synonyms:
                score += 0.25
            if score > 0:
                scored.append((op, round(min(score, 1.0), 3)))

        scored.sort(key=lambda pair: pair[1], reverse=True)

        wants_grouping = bool(query_tokens & _GROUPING_HINTS)
        n_match = _N_PATTERN.search((raw_query or "").lower())
        requested_n = None
        if n_match:
            for g in n_match.groups():
                if g is not None:
                    requested_n = int(g)
                    break

        return AggregationPlan(scored, wants_grouping, requested_n)


# ============================================================
# TEMPORAL PLANNING
# ============================================================

@dataclass
class TemporalPlan:
    table: Optional[str]
    column: Optional[str]
    granularity: Optional[str]


class TemporalPlanner:
    def __init__(self, cache: SchemaContextCache):
        self.cache = cache

    def plan(
        self, selected_tables: List[str], query_tokens: Set[str], requested: bool
    ) -> TemporalPlan:
        if not requested:
            return TemporalPlan(None, None, None)

        granularity = None
        for tok in query_tokens:
            if tok in _GRANULARITY_HINTS:
                granularity = _GRANULARITY_HINTS[tok]
                break

        for table_name in selected_tables:
            table = self.cache.tables.get(table_name)
            if not table:
                continue
            temporal_cols = table.columns_with_role(ColumnRole.TEMPORAL)
            if temporal_cols:
                return TemporalPlan(table_name, temporal_cols[0].name, granularity or "month")

        # Requested but genuinely unavailable — planner degrades
        # gracefully rather than fabricating a column.
        return TemporalPlan(None, None, granularity)


# ============================================================
# VECTOR PLANNING
# ============================================================

@dataclass
class VectorPlan:
    collection: Optional[str]
    enhanced_query: str
    payload_filters: Dict[str, Any]
    search_mode: str
    top_k: int


class VectorPlanner:
    def __init__(self, cache: SchemaContextCache, config: PlannerConfig):
        self.cache = cache
        self.config = config

    def plan(
        self,
        raw_query: str,
        query_tokens: Set[str],
        resolved_entities: List[ResolvedEntity],
        focus_mode: bool,
        top_k: int,
    ) -> VectorPlan:
        collections = self.cache.registry.get_vector_collections()
        if not collections:
            return VectorPlan(None, raw_query, {}, "dense", top_k)

        best_collection = self._select_collection(collections, query_tokens)

        payload_filters = self._build_payload_filters(best_collection, resolved_entities)

        if focus_mode:
            focus_column = self._find_focus_column(best_collection)
            if focus_column is not None:
                payload_filters[focus_column.name] = {"match": "negative"}

        enhanced_query = self._enhance_query(raw_query, resolved_entities, focus_mode)
        search_mode = "hybrid" if (payload_filters or resolved_entities) else "dense"

        return VectorPlan(
            collection=best_collection.name if best_collection else None,
            enhanced_query=enhanced_query,
            payload_filters=payload_filters,
            search_mode=search_mode,
            top_k=top_k,
        )

    def _select_collection(
        self, collections: List[VectorCollectionMeta], query_tokens: Set[str]
    ) -> Optional[VectorCollectionMeta]:
        if len(collections) == 1:
            return collections[0]

        def score(c: VectorCollectionMeta) -> float:
            name_tokens = expand_with_synonyms(split_identifier(c.name))
            field_tokens: Set[str] = set()
            for f in c.payload_fields:
                field_tokens |= f.tokens()
            return overlap_score(query_tokens, name_tokens | field_tokens)

        scored = [(c, score(c)) for c in collections]
        scored.sort(key=lambda pair: pair[1], reverse=True)

        best, best_score = scored[0]
        if best_score < self.config.min_vector_collection_score:
            # Below the configured relevance floor. Default floor is
            # 0.0 ("always pick the best available"), so this only
            # actually filters anything out if the config raises it —
            # in which case we still don't want to fail the whole plan,
            # just fall back to the top-ranked collection anyway rather
            # than returning None and silently dropping vector search.
            logger.debug(
                "VectorPlanner: best collection '%s' scored %.3f, below "
                "min_vector_collection_score=%.3f; using it anyway.",
                best.name, best_score, self.config.min_vector_collection_score,
            )
        return best

    def _build_payload_filters(
        self, collection: Optional[VectorCollectionMeta], resolved_entities: List[ResolvedEntity]
    ) -> Dict[str, Any]:
        if collection is None:
            return {}
        payload_names = {f.name for f in collection.payload_fields}
        filters: Dict[str, Any] = {}
        for entity in resolved_entities:
            if entity.column.name not in payload_names:
                continue
            value = entity.entity_value
            if isinstance(value, (list, tuple, set)):
                filters[entity.column.name] = {"any": list(value)}
            elif isinstance(value, dict):
                filters[entity.column.name] = value
            else:
                filters[entity.column.name] = {"match": value}
        return filters

    def _find_focus_column(self, collection: Optional[VectorCollectionMeta]) -> Optional[ColumnMeta]:
        if collection is None:
            return None
        focus_tokens = _SYNONYM_LOOKUP.get("negative", {"negative"})
        best: Optional[Tuple[float, ColumnMeta]] = None
        for f in collection.payload_fields:
            score = overlap_score(f.tokens(), focus_tokens)
            if score > 0 and (best is None or score > best[0]):
                best = (score, f)
        return best[1] if best else None

    def _enhance_query(
        self, raw_query: str, resolved_entities: List[ResolvedEntity], focus_mode: bool
    ) -> str:
        enhanced = raw_query or ""
        if focus_mode and "critical" not in enhanced.lower():
            enhanced = "critical or negative feedback focus: " + enhanced

        q_lower = enhanced.lower()
        injections = [
            str(e.entity_value) for e in resolved_entities
            if not isinstance(e.entity_value, (list, dict, set, tuple))
            and str(e.entity_value).lower() not in q_lower
        ]
        if injections:
            enhanced = enhanced.rstrip(". ") + " " + " ".join(injections)
        return enhanced.strip()


# ============================================================
# QUERY UNDERSTANDING (multi-intent scoring)
# ============================================================

class QueryIntent(str, Enum):
    LOOKUP = "lookup"
    AGGREGATION = "aggregation"
    COMPARISON = "comparison"
    SEMANTIC = "semantic"
    ANALYTICAL = "analytical"
    EXPLORATORY = "exploratory"
    RECOMMENDATION = "recommendation"
    TREND = "trend"
    QUESTION_ANSWERING = "question_answering"
    SUMMARIZATION = "summarization"
    FILTERING = "filtering"
    RANKING = "ranking"
    TEMPORAL = "temporal"


_QUERY_INTENT_SYNONYMS: Dict[QueryIntent, Set[str]] = {
    QueryIntent.LOOKUP: {"find", "get", "show", "lookup", "which", "what"},
    QueryIntent.AGGREGATION: {"count", "total", "average", "sum", "how", "many"},
    QueryIntent.COMPARISON: {"compare", "comparison", "versus", "vs", "difference"},
    QueryIntent.SEMANTIC: {"opinion", "feedback", "experience", "feel", "think", "say"},
    QueryIntent.ANALYTICAL: {"analyze", "analysis", "insight", "pattern", "correlation"},
    QueryIntent.EXPLORATORY: {"explore", "overview", "browse", "around"},
    QueryIntent.RECOMMENDATION: {"recommend", "suggest", "should", "best", "advice"},
    QueryIntent.TREND: {"trend", "trends", "overtime", "growth", "history"},
    QueryIntent.QUESTION_ANSWERING: {"why", "how", "when", "where", "who", "explain"},
    QueryIntent.SUMMARIZATION: {"summarize", "summary", "overview", "recap"},
    QueryIntent.FILTERING: {"only", "where", "filter", "exclude", "include"},
    QueryIntent.RANKING: {"top", "rank", "best", "worst", "highest", "lowest"},
    QueryIntent.TEMPORAL: set(_GRANULARITY_HINTS.keys()),
}


@dataclass
class QueryUnderstanding:
    intent_scores: Dict[QueryIntent, float]
    needs_temporal: bool
    tokens: Set[str]


class QueryUnderstandingEngine:
    def analyze(self, raw_query: str) -> QueryUnderstanding:
        tokens = set(tokenize(raw_query))
        expanded = expand_with_synonyms(tokens)
        scores: Dict[QueryIntent, float] = {}
        for intent, synonyms in _QUERY_INTENT_SYNONYMS.items():
            score = overlap_score(expanded, synonyms)
            if tokens & synonyms:
                score += 0.2
            if score > 0:
                scores[intent] = round(min(score, 1.0), 3)

        needs_temporal = bool(tokens & set(_GRANULARITY_HINTS.keys())) or QueryIntent.TREND in scores
        return QueryUnderstanding(scores, needs_temporal, tokens)


# ============================================================
# COST / COMPLEXITY ESTIMATION
# ============================================================

@dataclass
class CostEstimate:
    score: float
    breakdown: Dict[str, float]


class CostEstimator:
    def __init__(self, config: PlannerConfig):
        self.config = config

    def estimate(
        self,
        num_joins: int,
        num_aggregations: int,
        num_filters: int,
        vector_collection_size: Optional[int],
    ) -> CostEstimate:
        breakdown = {
            "joins": num_joins * self.config.cost_per_join,
            "aggregations": num_aggregations * self.config.cost_per_aggregation,
            "filters": num_filters * self.config.cost_per_filter,
        }
        if vector_collection_size:
            import math
            breakdown["vector_search"] = math.log10(max(vector_collection_size, 10)) * self.config.cost_per_vector_row_log
        else:
            breakdown["vector_search"] = 0.0

        return CostEstimate(round(sum(breakdown.values()), 3), breakdown)


class ComplexityEstimator:
    def __init__(self, config: PlannerConfig):
        self.config = config

    def estimate(
        self,
        num_entities: int,
        needs_time_series: bool,
        needs_focus: bool,
        is_hybrid: bool,
        num_active_intents: int,
    ) -> str:
        w = self.config.complexity_weights
        score = 0.0
        if num_entities > 1:
            score += w["multi_entity"]
        if needs_time_series:
            score += w["time_series"]
        if needs_focus:
            score += w["focus_mode"]
        if is_hybrid:
            score += w["hybrid_branch"]
        if num_active_intents >= 2:
            score += w["multi_intent"]

        if score <= self.config.complexity_simple_max:
            return "simple"
        if score <= self.config.complexity_medium_max:
            return "medium"
        return "complex"


# ============================================================
# STRATEGY SELECTION
# ============================================================

class StrategySelector:
    """Resolves the execution branch, degrading gracefully when a
    requested capability isn't actually available in the schema."""

    @staticmethod
    def resolve(
        needs_sql: bool,
        needs_vector: bool,
        sql_feasible: bool,
        vector_feasible: bool,
    ) -> Tuple[str, List[str]]:
        notes: List[str] = []

        want_sql = needs_sql and sql_feasible
        want_vector = needs_vector and vector_feasible

        if needs_sql and not sql_feasible:
            notes.append("SQL requested but no tables found in schema — dropped from plan")
        if needs_vector and not vector_feasible:
            notes.append("Vector search requested but no vector collection found — dropped from plan")

        if want_sql and want_vector:
            return "hybrid", notes
        if want_sql:
            return "sql", notes
        if want_vector:
            return "qdrant", notes

        # Nothing feasible — fall back to whichever capability actually
        # exists in the schema, rather than crashing.
        if sql_feasible:
            notes.append("No routing flags satisfiable — falling back to sql")
            return "sql", notes
        if vector_feasible:
            notes.append("No routing flags satisfiable — falling back to qdrant")
            return "qdrant", notes

        notes.append("No SQL tables or vector collections available in schema")
        return "qdrant", notes


# ============================================================
# DATA CONTRACTS (public, backward-compatible shapes)
# ============================================================

@dataclass
class IntentResult:
    query_type: str
    needs_sql: bool
    needs_vector: bool
    needs_time_series: bool
    needs_complaint_focus: bool
    normalized_query: str
    entities: Dict = field(default_factory=dict)


@dataclass
class SQLTask:
    tables: List[str]
    filters: Dict
    aggregations: List[str]
    time_series: bool
    prompt_hint: str
    schema_snippet: str


@dataclass
class VectorTask:
    enhanced_query: str
    qdrant_filters: Dict
    top_k: int
    complaint_focus: bool
    search_mode: str


@dataclass
class RetrievalPlan:
    branch: str
    sql_task: Optional[SQLTask]
    vector_task: Optional[VectorTask]
    execution_mode: str
    estimated_complexity: str
    query_type: str
    explanation: List[str] = field(default_factory=list)


# ============================================================
# PROMPT / SCHEMA CONTEXT BUILDER
# ============================================================

class PromptContextBuilder:
    """Builds the schema snippet and natural-language prompt hint for the
    SQL path entirely from discovered TableMeta — no static text tables."""

    def __init__(self, cache: SchemaContextCache):
        self.cache = cache

    def schema_snippet(self, tables: List[str], joins: List[RelationshipMeta]) -> str:
        blocks = []
        for name in tables:
            table = self.cache.tables.get(name)
            if not table:
                continue
            col_lines = "\n".join(
                f"  {c.name:<24s} {c.role.value.upper()}" for c in table.columns
            )
            blocks.append(f"TABLE {name} (\n{col_lines}\n)")

        if joins:
            join_lines = "\n".join(
                f"-- JOIN {j.from_table}.{j.from_column} = {j.to_table}.{j.to_column} "
                f"({j.cardinality}, confidence={j.confidence})"
                for j in joins
            )
            blocks.append(join_lines)

        return "\n\n".join(blocks)

    def sql_prompt_hint(
        self,
        joins: List[RelationshipMeta],
        filters: List[PlannedFilter],
        aggregation_plan: AggregationPlan,
        temporal_plan: TemporalPlan,
    ) -> str:
        parts: List[str] = []

        for j in joins:
            parts.append(f"Join {j.from_table} to {j.to_table} on {j.from_column} = {j.to_column}.")

        for f in filters:
            parts.append(self._describe_filter(f))

        top_ops = [op for op, score in aggregation_plan.operations if score >= 0.2]
        if AggregationOp.TOP_N in top_ops or AggregationOp.RANK in top_ops:
            n = aggregation_plan.requested_n or 10
            parts.append(f"Use ORDER BY … DESC LIMIT {n}.")
        if AggregationOp.BOTTOM_N in top_ops:
            n = aggregation_plan.requested_n or 10
            parts.append(f"Use ORDER BY … ASC LIMIT {n}.")
        if AggregationOp.DISTRIBUTION in top_ops:
            parts.append("Use GROUP BY with COUNT(*) and a percentage-of-total window function.")
        if AggregationOp.AVG in top_ops:
            parts.append("Use AVG() over the relevant numeric column.")
        if AggregationOp.COMPARE in top_ops:
            parts.append("GROUP BY the comparison dimension.")
        if aggregation_plan.wants_grouping and not any(
            op in top_ops for op in (AggregationOp.DISTRIBUTION, AggregationOp.COMPARE)
        ):
            parts.append("Group results by the most relevant categorical column.")

        if temporal_plan.column:
            parts.append(
                f"Bucket {temporal_plan.column} by {temporal_plan.granularity or 'month'} "
                f"and order chronologically."
            )

        return "  ".join(parts) if parts else "Generate a relevant SQL query."

    @staticmethod
    def _describe_filter(f: PlannedFilter) -> str:
        op_text = {
            FilterOperator.EQ: "=", FilterOperator.NEQ: "!=",
            FilterOperator.GT: ">", FilterOperator.GTE: ">=",
            FilterOperator.LT: "<", FilterOperator.LTE: "<=",
        }
        if f.operator == FilterOperator.IN:
            return f"Filter {f.column} IN {tuple(f.value)}."
        if f.operator == FilterOperator.BETWEEN:
            return f"Filter {f.column} BETWEEN {f.value[0]} AND {f.value[1]}."
        if f.operator in (FilterOperator.LIKE, FilterOperator.ILIKE):
            return f"Filter {f.column} {f.operator.value.upper()} '{f.value}'."
        if f.operator in op_text:
            return f"Filter {f.column} {op_text[f.operator]} {f.value!r}."
        return f"Filter {f.column} {f.operator.value}."


# ============================================================
# MAIN ORCHESTRATOR
# ============================================================

class RetrievalPlanBuilder:
    """
    Converts an IntentResult into a fully-specified RetrievalPlan, using
    only what the injected SchemaRegistry exposes. Zero dataset-specific
    knowledge lives in this class or anywhere it depends on. The exact
    same code path handles Amazon-reviews-shaped schemas, medical
    schemas, legal schemas, HR schemas, financial schemas, support
    tickets, or anything else — the only input that varies is what the
    Schema Analyzer put into the registry.
    """

    def __init__(self, schema_registry: SchemaRegistry, config: Optional[PlannerConfig] = None):
        self.registry = schema_registry
        self.config = config or DEFAULT_CONFIG

        self.cache = SchemaContextCache(schema_registry)
        self.entity_resolver = EntityResolver(self.cache, self.config)
        self.table_selector = TableSelectionEngine(self.cache, self.config)
        self.join_planner = JoinPlanner(self.cache, self.config)
        self.filter_planner = FilterPlanner()
        self.aggregation_planner = AggregationPlanner()
        self.temporal_planner = TemporalPlanner(self.cache)
        self.vector_planner = VectorPlanner(self.cache, self.config)
        self.query_engine = QueryUnderstandingEngine()
        self.cost_estimator = CostEstimator(self.config)
        self.complexity_estimator = ComplexityEstimator(self.config)
        self.prompt_builder = PromptContextBuilder(self.cache)

        logger.debug("RetrievalPlanBuilder ready (schema_version=%s).", getattr(schema_registry, "version", "unknown"))

    # ==========================================================
    # PUBLIC API
    # ==========================================================

    def build(self, intent: IntentResult) -> RetrievalPlan:
        explanation: List[str] = []
        query = intent.normalized_query or ""

        understanding = self.query_engine.analyze(query)
        query_tokens = understanding.tokens
        resolved_entities = self.entity_resolver.resolve(intent.entities or {})
        explanation.append(
            f"Resolved {len(resolved_entities)}/{len(intent.entities or {})} entities to schema columns."
        )

        needs_temporal = intent.needs_time_series or understanding.needs_temporal

        sql_feasible = bool(self.cache.tables)
        vector_feasible = bool(self.registry.get_vector_collections())

        branch, strategy_notes = StrategySelector.resolve(
            intent.needs_sql, intent.needs_vector, sql_feasible, vector_feasible
        )
        explanation.extend(strategy_notes)
        explanation.append(f"Branch resolved to '{branch}'.")

        sql_task: Optional[SQLTask] = None
        vector_task: Optional[VectorTask] = None
        num_joins = num_aggregations = num_filters = 0
        vector_collection_size = None

        if branch in ("sql", "hybrid"):
            table_selection = self.table_selector.select(query_tokens, resolved_entities, needs_temporal)
            explanation.append(
                f"Selected tables: {table_selection.tables} "
                f"({'; '.join(f'{t}: {r}' for t, r in table_selection.reasons.items() if t in table_selection.tables)})."
            )

            join_plan = self.join_planner.plan(table_selection.tables)
            if not join_plan.reachable:
                explanation.append("Warning: not all selected tables could be joined within max hop limit.")

            filters = self.filter_planner.build(
                [e for e in resolved_entities if e.table in table_selection.tables]
            )
            aggregation_plan = self.aggregation_planner.plan(query_tokens, query)
            temporal_plan = self.temporal_planner.plan(table_selection.tables, query_tokens, needs_temporal)

            num_joins = len(join_plan.edges)
            num_aggregations = len(aggregation_plan.operations)
            num_filters = len(filters)

            sql_task = SQLTask(
                tables=table_selection.tables,
                filters=self._filters_to_dict(filters),
                aggregations=[op.value for op, score in aggregation_plan.operations if score >= 0.2] or ["SELECT"],
                time_series=bool(temporal_plan.column),
                prompt_hint=self.prompt_builder.sql_prompt_hint(
                    join_plan.edges, filters, aggregation_plan, temporal_plan
                ),
                schema_snippet=self.prompt_builder.schema_snippet(table_selection.tables, join_plan.edges),
            )

        if branch in ("qdrant", "hybrid"):
            complexity_hint = self.complexity_estimator.estimate(
                len(resolved_entities), needs_temporal, intent.needs_complaint_focus,
                branch == "hybrid", len([s for s in understanding.intent_scores.values() if s >= 0.2]),
            )
            top_k = self.config.top_k_by_complexity.get(complexity_hint, 10)

            vector_plan = self.vector_planner.plan(
                query, query_tokens, resolved_entities, intent.needs_complaint_focus, top_k
            )
            vector_collection_size = self._collection_size(vector_plan.collection)

            vector_task = VectorTask(
                enhanced_query=vector_plan.enhanced_query,
                qdrant_filters=vector_plan.payload_filters,
                top_k=vector_plan.top_k,
                complaint_focus=intent.needs_complaint_focus,
                search_mode=vector_plan.search_mode,
            )
            explanation.append(
                f"Vector collection '{vector_plan.collection}' selected, mode={vector_plan.search_mode}."
            )

        active_intents = len([s for s in understanding.intent_scores.values() if s >= 0.2])
        complexity = self.complexity_estimator.estimate(
            len(resolved_entities), needs_temporal, intent.needs_complaint_focus,
            branch == "hybrid", active_intents,
        )

        cost = self.cost_estimator.estimate(num_joins, num_aggregations, num_filters, vector_collection_size)
        explanation.append(f"Estimated relative cost: {cost.score} ({cost.breakdown}).")

        execution_mode = "parallel" if branch == "hybrid" else "sequential"

        plan = RetrievalPlan(
            branch=branch,
            sql_task=sql_task,
            vector_task=vector_task,
            execution_mode=execution_mode,
            estimated_complexity=complexity,
            query_type=intent.query_type,
            explanation=explanation,
        )

        logger.debug(
            "RetrievalPlan | branch=%s | mode=%s | complexity=%s",
            plan.branch, plan.execution_mode, plan.estimated_complexity,
        )
        return plan

    # ==========================================================
    # HELPERS
    # ==========================================================

    @staticmethod
    def _filters_to_dict(filters: List[PlannedFilter]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for f in filters:
            if f.operator == FilterOperator.EQ:
                out[f.column] = f.value
            elif f.operator == FilterOperator.IN:
                out[f.column] = {"in": f.value}
            elif f.operator == FilterOperator.BETWEEN:
                out[f.column] = {"between": f.value}
            else:
                out[f.column] = {f.operator.value: f.value}
        return out

    def _collection_size(self, collection_name: Optional[str]) -> Optional[int]:
        if collection_name is None:
            return None
        for c in self.registry.get_vector_collections():
            if c.name == collection_name:
                return c.approx_size
        return None

    @staticmethod
    def plan_summary(plan: RetrievalPlan) -> str:
        parts = [
            f"branch={plan.branch}",
            f"mode={plan.execution_mode}",
            f"complexity={plan.estimated_complexity}",
        ]
        if plan.sql_task is not None:
            parts.append(f"tables={plan.sql_task.tables}")
            parts.append(f"aggs={plan.sql_task.aggregations}")
        if plan.vector_task is not None:
            parts.append(f"top_k={plan.vector_task.top_k}")
            parts.append(f"search={plan.vector_task.search_mode}")
        return " | ".join(parts)

    @staticmethod
    def extract_numeric_range_from_query(query: str) -> Optional[Dict[str, float]]:
        """
        Generic "above/below/at least <number>" phrasing recognizer,
        used only to help normalize free-text ranges into filter values.
        Makes no assumption about what the number refers to.
        """
        m = re.search(
            r"above\s+(\d+(?:\.\d+)?)|below\s+(\d+(?:\.\d+)?)|"
            r"(?:at least|minimum|min)\s+(\d+(?:\.\d+)?)|"
            r"(?:at most|maximum|max)\s+(\d+(?:\.\d+)?)",
            query or "", re.IGNORECASE,
        )
        if not m:
            return None
        above, below, at_least, at_most = m.groups()
        if above or at_least:
            return {"gte": float(above or at_least)}
        if below or at_most:
            return {"lte": float(below or at_most)}
        return None
