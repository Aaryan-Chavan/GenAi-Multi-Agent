# api/models.py

"""
==============================================================
API Models

Generic, reusable, dataset-independent Pydantic models
for Hybrid Retrieval-Augmented Generation (Hybrid RAG) APIs.

Author : Your Project
Python : 3.11+
Pydantic : v2+
==============================================================
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)


# ==========================================================
# ENUMS
# ==========================================================


class RetrievalMode(str, Enum):
    """Retrieval strategy selected by the system."""

    STRUCTURED = "structured"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"
    AUTO = "auto"


class QueryType(str, Enum):
    """High-level intent classification."""

    QUESTION = "question"
    ANALYTICS = "analytics"
    COMPARISON = "comparison"
    SUMMARY = "summary"
    SENTIMENT = "sentiment"
    KEYWORD = "keyword"
    FILTER = "filter"
    UNKNOWN = "unknown"


class ResponseStatus(str, Enum):
    """Standard API response status."""

    SUCCESS = "success"
    ERROR = "error"
    PARTIAL = "partial"


class SortOrder(str, Enum):
    """Generic sorting order."""

    ASC = "asc"
    DESC = "desc"


class HealthStatus(str, Enum):
    """Health state of API/service."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ComponentStatus(str, Enum):
    """Status of an individual service."""

    ONLINE = "online"
    OFFLINE = "offline"
    UNKNOWN = "unknown"


class FilterOperator(str, Enum):
    """
    Allowed operators for generic filter conditions.
    Keeps FilterCondition dataset-independent while still
    validating that downstream query builders receive a
    known, safe operator instead of an arbitrary string.
    """

    EQ = "eq"
    NEQ = "neq"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    IN = "in"
    NOT_IN = "not_in"
    CONTAINS = "contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    IS_NULL = "is_null"
    IS_NOT_NULL = "is_not_null"


# ==========================================================
# BASE MODEL
# ==========================================================


class APIBaseModel(BaseModel):
    """
    Base model inherited by every API schema.

    NOTE: `validate_default=True` is required alongside
    `use_enum_values=True`. Without it, enum-typed fields that
    fall back to their class default (e.g. `status: ResponseStatus =
    ResponseStatus.ERROR`) are never re-validated, so they stay as
    Enum members while user-supplied values get coerced to plain
    strings. That mismatch is silent and easy to miss in tests.
    Setting validate_default=True guarantees every instance is
    consistent, whether the value came from a default or from input.
    """

    model_config = ConfigDict(
        extra="ignore",
        validate_assignment=True,
        validate_default=True,
        populate_by_name=True,
        use_enum_values=True,
        frozen=False,
    )


def _utc_now() -> datetime:
    """
    Timezone-aware UTC timestamp factory.

    `datetime.utcnow()` is deprecated as of Python 3.12 and returns a
    naive datetime, which silently misbehaves when compared against or
    serialized alongside timezone-aware datetimes. This helper is used
    everywhere a timestamp default is needed.
    """
    return datetime.now(timezone.utc)


# ==========================================================
# STANDARD API METADATA
# ==========================================================


class ResponseMetadata(APIBaseModel):
    """Common metadata returned with every response."""

    request_id: UUID = Field(
        default_factory=uuid4,
        description="Unique request identifier."
    )

    timestamp: datetime = Field(
        default_factory=_utc_now,
        description="UTC timestamp."
    )

    processing_time: Optional[float] = Field(
        default=None,
        ge=0,
        description="Processing time in seconds."
    )

    api_version: Optional[str] = Field(
        default="1.0.0",
        description="API version."
    )


# ==========================================================
# PAGINATION
# ==========================================================


class Pagination(APIBaseModel):
    """Generic pagination information."""

    page: int = Field(default=1, ge=1)

    page_size: int = Field(default=25, ge=1, le=1000)

    total_records: int = Field(default=0, ge=0)

    total_pages: int = Field(default=0, ge=0)


# ==========================================================
# FILTER MODEL
# ==========================================================


class FilterCondition(APIBaseModel):
    """
    Generic filtering condition.

    `operator` is now a validated `FilterOperator` enum rather than a
    bare string, so downstream query builders never have to
    defensively re-check for garbage/unsupported operators.
    """

    field: str = Field(..., min_length=1)

    operator: FilterOperator = FilterOperator.EQ

    value: Any = None

    @field_validator("value")
    @classmethod
    def _validate_value_for_null_ops(cls, v: Any, info) -> Any:
        operator = info.data.get("operator")
        if operator in (FilterOperator.IS_NULL, FilterOperator.IS_NOT_NULL):
            # value is irrelevant for null checks; normalize to None
            return None
        return v


# ==========================================================
# SORT MODEL
# ==========================================================


class SortCondition(APIBaseModel):
    """Generic sorting configuration."""

    field: str = Field(..., min_length=1)

    order: SortOrder = SortOrder.ASC


# ==========================================================
# ERROR MODELS
# ==========================================================


class ErrorDetail(APIBaseModel):
    """Detailed error information."""

    code: str

    message: str

    field: Optional[str] = None

    details: Optional[Dict[str, Any]] = None


class ErrorResponse(APIBaseModel):
    """Generic API error response."""

    status: ResponseStatus = ResponseStatus.ERROR

    error: ErrorDetail

    metadata: ResponseMetadata = Field(default_factory=ResponseMetadata)


# ==========================================================
# SUCCESS RESPONSE BASE
# ==========================================================


class SuccessResponse(APIBaseModel):
    """Generic success response."""

    status: ResponseStatus = ResponseStatus.SUCCESS

    message: str = "Operation completed successfully."

    metadata: ResponseMetadata = Field(default_factory=ResponseMetadata)


# ==========================================================
# GENERIC KEY-VALUE MODEL
# ==========================================================


class KeyValue(APIBaseModel):
    """Generic key-value pair."""

    key: str

    value: Any


# ==========================================================
# GENERIC STATISTICS MODEL
# ==========================================================


class Statistics(APIBaseModel):
    """Generic statistics container."""

    count: int = 0

    minimum: Optional[float] = None

    maximum: Optional[float] = None

    average: Optional[float] = None

    median: Optional[float] = None

    std_dev: Optional[float] = Field(default=None, ge=0)


# ==========================================================
# GENERIC MESSAGE MODEL
# ==========================================================


class Message(APIBaseModel):
    """Generic informational message."""

    level: str = "info"

    text: str


# ==========================================================
# SOURCE & RETRIEVAL MODELS
# ==========================================================


class SourceMetadata(APIBaseModel):
    """Metadata associated with a retrieved document."""

    source_id: Optional[str] = None

    document_id: Optional[str] = None

    title: Optional[str] = None

    dataset: Optional[str] = None

    collection: Optional[str] = None

    file_name: Optional[str] = None

    page: Optional[int] = Field(default=None, ge=0)

    chunk_id: Optional[int] = Field(default=None, ge=0)

    row_id: Optional[int] = Field(default=None, ge=0)

    language: Optional[str] = None

    additional_metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievedDocument(APIBaseModel):
    """Generic retrieved document."""

    content: str

    score: Optional[float] = Field(default=None, ge=0.0)

    retrieval_mode: Optional[RetrievalMode] = None

    metadata: SourceMetadata = Field(default_factory=SourceMetadata)


class Citation(APIBaseModel):
    """
    Citation information returned with answers.

    `confidence` is now bounded [0, 1], matching the same convention
    used by `ConfidenceScore.score` elsewhere in this file. Previously
    the two fields had inconsistent validation rigor.
    """

    index: int = Field(..., ge=0)

    title: Optional[str] = None

    source: Optional[str] = None

    page: Optional[int] = Field(default=None, ge=0)

    chunk_id: Optional[int] = Field(default=None, ge=0)

    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class RetrievalMetadata(APIBaseModel):
    """Retrieval execution metadata."""

    retrieval_mode: RetrievalMode = RetrievalMode.AUTO

    query_type: Optional[QueryType] = None

    retrieved_documents: int = Field(default=0, ge=0)

    returned_documents: int = Field(default=0, ge=0)

    top_k: Optional[int] = Field(default=None, ge=1)

    reranked: bool = False

    compressed: bool = False


# ==========================================================
# TOKEN / LATENCY / CONFIDENCE
# ==========================================================


class TokenUsage(APIBaseModel):
    """LLM token usage."""

    prompt_tokens: int = Field(default=0, ge=0)

    completion_tokens: int = Field(default=0, ge=0)

    total_tokens: int = Field(default=0, ge=0)

    @field_validator("total_tokens")
    @classmethod
    def _default_total(cls, v: int, info) -> int:
        # If caller didn't set total explicitly (left at 0) but supplied
        # prompt/completion tokens, derive it instead of silently
        # returning an inconsistent 0.
        if v == 0:
            prompt = info.data.get("prompt_tokens", 0) or 0
            completion = info.data.get("completion_tokens", 0) or 0
            derived = prompt + completion
            if derived:
                return derived
        return v


class LatencyInfo(APIBaseModel):
    """Execution timing."""

    retrieval_seconds: Optional[float] = Field(default=None, ge=0)

    llm_seconds: Optional[float] = Field(default=None, ge=0)

    total_seconds: Optional[float] = Field(default=None, ge=0)


class ConfidenceScore(APIBaseModel):
    """Confidence estimation."""

    score: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    label: Optional[str] = None


# ==========================================================
# QUERY REQUEST
# ==========================================================


class QueryRequest(APIBaseModel):
    """Generic user query request."""

    query: str = Field(..., min_length=1, description="Natural language query.")

    retrieval_mode: RetrievalMode = Field(default=RetrievalMode.AUTO)

    top_k: Optional[int] = Field(default=None, ge=1)

    filters: List[FilterCondition] = Field(default_factory=list)

    sort: List[SortCondition] = Field(default_factory=list)

    conversation_id: Optional[str] = None

    user_id: Optional[str] = None

    include_sources: bool = True

    include_citations: bool = True

    include_metadata: bool = False

    stream: bool = False

    additional_parameters: Dict[str, Any] = Field(default_factory=dict)


# ==========================================================
# QUERY RESPONSE
# ==========================================================


class QueryResponse(APIBaseModel):
    """Generic query response."""

    status: ResponseStatus = ResponseStatus.SUCCESS

    answer: str

    retrieval: RetrievalMetadata = Field(default_factory=RetrievalMetadata)

    confidence: Optional[ConfidenceScore] = None

    latency: Optional[LatencyInfo] = None

    token_usage: Optional[TokenUsage] = None

    retrieved_documents: List[RetrievedDocument] = Field(default_factory=list)

    citations: List[Citation] = Field(default_factory=list)

    metadata: ResponseMetadata = Field(default_factory=ResponseMetadata)


# ==========================================================
# BATCH QUERY MODELS
# ==========================================================


class BatchQueryRequest(APIBaseModel):
    """Multiple query request."""

    queries: List[QueryRequest] = Field(..., min_length=1)


class BatchQueryResponse(APIBaseModel):
    """Multiple query response."""

    status: ResponseStatus = ResponseStatus.SUCCESS

    results: List[QueryResponse] = Field(default_factory=list)

    metadata: ResponseMetadata = Field(default_factory=ResponseMetadata)


# ==========================================================
# HEALTH CHECK MODELS
# ==========================================================


class ServiceHealth(APIBaseModel):
    """Individual service health."""

    name: str

    status: ComponentStatus = ComponentStatus.UNKNOWN

    message: Optional[str] = None

    latency_ms: Optional[float] = Field(default=None, ge=0)


class HealthResponse(APIBaseModel):
    """API health response."""

    status: HealthStatus = HealthStatus.HEALTHY

    services: List[ServiceHealth] = Field(default_factory=list)

    metadata: ResponseMetadata = Field(default_factory=ResponseMetadata)


# ==========================================================
# SYSTEM INFORMATION
# ==========================================================


class SystemInfoResponse(APIBaseModel):
    """Runtime system information."""

    application_name: str

    version: str

    model_name: Optional[str] = None

    embedding_model: Optional[str] = None

    retrieval_mode: RetrievalMode = RetrievalMode.AUTO

    uptime_seconds: Optional[float] = Field(default=None, ge=0)

    python_version: Optional[str] = None

    platform: Optional[str] = None

    metadata: ResponseMetadata = Field(default_factory=ResponseMetadata)


# ==========================================================
# CACHE MODELS
# ==========================================================


class CacheStatistics(APIBaseModel):
    """Cache statistics."""

    backend: Optional[str] = None

    total_keys: int = Field(default=0, ge=0)

    memory_usage_mb: Optional[float] = Field(default=None, ge=0)

    hit_rate: Optional[float] = Field(default=None, ge=0, le=1)


class CacheClearResponse(APIBaseModel):
    """Cache clear response."""

    status: ResponseStatus = ResponseStatus.SUCCESS

    cleared: bool = True

    message: str = "Cache cleared successfully."

    statistics: Optional[CacheStatistics] = None

    metadata: ResponseMetadata = Field(default_factory=ResponseMetadata)


# ==========================================================
# METRICS MODELS
# ==========================================================


class MetricsResponse(APIBaseModel):
    """Runtime metrics."""

    total_requests: int = Field(default=0, ge=0)

    successful_requests: int = Field(default=0, ge=0)

    failed_requests: int = Field(default=0, ge=0)

    average_latency: Optional[float] = Field(default=None, ge=0)

    average_confidence: Optional[float] = Field(default=None, ge=0, le=1)

    average_retrieval_time: Optional[float] = Field(default=None, ge=0)

    average_generation_time: Optional[float] = Field(default=None, ge=0)

    metadata: ResponseMetadata = Field(default_factory=ResponseMetadata)

    @field_validator("failed_requests")
    @classmethod
    def _sanity_check_counts(cls, v: int, info) -> int:
        total = info.data.get("total_requests", 0) or 0
        successful = info.data.get("successful_requests", 0) or 0
        if total and (successful + v) > total:
            raise ValueError(
                "successful_requests + failed_requests cannot exceed total_requests"
            )
        return v


# ==========================================================
# BENCHMARK MODELS
# ==========================================================


class BenchmarkRequest(APIBaseModel):
    """Benchmark execution request."""

    sample_size: Optional[int] = Field(default=None, ge=1)

    retrieval_mode: RetrievalMode = RetrievalMode.AUTO

    run_accuracy: bool = True

    run_latency: bool = True

    additional_parameters: Dict[str, Any] = Field(default_factory=dict)


class BenchmarkResult(APIBaseModel):
    """Single benchmark result."""

    metric: str

    value: float

    unit: str

    passed: bool


class BenchmarkResponse(APIBaseModel):
    """Benchmark execution response."""

    status: ResponseStatus = ResponseStatus.SUCCESS

    benchmark_name: str

    results: List[BenchmarkResult] = Field(default_factory=list)

    metadata: ResponseMetadata = Field(default_factory=ResponseMetadata)


# ==========================================================
# EVALUATION MODELS
# ==========================================================


class EvaluationRequest(APIBaseModel):
    """LLM evaluation request."""

    question: str

    generated_answer: str

    reference_answer: Optional[str] = None

    retrieved_documents: List[RetrievedDocument] = Field(default_factory=list)


class EvaluationScore(APIBaseModel):
    """Evaluation metric."""

    metric: str

    score: float = Field(ge=0, le=1)


class EvaluationResponse(APIBaseModel):
    """Evaluation response."""

    status: ResponseStatus = ResponseStatus.SUCCESS

    overall_score: float = Field(ge=0, le=1)

    metrics: List[EvaluationScore] = Field(default_factory=list)

    metadata: ResponseMetadata = Field(default_factory=ResponseMetadata)


# ==========================================================
# RELOAD MODELS
# ==========================================================


class ReloadResponse(APIBaseModel):
    """Dataset/model reload response."""

    status: ResponseStatus = ResponseStatus.SUCCESS

    reloaded: bool = True

    components: List[str] = Field(default_factory=list)

    metadata: ResponseMetadata = Field(default_factory=ResponseMetadata)


# ==========================================================
# CONVERSATION MODELS
# ==========================================================


class ConversationRole(str, Enum):
    """
    Allowed roles in a conversation message.
    Previously `ConversationMessage.role` was a bare string; any typo
    (e.g. "assitant") would pass validation silently.
    """

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ConversationMessage(APIBaseModel):
    """Single conversation message."""

    role: ConversationRole

    content: str = Field(..., min_length=1)

    timestamp: datetime = Field(default_factory=_utc_now)


class ConversationRequest(APIBaseModel):
    """Multi-turn conversation request."""

    conversation_id: Optional[str] = None

    messages: List[ConversationMessage] = Field(..., min_length=1)

    retrieval_mode: RetrievalMode = RetrievalMode.AUTO

    stream: bool = False

    additional_parameters: Dict[str, Any] = Field(default_factory=dict)


class ConversationResponse(APIBaseModel):
    """Multi-turn conversation response."""

    status: ResponseStatus = ResponseStatus.SUCCESS

    conversation_id: str

    answer: str

    retrieved_documents: List[RetrievedDocument] = Field(default_factory=list)

    citations: List[Citation] = Field(default_factory=list)

    metadata: ResponseMetadata = Field(default_factory=ResponseMetadata)


# ==========================================================
# STREAMING MODELS
# ==========================================================


class StreamChunk(APIBaseModel):
    """Single streamed response chunk."""

    chunk_id: int = Field(..., ge=0)

    text: str

    finished: bool = False


class StreamResponse(APIBaseModel):
    """Streaming response."""

    status: ResponseStatus = ResponseStatus.SUCCESS

    chunks: List[StreamChunk] = Field(default_factory=list)

    metadata: ResponseMetadata = Field(default_factory=ResponseMetadata)


# ==========================================================
# USER FEEDBACK
# ==========================================================


class FeedbackRequest(APIBaseModel):
    """User feedback."""

    request_id: Optional[str] = None

    rating: int = Field(ge=1, le=5)

    feedback: Optional[str] = None

    metadata: Dict[str, Any] = Field(default_factory=dict)


class FeedbackResponse(APIBaseModel):
    """Feedback acknowledgement."""

    status: ResponseStatus = ResponseStatus.SUCCESS

    stored: bool = True

    message: str = "Feedback recorded."

    metadata: ResponseMetadata = Field(default_factory=ResponseMetadata)


# ==========================================================
# GENERIC DATA MODELS
# ==========================================================


class GenericRecord(APIBaseModel):
    """Dataset-independent record."""

    data: Dict[str, Any] = Field(default_factory=dict)


class DatasetResponse(APIBaseModel):
    """Generic dataset response."""

    status: ResponseStatus = ResponseStatus.SUCCESS

    records: List[GenericRecord] = Field(default_factory=list)

    pagination: Optional[Pagination] = None

    metadata: ResponseMetadata = Field(default_factory=ResponseMetadata)


# ==========================================================
# SEARCH REQUEST
# ==========================================================


class SearchRequest(APIBaseModel):
    """Generic semantic search."""

    query: str = Field(..., min_length=1)

    top_k: int = Field(default=10, ge=1)

    filters: List[FilterCondition] = Field(default_factory=list)

    retrieval_mode: RetrievalMode = RetrievalMode.SEMANTIC


class SearchResponse(APIBaseModel):
    """Search response."""

    status: ResponseStatus = ResponseStatus.SUCCESS

    documents: List[RetrievedDocument] = Field(default_factory=list)

    metadata: ResponseMetadata = Field(default_factory=ResponseMetadata)


# ==========================================================
# STRUCTURED QUERY MODELS
# ==========================================================


class QueryLanguage(str, Enum):
    """
    Query language/dialect for a structured backend.
    Generalizes the old SQL-only assumption so this schema layer
    doesn't hardcode a relational backend.
    """

    SQL = "sql"
    NOSQL = "nosql"
    DSL = "dsl"          # e.g. Elasticsearch/OpenSearch query DSL
    GRAPHQL = "graphql"


class StructuredQueryRequest(APIBaseModel):
    """
    Generic structured query execution request.
    Backend-agnostic replacement/superset of the old SQLQueryRequest.
    """

    query: str = Field(..., min_length=1)

    query_language: QueryLanguage = QueryLanguage.SQL

    limit: Optional[int] = Field(default=None, ge=1)

    additional_parameters: Dict[str, Any] = Field(default_factory=dict)


class StructuredQueryResponse(APIBaseModel):
    """Generic structured query execution result."""

    status: ResponseStatus = ResponseStatus.SUCCESS

    rows: List[Dict[str, Any]] = Field(default_factory=list)

    row_count: int = Field(default=0, ge=0)

    metadata: ResponseMetadata = Field(default_factory=ResponseMetadata)


# Backwards-compatible aliases for callers still using the SQL-specific
# names. Prefer StructuredQueryRequest/StructuredQueryResponse in new code.
SQLQueryRequest = StructuredQueryRequest
SQLQueryResponse = StructuredQueryResponse


# ==========================================================
# API INFORMATION
# ==========================================================


class APIInfoResponse(APIBaseModel):
    """API information."""

    application: str

    version: str

    description: Optional[str] = None

    endpoints: List[str] = Field(default_factory=list)

    metadata: ResponseMetadata = Field(default_factory=ResponseMetadata)


# ==========================================================
# GENERIC RESPONSE WRAPPER
# ==========================================================


class APIResponse(APIBaseModel):
    """Universal API response wrapper."""

    status: ResponseStatus = ResponseStatus.SUCCESS

    message: Optional[str] = None

    data: Optional[Any] = None

    errors: List[ErrorDetail] = Field(default_factory=list)

    metadata: ResponseMetadata = Field(default_factory=ResponseMetadata)


# ==========================================================
# TYPE ALIASES
# ==========================================================

# Convenience alias for any of the "list-like" response payloads that
# wrap a collection of records/documents plus metadata.
JSONDict = Dict[str, Any]


# ==========================================================
# EXPORT LIST
# ==========================================================

__all__ = [
    # Enums
    "RetrievalMode",
    "QueryType",
    "ResponseStatus",
    "SortOrder",
    "HealthStatus",
    "ComponentStatus",
    "FilterOperator",
    "ConversationRole",
    "QueryLanguage",
    # Base
    "APIBaseModel",
    "JSONDict",
    # Metadata / pagination / filter / sort
    "ResponseMetadata",
    "Pagination",
    "FilterCondition",
    "SortCondition",
    # Errors / success
    "ErrorDetail",
    "ErrorResponse",
    "SuccessResponse",
    # Generic containers
    "KeyValue",
    "Statistics",
    "Message",
    # Source & retrieval
    "SourceMetadata",
    "RetrievedDocument",
    "Citation",
    "RetrievalMetadata",
    # Token / latency / confidence
    "TokenUsage",
    "LatencyInfo",
    "ConfidenceScore",
    # Query
    "QueryRequest",
    "QueryResponse",
    "BatchQueryRequest",
    "BatchQueryResponse",
    # Health
    "ServiceHealth",
    "HealthResponse",
    # System info
    "SystemInfoResponse",
    # Cache
    "CacheStatistics",
    "CacheClearResponse",
    # Metrics
    "MetricsResponse",
    # Benchmark
    "BenchmarkRequest",
    "BenchmarkResult",
    "BenchmarkResponse",
    # Evaluation
    "EvaluationRequest",
    "EvaluationScore",
    "EvaluationResponse",
    # Reload
    "ReloadResponse",
    # Conversation
    "ConversationMessage",
    "ConversationRequest",
    "ConversationResponse",
    # Streaming
    "StreamChunk",
    "StreamResponse",
    # Feedback
    "FeedbackRequest",
    "FeedbackResponse",
    # Generic data
    "GenericRecord",
    "DatasetResponse",
    # Search
    "SearchRequest",
    "SearchResponse",
    # Structured query
    "StructuredQueryRequest",
    "StructuredQueryResponse",
    "SQLQueryRequest",
    "SQLQueryResponse",
    # API info / wrapper
    "APIInfoResponse",
    "APIResponse",
]