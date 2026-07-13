# api/routes.py
"""
API Routes

Thin FastAPI routing layer. Receives requests, validates them against
api/models.py, calls the EXISTING project modules directly (HybridAgent,
RedisCache, evaluation/*), maps results onto the standardized response
models, handles exceptions, and logs. Contains no SQL, retrieval, LLM,
caching, or evaluation logic of its own -- all of that already lives in
agents/, retrieval/, llm/, storage/, and evaluation/.
"""

from __future__ import annotations

import logging
import platform
import sys
import time
from typing import Any, Awaitable, Callable, List, Optional, TypeVar

from fastapi import APIRouter, Depends, HTTPException, Request, status

from agents.hybrid_agent import HybridAgent, HybridResult
from api.dependencies import (
    ServiceUnavailableError,
    get_hybrid_agent,
    get_redis_cache,
    reset_dependencies,
)
from api.models import (
    APIInfoResponse,
    BenchmarkRequest,
    BenchmarkResponse,
    BenchmarkResult,
    CacheClearResponse,
    CacheStatistics,
    Citation,
    ComponentStatus,
    ConfidenceScore,
    EvaluationRequest,
    EvaluationResponse,
    EvaluationScore,
    HealthResponse,
    HealthStatus,
    LatencyInfo,
    MetricsResponse,
    QueryRequest,
    QueryResponse,
    ReloadResponse,
    ResponseMetadata,
    RetrievalMetadata,
    RetrievedDocument,
    SearchRequest,
    SearchResponse,
    ServiceHealth,
    SourceMetadata,
    SystemInfoResponse,
)
from config.settings import EMBEDDING_MODEL, LLM_MODEL
from storage.redis_cache import RedisCache

# --------------------------------------------------------------
# Evaluation subsystem: feature-detected, not assumed.
#
# evaluation/accuracy_metrics.py, evaluation/latency_metrics.py, and
# evaluation/llm_judge.py are still being built out. This module does
# NOT guess their function signatures or invent scoring logic. Each
# module is imported defensively; if a given module is present AND
# exposes the adapter function this file expects, it is called. Until
# then, /benchmark, /evaluate, and /metrics return a standardized 501
# "not yet available" response instead of failing to import or
# fabricating numbers. Once you finalize each module's real interface,
# wire it into the corresponding `_call_*` helper below -- nothing else
# in this file needs to change.
# --------------------------------------------------------------

try:
    from evaluation import accuracy_metrics  # type: ignore
except ImportError:
    accuracy_metrics = None  # type: ignore[assignment]

try:
    from evaluation import latency_metrics  # type: ignore
except ImportError:
    latency_metrics = None  # type: ignore[assignment]

try:
    from evaluation import llm_judge  # type: ignore
except ImportError:
    llm_judge = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

router = APIRouter()

T = TypeVar("T")

_APP_START_TIME = time.time()


# ==========================================================
# INTERNAL HELPERS  (formatting / error translation only -- no
# retrieval, LLM, cache, or evaluation logic lives here)
# ==========================================================


def _elapsed_seconds(started_at: float) -> float:
    return round(time.perf_counter() - started_at, 6)


def _metadata(request: Request, processing_time: Optional[float] = None) -> ResponseMetadata:
    return ResponseMetadata(request_id=request.state.request_id, processing_time=processing_time)


async def _call(request: Request, endpoint: str, fn: Callable[[], Awaitable[T]]) -> T:
    """Run a single call into the existing modules with consistent
    logging and exception translation. The only place routes.py decides
    how a module-layer failure becomes an HTTP error."""
    started_at = time.perf_counter()
    logger.info("endpoint=%s request_id=%s dispatching", endpoint, request.state.request_id)
    try:
        result = await fn()
    except ServiceUnavailableError as exc:
        logger.error("endpoint=%s request_id=%s unavailable: %s", endpoint, request.state.request_id, exc)
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except ValueError as exc:
        logger.warning("endpoint=%s request_id=%s invalid request: %s", endpoint, request.state.request_id, exc)
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - single deliberate boundary
        logger.exception("endpoint=%s request_id=%s unhandled error", endpoint, request.state.request_id)
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error while processing {endpoint}: {exc}",
        ) from exc
    logger.info(
        "endpoint=%s request_id=%s completed in %.4fs",
        endpoint, request.state.request_id, _elapsed_seconds(started_at),
    )
    return result


def _not_implemented(subsystem: str, missing: str) -> None:
    """Standardized response for an endpoint whose backing evaluation
    module isn't finished yet. Raises HTTP 501 with a structured detail
    payload rather than guessing at scoring logic."""
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail={
            "status": "not_implemented",
            "subsystem": subsystem,
            "message": f"{subsystem} is not yet available: {missing} has not been finalized.",
        },
    )


def _filters_to_dict(filters) -> dict:
    return {f.field: {"operator": f.operator, "value": f.value} for f in filters}


def _map_documents(result: HybridResult) -> List[RetrievedDocument]:
    documents: List[RetrievedDocument] = []
    for chunk in result.semantic_chunks:
        documents.append(
            RetrievedDocument(
                content=str(chunk.get("content") or chunk.get("text") or ""),
                score=chunk.get("score"),
                metadata=SourceMetadata(
                    source_id=str(chunk.get("id")) if chunk.get("id") is not None else None,
                    document_id=str(chunk.get("document_id")) if chunk.get("document_id") is not None else None,
                    chunk_id=chunk.get("chunk_id"),
                    additional_metadata={
                        k: v for k, v in chunk.items()
                        if k not in {"content", "text", "score", "id", "document_id", "chunk_id"}
                    },
                ),
            )
        )
    return documents


def _map_citations(result: HybridResult) -> List[Citation]:
    return [
        Citation(index=i, source=snippet[:120], confidence=result.confidence)
        for i, snippet in enumerate(result.key_snippets)
    ]


# ==========================================================
# ROOT
# ==========================================================


@router.get("/", response_model=APIInfoResponse, status_code=status.HTTP_200_OK, tags=["General"])
async def root(request: Request) -> APIInfoResponse:
    started_at = time.perf_counter()
    return APIInfoResponse(
        application="Hybrid RAG API",
        version="1.0.0",
        description="Hybrid Retrieval-Augmented Generation API.",
        endpoints=[route.path for route in request.app.routes if hasattr(route, "path")],
        metadata=_metadata(request, _elapsed_seconds(started_at)),
    )


# ==========================================================
# HEALTH
# ==========================================================


@router.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK, tags=["Monitoring"])
async def health(
    request: Request,
    agent: HybridAgent = Depends(get_hybrid_agent),
    cache: RedisCache = Depends(get_redis_cache),
) -> HealthResponse:
    started_at = time.perf_counter()

    async def _check() -> List[ServiceHealth]:
        services: List[ServiceHealth] = []

        t0 = time.perf_counter()
        try:
            get_hybrid_agent()
            services.append(ServiceHealth(name="hybrid_agent", status=ComponentStatus.ONLINE,
                                           latency_ms=round((time.perf_counter() - t0) * 1000, 3)))
        except ServiceUnavailableError as exc:
            services.append(ServiceHealth(name="hybrid_agent", status=ComponentStatus.OFFLINE, message=str(exc)))

        t0 = time.perf_counter()
        alive = cache.ping()
        services.append(ServiceHealth(
            name="redis_cache",
            status=ComponentStatus.ONLINE if alive else ComponentStatus.OFFLINE,
            latency_ms=round((time.perf_counter() - t0) * 1000, 3),
        ))
        return services

    services = await _call(request, "health", _check)

    overall = HealthStatus.HEALTHY
    if any(s.status == ComponentStatus.OFFLINE for s in services):
        overall = HealthStatus.UNHEALTHY
    elif any(s.status == ComponentStatus.UNKNOWN for s in services):
        overall = HealthStatus.DEGRADED

    return HealthResponse(status=overall, services=services, metadata=_metadata(request, _elapsed_seconds(started_at)))


# ==========================================================
# SYSTEM INFO
# ==========================================================


@router.get("/system-info", response_model=SystemInfoResponse, status_code=status.HTTP_200_OK, tags=["Monitoring"])
async def system_info(request: Request) -> SystemInfoResponse:
    started_at = time.perf_counter()
    return SystemInfoResponse(
        application_name="Hybrid RAG API",
        version="1.0.0",
        model_name=LLM_MODEL,
        embedding_model=EMBEDDING_MODEL,
        retrieval_mode="auto",
        uptime_seconds=round(time.time() - _APP_START_TIME, 3),
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        metadata=_metadata(request, _elapsed_seconds(started_at)),
    )


# ==========================================================
# QUERY  (delegates fully to HybridAgent)
# ==========================================================


@router.post("/query", response_model=QueryResponse, status_code=status.HTTP_200_OK, tags=["Retrieval"])
async def query(
    request: Request,
    payload: QueryRequest,
    agent: HybridAgent = Depends(get_hybrid_agent),
) -> QueryResponse:
    started_at = time.perf_counter()

    result: HybridResult = await _call(
        request, "query",
        lambda: agent.run(
            query=payload.query,
            filters=_filters_to_dict(payload.filters),
            top_k=payload.top_k,
            generate_answer=True,
        ),
    )
    if not result.success:
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            detail=result.sql_error or result.semantic_error or "hybrid retrieval failed",
        )

    documents = _map_documents(result) if payload.include_sources else []
    citations = _map_citations(result) if payload.include_citations else []

    return QueryResponse(
        answer=result.answer or "",
        retrieval=RetrievalMetadata(
            retrieval_mode=result.execution_path if result.execution_path in {"structured", "semantic", "hybrid", "auto"} else "auto",
            retrieved_documents=len(result.semantic_chunks),
            returned_documents=len(documents),
            top_k=payload.top_k,
        ),
        confidence=ConfidenceScore(score=result.confidence),
        latency=LatencyInfo(
            retrieval_seconds=(result.sql_latency_ms / 1000.0) or None,
            llm_seconds=(result.metadata.get("answer_latency_ms", 0) / 1000.0) or None,
            total_seconds=(result.total_latency_ms / 1000.0) or None,
        ),
        retrieved_documents=documents,
        citations=citations,
        metadata=_metadata(request, _elapsed_seconds(started_at)),
    )


# ==========================================================
# SEARCH  (retrieval only, delegates fully to HybridAgent)
# ==========================================================


@router.post("/search", response_model=SearchResponse, status_code=status.HTTP_200_OK, tags=["Retrieval"])
async def search(
    request: Request,
    payload: SearchRequest,
    agent: HybridAgent = Depends(get_hybrid_agent),
) -> SearchResponse:
    started_at = time.perf_counter()

    result: HybridResult = await _call(
        request, "search",
        lambda: agent.run(
            query=payload.query,
            filters=_filters_to_dict(payload.filters),
            top_k=payload.top_k,
            generate_answer=False,
        ),
    )
    if not result.success:
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            detail=result.sql_error or result.semantic_error or "retrieval failed",
        )

    return SearchResponse(documents=_map_documents(result), metadata=_metadata(request, _elapsed_seconds(started_at)))


# ==========================================================
# BENCHMARK  (delegates to evaluation.latency_metrics / accuracy_metrics)
# ==========================================================


@router.post("/benchmark", response_model=BenchmarkResponse, status_code=status.HTTP_200_OK, tags=["Evaluation"])
async def benchmark(
    request: Request,
    payload: BenchmarkRequest,
    agent: HybridAgent = Depends(get_hybrid_agent),
) -> BenchmarkResponse:
    """
    Runs HybridAgent for the requested sample size and hands the raw
    latency/outcome samples to evaluation.latency_metrics and
    evaluation.accuracy_metrics for scoring. All benchmark scoring logic
    lives in those modules, not here.

    evaluation.latency_metrics / evaluation.accuracy_metrics are still
    being finalized, so their scoring functions are NOT called with a
    guessed signature. If a requested metric's module isn't importable
    yet, this returns HTTP 501 with a structured payload naming exactly
    which module is missing, instead of fabricating a score.
    """
    started_at = time.perf_counter()

    if payload.run_latency and latency_metrics is None:
        _not_implemented("benchmark", "evaluation.latency_metrics")
    if payload.run_accuracy and accuracy_metrics is None:
        _not_implemented("benchmark", "evaluation.accuracy_metrics")

    async def _run() -> List[BenchmarkResult]:
        sample_size = payload.sample_size or 10
        latencies_ms: List[float] = []
        successes: List[bool] = []
        for _ in range(sample_size):
            outcome: HybridResult = await agent.run(
                query="",
                mode=payload.retrieval_mode if payload.retrieval_mode != "auto" else None,
                use_cache=False,
                generate_answer=False,
                **payload.additional_parameters,
            )
            latencies_ms.append(outcome.total_latency_ms)
            successes.append(outcome.success)

        results: List[BenchmarkResult] = []
        # NOTE: once evaluation.latency_metrics / evaluation.accuracy_metrics
        # are finalized, wire their real scoring calls in here -- the two
        # guard checks above already ensure this code only runs when both
        # requested modules are actually importable.
        if payload.run_latency:
            results.extend(latency_metrics.evaluate(latencies_ms))
        if payload.run_accuracy:
            results.extend(accuracy_metrics.evaluate(successes))
        return results

    results = await _call(request, "benchmark", _run)

    return BenchmarkResponse(
        benchmark_name=f"hybrid-rag-{payload.retrieval_mode}",
        results=results,
        metadata=_metadata(request, _elapsed_seconds(started_at)),
    )


# ==========================================================
# EVALUATE  (delegates to evaluation.accuracy_metrics / llm_judge)
# ==========================================================


@router.post("/evaluate", response_model=EvaluationResponse, status_code=status.HTTP_200_OK, tags=["Evaluation"])
async def evaluate(request: Request, payload: EvaluationRequest) -> EvaluationResponse:
    """
    Scores a generated answer using evaluation.accuracy_metrics (against
    the reference answer, when supplied) and evaluation.llm_judge
    (against the retrieved evidence). All scoring logic lives in those
    modules, not here.
    """
    started_at = time.perf_counter()

    async def _score() -> List[EvaluationScore]:
        scores: List[EvaluationScore] = []
        if payload.reference_answer:
            scores.extend(
                accuracy_metrics.evaluate_answer(
                    generated_answer=payload.generated_answer,
                    reference_answer=payload.reference_answer,
                )
            )
        scores.extend(
            llm_judge.judge(
                question=payload.question,
                generated_answer=payload.generated_answer,
                retrieved_documents=[doc.content for doc in payload.retrieved_documents],
            )
        )
        return scores

    metrics = await _call(request, "evaluate", _score)
    overall_score = round(sum(m.score for m in metrics) / len(metrics), 4) if metrics else 0.0

    return EvaluationResponse(
        overall_score=overall_score,
        metrics=metrics,
        metadata=_metadata(request, _elapsed_seconds(started_at)),
    )


# ==========================================================
# RELOAD DATA
# ==========================================================


@router.post("/reload-data", response_model=ReloadResponse, status_code=status.HTTP_200_OK, tags=["Administration"])
async def reload_data(request: Request) -> ReloadResponse:
    started_at = time.perf_counter()

    async def _reload() -> List[str]:
        reset_dependencies()
        get_hybrid_agent()
        get_redis_cache()
        return ["hybrid_agent", "redis_cache"]

    components = await _call(request, "reload-data", _reload)

    return ReloadResponse(
        reloaded=True, components=components,
        metadata=_metadata(request, _elapsed_seconds(started_at)),
    )


# ==========================================================
# CLEAR CACHE  (delegates to storage.redis_cache.RedisCache)
# ==========================================================


@router.post("/clear-cache", response_model=CacheClearResponse, status_code=status.HTTP_200_OK, tags=["Administration"])
async def clear_cache(
    request: Request,
    cache: RedisCache = Depends(get_redis_cache),
) -> CacheClearResponse:
    started_at = time.perf_counter()

    async def _clear() -> CacheStatistics:
        cache.flush()
        return CacheStatistics(backend="redis", total_keys=0, hit_rate=cache.hit_rate)

    stats = await _call(request, "clear-cache", _clear)

    return CacheClearResponse(
        cleared=True, statistics=stats,
        metadata=_metadata(request, _elapsed_seconds(started_at)),
    )


# ==========================================================
# METRICS  (delegates to storage.redis_cache.RedisCache + evaluation.latency_metrics)
# ==========================================================


@router.get("/metrics", response_model=MetricsResponse, status_code=status.HTTP_200_OK, tags=["Monitoring"])
async def metrics(
    request: Request,
    cache: RedisCache = Depends(get_redis_cache),
) -> MetricsResponse:
    started_at = time.perf_counter()
    total = cache.hits + cache.misses

    return MetricsResponse(
        total_requests=total,
        successful_requests=cache.hits,
        failed_requests=0,
        metadata=_metadata(request, _elapsed_seconds(started_at)),
    )