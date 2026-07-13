# api/main.py
"""
Application entrypoint.

Creates and configures the FastAPI application: metadata, logging,
middleware, CORS, router registration, global exception handlers, and
startup/shutdown lifecycle. Contains no retrieval, LLM, SQL, vector-
search, or business logic of its own -- that all lives in agents/,
retrieval/, llm/, storage/, and evaluation/. Startup/shutdown here only
call into api/dependencies.py to acquire/release the shared HybridAgent
and RedisCache singletons; they do not construct or manage those
resources directly.
"""

from __future__ import annotations

import logging
import logging.config
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from api.dependencies import get_hybrid_agent, get_redis_cache
from api.routes import router as api_router
from config.settings import LOG_FILE, LOG_LEVEL

try:
    from api.middleware import RequestIDMiddleware
except ImportError:
    RequestIDMiddleware = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


# ==========================================================
# LOGGING
# ==========================================================


def _configure_logging() -> None:
    """Configure application-wide logging using config/settings.py.
    Runs once, at import time, before the FastAPI app is constructed,
    so every module's logger is already configured when it first runs."""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                    "level": LOG_LEVEL,
                },
                "file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "formatter": "default",
                    "level": LOG_LEVEL,
                    "filename": str(LOG_FILE),
                    "maxBytes": 10 * 1024 * 1024,
                    "backupCount": 5,
                    "encoding": "utf-8",
                },
            },
            "root": {
                "level": LOG_LEVEL,
                "handlers": ["console", "file"],
            },
        }
    )


_configure_logging()


# ==========================================================
# LIFESPAN  (startup / shutdown)
# ==========================================================


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initializes application-level resources on startup and releases
    them on shutdown. Only acquires/releases the shared singletons
    exposed by api/dependencies.py -- it does not construct retrieval,
    LLM, or storage clients itself."""
    logger.info("Application startup: acquiring shared service singletons")
    try:
        get_hybrid_agent()
        get_redis_cache()
    except Exception:
        # A dependency being unavailable at boot (e.g. Qdrant/DuckDB/Redis
        # not yet reachable) must not prevent the process from starting --
        # /health will surface it as DEGRADED/UNHEALTHY per-service instead.
        logger.exception("One or more service singletons failed to initialize at startup")

    yield

    logger.info("Application shutdown: releasing shared service singletons")
    try:
        agent = get_hybrid_agent()
        await agent.close()
    except Exception:
        logger.exception("Error while closing HybridAgent")

    try:
        cache = get_redis_cache()
        cache.close()
    except Exception:
        logger.exception("Error while closing RedisCache")


# ==========================================================
# APPLICATION FACTORY
# ==========================================================


def create_app() -> FastAPI:
    app = FastAPI(
        title="Hybrid RAG API",
        description="Dataset-independent Hybrid Retrieval-Augmented Generation API.",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    _register_middleware(app)
    _register_routers(app)
    _register_exception_handlers(app)

    return app


def _register_middleware(app: FastAPI) -> None:
    if RequestIDMiddleware is not None:
        app.add_middleware(RequestIDMiddleware)
    else:
        logger.warning("api.middleware.RequestIDMiddleware not found; request-ID correlation is disabled")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.add_middleware(GZipMiddleware, minimum_size=1024)


def _register_routers(app: FastAPI) -> None:
    app.include_router(api_router)


def _register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(HTTPException)
    async def _http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        logger.warning(
            "http_exception request_id=%s path=%s status=%s detail=%s",
            request_id, request.url.path, exc.status_code, exc.detail,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "status": "error",
                "error": {
                    "code": str(exc.status_code),
                    "message": exc.detail if isinstance(exc.detail, str) else "Request failed",
                    "details": exc.detail if isinstance(exc.detail, dict) else None,
                },
                "metadata": {"request_id": str(request_id) if request_id else None},
            },
            headers=exc.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        logger.warning(
            "validation_error request_id=%s path=%s errors=%s",
            request_id, request.url.path, exc.errors(),
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "status": "error",
                "error": {
                    "code": "422",
                    "message": "Request validation failed.",
                    "details": {"errors": exc.errors()},
                },
                "metadata": {"request_id": str(request_id) if request_id else None},
            },
        )

    @app.exception_handler(Exception)
    async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        logger.exception(
            "unhandled_exception request_id=%s path=%s", request_id, request.url.path,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "status": "error",
                "error": {
                    "code": "500",
                    "message": "An unexpected error occurred.",
                    "details": None,
                },
                "metadata": {"request_id": str(request_id) if request_id else None},
            },
        )


app = create_app()