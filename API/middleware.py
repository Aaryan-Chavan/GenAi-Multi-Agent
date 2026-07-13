# api/middleware.py
"""
Request-ID middleware.

Stamps every inbound request with a unique ID (request.state.request_id),
used by api/routes.py for log correlation and for populating
ResponseMetadata.request_id on every response.
"""

from __future__ import annotations

import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Assigns request.state.request_id and logs method/path/status/latency."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request.state.request_id = uuid.uuid4()
        started_at = time.perf_counter()

        response = await call_next(request)

        elapsed_ms = round((time.perf_counter() - started_at) * 1000, 3)
        response.headers["X-Request-ID"] = str(request.state.request_id)
        logger.info(
            "request_id=%s method=%s path=%s status=%s latency_ms=%.3f",
            request.state.request_id, request.method, request.url.path,
            response.status_code, elapsed_ms,
        )
        return response