"""Request IDs, access logging and a safe catch-all error handler."""

from __future__ import annotations

import logging
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("api.access")

REQUEST_ID_HEADER = "X-Request-ID"


def install_middleware(app: FastAPI) -> None:
    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex[:16]
        request.state.request_id = request_id
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.headers[REQUEST_ID_HEADER] = request_id
        # Path only — query strings may carry session IDs.
        logger.info(
            "%s %s -> %s %.0fms rid=%s",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
            request_id,
        )
        return response

    @app.exception_handler(Exception)
    async def unhandled_error(request: Request, exc: Exception) -> JSONResponse:
        request_id = getattr(request.state, "request_id", "-")
        logger.exception("Unhandled error on %s rid=%s", request.url.path, request_id)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error.", "request_id": request_id},
            headers={REQUEST_ID_HEADER: request_id},
        )
