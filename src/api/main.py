"""Veriscan — FastAPI gateway for fraud ML, agents and local RAG.

Run with ``./run_api.sh`` (sets ``PYTHONPATH=src``). Routes live in ``api/routers/``;
heavy services are loaded once by ``api.lifespan`` into ``api.state``.
"""

from __future__ import annotations

import os

# macOS stability guards (SIGABRT with duplicate OpenMP / fork safety). Must be set
# before torch / MLX / sentence-transformers are imported by the lifespan hook.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.lifespan import lifespan
from api.middleware import install_middleware
from api.routers import ALL_ROUTERS
from api.routers.system import API_VERSION
from core.logging_config import setup_logging
from core.settings import get_settings


def create_app() -> FastAPI:
    """Application factory; tests build an app without triggering model loading."""
    setup_logging()
    settings = get_settings()
    app = FastAPI(
        title="Veriscan — Fraud Intelligence API",
        description="Gateway for ML fraud prediction, agentic investigation and local RAG retrieval.",
        version=API_VERSION,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    install_middleware(app)
    for router in ALL_ROUTERS:
        app.include_router(router)
    return app


app = create_app()
