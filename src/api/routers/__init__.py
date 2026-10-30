"""Feature routers mounted by :func:`api.main.create_app`."""

from __future__ import annotations

from api.routers import advisor, auth, dna, fraud, rag, security, system

ALL_ROUTERS = [
    system.router,
    auth.router,
    fraud.router,
    rag.router,
    advisor.router,
    security.router,
    dna.router,
]
