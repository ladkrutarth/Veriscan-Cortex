"""Health and readiness probes."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.schemas import HealthResponse
from api.state import AppState, get_state

API_VERSION = "2.0.0"

router = APIRouter(tags=["System"])


@router.get("/api/health", response_model=HealthResponse)
async def health_check(state: AppState = Depends(get_state)) -> HealthResponse:
    """Return API health status and loaded services."""
    advisor = "loaded" if state.advisor_agent else "unavailable"
    if not state.advisor_agent and state.advisor_load_error:
        advisor = f"unavailable: {state.advisor_load_error}"
    return HealthResponse(
        status="operational",
        version=API_VERSION,
        services={
            "guard_agent": "loaded" if state.guard_agent else "unavailable",
            "rag_engine": "loaded" if state.rag_engine else "unavailable",
            "advisor_agent": advisor,
            "dna_agent": "loaded" if state.dna_agent else "unavailable",
        },
    )


@router.get("/healthz", include_in_schema=False)
async def liveness() -> dict:
    """Liveness: the process is up (touches no models)."""
    return {"status": "ok"}
