"""Fraud ML and per-user risk lookups."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Query

from api.schemas import HighRiskTransactionsResponse, UserRiskResponse

router = APIRouter()


@router.get("/api/fraud/high-risk", response_model=HighRiskTransactionsResponse, tags=["Fraud ML"])
async def get_high_risk_transactions(
    limit: int = Query(default=10, ge=1, le=100),
    session_id: Optional[str] = Query(None),
) -> HighRiskTransactionsResponse:
    """Get the top N highest-risk transactions."""
    from models.agent_tools_data import tool_get_high_risk_transactions

    results = tool_get_high_risk_transactions(limit=limit)
    return HighRiskTransactionsResponse(count=len(results), transactions=results)


@router.get("/api/user/{user_id}/risk", response_model=UserRiskResponse, tags=["User Intelligence"])
async def get_user_risk(user_id: str, session_id: Optional[str] = Query(None)) -> UserRiskResponse:
    """Retrieve the risk profile for a specific user."""
    from models.agent_tools_data import tool_get_user_risk_profile

    return UserRiskResponse(**tool_get_user_risk_profile(user_id))
