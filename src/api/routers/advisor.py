"""AI Financial Advisor: agentic chat, user list, memory reset."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Query

from api.schemas import AdvisorChatRequest, AdvisorChatResponse
from api.state import AppState, get_state

router = APIRouter(tags=["AI Financial Advisor"])


@router.post("/api/advisor/chat", response_model=AdvisorChatResponse)
async def advisor_chat(req: AdvisorChatRequest, state: AppState = Depends(get_state)) -> AdvisorChatResponse:
    """Agentic (ReAct / supervisor) or keyword financial advice.

    Deliberately ``async def``: MLX inference must stay on the main thread, otherwise
    Metal can abort with completed-handler assertions.
    """
    advisor = state.require_advisor()
    result = advisor.chat(req.message, req.user_id, session_id=req.session_id, approved=req.approved)
    return AdvisorChatResponse(
        user_id=req.user_id,
        message=req.message,
        reply=result.get("reply", ""),
        tool_results=result.get("tool_results", []),
        trace=result.get("trace"),
        step_count=result.get("step_count"),
        tools_used=result.get("tools_used"),
        pending_approval=result.get("pending_approval"),
        status=result.get("status"),
    )


@router.get("/api/advisor/users")
async def advisor_users(
    session_id: Optional[str] = Query(None), state: AppState = Depends(get_state)
) -> dict:
    """Return all user IDs in the financial advisor dataset."""
    return {"users": state.require_advisor().get_all_users()}


@router.post("/api/advisor/reset")
async def advisor_reset(session_id: str = Query(...)) -> dict:
    """Clear conversation history for a specific session."""
    from agents.memory import get_memory

    get_memory().clear(session_id)
    return {"status": "cleared", "session_id": session_id}
