"""AI Security Analyst: tool-calling ReAct loop with HITL-gated actions."""

from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, Depends

from api.schemas import AgentActionStep, SecurityChatRequest, SecurityChatResponse
from api.state import AppState, get_state

router = APIRouter(tags=["AI Security Analyst"])


def _to_action_steps(raw_actions: List[Any]) -> List[AgentActionStep]:
    return [
        AgentActionStep(
            step=int(a.get("step", 0)),
            tool=str(a.get("tool", "")),
            args=a.get("args") or {},
            result=a.get("result"),
        )
        for a in raw_actions
        if isinstance(a, dict)
    ]


@router.post("/api/security/chat", response_model=SecurityChatResponse)
async def security_chat(
    req: SecurityChatRequest, state: AppState = Depends(get_state)
) -> SecurityChatResponse:
    """Security analyst (RAG, risk profile, HITL challenge). ``async def`` keeps MLX on the main thread."""
    agent = state.require("guard_agent", "GuardAgent")
    result: Dict[str, Any] = agent.analyze(req.message, session_id=req.session_id, approved=req.approved)
    return SecurityChatResponse(
        reply=result.get("reply") or result.get("answer") or "",
        actions=_to_action_steps(result.get("actions") or []),
        status=result.get("status", "completed"),
        session_id=req.session_id,
        trace=result.get("trace"),
        tools_used=result.get("tools_used"),
        step_count=result.get("step_count"),
        pending_approval=result.get("pending_approval"),
    )
