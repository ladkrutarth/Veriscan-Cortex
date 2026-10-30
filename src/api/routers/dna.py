"""Spending DNA: behavioural profile, session comparison and HITL Challenge Auth."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from api.schemas import (
    DNAChallengeRequest,
    DNAChallengeResponse,
    DNACompareRequest,
    DNACompareResponse,
    SpendingDNAResponse,
)
from api.state import AppState, get_state

router = APIRouter(tags=["Spending DNA"])

# Sessions whose trust score falls below this need a human-approved Challenge Auth.
CHALLENGE_TRUST_THRESHOLD = 0.55


@router.get("/api/dna/profile/{user_id}", response_model=SpendingDNAResponse)
async def get_dna_profile(
    user_id: str, session_id: Optional[str] = Query(None), state: AppState = Depends(get_state)
) -> SpendingDNAResponse:
    """8-axis Spending DNA for a user."""
    result = state.require("dna_agent", "DNA Agent").compute_dna(user_id)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return SpendingDNAResponse(**result)


@router.post("/api/dna/compare", response_model=DNACompareResponse)
async def compare_dna(req: DNACompareRequest, state: AppState = Depends(get_state)) -> DNACompareResponse:
    """Compare a session against the DNA baseline; high deviation needs approved Challenge Auth."""
    agent = state.require("dna_agent", "DNA Agent")
    result = agent.compare_session(req.user_id, session_overrides=req.session_overrides)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])

    needs_challenge = float(result.get("session_trust_score", 1.0)) < CHALLENGE_TRUST_THRESHOLD
    challenge = None
    if needs_challenge and req.approved:
        from models.guard_agent_local import tool_challenge_auth

        challenge = tool_challenge_auth(req.user_id, reason="high_dna_deviation")
    result["requires_approval"] = needs_challenge and not req.approved
    result["challenge"] = challenge
    return DNACompareResponse(**result)


@router.post("/api/dna/challenge", response_model=DNAChallengeResponse)
async def dna_challenge_auth(req: DNAChallengeRequest) -> DNAChallengeResponse:
    """HITL-gated Challenge Auth — never executes unless the request has approved=true."""
    if not req.approved:
        return DNAChallengeResponse(
            user_id=req.user_id,
            status="pending_approval",
            challenge=None,
            message="Challenge Auth requires human approval. Re-submit with approved=true.",
            pending_approval=True,
        )
    from models.guard_agent_local import tool_challenge_auth

    challenge = tool_challenge_auth(req.user_id, reason=req.reason)
    return DNAChallengeResponse(
        user_id=req.user_id,
        status="challenge_issued",
        challenge=challenge,
        message=challenge.get("message", "Challenge Auth issued."),
        pending_approval=False,
    )
