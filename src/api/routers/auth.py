"""Demo login. See CLAUDE.md "Known production gaps": the user store is in-memory."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends

from api.schemas import AuthLoginRequest, AuthLoginResponse
from api.state import AppState, get_state
from models.auth_store import get_user_store

router = APIRouter(tags=["Auth"])

SUSPICIOUS_USERNAMES = frozenset({"root", "admin_test", "hacker", "pentest"})
WEAK_PASSWORDS = frozenset({"password", "password123", "123456", "admin"})
FAILURE_THRESHOLD = 3


def login_risk_score(username: str, password: str, prior_failures: int) -> float:
    """Heuristic 0-55 login risk (demo). Feeds the planned deception routing."""
    risk = 0.0
    if username.lower() in SUSPICIOUS_USERNAMES:
        risk += 30.0
    if password.lower() in WEAK_PASSWORDS:
        risk += 15.0
    if prior_failures >= FAILURE_THRESHOLD:
        risk += 10.0
    return risk


@router.post("/api/auth/login", response_model=AuthLoginResponse)
async def auth_login(req: AuthLoginRequest, state: AppState = Depends(get_state)) -> AuthLoginResponse:
    """Verify demo credentials and issue a session ID.

    The risk score is computed but deception routing is not wired in yet, so
    ``diverted`` is always False.
    """
    username = req.username.strip()
    if not get_user_store().verify_user(username, req.password):
        state.login_failures[username] = state.login_failures.get(username, 0) + 1
        return AuthLoginResponse(
            authenticated=False, session_id=None, diverted=False, message="Invalid username or password."
        )

    login_risk_score(username, req.password, state.login_failures.get(username, 0))
    state.login_failures[username] = 0
    return AuthLoginResponse(
        authenticated=True, session_id=str(uuid.uuid4()), diverted=False, message="Login successful."
    )
