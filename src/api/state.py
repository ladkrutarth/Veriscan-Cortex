"""Process-wide service container for the API.

Heavy resources (LLMs, RAG engines, agents) are loaded once in the lifespan hook and
stored here. Routers read them through :func:`get_state`; tests replace attributes directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from fastapi import HTTPException


@dataclass
class AppState:
    guard_agent: Any = None
    rag_engine: Any = None
    multimodal_rag: Any = None
    advisor_agent: Any = None
    advisor_load_error: Optional[str] = None
    dna_agent: Any = None
    vision_llm: Any = None
    login_failures: Dict[str, int] = field(default_factory=dict)

    @property
    def llm(self) -> Any:
        """The shared text LLM (owned by the guard agent), or None."""
        return getattr(self.guard_agent, "llm", None)

    def reset(self) -> None:
        """Drop every loaded service (used on shutdown and in tests)."""
        self.__init__()  # type: ignore[misc]

    # -- availability guards used by routers ---------------------------------
    def require(self, attr: str, label: str) -> Any:
        service = getattr(self, attr)
        if service is None:
            raise HTTPException(status_code=503, detail=f"{label} not loaded.")
        return service

    def require_advisor(self) -> Any:
        if self.advisor_agent is None:
            detail = "Financial Advisor not loaded."
            if self.advisor_load_error:
                detail = f"Financial Advisor not loaded: {self.advisor_load_error}"
            raise HTTPException(status_code=503, detail=detail)
        return self.advisor_agent


_state = AppState()


def get_state() -> AppState:
    return _state
