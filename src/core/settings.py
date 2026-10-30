"""Typed runtime settings for Veriscan-Cortex, read once from environment variables.

Every flag documented in CLAUDE.md lives here so callers don't parse ``os.environ`` ad hoc.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional, Tuple

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def env_flag(name: str, default: bool = False) -> bool:
    """Parse a boolean env var; unknown or empty values fall back to ``default``."""
    raw = os.environ.get(name, "").strip().lower()
    if raw in _TRUE:
        return True
    if raw in _FALSE:
        return False
    return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip())
    except ValueError:
        return default


def _env_list(name: str, default: Tuple[str, ...]) -> Tuple[str, ...]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return tuple(item.strip() for item in raw.split(",") if item.strip())


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of configuration; build with :func:`get_settings`."""

    fast_mode: bool = False
    llm_model: Optional[str] = None
    skip_vision: bool = False
    enable_agentic_advisor: bool = True
    enable_multi_agent_advisor: bool = False
    trajectory_log: bool = True
    log_level: str = "INFO"
    log_format: str = "text"
    cors_origins: Tuple[str, ...] = field(
        default=("http://localhost:8502", "http://127.0.0.1:8502", "http://localhost:8501")
    )
    max_upload_files: int = 10
    max_upload_bytes: int = 25 * 1024 * 1024

    @classmethod
    def from_env(cls) -> Settings:
        defaults = cls()
        return cls(
            fast_mode=env_flag("VERISCAN_FAST_MODE"),
            llm_model=os.environ.get("VERISCAN_LLM_MODEL") or None,
            skip_vision=env_flag("VERISCAN_SKIP_VISION"),
            enable_agentic_advisor=env_flag("ENABLE_AGENTIC_ADVISOR", True),
            enable_multi_agent_advisor=env_flag("ENABLE_MULTI_AGENT_ADVISOR"),
            trajectory_log=env_flag("VERISCAN_TRAJECTORY_LOG", True),
            log_level=os.environ.get("VERISCAN_LOG_LEVEL", "INFO").upper(),
            log_format=os.environ.get("VERISCAN_LOG_FORMAT", "text").lower(),
            cors_origins=_env_list("VERISCAN_CORS_ORIGINS", defaults.cors_origins),
            max_upload_files=_env_int("VERISCAN_MAX_UPLOAD_FILES", defaults.max_upload_files),
            max_upload_bytes=_env_int("VERISCAN_MAX_UPLOAD_MB", 25) * 1024 * 1024,
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings. Tests may call ``get_settings.cache_clear()``."""
    return Settings.from_env()
