"""Logging setup: readable text in dev, one-line JSON when ``VERISCAN_LOG_FORMAT=json``.

Never log passwords, tokens, raw session IDs, uploaded content or full prompts.
Use :func:`redact_session` when a session needs to be correlated in logs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from typing import Optional

from core.settings import get_settings

_CONFIGURED = False


class JsonFormatter(logging.Formatter):
    """Minimal structured formatter (no third-party dependency)."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(level: Optional[str] = None, fmt: Optional[str] = None) -> None:
    """Configure the root logger once; safe to call repeatedly (e.g. uvicorn reload)."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    settings = get_settings()
    handler = logging.StreamHandler(sys.stderr)
    if (fmt or settings.log_format) == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S")
        )
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level or settings.log_level)
    _CONFIGURED = True


def redact_session(session_id: Optional[str]) -> str:
    """Short stable hash of a session ID, safe to put in logs."""
    if not session_id:
        return "-"
    return hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:10]
