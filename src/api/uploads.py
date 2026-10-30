"""Validated, session-scoped storage for user uploads."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import HTTPException, UploadFile

from core.logging_config import redact_session
from core.settings import get_settings
from paths import UPLOADS_DIR

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = frozenset({".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".webp", ".csv", ".txt", ".json"})
# Legacy shared bucket for uploads sent without a session (see CLAUDE.md "Known production gaps").
FALLBACK_SESSION = "global"

_SESSION_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._ -]")


def validate_session_id(session_id: Optional[str]) -> str:
    """Return a session ID that is safe to use as a directory name, or raise 400."""
    if not session_id:
        return FALLBACK_SESSION
    if not _SESSION_RE.match(session_id):
        raise HTTPException(status_code=400, detail="Invalid session_id.")
    return session_id


def safe_filename(name: Optional[str]) -> Optional[str]:
    """Strip directories and unusual characters; None if nothing usable remains."""
    base = Path((name or "").replace("\\", "/")).name
    cleaned = _UNSAFE_CHARS.sub("_", base).strip(" .")
    return cleaned[:128] or None


def session_upload_dir(session_id: str) -> Path:
    path = UPLOADS_DIR / session_id
    path.mkdir(parents=True, exist_ok=True)
    return path


async def save_and_index(
    files: List[UploadFile], session_id: Optional[str], multimodal_rag: Any
) -> List[Dict[str, Any]]:
    """Persist each allowed file under the session directory and index it if RAG is loaded."""
    settings = get_settings()
    if len(files) > settings.max_upload_files:
        raise HTTPException(status_code=413, detail=f"At most {settings.max_upload_files} files per upload.")

    sid = validate_session_id(session_id)
    target_dir = session_upload_dir(sid)
    results: List[Dict[str, Any]] = []

    for upload in files:
        name = safe_filename(upload.filename)
        if not name or Path(name).suffix.lower() not in ALLOWED_EXTENSIONS:
            results.append(
                {"filename": upload.filename, "status": "skipped", "error": "Unsupported file type."}
            )
            continue

        content = await upload.read(settings.max_upload_bytes + 1)
        if len(content) > settings.max_upload_bytes:
            results.append({"filename": name, "status": "skipped", "error": "File too large."})
            continue

        try:
            (target_dir / name).write_bytes(content)
        except OSError:
            logger.exception("Failed to store upload for session %s", redact_session(sid))
            results.append({"filename": name, "status": "failed", "error": "Could not store file."})
            continue

        entry: Dict[str, Any] = {"filename": name, "status": "indexed successfully"}
        if multimodal_rag is not None:
            try:
                entry["index"] = multimodal_rag.index_file_bytes(name, content, session_id=sid)
            except Exception:
                logger.exception("Indexing failed for an upload in session %s", redact_session(sid))
                entry["index"] = {"status": "failed"}
        results.append(entry)

    logger.info(
        "Stored %d/%d uploads for session %s",
        sum(r["status"] != "skipped" for r in results),
        len(files),
        redact_session(sid),
    )
    return results
