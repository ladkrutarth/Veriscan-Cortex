"""Grounded chat over a session's uploaded documents (multimodal RAG + vision/OCR)."""

from __future__ import annotations

import base64
import binascii
import contextlib
import logging
import os
import tempfile
from typing import Any, Dict, List, Optional

from api.schemas import DocChatRequest, DocChatResponse
from api.state import AppState
from core.logging_config import redact_session
from models.rag_relevance import (
    format_grounded_context,
    grounded_system_prompt,
    is_unreadable_visual_evidence,
    is_visual_query,
    snippet_fallback_answer,
)

logger = logging.getLogger(__name__)

DEFAULT_DOC_TYPES = ["pdf_doc", "image_doc", "csv_doc", "text_doc", "pdf_summary", "csv_summary"]
MAX_INLINE_IMAGES = 3
MAX_SESSION_IMAGES = 2
VISION_PROMPT = (
    "Task: Extract detailed information relevant to this query: {query}\n"
    "If this is a bank statement, receipt, UI screenshot, or photo, describe what is visible "
    "and list every readable date, amount, merchant, label, and notable UI element. "
    "Do not invent text that is not visible."
)
PLOTLY_INSTRUCTIONS = (
    "\n\nDATA VISUALIZATION:\n"
    "If the user asks for a graph/chart, include Plotly JSON between [PLOTLY_START] and [PLOTLY_END] "
    "with ONLY the raw JSON object (no markdown fences)."
)


def _meta(hit: Dict[str, Any]) -> Dict[str, Any]:
    return hit.get("metadata") or {}


def _is_weak(hit: Dict[str, Any]) -> bool:
    return _meta(hit).get("ocr_weak") == "true" or is_unreadable_visual_evidence(hit.get("text") or "")


def _retrieve(rag: Any, req: DocChatRequest) -> List[Dict[str, Any]]:
    """Dense retrieval scoped to the session, with image/document fallbacks when it misses."""
    results = rag.query(
        req.message,
        n_results=6,
        include_types=req.file_types or DEFAULT_DOC_TYPES,
        session_id=req.session_id,
    )
    if req.session_id and not results and is_visual_query(req.message):
        results = rag.force_image_hits(req.session_id)
    if req.session_id and not results:
        results = rag.force_document_hits(req.session_id)
    logger.debug("doc_chat retrieved %d hits for session %s", len(results), redact_session(req.session_id))
    return results or []


def _prefer_readable(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop weak OCR stubs / empty chunks when real document text is also available."""
    readable = [r for r in results if not _is_weak(r) and _meta(r).get("empty") != "true"]
    return readable if readable and len(readable) < len(results) else results


def _decode_inline_images(images: Optional[List[str]]) -> List[str]:
    """Write base64 images (optionally data-URLs) to temp files; returns their paths."""
    paths: List[str] = []
    for img_b64 in (images or [])[:MAX_INLINE_IMAGES]:
        try:
            data = base64.b64decode(img_b64.split(",")[-1])
        except (binascii.Error, ValueError):
            logger.warning("Skipping undecodable inline image")
            continue
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(data)
            paths.append(tmp.name)
    return paths


def _describe_images(state: AppState, paths: List[str], query: str, temp_paths: List[str]) -> str:
    """Run the vision model over ``paths``; temp files in ``temp_paths`` are always removed."""
    visual_context = ""
    try:
        if state.vision_llm is None:
            return ""
        prompt = VISION_PROMPT.format(query=query)
        for path in paths:
            try:
                desc = state.vision_llm.analyze_image(path, prompt, max_tokens=500)
                visual_context += f"\n[VISUAL EVIDENCE]: {desc}"
            except Exception:
                logger.exception("Vision analysis failed")
        return visual_context
    finally:
        for path in temp_paths:
            with contextlib.suppress(OSError):
                os.unlink(path)


def _sources(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "text": r.get("text"),
            "metadata": r.get("metadata", {}),
            "confidence": r.get("confidence"),
            "filename": r.get("filename") or _meta(r).get("filename"),
            "page": r.get("page") or _meta(r).get("page"),
        }
        for r in results
    ]


def _unreadable_reply(rag: Any, req: DocChatRequest) -> str:
    inventory = rag.get_file_inventory(req.session_id) if req.session_id else []
    names = ", ".join(f.get("filename", "?") for f in inventory) or "(none indexed)"
    has_images = bool(req.images or (req.session_id and rag.resolve_session_image_paths(req.session_id)))
    kind = "image" if has_images else "document"
    return (
        f"I can see that a {kind} was uploaded, but I cannot read its contents yet "
        "(no usable OCR text and Vision LLM did not return a description).\n\n"
        f"**Indexed files:** {names}\n\n"
        "**Fix:** re-index after OCR is available (Tesseract via `brew install tesseract`, "
        "or the bundled RapidOCR fallback), restart the API **without** "
        "`VERISCAN_SKIP_VISION=1`, click **Index All Documents** again, and re-ask. "
        "I will not invent tables or amounts from a blank OCR result."
    )


def _snippet_reply(message: str, results: List[Dict[str, Any]], visual_context: str) -> str:
    """Answer without an LLM: quote the best snippets plus any visual notes."""
    reply = snippet_fallback_answer(message, results)
    if visual_context and "OCR extracted little text" not in visual_context:
        return f"{reply}\n\n**Visual notes:**\n{visual_context[:1200]}"
    if visual_context:
        return (
            "I can see an uploaded image, but could not read enough text from it yet.\n\n"
            f"{visual_context[:800]}"
        )
    return reply


def _generate(llm: Any, system_content: str, message: str) -> str:
    if hasattr(llm, "generate_chat"):
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": message},
        ]
        return llm.generate_chat(messages, max_tokens=700, temp=0.1)
    prompt = f"{system_content}\n\nUser: {message}\nAssistant:"
    return llm.generate(prompt, max_tokens=700, temp=0.1)


def answer_document_question(state: AppState, req: DocChatRequest) -> DocChatResponse:
    """Retrieve → (optionally) look at images → refuse, quote, or generate a grounded answer."""
    rag = state.multimodal_rag
    results = _retrieve(rag, req)
    visual_q = is_visual_query(req.message)

    # Vision: inline images first, otherwise the session's stored images.
    temp_paths = _decode_inline_images(req.images)
    vision_targets = list(temp_paths)
    want_vision = bool(rag.needs_vision(req.message, results, req.session_id) or visual_q or req.images)
    if want_vision and req.session_id and not vision_targets:
        vision_targets = [
            str(p) for p in rag.resolve_session_image_paths(req.session_id)[:MAX_SESSION_IMAGES]
        ]
    visual_context = (
        _describe_images(state, vision_targets, req.message, temp_paths) if vision_targets else ""
    )

    # No vision model / no description: fall back to live OCR on session images.
    if want_vision and req.session_id and not visual_context.strip():
        live = rag.live_analyze_session_images(req.session_id, req.message)
        visual_context = live.get("text") or ""

    results = _prefer_readable(results)
    context = format_grounded_context(results)
    weak_stubs = bool(results) and all(_is_weak(r) for r in results)
    vision_useful = bool(visual_context.strip()) and not is_unreadable_visual_evidence(visual_context)
    if not ((context.strip() and not weak_stubs) or vision_useful):
        return DocChatResponse(
            reply=_unreadable_reply(rag, req), sources=_sources(results), session_id=req.session_id
        )

    sources = _sources(results)
    llm = state.llm
    if llm is None:
        reply = _snippet_reply(req.message, results, visual_context)
    else:
        system_content = grounded_system_prompt(context, visual_context) + PLOTLY_INSTRUCTIONS
        reply = _generate(llm, system_content, req.message)
    return DocChatResponse(reply=reply, sources=sources, session_id=req.session_id)
