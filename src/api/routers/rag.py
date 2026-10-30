"""Knowledge base: semantic search, session uploads and grounded document chat."""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, File, Query, UploadFile

from api.doc_chat import answer_document_question
from api.schemas import DocChatRequest, DocChatResponse, RAGQueryRequest, RAGQueryResponse, RAGResult
from api.state import AppState, get_state
from api.uploads import save_and_index

router = APIRouter(tags=["Knowledge Base"])


@router.post("/api/rag/query", response_model=RAGQueryResponse)
async def rag_query(req: RAGQueryRequest, state: AppState = Depends(get_state)) -> RAGQueryResponse:
    """Semantic search over the system knowledge base."""
    engine = state.require("rag_engine", "RAG engine")
    parsed = [
        RAGResult(text=r["text"], confidence=r["confidence"], metadata=r.get("metadata"))
        for r in engine.query(req.query, n_results=req.n_results)
    ]
    return RAGQueryResponse(query=req.query, count=len(parsed), results=parsed)


@router.post("/api/rag/upload")
async def rag_upload(
    files: List[UploadFile] = File(...),
    session_id: Optional[str] = Query(None),
    state: AppState = Depends(get_state),
) -> dict:
    """Upload PDFs, images, CSVs or text into the session's store and index them."""
    return {"uploads": await save_and_index(files, session_id, state.multimodal_rag)}


@router.post("/api/rag/chat", response_model=DocChatResponse)
async def rag_chat(req: DocChatRequest, state: AppState = Depends(get_state)) -> DocChatResponse:
    """Conversational interface grounded in the session's indexed documents.

    Deliberately ``async def``: MLX must run on the main (event-loop) thread, not a worker.
    """
    state.require("multimodal_rag", "Multimodal RAG engine")
    return answer_document_question(state, req)
