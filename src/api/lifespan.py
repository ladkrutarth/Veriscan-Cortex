"""Startup / shutdown: load every heavy service once, tolerating individual failures."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from api.state import AppState, get_state
from core.settings import get_settings

logger = logging.getLogger(__name__)


def load_services(state: AppState) -> None:
    """Populate ``state``. A failing component is logged and left as None (routes return 503)."""
    settings = get_settings()

    try:
        from models.rag_engine_local import RAGEngineLocal

        state.rag_engine = RAGEngineLocal()
        state.rag_engine.index_data()
        logger.info("System RAG engine loaded")

        from models.multimodal_rag import MultimodalRAG

        state.multimodal_rag = MultimodalRAG()
        state.multimodal_rag.index_data()
        logger.info("Multimodal RAG engine loaded")
    except Exception:
        logger.exception("RAG engine failed to load")

    try:
        from models.guard_agent_local import LocalGuardAgent

        state.guard_agent = LocalGuardAgent()
        logger.info("GuardAgent (security analyst) loaded")
    except Exception as e:
        logger.warning("GuardAgent not loaded: %s", e)

    try:
        from agents.financial_advisor_agent import FinancialAdvisorAgent

        state.advisor_agent = FinancialAdvisorAgent(llm=state.llm)
        _ = state.advisor_agent.df  # load the dataset now, not on the first request
        state.advisor_load_error = None
        logger.info("Financial Advisor agent loaded")
    except Exception as e:
        state.advisor_agent = None
        state.advisor_load_error = str(e)
        logger.exception("Financial Advisor failed to load")

    try:
        from agents.spending_dna_agent import SpendingDNAAgent

        state.dna_agent = SpendingDNAAgent()
        logger.info("Spending DNA agent loaded")
    except Exception:
        logger.exception("Spending DNA agent failed to load")

    if settings.skip_vision:
        logger.info("Vision LLM skipped (VERISCAN_SKIP_VISION=1)")
    else:
        try:
            from models.vision_llm import VisionLLM

            state.vision_llm = VisionLLM()
            logger.info("Vision LLM loaded")
        except Exception as e:
            logger.warning("Vision LLM not loaded: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    state = get_state()
    logger.info("Veriscan API loading resources...")
    load_services(state)
    logger.info("Veriscan API ready")
    yield
    logger.info("Veriscan API shutting down")
    state.reset()
