"""
Veriscan — Multi-Agent GuardAgent: Clean Architecture Facade
This file acts as the primary entry point for the agent system, coordinating
data loading, tool definitions, and routing to specialized modular agents.
"""

import json
import os
import re
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from models.local_llm import LocalLLM
from models.rag_engine_local import RAGEngineLocal
from agents import (
    KnowledgeAgent,
    RiskScannerAgent,
    ProfileAgent,
    SynthesisAgent,
    AgentResult
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data paths
DATA_PATHS = {
    "features": PROJECT_ROOT / "dataset" / "csv_data" / "features_output.csv",
    "fraud_scores": PROJECT_ROOT / "dataset" / "csv_data" / "fraud_scores_output.csv",
    "auth_profiles": PROJECT_ROOT / "dataset" / "csv_data" / "auth_profiles_output.csv",
    "transactions": PROJECT_ROOT / "dataset" / "csv_data" / "transactions_3000.csv",
}

# ---------------------------------------------------------------------------
# PRE-CACHED DATA (shared by all agents)
# ---------------------------------------------------------------------------
_fraud_df = None
_auth_df = None

def _load_cache():
    global _fraud_df, _auth_df
    fraud_path = DATA_PATHS["fraud_scores"]
    if fraud_path.exists() and _fraud_df is None:
        _fraud_df = pd.read_csv(fraud_path)
        _fraud_df.columns = [c.upper() for c in _fraud_df.columns]
    auth_path = DATA_PATHS["auth_profiles"]
    if auth_path.exists() and _auth_df is None:
        _auth_df = pd.read_csv(auth_path)
        _auth_df.columns = [c.upper() for c in _auth_df.columns]

_load_cache()

_rag_engine = None
def _get_rag():
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngineLocal()
        _rag_engine.index_data()
    return _rag_engine

# ---------------------------------------------------------------------------
# TOOLS (Used by modular agents)
# ---------------------------------------------------------------------------

def tool_get_user_risk_profile(user_id: str) -> Dict[str, Any]:
    _load_cache()
    result = {"user_id": user_id, "found": False}
    if _auth_df is not None:
        user_data = _auth_df[_auth_df["USER_ID"] == user_id]
        if not user_data.empty:
            row = user_data.iloc[0]
            result.update({
                "found": True,
                "security_level": str(row.get("RECOMMENDED_SECURITY_LEVEL", "N/A")),
                "avg_risk": float(row.get("AVG_RISK", 0)),
                "high_risk_count": int(row.get("HIGH_RISK_COUNT", 0))
            })
    if _fraud_df is not None:
        user_data = _fraud_df[_fraud_df["USER_ID"] == user_id]
        if not user_data.empty:
            result.update({
                "found": True,
                "txn_count": len(user_data),
                "avg_combined_score": float(user_data["COMBINED_RISK_SCORE"].mean()),
                "risk_distribution": user_data["RISK_LEVEL"].value_counts().to_dict()
            })
    return result

def tool_get_high_risk_transactions(limit: int = 10) -> List[Dict]:
    _load_cache()
    if _fraud_df is None: return []
    df = _fraud_df.sort_values("COMBINED_RISK_SCORE", ascending=False).head(limit)
    return df.to_dict("records")

def tool_query_rag(question: str) -> str:
    try:
        return _get_rag().get_context_for_query(question)
    except Exception as e:
        return f"RAG tool error: {e}"

# ---------------------------------------------------------------------------
# MASTER AGENT: Clean Architecture Facade
# ---------------------------------------------------------------------------

class LocalGuardAgent:
    def __init__(self):
        self.llm = LocalLLM()
        self.rag_agent = KnowledgeAgent(self.llm)
        self.risk_agent = RiskScannerAgent()
        self.user_agent = ProfileAgent()
        self.synthesis_agent = SynthesisAgent(self.llm)

    def _classify_query(self, query: str) -> str:
        q = query.upper()
        user_ids = re.findall(r'USER[_\s]?\d+', q)
        if len(user_ids) >= 2: return "synthesis"

        is_risk_scan = any(kw in q for kw in ["HIGH RISK", "TOP TRANSACTION", "MOST DANGEROUS", "SYSTEM SCAN", "STATE OF"])
        is_knowledge = any(kw in q for kw in ["WHAT IS", "EXPLAIN", "HOW TO", "DEFINE", "SEARCH", "CFPB"])

        if (int(len(user_ids) > 0) + int(is_risk_scan) + int(is_knowledge)) >= 2:
            return "synthesis"

        if len(user_ids) > 0: return "user_profile"
        if is_risk_scan: return "risk_scan"
        if is_knowledge: return "knowledge"
        return "synthesis"

    def analyze(self, question: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        agent_type = self._classify_query(question)
        print(f"⚡ GuardAgent Facade → {agent_type.upper()} agent (Session: {session_id})")

        try:
            if agent_type == "knowledge":
                res = self.rag_agent.run(question, session_id)
            elif agent_type == "risk_scan":
                res = self.risk_agent.run(question, session_id)
            elif agent_type == "user_profile":
                res = self.user_agent.run(question, session_id)
            else:
                res = self.synthesis_agent.run(question, session_id)

            return {
                "answer": res.answer,
                "actions": [a.dict() for a in res.actions],
                "status": res.status,
                "session_id": res.session_id,
                "trace": res.trace
            }
        except Exception as e:
            traceback.print_exc()
            return {"answer": f"Error in {agent_type} agent: {e}", "actions": [], "status": "error"}

if __name__ == "__main__":
    agent = LocalGuardAgent()
    print(agent.analyze("Investigate USER_1")["answer"])
