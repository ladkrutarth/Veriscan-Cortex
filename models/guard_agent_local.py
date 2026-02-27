"""
Veriscan — Local GuardAgent: Autonomous Private Security Analyst
Implements a tool-using reasoning loop using a local MLX-LM model.
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data paths
DATA_PATHS = {
    "features": PROJECT_ROOT / "dataset" / "csv_data" / "features_output.csv",
    "fraud_scores": PROJECT_ROOT / "dataset" / "csv_data" / "fraud_scores_output.csv",
    "auth_profiles": PROJECT_ROOT / "dataset" / "csv_data" / "auth_profiles_output.csv",
    "transactions": PROJECT_ROOT / "dataset" / "csv_data" / "transactions_3000.csv",
}

# ---------------------------------------------------------------------------
# TOOLS
# ---------------------------------------------------------------------------

def tool_get_user_risk_profile(user_id: str) -> Dict[str, Any]:
    """Retrieve risk profile for a specific user."""
    result = {"user_id": user_id, "found": False}
    
    # Auth profiles
    auth_path = DATA_PATHS["auth_profiles"]
    if auth_path.exists():
        df = pd.read_csv(auth_path)
        df.columns = [c.upper() for c in df.columns]
        user_data = df[df["USER_ID"] == user_id]
        if not user_data.empty:
            row = user_data.iloc[0]
            result.update({
                "found": True,
                "security_level": str(row.get("RECOMMENDED_SECURITY_LEVEL", "N/A")),
                "avg_risk": float(row.get("AVG_RISK", 0)),
                "high_risk_count": int(row.get("HIGH_RISK_COUNT", 0))
            })
            
    # Fraud scores
    fraud_path = DATA_PATHS["fraud_scores"]
    if fraud_path.exists():
        df = pd.read_csv(fraud_path)
        df.columns = [c.upper() for c in df.columns]
        user_data = df[df["USER_ID"] == user_id]
        if not user_data.empty:
            result.update({
                "found": True,
                "txn_count": len(user_data),
                "avg_combined_score": float(user_data["COMBINED_RISK_SCORE"].mean()),
                "risk_distribution": user_data["RISK_LEVEL"].value_counts().to_dict()
            })
            
    return result

def tool_get_high_risk_transactions(limit: int = 10) -> List[Dict]:
    """Get the top highest-risk transactions across the system."""
    fraud_path = DATA_PATHS["fraud_scores"]
    if not fraud_path.exists(): return []
    df = pd.read_csv(fraud_path)
    df.columns = [c.upper() for c in df.columns]
    df = df.sort_values("COMBINED_RISK_SCORE", ascending=False).head(limit)
    return df.to_dict("records")

def tool_query_rag(question: str) -> str:
    """Search local knowledge base for context."""
    try:
        engine = RAGEngineLocal()
        return engine.get_context_for_query(question)
    except Exception as e:
        return f"RAG tool error: {e}"

# ---------------------------------------------------------------------------
# REGISTRY
# ---------------------------------------------------------------------------

TOOL_REGISTRY = {
    "get_user_risk_profile": {
        "fn": tool_get_user_risk_profile,
        "description": "Get risk profile for a specific user. Args: user_id (str)."
    },
    "get_high_risk_transactions": {
        "fn": tool_get_high_risk_transactions,
        "description": "List top N high-risk transactions. Args: limit (int)."
    },
    "query_rag": {
        "fn": tool_query_rag,
        "description": "Query local RAG knowledge base. Args: question (str)."
    }
}

SYSTEM_PROMPT = """You are the Veriscan GuardAgent — an autonomous Private Financial Security Specialist.
You investigate fraud risk using data from your tools.

## Tools Available
{tool_descriptions}

## Workflow
1. DECIDE: Which tool do you need?
2. CALL: Respond with ONLY a JSON block like: {{"tool": "tool_name", "args": {{"arg": "val"}}}}
3. ANALYZE: When you have enough info, provide a final REPORT with Findings, Risk Level (LOW/MED/HIGH), and Recommended Action.

## Rules
- ALWAYS cite numbers from the tool results.
- NEVER invent data.
- If you need to call a tool, ONLY output the JSON block. No conversational filler.
"""

class LocalGuardAgent:
    """Agentic loop using LocalLLM and local tools."""
    
    def __init__(self):
        self.llm = LocalLLM()
        
    def _build_tool_desc(self) -> str:
        return "\n".join([f"- {name}: {info['description']}" for name, info in TOOL_REGISTRY.items()])

    def _clean_final_answer(self, text: str) -> str:
        """Strip JSON blocks and any lingering special tokens from final report."""
        # Remove JSON blocks
        clean_text = re.sub(r'\{.*\}', '', text, flags=re.DOTALL)
        # Remove common special tokens or markers
        clean_text = re.sub(r'<\|.*?\|>', '', clean_text)
        # Remove turn headers
        clean_text = clean_text.replace("assistant", "").replace("user", "").replace("Decision:", "")
        return clean_text.strip()

    def analyze(self, question: str) -> Dict[str, Any]:
        tool_desc = self._build_tool_desc()
        prompt = SYSTEM_PROMPT.format(tool_descriptions=tool_desc)
        
        actions_log = []
        current_context = f"User Request: {question}"
        
        for i in range(5): # Limit steps
            full_prompt = f"{prompt}\n\nCurrent Investigation Status:\n{current_context}\n\nDecision:"
            response = self.llm.generate(full_prompt)
            
            # Parse for tool call
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                try:
                    call = json.loads(match.group(0))
                    tool_name = call.get("tool")
                    args = call.get("args", {})
                    
                    if tool_name in TOOL_REGISTRY:
                        print(f"🛠️ Calling tool: {tool_name}({args})")
                        result = TOOL_REGISTRY[tool_name]["fn"](**args)
                        result_str = json.dumps(result, indent=2)
                        
                        actions_log.append({"step": i+1, "tool": tool_name, "args": args, "result": result_str[:200]+"..."})
                        current_context += f"\nStep {i+1}: Called {tool_name}. Result: {result_str}"
                        continue
                except Exception as e:
                    print(f"⚠️ Tool parse error: {e}")
                    
            # If no tool call found OR tool execution failed, this might be the final answer
            final_report = self._clean_final_answer(response)
            
            # If the response was purely a tool call that we failed to process, 
            # and there's no text left, don't return it as a success yet unless it's the last turn
            if not final_report and i < 4:
                continue

            return {
                "answer": final_report if final_report else response,
                "actions": actions_log,
                "status": "success"
            }
            
        return {"answer": "Investigation timed out.", "actions": actions_log, "status": "timeout"}

if __name__ == "__main__":
    agent = LocalGuardAgent()
    print(agent.analyze("Investigate risk for USER_0")["answer"])
