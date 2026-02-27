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
        "description": "CRITICAL: Use this FIRST for ANY query about a specific User ID (e.g., USER_123). Returns security level and risk scores. Args: user_id (str)."
    },
    "get_high_risk_transactions": {
        "fn": tool_get_high_risk_transactions,
        "description": "Use to list the most dangerous transactions across the whole system. Args: limit (int)."
    },
    "query_rag": {
        "fn": tool_query_rag,
        "description": "Use for general questions about fraud trends, CFPB policies, or explaining terms like 'velocity'. Args: question (str)."
    }
}

SYSTEM_PROMPT = """You are the Veriscan GuardAgent — an autonomous Private Financial Security Specialist.
You investigate fraud risk using data from your tools.

## Primary Policy: Mandatory Tool Usage
- If a query contains "USER_X" (where X is any number), you MUST call 'get_user_risk_profile' as your very first action.
- If a query asks for information or explanations, you MUST call 'query_rag' as your very first action.
- STOP and CALL a tool before providing any commentary.

## Tools Available
{tool_descriptions}

## Operational Workflow
1. Thought: Analyze the query intent.
2. Call: Provide the tool call in a single JSON block.

## Example 1
User: Check USER_123 for risk.
Thought: The request is about a specific User ID. Per policy, I must call get_user_risk_profile first.
{{ "tool": "get_user_risk_profile", "args": {{ "user_id": "USER_123" }} }}

## Example 2
User: What is velocity?
Thought: This is a request for an explanation. I must query the market intelligence RAG.
{{ "tool": "query_rag", "args": {{ "question": "what is velocity in fraud detection" }} }}

## Final Rule
- NEVER provide a final answer until you have called at least one tool.
- ALWAYS output the JSON block for tool calls.
"""

class LocalGuardAgent:
    """Agentic loop using LocalLLM and local tools with hybrid intent routing."""
    
    def __init__(self):
        self.llm = LocalLLM()
        
    def _build_tool_desc(self) -> str:
        return "\n".join([f"- {name}: {info['description']}" for name, info in TOOL_REGISTRY.items()])

    def _clean_final_answer(self, text: str) -> str:
        """Strip Thought, JSON blocks and tokens."""
        clean_text = re.sub(r'<\|.*?\|>', '', text)
        clean_text = re.sub(r'\{.*\}', '', clean_text, flags=re.DOTALL)
        clean_text = re.sub(r'(?i)thought:', '', clean_text)
        clean_text = clean_text.replace("assistant", "").replace("user", "").replace("Decision:", "")
        return clean_text.strip()

    def _extract_json(self, text: str) -> Optional[str]:
        """Find the first complete JSON object using brace matching."""
        start = text.find('{')
        if start == -1: return None
        count = 0
        for i in range(start, len(text)):
            if text[i] == '{': count += 1
            elif text[i] == '}':
                count -= 1
                if count == 0:
                    return text[start:i+1]
        return None

    def _route_intent(self, query: str) -> Optional[Dict]:
        """Deterministic keyword-based router — runs BEFORE the LLM.
        Returns a tool call dict if a pattern matches, else None (LLM fallback)."""
        q = query.upper()

        # Pattern 1: User-specific queries (e.g. "USER_123", "USER 0", "user_789")
        match = re.search(r'USER[_\s]?(\d+)', q)
        if match:
            user_id = f"USER_{match.group(1)}"
            return {"tool": "get_user_risk_profile", "args": {"user_id": user_id}}

        # Pattern 2: High-risk / top transactions
        if any(kw in q for kw in ["HIGH RISK", "TOP TRANSACTION", "MOST DANGEROUS", "HIGHEST RISK", "RISKIEST"]):
            return {"tool": "get_high_risk_transactions", "args": {"limit": 10}}

        # Pattern 3: General knowledge / explanation queries → RAG
        if any(kw in q for kw in ["WHAT IS", "WHAT ARE", "EXPLAIN", "HOW TO", "DEFINE", "DESCRIBE", "TREND", "CFPB", "DISPUTE", "IDENTITY THEFT", "VELOCITY"]):
            return {"tool": "query_rag", "args": {"question": query}}

        # No match → fall back to LLM reasoning
        return None

    def analyze(self, question: str) -> Dict[str, Any]:
        tool_desc = self._build_tool_desc()
        prompt = SYSTEM_PROMPT.format(tool_descriptions=tool_desc)
        
        actions_log = []
        current_context = f"User Request: {question}"

        # ── Step 0: Deterministic Intent Router ──
        # If router matches, execute the tool and IMMEDIATELY return — no LLM needed.
        routed_call = self._route_intent(question)
        if routed_call:
            tool_name = routed_call["tool"]
            args = routed_call["args"]
            if tool_name in TOOL_REGISTRY:
                print(f"🎯 Router matched → {tool_name}({args})")
                result = TOOL_REGISTRY[tool_name]["fn"](**args)
                result_str = json.dumps(result, indent=2)
                actions_log.append({"step": 0, "tool": tool_name, "args": args, "result": result_str[:400] + "..."})

                # Format a clean plain-text report without invoking the LLM
                if tool_name == "query_rag":
                    answer = f"**Knowledge Base Result:**\n\n{result}"
                elif tool_name == "get_user_risk_profile":
                    found = result.get("found", False)
                    if found:
                        answer = (
                            f"**User Risk Profile: {args.get('user_id')}**\n\n"
                            f"- Security Level: `{result.get('security_level', 'N/A')}`\n"
                            f"- Average Risk Score: `{result.get('avg_risk', 'N/A')}`\n"
                            f"- High Risk Transactions: `{result.get('high_risk_count', 'N/A')}`\n"
                            f"- Total Transactions: `{result.get('txn_count', 'N/A')}`\n"
                            f"- Risk Distribution: `{result.get('risk_distribution', 'N/A')}`"
                        )
                    else:
                        answer = f"⚠️ User `{args.get('user_id')}` was not found in the database."
                else:
                    rows = result if isinstance(result, list) else [result]
                    lines = [f"**Top {len(rows)} High Risk Transactions:**\n"]
                    for r in rows[:5]:
                        lines.append(
                            f"- User: `{r.get('USER_ID','?')}` | Score: `{r.get('COMBINED_RISK_SCORE','?'):.2f}` | Level: `{r.get('RISK_LEVEL','?')}`"
                        )
                    answer = "\n".join(lines)

                return {
                    "answer": answer,
                    "actions": actions_log,
                    "status": "success"
                }

        # ── Steps 1-5: LLM Reasoning Loop (Fallback for unknown queries) ──
        for i in range(5):
            full_prompt = f"{prompt}\n\nExisting Context:\n{current_context}\n\nDecision:"
            response = self.llm.generate(full_prompt, max_tokens=256)
            
            # Try to parse a tool call from LLM output
            json_str = self._extract_json(response)
            if json_str:
                try:
                    call = json.loads(json_str)
                    tool_name = call.get("tool")
                    args = call.get("args", {})
                    
                    if tool_name in TOOL_REGISTRY:
                        print(f"🛠️ Calling tool: {tool_name}({args})")
                        result = TOOL_REGISTRY[tool_name]["fn"](**args)
                        result_str = json.dumps(result, indent=2)
                        actions_log.append({"step": i+1, "tool": tool_name, "args": args, "result": result_str[:200]+"..."})
                        current_context += f"\nTurn {i+1}: Called {tool_name}. Result: {result_str}"
                        continue
                except Exception as e:
                    print(f"⚠️ Tool parse error: {e}")
                    
            # Final answer
            final_report = self._clean_final_answer(response)
            
            if not final_report and i < 4:
                current_context += f"\nTurn {i+1} Reasoning: {response.strip()}"
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
