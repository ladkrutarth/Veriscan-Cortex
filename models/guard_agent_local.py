"""
Veriscan — Multi-Agent GuardAgent: Fast Autonomous Private Security Analyst
Uses specialized sub-agents for instant responses with LLM-powered synthesis.

Architecture:
  ┌─ RAGAgent ────────── Knowledge & Policy queries (instant)
  ├─ RiskScanAgent ──── System-wide risk scanning (instant)
  ├─ UserProfileAgent ─ User-specific investigations (instant)
  └─ SynthesisAgent ─── Complex reasoning (1 LLM call)
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
# PRE-CACHED DATA (loaded ONCE at import time for instant access)
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

# Singleton RAG engine
_rag_engine = None
def _get_rag():
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngineLocal()
        _rag_engine.index_data()
    return _rag_engine

# ---------------------------------------------------------------------------
# FAST TOOLS (use cached DataFrames — no file I/O)
# ---------------------------------------------------------------------------

def tool_get_user_risk_profile(user_id: str) -> Dict[str, Any]:
    """Retrieve risk profile for a specific user (cached — instant)."""
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
    """Get the top highest-risk transactions (cached — instant)."""
    _load_cache()
    if _fraud_df is None: return []
    df = _fraud_df.sort_values("COMBINED_RISK_SCORE", ascending=False).head(limit)
    return df.to_dict("records")

def tool_query_rag(question: str) -> str:
    """Search local knowledge base (singleton engine — fast)."""
    try:
        return _get_rag().get_context_for_query(question)
    except Exception as e:
        return f"RAG tool error: {e}"

# ---------------------------------------------------------------------------
# TOOL REGISTRY
# ---------------------------------------------------------------------------
TOOL_REGISTRY = {
    "get_user_risk_profile": {
        "fn": tool_get_user_risk_profile,
        "description": "Risk profile for a specific User ID. Args: user_id (str)."
    },
    "get_high_risk_transactions": {
        "fn": tool_get_high_risk_transactions,
        "description": "List the most dangerous transactions. Args: limit (int)."
    },
    "query_rag": {
        "fn": tool_query_rag,
        "description": "Search knowledge base for fraud trends or CFPB policies. Args: question (str)."
    }
}

# ---------------------------------------------------------------------------
# SUB-AGENT: RAG Knowledge Agent (routes to SynthesisAgent for expert answers)
# ---------------------------------------------------------------------------
class RAGAgent:
    """Handles policy/knowledge queries by routing through SynthesisAgent for LLM synthesis."""

    def __init__(self, llm: LocalLLM):
        self.llm = llm

    def run(self, query: str) -> Dict[str, Any]:
        """Route knowledge queries through LLM synthesis for expert-level answers."""
        # Get RAG context
        rag = _get_rag()
        results = rag.query(query, n_results=5)
        raw_context = "\n".join([r["text"] for r in results]) if results else "No documents found."
        actions = [{"step": 0, "tool": "query_rag", "args": {"question": query}, "result": raw_context[:500]}]

        # Use LLM to synthesize expert answer
        prompt = f"""You are a Senior Fraud Intelligence Analyst. Answer this question in 150-200 words.

Question: {query}

Evidence from CFPB database and fraud intelligence:
{raw_context[:1500]}

Write a professional, detailed analysis. Include:
- Clear definitions for any fraud terms mentioned
- Specific references to data (companies, states, amounts from the evidence)
- How Veriscan's ML detection capabilities relate to this topic
- A concrete recommendation at the end

Analysis:"""

        answer = self.llm.generate(prompt, max_tokens=300)
        answer = re.sub(r'<\|.*?\|>', '', answer).strip()
        for artifact in ["assistant", "user", "Question:", "Context:", "Evidence:", "Analysis:"]:
            answer = answer.replace(artifact, "")
        answer = answer.strip()

        if not answer or len(answer) < 30:
            # Structured fallback with query-aware analysis
            if results:
                top_conf = results[0]["confidence"]
                metadata = results[0].get("metadata", {})
                states = set()
                companies = set()
                for r in results:
                    m = r.get("metadata", {})
                    if m.get("state"): states.add(m["state"])
                    if m.get("company"): companies.add(m["company"])

                answer = f"""**🔍 Knowledge Base Intelligence Report**

**Query:** {query}

**Evidence Retrieved:** {len(results)} documents (Top confidence: {top_conf:.0%})

**Key Sources:**
"""
                for i, r in enumerate(results[:3], 1):
                    m = r.get("metadata", {})
                    company = m.get("company", "Unknown")
                    state = m.get("state", "N/A")
                    answer += f"\n{i}. **{company}** (State: {state}, Relevance: {r['confidence']:.0%})\n   {r['text'][:200]}...\n"

                if companies:
                    answer += f"\n**Entities Involved:** {', '.join(list(companies)[:5])}"
                if states:
                    answer += f"\n**Geographic Coverage:** {', '.join(list(states)[:5])}"
                answer += f"\n\n**Recommendation:** Based on {len(results)} evidence documents, further investigation into the identified patterns is recommended. The data suggests active monitoring of identified entities and geographic regions."

        return {"answer": answer, "actions": actions, "status": "success"}



# ---------------------------------------------------------------------------
# SUB-AGENT: Risk Scanner (NO LLM — instant)
# ---------------------------------------------------------------------------
class RiskScanAgent:
    """Handles system-wide risk scanning with pre-computed data."""

    def run(self, query: str, limit: int = 10) -> Dict[str, Any]:
        txns = tool_get_high_risk_transactions(limit=limit)
        actions = [{"step": 0, "tool": "get_high_risk_transactions", "args": {"limit": limit}, "result": json.dumps(txns[:3], indent=2)[:400]}]

        if not txns:
            return {"answer": "No transaction data available.", "actions": actions, "status": "success"}

        # Build detailed analytical report
        critical = [t for t in txns if t.get("RISK_LEVEL") == "CRITICAL"]
        high = [t for t in txns if t.get("RISK_LEVEL") == "HIGH"]
        categories = {}
        merchants = {}
        users = set()
        for t in txns:
            cat = t.get("CATEGORY", "Unknown")
            mer = t.get("MERCHANT", "Unknown")
            categories[cat] = categories.get(cat, 0) + 1
            merchants[mer] = merchants.get(mer, 0) + 1
            users.add(t.get("USER_ID", "Unknown"))

        top_cat = max(categories, key=categories.get) if categories else "N/A"
        top_mer = max(merchants, key=merchants.get) if merchants else "N/A"

        report = f"""**🛡️ System-Wide Risk Scan Report**

**Threat Summary:** Scanned the top {limit} highest-risk transactions across the entire system.

| Metric | Value |
|--------|-------|
| CRITICAL Risk | {len(critical)} transactions |
| HIGH Risk | {len(high)} transactions |
| Unique Users Affected | {len(users)} |
| Most Targeted Category | {top_cat} ({categories.get(top_cat, 0)} occurrences) |
| Most Targeted Merchant | {top_mer[:40]} |

**Top 3 Threats:**
"""
        for i, t in enumerate(txns[:3], 1):
            report += f"\n{i}. **{t.get('TRANSACTION_ID', 'N/A')}** — User: `{t.get('USER_ID', 'N/A')}`, Category: {t.get('CATEGORY', 'N/A')}, Risk Score: {t.get('COMBINED_RISK_SCORE', 0):.1f}, Z-Score: {t.get('ZSCORE_FLAG', 0):.2f}"

        report += f"\n\n**Recommendation:** Immediate review of {len(critical)} CRITICAL-risk transactions is advised. The {top_cat} category shows elevated fraud activity and should be flagged for enhanced monitoring."

        return {"answer": report, "actions": actions, "status": "success"}

# ---------------------------------------------------------------------------
# SUB-AGENT: User Profile Agent (NO LLM — instant)
# ---------------------------------------------------------------------------
class UserProfileAgent:
    """Handles user-specific investigations with pre-computed data."""

    def run(self, user_id: str) -> Dict[str, Any]:
        profile = tool_get_user_risk_profile(user_id)
        actions = [{"step": 0, "tool": "get_user_risk_profile", "args": {"user_id": user_id}, "result": json.dumps(profile, indent=2)[:400]}]

        if not profile.get("found"):
            return {"answer": f"User `{user_id}` was not found in the database.", "actions": actions, "status": "success"}

        # Build detailed user report
        sec_level = profile.get("security_level", "N/A")
        avg_risk = profile.get("avg_risk", 0)
        avg_combined = profile.get("avg_combined_score", 0)
        high_risk_count = profile.get("high_risk_count", 0)
        txn_count = profile.get("txn_count", 0)
        risk_dist = profile.get("risk_distribution", {})

        risk_label = "🔴 HIGH" if avg_risk > 50 or high_risk_count > 5 else "🟡 MEDIUM" if avg_risk > 20 else "🟢 LOW"

        report = f"""**🛡️ Security Investigation: {user_id}**

**Overall Threat Level: {risk_label}**

| Attribute | Value |
|-----------|-------|
| Security Level | {sec_level} |
| Average Risk Score | {avg_risk:.2f} |
| Average Combined Score | {avg_combined:.2f} |
| Total Transactions | {txn_count} |
| High-Risk Transactions | {high_risk_count} |

**Risk Distribution:**
"""
        for level, count in risk_dist.items():
            pct = (count / txn_count * 100) if txn_count > 0 else 0
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            report += f"- {level}: {count} ({pct:.1f}%) {bar}\n"

        if high_risk_count > 5:
            report += f"\n**⚠️ Alert:** This user has {high_risk_count} high-risk transactions. Recommend immediate review, enhanced authentication, and potential account restriction."
        elif avg_risk > 20:
            report += f"\n**🟡 Watch:** Elevated average risk score. Recommend monitoring for unusual patterns and velocity changes."
        else:
            report += f"\n**✅ Status:** User appears to be operating within normal parameters. No immediate action required."

        return {"answer": report, "actions": actions, "status": "success"}

# ---------------------------------------------------------------------------
# SUB-AGENT: Synthesis Agent (1 LLM call for complex multi-tool queries)
# ---------------------------------------------------------------------------
class SynthesisAgent:
    """Handles complex queries that need multiple tools + LLM reasoning."""

    def __init__(self, llm: LocalLLM):
        self.llm = llm

    def run(self, query: str) -> Dict[str, Any]:
        actions = []
        context_parts = []

        # Execute ALL relevant tools in sequence (deterministic, no LLM for tool selection)
        # 1. Check for user IDs
        user_matches = re.findall(r'USER[_\s]?(\d+)', query.upper())
        for uid in user_matches[:3]:  # Max 3 users
            user_id = f"USER_{uid}"
            result = tool_get_user_risk_profile(user_id)
            actions.append({"step": len(actions), "tool": "get_user_risk_profile", "args": {"user_id": user_id}, "result": json.dumps(result, indent=2)[:400]})
            context_parts.append(f"User Profile ({user_id}): {json.dumps(result)}")

        # 2. Get high-risk transactions
        q = query.upper()
        if any(kw in q for kw in ["RISK", "DANGEROUS", "THREAT", "FRAUD", "TRANSACTION", "SCAN", "REPORT", "TOP", "MERCHANT", "COMMON", "BASED ON", "PREDICTIVE", "ANALYTICS", "HEALTH", "STATE"]):
            txns = tool_get_high_risk_transactions(limit=10)
            actions.append({"step": len(actions), "tool": "get_high_risk_transactions", "args": {"limit": 10}, "result": json.dumps(txns[:3], indent=2)[:400]})
            # Build merchant/category summary
            merchants = {}
            categories = {}
            for t in txns:
                m = t.get("MERCHANT", "Unknown")
                c = t.get("CATEGORY", "Unknown")
                merchants[m] = merchants.get(m, 0) + 1
                categories[c] = categories.get(c, 0) + 1
            context_parts.append(f"Top 10 High-Risk Transactions: {json.dumps(txns)}")
            context_parts.append(f"Merchant Frequency in High-Risk: {json.dumps(merchants)}")
            context_parts.append(f"Category Frequency in High-Risk: {json.dumps(categories)}")

        # 3. Get RAG context (always for synthesis queries)
        rag_result = tool_query_rag(query)
        actions.append({"step": len(actions), "tool": "query_rag", "args": {"question": query}, "result": rag_result[:400]})
        context_parts.append(f"Knowledge Base Intel: {rag_result}")

        # 4. ONE LLM call to synthesize — with expert prompt
        combined_context = "\n\n".join(context_parts)

        synthesis_prompt = f"""You are a Senior Financial Crime Investigator and Compliance Officer at Veriscan, a leading fraud detection platform.

The analyst asked: "{query}"

Here is all the intelligence gathered from our multi-agent system:
{combined_context[:3000]}

INSTRUCTIONS:
- Write a comprehensive 200-300 word professional security analysis.
- Structure your response with clear sections: Findings, Risk Assessment, and Recommendations.
- Reference SPECIFIC data: user IDs, transaction IDs, risk scores, merchant names, categories, percentages.
- If comparing users, create a side-by-side comparison with specific metrics.
- If analyzing merchants, explain WHY certain merchants are targeted (high-value categories, online vs POS, geographic patterns).
- If asked about compliance/policy, cite CFPB regulations and explain how Veriscan's scoring helps.
- If asked about technical concepts (Z-Scores, velocity, etc.), give a textbook definition AND practical application.
- End with 2-3 specific, actionable recommendations.
- Use professional security analyst tone throughout.

Your detailed investigation report:"""

        answer = self.llm.generate(synthesis_prompt, max_tokens=350)
        answer = re.sub(r'<\|.*?\|>', '', answer).replace("assistant", "").replace("user", "").strip()

        if not answer or len(answer) < 30:
            answer = self._build_fallback_report(query, actions, context_parts)

        return {"answer": answer, "actions": actions, "status": "success"}

    def _build_fallback_report(self, query: str, actions: list, context_parts: list) -> str:
        """Generate a structured report if LLM fails."""
        report = f"**🛡️ Multi-Source Investigation Report**\n\n"
        report += f"**Query:** {query}\n\n"
        report += f"**Data Sources Analyzed:** {len(actions)}\n\n"
        for a in actions:
            report += f"- **{a['tool']}**: {a['result'][:200]}...\n\n"
        report += "**Recommendation:** Review the tool results above for detailed findings. The multi-agent system gathered intelligence from user profiles, transaction databases, and the knowledge base."
        return report

# ---------------------------------------------------------------------------
# MASTER AGENT: Multi-Agent Router
# ---------------------------------------------------------------------------

class LocalGuardAgent:
    """Multi-Agent GuardAgent with specialized sub-agents for fast responses.
    
    Routes queries to the fastest appropriate sub-agent:
    - RAGAgent: Knowledge/policy queries (instant, no LLM)
    - RiskScanAgent: System-wide scans (instant, no LLM)
    - UserProfileAgent: User investigations (instant, no LLM)
    - SynthesisAgent: Complex multi-tool reasoning (1 LLM call)
    """

    def __init__(self):
        self.llm = LocalLLM()
        self.rag_agent = RAGAgent(self.llm)
        self.risk_agent = RiskScanAgent()
        self.user_agent = UserProfileAgent()
        self.synthesis_agent = SynthesisAgent(self.llm)

    def _classify_query(self, query: str) -> str:
        """Classify query intent using deterministic rules (no LLM)."""
        q = query.upper()

        # Check for user IDs
        user_ids = re.findall(r'USER[_\s]?\d+', q)
        has_user = len(user_ids) > 0
        has_multi_user = len(user_ids) >= 2

        # Multi-user comparison always goes to synthesis
        if has_multi_user:
            return "synthesis"

        # Check for risk scan keywords
        is_risk_scan = any(kw in q for kw in [
            "HIGH RISK", "TOP TRANSACTION", "MOST DANGEROUS", "HIGHEST RISK",
            "RISKIEST", "SYSTEM SCAN", "STATE OF", "FULL REPORT"
        ])

        # Check for knowledge/RAG keywords — BROAD matching
        is_knowledge = any(kw in q for kw in [
            "WHAT IS", "WHAT ARE", "EXPLAIN", "HOW TO", "DEFINE", "DESCRIBE",
            "TREND", "CFPB", "DISPUTE", "IDENTITY THEFT", "VELOCITY",
            "LATE FEE", "REWARD", "COMPLAINT", "PHISHING", "SCAM",
            "BILLING", "CREDIT REPORT", "BALANCE TRANSFER", "UNAUTHORIZED",
            "INTEREST RATE", "CUSTOMER SERVICE", "SYNTHETIC", "TRUE NAME",
            "Z-SCORE", "COMPLIANCE", "REGULATION", "POLICY", "SEARCH",
            "DIFFERENCE", "HARDER", "RESOLUTION", "TIMELINE",
            "COMMON", "CONSUMER", "PROCEDURE", "STANDARD"
        ])

        # Complex queries need multiple tools
        complexity = int(has_user) + int(is_risk_scan) + int(is_knowledge)
        if complexity >= 2:
            return "synthesis"

        # Single-purpose queries
        if has_user and not is_risk_scan and not is_knowledge:
            return "user_profile"
        if is_risk_scan and not has_user:
            return "risk_scan"
        if is_knowledge and not has_user:
            return "knowledge"

        # Default to synthesis for ambiguous queries
        return "synthesis"

    def analyze(self, question: str) -> Dict[str, Any]:
        """Route to the fastest appropriate sub-agent."""
        agent_type = self._classify_query(question)
        print(f"⚡ Multi-Agent Router → {agent_type.upper()} agent")

        if agent_type == "knowledge":
            return self.rag_agent.run(question)
        elif agent_type == "risk_scan":
            return self.risk_agent.run(question)
        elif agent_type == "user_profile":
            user_match = re.search(r'USER[_\s]?(\d+)', question.upper())
            user_id = f"USER_{user_match.group(1)}" if user_match else "USER_001"
            return self.user_agent.run(user_id)
        else:  # synthesis
            return self.synthesis_agent.run(question)


if __name__ == "__main__":
    agent = LocalGuardAgent()
    print("--- User Profile Query ---")
    print(agent.analyze("Investigate risk for USER_0")["answer"])
    print("\n--- Risk Scan Query ---")
    print(agent.analyze("Show me the top 5 most dangerous transactions")["answer"])
    print("\n--- Knowledge Query ---")
    print(agent.analyze("What is velocity in fraud detection?")["answer"])
