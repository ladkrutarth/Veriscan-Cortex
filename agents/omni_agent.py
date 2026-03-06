"""
SupremeOmniAgent — The master orchestrator for Veriscan's agentic intelligence.
Combines Security Analysis (LocalGuardAgent) and Financial Advice (FinancialAdvisorAgent).
"""

import re
import traceback
from typing import Any, Dict, List, Optional
from models.guard_agent_local import LocalGuardAgent
from agents.financial_advisor_agent import FinancialAdvisorAgent

class SupremeOmniAgent:
    def __init__(self):
        self.guard_agent = LocalGuardAgent()
        self.advisor_agent = FinancialAdvisorAgent()

    def _classify_domain(self, query: str) -> str:
        q = query.upper()
        
        # Security keywords
        security_kw = ["INVESTIGATE", "RISK", "FRAUD", "ANOMALY", "SCAN", "SECURITY", "THREAT", "BUREAU", "CFPB"]
        # Financial keywords
        financial_kw = ["SAVE", "SPENDING", "BUDGET", "ADVICE", "MONEY", "COFFEE", "DINING", "SUBSCRIPTION", "TRANSACTION", "HISTORY"]
        
        has_security = any(kw in q for kw in security_kw) or bool(re.search(r'USER[_\s]?\d+', q))
        has_financial = any(kw in q for kw in financial_kw)
        
        if has_security and has_financial:
            return "hybrid"
        if has_security:
            return "security"
        if has_financial:
            return "financial"
        
        return "hybrid"  # Default to hybrid for advanced reasoning

    def chat(self, user_id: str, message: str) -> Dict[str, Any]:
        domain = self._classify_domain(message)
        print(f"🧠 SupremeOmniAgent Orchestration → {domain.upper()} domain")
        
        actions = []
        reply = ""
        status = "success"
        show_chart = False
        chart_data = {}
        
        # Enforce detailed reporting for anomaly protocol
        is_diagnostic = "Initiate system-wide anomaly detection" in message
        if is_diagnostic:
            message += " Please provide a detailed professional report of approximately 100 words explaining any detected risks or anomalies."

        try:
            # 1. Active Monitoring / Safety Check
            if domain in ["security", "hybrid"]:
                print("🛡️ Running background safety scan...")
                security_res = self.guard_agent.analyze(message)
                actions.extend(security_res.get("actions", []))
                if domain == "security":
                    reply = security_res.get("answer", "")
            
            # 2. Financial Context / Advice
            if domain in ["financial", "hybrid"]:
                print("💰 Retrieving financial insights...")
                financial_res = self.advisor_agent.chat(message, user_id)
                actions.extend(financial_res.get("tool_results", []))
                
                # Propagate chart metadata
                show_chart = financial_res.get("show_chart", False)
                if show_chart:
                    chart_data = self.advisor_agent.get_chart_data(user_id)

                if domain == "financial":
                    reply = financial_res.get("reply", "")
                elif domain == "hybrid":
                    security_answer = security_res.get("answer", "No security threats detected.")
                    financial_answer = financial_res.get("reply", "No specific financial advice found.")
                    reply = f"### 🛡️ Security Analysis\n{security_answer}\n\n### 💰 Financial Insights\n{financial_answer}"

            return {
                "user_id": user_id,
                "reply": reply or "Analysis complete. System status is within normal operational parameters.",
                "actions": actions,
                "domain": domain,
                "status": status,
                "show_chart": show_chart,
                "chart_data": chart_data
            }

        except Exception as e:
            traceback.print_exc()
            return {
                "user_id": user_id,
                "reply": f"Omni-Agent Error: {str(e)}",
                "actions": [],
                "domain": domain,
                "status": "error"
            }

if __name__ == "__main__":
    omni = SupremeOmniAgent()
    print(omni.chat("USER_0001", "Check my risk and suggest a coffee budget"))
