import json
from typing import Dict, Any, Optional
from .base import BaseAgent, AgentResult, AgentAction

class RiskScannerAgent(BaseAgent):
    """Handles system-wide risk scanning with pre-computed data."""
    
    def run(self, query: str, session_id: Optional[str] = None) -> AgentResult:
        from models.guard_agent_local import tool_get_high_risk_transactions # Dependency on tools
        
        limit = 10
        txns = tool_get_high_risk_transactions(limit=limit)
        actions = [AgentAction(step=0, tool="get_high_risk_transactions", args={"limit": limit}, result=json.dumps(txns[:3], indent=2)[:400])]

        if not txns:
            return AgentResult(answer="No transaction data available.", actions=actions, session_id=session_id)

        critical = [t for t in txns if t.get("RISK_LEVEL") == "CRITICAL"]
        high = [t for t in txns if t.get("RISK_LEVEL") == "HIGH"]
        categories = {}
        for t in txns:
            cat = t.get("CATEGORY", "Unknown")
            categories[cat] = categories.get(cat, 0) + 1

        top_cat = max(categories, key=categories.get) if categories else "N/A"
        
        report = f"**🛡️ System-Wide Risk Scan Report**\n\nDetected {len(critical)} CRITICAL transactions. Most targeted category: {top_cat}."
        return AgentResult(answer=report, actions=actions, session_id=session_id)
