import json
import re
from typing import Dict, Any, Optional
from .base import BaseAgent, AgentResult, AgentAction

class ProfileAgent(BaseAgent):
    """Handles user-specific investigations with pre-computed data."""
    
    def run(self, query: str, session_id: Optional[str] = None) -> AgentResult:
        from models.guard_agent_local import tool_get_user_risk_profile
        
        user_match = re.search(r'USER[_\s]?(\d+)', query.upper())
        user_id = f"USER_{user_match.group(1)}" if user_match else "USER_001"
        
        profile = tool_get_user_risk_profile(user_id)
        actions = [AgentAction(step=0, tool="get_user_risk_profile", args={"user_id": user_id}, result=json.dumps(profile, indent=2)[:400])]

        if not profile.get("found"):
            return AgentResult(answer=f"User `{user_id}` not found.", actions=actions, session_id=session_id)

        report = f"**🛡️ Security Investigation: {user_id}**\n\nRisk Score: {profile.get('avg_risk', 0):.2f}. Security Level: {profile.get('security_level', 'N/A')}."
        return AgentResult(answer=report, actions=actions, session_id=session_id)
