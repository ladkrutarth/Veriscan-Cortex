"""Regression guards: model output must never be able to run a HITL-gated tool."""

from __future__ import annotations

import json

from agents.financial_advisor_agent import FinancialAdvisorAgent
from agents.tool_registry import ToolRegistry


def test_advisor_registry_exposes_no_hitl_tools():
    registry = FinancialAdvisorAgent(llm=None).build_tool_registry("USER_0001")
    assert len(registry) > 0
    assert [s["name"] for s in registry.to_json_schema_list() if s["requires_hitl"]] == []


def test_advisor_oneshot_plan_does_not_auto_approve(monkeypatch, fake_llm_factory):
    executed: list = []
    reg = ToolRegistry()
    reg.register("challenge_auth", "Sensitive.", lambda **_: executed.append("challenge") or {"ok": True})
    reg.register("spending_summary", "Summary.", lambda **_: {"total": 1})

    agent = FinancialAdvisorAgent(llm=fake_llm_factory([json.dumps({"tools": [{"name": "challenge_auth"}]})]))
    monkeypatch.setattr(agent, "build_tool_registry_for_intent", lambda user_id, message: (reg, "general"))

    results = agent._oneshot_tool_plan("lock my account", "USER_0001")
    assert executed == []
    assert any(r.get("pending_approval") for r in results)
