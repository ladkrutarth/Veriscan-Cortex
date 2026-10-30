from __future__ import annotations

import json

import pytest

from agents.react_loop import ReactLoop, extract_json
from agents.tool_registry import ToolRegistry


def tool(name: str, **args) -> str:
    return json.dumps({"action": "tool", "tool": name, "args": args})


def final(answer: str) -> str:
    return json.dumps({"action": "final", "answer": answer})


@pytest.fixture
def registry():
    calls: list = []
    reg = ToolRegistry()
    reg.register(
        "balance",
        "Account balance.",
        lambda user_id="?": calls.append(("balance", user_id)) or {"balance": 100, "user_id": user_id},
        {"type": "object", "properties": {"user_id": {"type": "string"}}},
    )
    reg.register("fails", "Always errors.", lambda: {"error": "nope"})
    reg.register("challenge_auth", "Sensitive.", lambda: calls.append(("challenge",)) or {"ok": True})
    reg.calls = calls
    return reg


@pytest.mark.parametrize(
    "text, expected",
    [
        ('{"action":"final","answer":"hi"}', {"action": "final", "answer": "hi"}),
        ('```json\n{"a": 1}\n```', {"a": 1}),
        ('Sure! {"a": {"b": 2}} hope that helps', {"a": {"b": 2}}),
        ("not json at all", None),
        ("[1, 2, 3]", None),
        ("", None),
    ],
)
def test_extract_json(text, expected):
    assert extract_json(text) == expected


def test_tool_then_final(registry, fake_llm_factory):
    llm = fake_llm_factory([tool("balance"), final("You have 100.")])
    out = ReactLoop(llm, registry, default_args={"user_id": "U7"}).run("balance?")
    assert out["status"] == "success"
    assert out["reply"] == "You have 100."
    assert out["tools_used"] == ["balance"]
    assert registry.calls == [("balance", "U7")]  # default_args injected


def test_hitl_tool_requested_by_model_is_not_executed(registry, fake_llm_factory):
    llm = fake_llm_factory([tool("challenge_auth")])
    out = ReactLoop(llm, registry).run("lock the account")
    assert out["status"] == "pending_approval"
    assert out["pending_approval"]["tool"] == "challenge_auth"
    assert registry.calls == []


def test_hitl_tool_runs_when_caller_approved(registry, fake_llm_factory):
    llm = fake_llm_factory([tool("challenge_auth"), final("Challenge sent.")])
    out = ReactLoop(llm, registry).run("lock the account", approved=True)
    assert out["status"] == "success"
    assert registry.calls == [("challenge",)]


def test_duplicate_tool_call_is_blocked(registry, fake_llm_factory):
    llm = fake_llm_factory([tool("balance"), tool("balance"), final("ok")])
    out = ReactLoop(llm, registry, max_tool_calls=3).run("q")
    assert len(registry.calls) == 1
    assert any("duplicate" in t for t in out["trace"])


def test_tool_budget_forces_final(registry, fake_llm_factory):
    llm = fake_llm_factory([tool("balance", user_id="a"), tool("balance", user_id="b")])
    out = ReactLoop(llm, registry, max_tool_calls=1).run("q")
    assert out["status"] == "needs_synth"
    assert len(registry.calls) == 1


def test_step_limit_is_respected(registry, fake_llm_factory):
    llm = fake_llm_factory([tool("missing_tool")] * 10)
    out = ReactLoop(llm, registry, max_steps=3).run("q")
    assert out["status"] == "max_steps"
    assert len(llm.prompts) == 3


def test_parse_failure_uses_caller_fallback(registry, fake_llm_factory):
    llm = fake_llm_factory(["I am not JSON"])
    out = ReactLoop(llm, registry).run("q", on_parse_fail=lambda: {"reply": "keyword answer"})
    assert out["reply"] == "keyword answer"
    assert out["used_fallback"] is True
    assert out["status"] == "fallback_parse"


def test_llm_exception_without_fallback_is_error(registry, fake_llm_factory):
    llm = fake_llm_factory([RuntimeError("metal crashed")])
    out = ReactLoop(llm, registry).run("q")
    assert out["status"] == "error"
    assert "metal crashed" in out["reply"]


def test_repeated_tool_errors_stop_the_loop(registry, fake_llm_factory):
    llm = fake_llm_factory([tool("fails"), tool("fails", retry=1), final("unused")])
    out = ReactLoop(llm, registry, max_tool_calls=3).run("q")
    assert out["status"] == "needs_synth"
    assert out["tools_used"] == ["fails", "fails"]
