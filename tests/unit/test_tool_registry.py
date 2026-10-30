from __future__ import annotations

from agents.tool_registry import REQUIRE_HITL, ToolRegistry, truncate_observation


def _registry() -> tuple[ToolRegistry, list]:
    calls: list = []
    reg = ToolRegistry()
    reg.register(
        "add",
        "Add two ints.",
        lambda a=0, b=0: {"sum": int(a) + int(b)},
        {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}},
    )
    reg.register("listing", "Returns a list.", lambda: [1, 2])
    reg.register("scalar", "Returns a scalar.", lambda: 7)
    reg.register("boom", "Raises.", lambda: 1 / 0)
    reg.register("challenge_auth", "Sensitive.", lambda user_id="u": calls.append(user_id) or {"ok": True})
    return reg, calls


def test_call_returns_dict_result():
    reg, _ = _registry()
    assert reg.call("add", {"a": 2, "b": 3}) == {"sum": 5}


def test_call_wraps_list_and_scalar_results():
    reg, _ = _registry()
    assert reg.call("listing") == {"tool": "listing", "items": [1, 2]}
    assert reg.call("scalar") == {"tool": "scalar", "result": 7}


def test_unknown_tool_returns_error():
    reg, _ = _registry()
    assert reg.call("nope")["error"] == "Unknown tool: nope"


def test_tool_exception_becomes_error_dict():
    reg, _ = _registry()
    out = reg.call("boom")
    assert out["tool"] == "boom" and "division" in out["error"]


def test_undeclared_args_from_llm_are_dropped_on_retry():
    reg, _ = _registry()
    assert reg.call("add", {"a": 1, "b": 1, "invented": True}) == {"sum": 2}


def test_challenge_auth_is_always_hitl_gated():
    assert "challenge_auth" in REQUIRE_HITL
    reg, _ = _registry()
    assert reg.get("challenge_auth").requires_hitl


def test_hitl_tool_refused_without_approval():
    reg, calls = _registry()
    out = reg.call("challenge_auth", {"user_id": "U1"})
    assert out["pending_approval"] is True
    assert calls == []


def test_hitl_tool_runs_with_explicit_approval():
    reg, calls = _registry()
    assert reg.call("challenge_auth", {"user_id": "U1"}, approved=True) == {"ok": True}
    assert calls == ["U1"]


def test_subset_keeps_order_and_skips_unknown():
    reg, _ = _registry()
    sub = reg.subset(["scalar", "missing", "add"])
    assert sub.names() == ["scalar", "add"]
    assert len(sub) == 2 and "add" in sub and "boom" not in sub


def test_prompt_schema_marks_approval_tools():
    reg, _ = _registry()
    text = reg.schemas_for_prompt()
    assert "add(a: integer, b: integer)" in text
    assert "challenge_auth(" in text and "[REQUIRES_APPROVAL]" in text


def test_truncate_observation_caps_length():
    out = truncate_observation({"x": "y" * 5000}, max_chars=100)
    assert len(out) == 100 and out.endswith("...")
