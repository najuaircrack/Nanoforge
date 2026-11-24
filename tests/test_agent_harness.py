from __future__ import annotations

from nanoforge.agent import (
    AgentHarness,
    ExecutionBudget,
    ReplayRunner,
    StaticApproval,
    ToolExecutor,
    ToolRegistry,
    ToolSpec,
)


def _echo_spec(side_effect: str = "read", approval_policy: str = "never") -> ToolSpec:
    return ToolSpec(
        name="echo",
        description="Echo a value.",
        input_schema={"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"], "additionalProperties": False},
        fn=lambda value: {"value": value},
        capabilities=frozenset({"workspace.read"} if side_effect == "read" else {"workspace.write"}),
        approval_policy=approval_policy,
        side_effect=side_effect,
    )


def test_schema_validation_and_duplicate_registration():
    registry = ToolRegistry([_echo_spec()])
    try:
        registry.register(_echo_spec())
    except KeyError:
        pass
    else:
        raise AssertionError("duplicate tools must fail")
    result = ToolExecutor(registry).execute({"name": "echo", "arguments": {"value": 3}})
    assert result.status == "error"
    assert "must be string" in (result.error or "")


def test_write_requires_approval_and_can_be_allowed():
    registry = ToolRegistry([_echo_spec("write", "always")])
    denied = ToolExecutor(registry, capabilities={"workspace.write"}).execute({"name": "echo", "arguments": {"value": "x"}})
    assert denied.status == "denied"
    allowed = ToolExecutor(
        registry,
        approval=StaticApproval({"echo"}),
        capabilities={"workspace.write"},
    ).execute({"name": "echo", "arguments": {"value": "x"}})
    assert allowed.status == "ok"


def test_explicit_empty_capabilities_do_not_gain_read_access():
    registry = ToolRegistry([_echo_spec()])
    denied = ToolExecutor(registry, capabilities=set()).execute(
        {"name": "echo", "arguments": {"value": "x"}}
    )
    assert denied.status == "denied"
    assert "workspace.read" in (denied.error or "")


def test_harness_runs_tools_then_final_and_replays_without_side_effects():
    harness = AgentHarness(ToolRegistry([_echo_spec()]))
    actions = iter([
        {"type": "tool_call", "name": "echo", "arguments": {"value": "hello"}},
        {"type": "final", "content": "done"},
    ])
    result = harness.run("request", lambda request, observations, tools: next(actions))
    assert result.status == "completed"
    assert result.answer == "done"
    assert result.observations[0]["status"] == "ok"
    replay = ReplayRunner().replay(result.trace)
    assert replay["side_effects_applied"] is False


def test_harness_budget_stops_before_another_action():
    harness = AgentHarness(ToolRegistry([_echo_spec()]))
    result = harness.run(
        "request",
        lambda request, observations, tools: {"type": "tool_call", "name": "echo", "arguments": {"value": "x"}},
        budget=ExecutionBudget(max_steps=1, max_tool_calls=5),
    )
    assert result.status == "blocked"
    assert result.error_code == "budget_exceeded"
    assert "budget" in (result.error or "")
    assert any(event["details"].get("error", {}).get("redacted") for event in result.trace if isinstance(event["details"].get("error"), dict))
