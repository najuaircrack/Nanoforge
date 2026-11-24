from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from nanoforge.agent import (
    AgentHarness,
    CancellationToken,
    ExecutionBudget,
    InMemoryTrace,
    ManualActionProvider,
    ReplayRunner,
    StaticApproval,
    ToolExecutor,
    ToolRegistry,
    ToolSpec,
)
from nanoforge.agent.protocol import PolicyDecision


def _spec(fn, **kwargs):
    return ToolSpec("probe", "Synthetic probe", {"type": "object"}, fn, **kwargs)


def _process_alive(pid):
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        try:
            code = wintypes.DWORD()
            assert kernel.GetExitCodeProcess(handle, ctypes.byref(code))
            return code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    state = Path(f"/proc/{pid}/stat")
    if state.exists():
        try:
            return state.read_text(encoding="utf-8").partition(") ")[2].split()[0] != "Z"
        except FileNotFoundError:
            return False
    return True


def test_tool_timeout_settles_late_file_effect(tmp_path):
    marker = tmp_path / "late.txt"
    started = tmp_path / "started.txt"

    def late():
        started.write_text(str(os.getpid()), encoding="utf-8")
        time.sleep(4)
        marker.write_text("effect", encoding="utf-8")

    result = ToolExecutor(ToolRegistry([_spec(late, timeout_seconds=2.5)])).execute({"name": "probe"})
    assert result.status == "timeout"
    assert result.effects_settled
    assert started.exists(), "positive control: tool body actually started"
    assert not _process_alive(int(started.read_text(encoding="utf-8")))
    time.sleep(.1)
    assert not marker.exists()


def test_active_cancellation_settles_tool(tmp_path):
    marker = tmp_path / "late.txt"
    started = tmp_path / "started.txt"

    def late():
        started.write_text(str(os.getpid()), encoding="utf-8")
        time.sleep(4)
        marker.write_text("effect", encoding="utf-8")

    token = CancellationToken()
    def cancel_started():
        deadline = time.monotonic() + 10
        while not started.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        token.cancel()

    timer = threading.Thread(target=cancel_started)
    timer.start()
    try:
        result = ToolExecutor(ToolRegistry([_spec(late)]), cancellation=token).execute({"name": "probe"})
    finally:
        timer.join()
    assert result.status == "cancelled"
    assert result.effects_settled
    assert started.exists(), "positive control: tool body actually started"
    assert not _process_alive(int(started.read_text(encoding="utf-8")))
    assert not marker.exists()


def test_expired_budget_prevents_effect(tmp_path):
    marker = tmp_path / "denied.txt"
    budget = ExecutionBudget(max_wall_seconds=.01)
    time.sleep(.02)
    result = ToolExecutor(
        ToolRegistry([_spec(lambda: marker.write_text("no", encoding="utf-8"))]), budget=budget
    ).execute({"name": "probe"})
    assert result.status == "timeout"
    assert not marker.exists()


def test_provider_deadline_prevents_late_final():
    def slow(*args):
        time.sleep(2)
        return {"type": "final", "content": "late"}

    started = time.monotonic()
    result = AgentHarness(ToolRegistry()).run("request", slow, budget=ExecutionBudget(max_wall_seconds=.5))
    assert result.status == "timeout"
    assert result.answer is None
    assert time.monotonic() - started < 1.5


def test_policy_approval_flag_cannot_be_bypassed():
    class RequireApproval:
        def decide(self, spec):
            return PolicyDecision(True, "human confirmation", True)

    executor = ToolExecutor(ToolRegistry([_spec(lambda: "no")]), policy=RequireApproval())
    result = executor.execute({"name": "probe"})
    assert result.status == "denied"


def test_approved_write_and_denied_capability(tmp_path):
    marker = tmp_path / "approved.txt"
    spec = _spec(
        lambda: marker.write_text("yes", encoding="utf-8"),
        capabilities=frozenset({"workspace.write"}), side_effect="write", approval_policy="always",
    )
    denied = ToolExecutor(ToolRegistry([spec]), approval=StaticApproval({"probe"})).execute({"name": "probe"})
    assert denied.status == "denied" and not marker.exists()
    approved = ToolExecutor(
        ToolRegistry([spec]), approval=StaticApproval({"probe"}), capabilities={"workspace.write"}
    ).execute({"name": "probe"})
    assert approved.status == "ok" and marker.read_text(encoding="utf-8") == "yes"


@pytest.mark.parametrize("value", [99, True, float("nan"), 3])
def test_schema_range_enum_and_nonfinite_refused(value):
    schema = {"type": "object", "properties": {"x": {"type": "integer", "minimum": 1, "maximum": 5, "enum": [1, 2]}}, "required": ["x"]}
    registry = ToolRegistry([ToolSpec("probe", "probe", schema, lambda x: x)])
    assert ToolExecutor(registry).execute({"name": "probe", "arguments": {"x": value}}).status == "error"


def test_schema_nested_array_and_unknown_dialect_rejected():
    schema = {"type": "object", "properties": {"values": {"type": "array", "items": {"type": "string", "minLength": 2}, "uniqueItems": True}}, "required": ["values"], "additionalProperties": False}
    registry = ToolRegistry([ToolSpec("probe", "probe", schema, lambda values: values)])
    for values in (["x"], ["ok", "ok"], [1]):
        assert ToolExecutor(registry).execute({"name": "probe", "arguments": {"values": values}}).status == "error"
    with pytest.raises(ValueError, match="unsupported"):
        ToolRegistry([ToolSpec("bad", "bad", {"type": "object", "$ref": "elsewhere"}, lambda: None)])


@pytest.mark.parametrize("limit", [2, 8, 100, 256])
def test_utf8_and_run_output_cap(limit):
    registry = ToolRegistry([_spec(lambda: "\u20ac" * 200, max_output_bytes=10000)])
    result = ToolExecutor(registry, budget=ExecutionBudget(max_output_bytes=limit)).execute({"name": "probe"})
    assert result.status == "ok" and result.truncated
    assert len(json.dumps(result.output, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) <= limit


def test_reconstructible_redacted_persistent_trace(tmp_path):
    path = tmp_path / "trace.json"
    registry = ToolRegistry([_spec(lambda: {"text": "private-body", "status": "found"})])
    result = AgentHarness(registry).run(
        "private-request",
        ManualActionProvider([{"type": "tool_call", "name": "probe", "arguments": {}}, {"type": "final", "content": "private-answer"}]),
        trace_path=path,
    )
    encoded = path.read_text(encoding="utf-8")
    for private in ("private-body", "private-request", "private-answer"):
        assert private not in encoded
    loaded = InMemoryTrace.load(path)
    replay = ReplayRunner().replay(loaded.to_list())
    assert result.status == "completed"
    assert replay["side_effects_applied"] is False
    assert replay["state"]["status"] == "completed"
    assert replay["state"]["actions"][0]["name"] == "probe"
    assert replay["state"]["observations"][0]["tool_name"] == "probe"
    document = json.loads(encoded)
    document["events"][0]["action"] = "tampered"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        InMemoryTrace.load(path)


def test_token_context_and_budget_validation():
    with pytest.raises(ValueError):
        ExecutionBudget(max_wall_seconds=float("nan"))
    with pytest.raises(ValueError):
        ExecutionBudget(max_memory_bytes=-1)
    result = AgentHarness(ToolRegistry()).run("too long", ManualActionProvider([{"type": "final", "content": "done"}]), budget=ExecutionBudget(max_context_bytes=4))
    assert result.status == "blocked"


def test_strict_wire_version_rejects_boolean():
    from nanoforge.agent.actions import ActionDecodeError, decode_action

    with pytest.raises(ActionDecodeError):
        decode_action('{"schema_version":true,"type":"final","content":"no"}')


def test_process_tree_child_is_settled_on_timeout(tmp_path):
    started = tmp_path / "child-started.txt"
    late = tmp_path / "child-late.txt"

    def launch_child():
        code = "import os,pathlib,time;pathlib.Path(" + repr(str(started)) + ").write_text(str(os.getpid()));time.sleep(4);pathlib.Path(" + repr(str(late)) + ").write_text('late')"
        subprocess.Popen([sys.executable, "-c", code])
        time.sleep(8)

    result = ToolExecutor(ToolRegistry([_spec(launch_child, timeout_seconds=2.5)])).execute({"name": "probe"})
    assert result.status == "timeout" and result.effects_settled
    assert started.exists(), "positive control: grandchild actually started"
    assert not _process_alive(int(started.read_text(encoding="utf-8")))
    time.sleep(.1)
    assert not late.exists()


def test_worker_memory_limit_refuses_large_allocation():
    registry = ToolRegistry([_spec(lambda: len(bytearray(512 * 1024 * 1024)), max_memory_bytes=128 * 1024 * 1024)])
    result = ToolExecutor(registry).execute({"name": "probe"})
    assert result.status == "error"
    assert result.effects_settled
    assert "MemoryError" in (result.error or "") or "resource limit" in (result.error or "")


def test_tool_context_and_persistent_provider_state():
    def with_context(context):
        return {"cancelled": context.cancellation.cancelled, "output_limit": context.max_output_bytes, "deadline_future": context.deadline > time.monotonic()}

    result = ToolExecutor(ToolRegistry([_spec(with_context, inject_context=True)])).execute({"name": "probe"})
    assert result.status == "ok"
    assert result.output["cancelled"] is False and result.output["deadline_future"] is True
    actions = iter([{"type": "tool_call", "name": "probe", "arguments": {}}, {"type": "final", "content": "done"}])
    run = AgentHarness(ToolRegistry([_spec(lambda: "observation")])).run("request", lambda *args: next(actions))
    assert run.status == "completed" and len(run.observations) == 1


def test_pathological_schema_pattern_is_time_bounded():
    schema = {"type": "object", "properties": {"value": {"type": "string", "pattern": "(a+)+$"}}, "required": ["value"]}
    spec = ToolSpec("probe", "probe", schema, lambda value: "must not execute", timeout_seconds=2)
    started = time.monotonic()
    result = ToolExecutor(ToolRegistry([spec])).execute({"name": "probe", "arguments": {"value": "a" * 200 + "!"}})
    assert result.status == "timeout" and result.effects_settled
    assert time.monotonic() - started < 3.5


def test_active_provider_cancellation_settles_late_effect(tmp_path):
    started = tmp_path / "provider-started.txt"
    late = tmp_path / "provider-late.txt"
    token = CancellationToken()

    def provider(*args):
        started.write_text(str(os.getpid()), encoding="utf-8")
        time.sleep(4)
        late.write_text("late", encoding="utf-8")
        return {"type": "final", "content": "late"}

    def cancel_started():
        deadline = time.monotonic() + 10
        while not started.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        token.cancel()

    thread = threading.Thread(target=cancel_started)
    thread.start()
    try:
        result = AgentHarness(ToolRegistry()).run("request", provider, cancellation=token)
    finally:
        thread.join()
    assert started.exists() and result.status == "cancelled"
    assert not _process_alive(int(started.read_text(encoding="utf-8")))
    assert not late.exists()


def test_token_budget_and_denied_replay_are_explicit(tmp_path):
    from nanoforge.agent.provider import ScriptedProvider

    result = AgentHarness(ToolRegistry()).run(
        "request", ScriptedProvider(['{"schema_version":1,"type":"final","content":"done"}']),
        budget=ExecutionBudget(max_tokens=5),
    )
    assert result.status == "blocked" and result.error_code == "budget_exceeded"
    spec = _spec(lambda: "no", capabilities=frozenset({"workspace.write"}), side_effect="write", approval_policy="always")
    path = tmp_path / "denial.json"
    result = AgentHarness(ToolRegistry([spec])).run(
        "request", ManualActionProvider([{"type": "tool_call", "name": "probe", "arguments": {}}, {"type": "final", "content": "denied"}]), trace_path=path,
    )
    replay = ReplayRunner().replay(path)
    assert replay["trace_complete"]
    assert result.observations[0]["status"] == replay["state"]["observations"][0]["status"] == "denied"


def test_late_approval_cannot_dispatch_effect(tmp_path):
    marker = tmp_path / "must-not-write.txt"

    class SlowApproval:
        def approve(self, call, spec):
            time.sleep(2)
            return True

    spec = _spec(lambda: marker.write_text("no", encoding="utf-8"), side_effect="write", approval_policy="always")
    result = ToolExecutor(ToolRegistry([spec]), approval=SlowApproval(), budget=ExecutionBudget(max_wall_seconds=.1)).execute({"name": "probe"})
    assert result.status == "timeout" and not marker.exists()


def test_legacy_registry_requires_explicit_policy_and_uses_isolation():
    from nanoforge.tools import ToolRegistry as LegacyRegistry
    from nanoforge.tools import ToolSpec as LegacySpec

    legacy = LegacyRegistry()
    legacy.register(LegacySpec("legacy", "legacy", {"type": "object"}, lambda: "done"))
    with pytest.raises(PermissionError):
        legacy.call_json('{"name":"legacy"}')
    allowed = LegacyRegistry(executor_kwargs={"approval": StaticApproval({"legacy"}), "capabilities": {"legacy.execute"}})
    allowed.register(LegacySpec("legacy", "legacy", {"type": "object"}, lambda: "done"))
    assert allowed.call_json('{"name":"legacy"}') == "done"


def test_local_tool_worker_does_not_inherit_ambient_credential(monkeypatch):
    monkeypatch.setenv("LATTICEKIN_SYNTHETIC_SECRET", "synthetic-sensitive-value")
    result = ToolExecutor(ToolRegistry([_spec(lambda: os.environ.get("LATTICEKIN_SYNTHETIC_SECRET"))])).execute({"name": "probe"})
    assert result.status == "ok" and result.output is None


def test_explicit_governed_context_is_bounded_and_redacted(tmp_path):
    from nanoforge.agent.provider import ScriptedProvider

    provider = ScriptedProvider(['{"schema_version":1,"type":"final","content":"done"}'])
    path = tmp_path / "memory-trace.json"
    items = [{"id": "memory-1", "text": "private-memory", "provenance": {"source": "user-consented"}, "owner": "principal", "scope": "project"}]
    result = AgentHarness(ToolRegistry()).run("request", provider, context_items=items, trace_path=path)
    assert result.status == "completed"
    assert provider.requests[0].context_items[0]["id"] == "memory-1"
    assert "private-memory" not in path.read_text(encoding="utf-8")
    rejected = AgentHarness(ToolRegistry()).run("request", provider, context_items=items, budget=ExecutionBudget(max_context_bytes=8))
    assert rejected.status == "blocked" and rejected.error_code == "budget_exceeded"


def test_persistence_reapplies_redaction_after_public_event_mutation(tmp_path):
    trace = InMemoryTrace()
    trace.emit("request", "accepted", "run", {"request": "synthetic-private"})
    trace.events[0].details["error"] = "synthetic-mutated-private"
    path = tmp_path / "mutated-trace.json"
    trace.save(path)
    raw = path.read_text(encoding="utf-8")
    assert "synthetic-private" not in raw and "synthetic-mutated-private" not in raw
    restored = InMemoryTrace.load(path)
    assert restored.events[0].details["request"]["redacted"]
    assert restored.events[0].details["error"]["redacted"]


def test_wire_surrogate_is_rejected_before_trace_or_tool_execution():
    from nanoforge.agent.actions import ActionDecodeError, decode_action

    with pytest.raises(ActionDecodeError) as error:
        decode_action('{"schema_version":1,"type":"final","content":"\\ud800"}')
    assert error.value.code == "invalid_unicode"
