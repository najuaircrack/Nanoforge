"""Schema-validating and policy-checked tool execution."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Any, Iterable

from nanoforge.agent.protocol import (
    AllowReadOnlyPolicy,
    ApprovalProvider,
    CancellationToken,
    CapabilityPolicy,
    DenyApproval,
    ExecutionBudget,
    InMemoryTrace,
    ToolCall,
    ToolResult,
    ToolSpec,
)


def _validate_schema(value: Any, schema: dict[str, Any], path: str = "arguments") -> None:
    expected = schema.get("type")
    if isinstance(expected, list):
        if value is None and "null" in expected:
            return
        for alternative in expected:
            if alternative == "null":
                continue
            try:
                _validate_schema(value, {**schema, "type": alternative}, path)
                return
            except ValueError:
                continue
        raise ValueError(f"{path} has an invalid type")
    if expected == "object":
        if not isinstance(value, dict):
            raise ValueError(f"{path} must be an object")
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        for key in required:
            if key not in value:
                raise ValueError(f"{path}.{key} is required")
        if schema.get("additionalProperties", True) is False:
            unknown = sorted(set(value) - set(properties))
            if unknown:
                raise ValueError(f"{path} contains unknown field(s): {', '.join(unknown)}")
        for key, child_schema in properties.items():
            if key in value:
                _validate_schema(value[key], child_schema, f"{path}.{key}")
        return
    checks = {
        "string": isinstance(value, str),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "array": isinstance(value, list),
    }
    if expected in checks and not checks[expected]:
        raise ValueError(f"{path} must be {expected}")


class ToolRegistry:
    def __init__(self, specs: Iterable[ToolSpec] = ()):
        self._tools: dict[str, ToolSpec] = {}
        for spec in specs:
            self.register(spec)

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise KeyError(f"Tool already registered: {spec.name}")
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"Unknown tool: {name}") from exc

    def schema(self) -> list[dict[str, Any]]:
        return [self._tools[name].schema() for name in sorted(self._tools)]


class ToolExecutor:
    def __init__(
        self,
        registry: ToolRegistry,
        *,
        policy: CapabilityPolicy | None = None,
        approval: ApprovalProvider | None = None,
        budget: ExecutionBudget | None = None,
        cancellation: CancellationToken | None = None,
        trace: InMemoryTrace | None = None,
        capabilities: set[str] | None = None,
    ):
        self.registry = registry
        self.policy = policy or AllowReadOnlyPolicy()
        self.approval = approval or DenyApproval()
        self.budget = budget or ExecutionBudget()
        self.cancellation = cancellation or CancellationToken()
        self.trace = trace or InMemoryTrace()
        self.capabilities = capabilities or {"workspace.read"}

    def execute(self, call: ToolCall | dict[str, Any] | str) -> ToolResult:
        started = time.perf_counter()
        try:
            self.cancellation.raise_if_cancelled()
            self.budget.check_tool_call()
            parsed = ToolCall.from_value(call)
            spec = self.registry.get(parsed.name)
            _validate_schema(parsed.arguments, spec.input_schema)
            decision = self.policy.decide(spec)
            self.trace.emit("policy", "allowed" if decision.allowed else "denied", parsed.name, {"reason": decision.reason})
            missing_capabilities = sorted(spec.capabilities - self.capabilities)
            if missing_capabilities:
                return self._result(
                    parsed,
                    "denied",
                    error=f"missing capability: {', '.join(missing_capabilities)}",
                    started=started,
                )
            if not decision.allowed:
                if not decision.requires_approval or not self.approval.approve(parsed, spec):
                    return self._result(parsed, "denied", error=decision.reason, started=started)
                self.trace.emit("approval", "approved", parsed.name)
            elif spec.approval_policy == "always" and not self.approval.approve(parsed, spec):
                self.trace.emit("approval", "denied", parsed.name)
                return self._result(parsed, "denied", error="approval denied", started=started)
            self.cancellation.raise_if_cancelled()
            self.trace.emit("tool", "started", parsed.name, {"call_id": parsed.call_id})
            pool = ThreadPoolExecutor(max_workers=1)
            future = pool.submit(spec.fn, **parsed.arguments)
            try:
                output = future.result(timeout=spec.timeout_seconds)
            finally:
                pool.shutdown(wait=False, cancel_futures=True)
            result = self._result(parsed, "ok", output=output, started=started).bounded(spec.max_output_bytes)
            self.trace.emit("tool", "completed", parsed.name, {"status": result.status, "truncated": result.truncated})
            return result
        except TimeoutError:
            self.trace.emit("tool", "timeout", getattr(call, "name", "unknown"))
            return self._result(ToolCall.from_value(call), "timeout", error="tool timeout", started=started)
        except Exception as exc:
            name = call.name if isinstance(call, ToolCall) else "unknown"
            self.trace.emit("tool", "error", name, {"error": f"{type(exc).__name__}: {exc}"})
            try:
                parsed_call = call if isinstance(call, ToolCall) else ToolCall.from_value(call)
            except Exception:
                parsed_call = ToolCall(name=name)
            return self._result(
                parsed_call,
                "error",
                error=f"{type(exc).__name__}: {exc}",
                started=started,
            )

    @staticmethod
    def _result(call: ToolCall, status: str, *, output: Any = None, error: str | None = None, started: float) -> ToolResult:
        return ToolResult(
            call_id=call.call_id,
            tool_name=call.name,
            status=status,
            output=output,
            error=error,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )
