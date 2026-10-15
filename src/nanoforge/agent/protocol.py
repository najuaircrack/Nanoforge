"""Shared protocol objects for the coding-agent harness.

The protocol is intentionally model-agnostic. A local Nanoforge model, an
imported model, or a deterministic test provider can propose the same
structured action. The executor remains the authority over permissions,
budgets, validation, and side effects.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol


def _bounded(value: Any, limit: int) -> Any:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    if len(text) <= limit:
        return value
    return {"truncated": True, "text": text[:limit], "original_bytes": len(text.encode("utf-8"))}


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    fn: Any
    capabilities: frozenset[str] = frozenset()
    timeout_seconds: float = 30.0
    max_output_bytes: int = 64 * 1024
    approval_policy: str = "never"
    side_effect: str = "read"
    audit_name: str | None = None

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.input_schema,
            "capabilities": sorted(self.capabilities),
            "approval_policy": self.approval_policy,
            "side_effect": self.side_effect,
        }


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    call_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    @classmethod
    def from_value(cls, value: "ToolCall | str | dict[str, Any]") -> "ToolCall":
        if isinstance(value, ToolCall):
            return value
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Tool action is not valid JSON: {exc.msg}") from exc
        if not isinstance(value, dict):
            raise ValueError("Tool action must be a JSON object.")
        name = value.get("name")
        arguments = value.get("arguments", {})
        if not isinstance(name, str) or not name:
            raise ValueError("Tool action requires a non-empty string name.")
        if not isinstance(arguments, dict):
            raise ValueError("Tool action arguments must be an object.")
        return cls(name=name, arguments=arguments, call_id=str(value.get("call_id") or uuid.uuid4()))


@dataclass
class ToolResult:
    call_id: str
    tool_name: str
    status: str
    output: Any = None
    error: str | None = None
    truncated: bool = False
    elapsed_ms: float = 0.0

    def bounded(self, max_bytes: int) -> "ToolResult":
        encoded = json.dumps(self.output, ensure_ascii=False, default=str).encode("utf-8")
        if len(encoded) <= max_bytes:
            return self
        self.output = _bounded(self.output, max_bytes)
        self.truncated = True
        return self


@dataclass
class ExecutionBudget:
    max_steps: int = 12
    max_tool_calls: int = 12
    max_wall_seconds: float = 120.0
    max_output_bytes: int = 64 * 1024
    steps_used: int = 0
    tool_calls_used: int = 0
    started_at: float = field(default_factory=time.monotonic)

    def check_step(self) -> None:
        if self.steps_used >= self.max_steps:
            raise RuntimeError("agent step budget exceeded")
        if time.monotonic() - self.started_at >= self.max_wall_seconds:
            raise RuntimeError("agent wall-clock budget exceeded")
        self.steps_used += 1

    def check_tool_call(self) -> None:
        if self.tool_calls_used >= self.max_tool_calls:
            raise RuntimeError("agent tool-call budget exceeded")
        self.tool_calls_used += 1


class CancellationToken:
    def __init__(self) -> None:
        self._reason: str | None = None

    @property
    def cancelled(self) -> bool:
        return self._reason is not None

    @property
    def reason(self) -> str | None:
        return self._reason

    def cancel(self, reason: str = "cancelled") -> None:
        self._reason = reason

    def raise_if_cancelled(self) -> None:
        if self._reason is not None:
            raise RuntimeError(self._reason)


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    reason: str
    requires_approval: bool = False


class CapabilityPolicy(Protocol):
    def decide(self, spec: ToolSpec) -> PolicyDecision: ...


class ApprovalProvider(Protocol):
    def approve(self, call: ToolCall, spec: ToolSpec) -> bool: ...


class AllowReadOnlyPolicy:
    def decide(self, spec: ToolSpec) -> PolicyDecision:
        if spec.side_effect == "read" and spec.approval_policy == "never":
            return PolicyDecision(True, "read-only tool allowed")
        return PolicyDecision(False, "tool requires an explicit approval policy", requires_approval=True)


class DenyApproval:
    def approve(self, call: ToolCall, spec: ToolSpec) -> bool:
        return False


class StaticApproval:
    def __init__(self, allowed_tools: set[str] | None = None):
        self.allowed_tools = allowed_tools or set()

    def approve(self, call: ToolCall, spec: ToolSpec) -> bool:
        return call.name in self.allowed_tools


@dataclass
class TraceEvent:
    run_id: str
    sequence: int
    phase: str
    status: str
    action: str
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class InMemoryTrace:
    def __init__(self, run_id: str | None = None, max_events: int = 500):
        self.run_id = run_id or str(uuid.uuid4())
        self.max_events = max_events
        self.events: list[TraceEvent] = []

    def emit(self, phase: str, status: str, action: str, details: dict[str, Any] | None = None) -> None:
        if len(self.events) >= self.max_events:
            return
        self.events.append(
            TraceEvent(
                run_id=self.run_id,
                sequence=len(self.events),
                phase=phase,
                status=status,
                action=action,
                details=_bounded(details or {}, 16 * 1024),
            )
        )

    def to_list(self) -> list[dict[str, Any]]:
        return [event.to_dict() for event in self.events]
