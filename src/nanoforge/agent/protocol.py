"""Shared protocol objects for the coding-agent harness.

The protocol is intentionally model-agnostic. A local Nanoforge model, an
imported model, or a deterministic test provider can propose the same
structured action. The executor remains the authority over permissions,
budgets, validation, and side effects.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str, allow_nan=False).encode("utf-8")


def _bounded(value: Any, limit: int) -> Any:
    """Bound the complete JSON value, including UTF-8 and envelope overhead."""
    if limit < 2:
        raise ValueError("JSON output limit must be at least 2 bytes")
    encoded = json_bytes(value)
    if len(encoded) <= limit:
        return value
    text = value if isinstance(value, str) else encoded.decode("utf-8")
    template = {"truncated": True, "text": "", "original_bytes": len(encoded)}
    envelope = len(json_bytes(template)) <= limit
    low, high = 0, min(len(text), limit)
    while low < high:
        middle = (low + high + 1) // 2
        candidate = {**template, "text": text[:middle]} if envelope else text[:middle]
        if len(json_bytes(candidate)) <= limit:
            low = middle
        else:
            high = middle - 1
    return {**template, "text": text[:low]} if envelope else text[:low]


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
    inject_context: bool = False
    max_memory_bytes: int | None = None

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": copy.deepcopy(self.input_schema),
            "capabilities": sorted(self.capabilities),
            "approval_policy": self.approval_policy,
            "side_effect": self.side_effect,
            "timeout_seconds": self.timeout_seconds,
            "max_output_bytes": self.max_output_bytes,
            "isolation": "trusted-process-resource-boundary",
            "max_input_bytes": 16 * 1024,
            "memory_limit_scope": "windows-job-tree-commit" if os.name == "nt" else "posix-per-process-address-space",
        }


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    call_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    @classmethod
    def from_value(cls, value: "ToolCall | str | dict[str, Any]") -> "ToolCall":
        if isinstance(value, ToolCall):
            value = {"name": value.name, "arguments": value.arguments, "call_id": value.call_id}
        if isinstance(value, str):
            from nanoforge.agent.actions import _duplicate_key_guard, _reject_constant
            if len(value.encode("utf-8")) > 16 * 1024:
                raise ValueError("Tool action exceeds 16384 bytes")
            try:
                value = json.loads(value, object_pairs_hook=_duplicate_key_guard, parse_constant=_reject_constant)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Tool action is not valid JSON: {exc.msg}") from exc
        if not isinstance(value, dict):
            raise ValueError("Tool action must be a JSON object.")
        if len(json_bytes(value)) > 16 * 1024:
            raise ValueError("Tool action exceeds 16384 bytes")
        from nanoforge.agent.actions import _check_depth
        _check_depth(value, 1, 16)
        if set(value) - {"name", "arguments", "call_id", "type", "schema_version"}:
            raise ValueError("Tool action contains unknown fields")
        name = value.get("name")
        arguments = value.get("arguments", {})
        if not isinstance(name, str) or not name:
            raise ValueError("Tool action requires a non-empty string name.")
        if not isinstance(arguments, dict):
            raise ValueError("Tool action arguments must be an object.")
        return cls(name=name, arguments=copy.deepcopy(arguments), call_id=str(value.get("call_id") or uuid.uuid4()))


@dataclass
class ToolResult:
    call_id: str
    tool_name: str
    status: str
    output: Any = None
    error: str | None = None
    truncated: bool = False
    elapsed_ms: float = 0.0
    effects_settled: bool = True
    isolation: str = "trusted-process-resource-boundary"

    def bounded(self, max_bytes: int) -> "ToolResult":
        if self.output is not None:
            oversized = len(json_bytes(self.output)) > max_bytes
            self.output = _bounded(self.output, max_bytes)
            self.truncated = self.truncated or oversized
        if self.error is not None:
            bounded_error = _bounded(self.error, max_bytes)
            self.error = bounded_error if isinstance(bounded_error, str) else str(bounded_error.get("text", ""))
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
    max_tokens: int = 4096
    max_memory_bytes: int | None = None
    max_context_bytes: int = 256 * 1024
    tokens_used: int = 0
    output_bytes_used: int = 0
    token_accounting: str = "reported-or-conservative-utf8-bytes"
    _lock: Any = field(default_factory=threading.Lock, repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in ("max_steps", "max_tool_calls", "max_output_bytes", "max_tokens", "max_context_bytes"):
            value = getattr(self, name)
            minimum = 2 if name == "max_output_bytes" else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if isinstance(self.max_wall_seconds, bool) or not isinstance(self.max_wall_seconds, (int, float)) or not math.isfinite(self.max_wall_seconds) or self.max_wall_seconds <= 0:
            raise ValueError("max_wall_seconds must be positive and finite")
        if self.max_memory_bytes is not None and (isinstance(self.max_memory_bytes, bool) or not isinstance(self.max_memory_bytes, int) or self.max_memory_bytes <= 0):
            raise ValueError("max_memory_bytes must be a positive integer or None")

    @property
    def deadline(self) -> float:
        return self.started_at + self.max_wall_seconds

    def remaining_seconds(self) -> float:
        return max(0.0, self.deadline - time.monotonic())

    def check_wall(self) -> None:
        if self.remaining_seconds() <= 0:
            raise TimeoutError("agent wall-clock budget exceeded")

    def check_step(self) -> None:
        with self._lock:
            if self.steps_used >= self.max_steps:
                raise RuntimeError("agent step budget exceeded")
            self.check_wall()
            self.steps_used += 1

    def check_tool_call(self) -> None:
        with self._lock:
            self.check_wall()
            if self.tool_calls_used >= self.max_tool_calls:
                raise RuntimeError("agent tool-call budget exceeded")
            self.tool_calls_used += 1

    def charge_tokens(self, count: int) -> None:
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("token usage must be a nonnegative integer")
        with self._lock:
            if self.tokens_used + count > self.max_tokens:
                raise RuntimeError("agent token budget exceeded")
            self.tokens_used += count

    def snapshot(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in ("max_steps", "max_tool_calls", "max_wall_seconds", "max_output_bytes", "max_tokens", "max_memory_bytes", "max_context_bytes", "steps_used", "tool_calls_used", "tokens_used", "output_bytes_used", "token_accounting")} | {"remaining_seconds": self.remaining_seconds()}


class CancellationToken:
    def __init__(self, event: Any = None) -> None:
        self._reason: str | None = None
        self._event = event

    @property
    def cancelled(self) -> bool:
        return self._reason is not None or (self._event is not None and self._event.is_set())

    @property
    def reason(self) -> str | None:
        return self._reason or ("cancelled" if self.cancelled else None)

    def cancel(self, reason: str = "cancelled") -> None:
        self._reason = reason
        if self._event is not None:
            self._event.set()

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise RuntimeError(self.reason or "cancelled")


@dataclass(frozen=True)
class ToolContext:
    cancellation: CancellationToken
    deadline: float
    max_output_bytes: int


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


_PRIVATE_FIELDS = {"request", "user_request", "answer", "content", "text", "replacement_text", "output", "error"}
_SECRET_FIELDS = re.compile(r"^(token|secret|password|api.?key|credential|authorization)$|secret|password|api.?key|access_token|refresh_token", re.I)


def redact_trace(value: Any, *, include_payloads: bool = False) -> Any:
    """Persist control flow and fingerprints; private payload inclusion is opt-in."""
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            if key == "arguments" and isinstance(child, dict) and not include_payloads:
                # Keep argument structure while treating arbitrary tool values as private.
                result[key] = {argument: item if _redaction_marker(item) else {"redacted": True, "sha256": hashlib.sha256(json_bytes(item)).hexdigest(), "bytes": len(json_bytes(item))} for argument, item in child.items()}
                continue
            if _SECRET_FIELDS.search(str(key)) or (not include_payloads and key in _PRIVATE_FIELDS):
                result[key] = child if _redaction_marker(child) else {"redacted": True, "sha256": hashlib.sha256(json_bytes(child)).hexdigest(), "bytes": len(json_bytes(child))}
            else:
                result[key] = redact_trace(child, include_payloads=include_payloads)
        return result
    if isinstance(value, (tuple, list)):
        return [redact_trace(child, include_payloads=include_payloads) for child in value]
    return value


def _redaction_marker(value: Any) -> bool:
    return isinstance(value, dict) and set(value) == {"redacted", "sha256", "bytes"} and value["redacted"] is True and isinstance(value["sha256"], str) and re.fullmatch("[a-f0-9]{64}", value["sha256"]) is not None and type(value["bytes"]) is int and value["bytes"] >= 0


class InMemoryTrace:
    SCHEMA_VERSION = 1

    def __init__(self, run_id: str | None = None, max_events: int = 500, *, path: str | Path | None = None, include_payloads: bool = False, redactor: Callable[[Any], Any] | None = None):
        if max_events < 1:
            raise ValueError("max_events must be positive")
        self.run_id = run_id or str(uuid.uuid4())
        self.max_events = max_events
        self.events: list[TraceEvent] = []
        self.path = Path(path) if path is not None else None
        self.include_payloads = include_payloads
        self.redactor = redactor
        self.dropped_events = 0
        self._lock = threading.RLock()

    def emit(self, phase: str, status: str, action: str, details: dict[str, Any] | None = None) -> None:
        with self._lock:
            if len(self.events) >= self.max_events:
                self.dropped_events += 1
            else:
                safe = redact_trace(details or {}, include_payloads=self.include_payloads)
                if self.redactor is not None:
                    safe = self.redactor(safe)
                bounded = _bounded(safe, 64 * 1024)
                if not isinstance(bounded, dict):
                    bounded = {"truncated": True, "text": bounded}
                self.events.append(TraceEvent(self.run_id, len(self.events), phase, status, action, bounded))
            if self.path is not None:
                self.save(self.path)

    def to_list(self) -> list[dict[str, Any]]:
        with self._lock:
            return [event.to_dict() for event in self.events]

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Reapply the persistence policy even if a trusted caller mutated the public events.
        body = {"schema_version": self.SCHEMA_VERSION, "run_id": self.run_id, "events": redact_trace(self.to_list(), include_payloads=self.include_payloads), "dropped_events": self.dropped_events, "payloads_included": self.include_payloads}
        document = {**body, "sha256": hashlib.sha256(json_bytes(body)).hexdigest()}
        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        try:
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as stream:
                stream.write(json_bytes(document))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: str | Path) -> "InMemoryTrace":
        target = Path(path)
        if target.stat().st_size > 64 * 1024 * 1024:
            raise ValueError("trace exceeds load limit")
        document = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(document, dict) or type(document.get("schema_version")) is not int or document.get("schema_version") != cls.SCHEMA_VERSION:
            raise ValueError("unsupported trace schema")
        digest = document.pop("sha256", None)
        if not isinstance(digest, str) or hashlib.sha256(json_bytes(document)).hexdigest() != digest:
            raise ValueError("trace integrity check failed")
        events = document.get("events")
        if not isinstance(events, list) or not isinstance(document.get("run_id"), str):
            raise ValueError("trace events must be an array")
        if type(document.get("dropped_events")) is not int or document["dropped_events"] < 0:
            raise ValueError("trace dropped event count must be nonnegative")
        trace = cls(document["run_id"], max_events=max(500, len(events)), include_payloads=bool(document.get("payloads_included")))
        for sequence, event in enumerate(events):
            if not isinstance(event, dict) or type(event.get("sequence")) is not int or event.get("sequence") != sequence or event.get("run_id") != trace.run_id:
                raise ValueError("trace event integrity/order mismatch")
            if set(event) != {"run_id", "sequence", "phase", "status", "action", "details", "timestamp"} or any(not isinstance(event[key], str) for key in ("phase", "status", "action")) or not isinstance(event["details"], dict) or isinstance(event["timestamp"], bool) or not isinstance(event["timestamp"], (int, float)) or not math.isfinite(event["timestamp"]):
                raise ValueError("trace event schema is invalid")
            trace.events.append(TraceEvent(**event))
        trace.dropped_events = document.get("dropped_events", 0)
        return trace
