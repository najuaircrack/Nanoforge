"""Registered tool schema validation, capability policy, and isolated execution."""

from __future__ import annotations

import copy
import math
import queue
import re
import threading
import time
from dataclasses import replace
from typing import Any, Iterable

from nanoforge.agent.isolation import ExecutionCancelled, IsolatedWorker
from nanoforge.agent.protocol import (
    AllowReadOnlyPolicy,
    ApprovalProvider,
    CancellationToken,
    CapabilityPolicy,
    DenyApproval,
    ExecutionBudget,
    InMemoryTrace,
    ToolCall,
    ToolContext,
    ToolResult,
    ToolSpec,
    json_bytes,
)

# A deliberately closed JSON Schema dialect. Unsupported keywords fail registration.
_SCHEMA_KEYS = {
    "$schema", "title", "description", "default", "examples", "deprecated", "readOnly", "writeOnly",
    "type", "properties", "required", "additionalProperties", "enum", "const", "items",
    "minItems", "maxItems", "uniqueItems", "minLength", "maxLength", "pattern",
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
    "minProperties", "maxProperties", "allOf", "anyOf", "oneOf", "not",
}
_TYPES = {"object", "array", "string", "integer", "number", "boolean", "null"}


def _check_schema(schema: dict[str, Any], path: str = "schema") -> None:
    if not isinstance(schema, dict):
        raise ValueError(f"{path} must be an object")
    unknown = set(schema) - _SCHEMA_KEYS
    if unknown:
        raise ValueError(f"{path} contains unsupported keyword(s): {', '.join(sorted(unknown))}")
    expected = schema.get("type")
    if expected is not None:
        alternatives = expected if isinstance(expected, list) else [expected]
        if not alternatives or any(not isinstance(item, str) or item not in _TYPES for item in alternatives):
            raise ValueError(f"{path} has an unsupported type")
    properties = schema.get("properties", {})
    if not isinstance(properties, dict) or any(not isinstance(key, str) for key in properties):
        raise ValueError(f"{path}.properties must be an object")
    for key, child in properties.items():
        _check_schema(child, f"{path}.{key}")
    required = schema.get("required", [])
    if not isinstance(required, list) or any(not isinstance(key, str) for key in required) or len(set(required)) != len(required):
        raise ValueError(f"{path}.required must contain unique strings")
    additional = schema.get("additionalProperties", True)
    if not isinstance(additional, (bool, dict)):
        raise ValueError(f"{path}.additionalProperties must be boolean or schema")
    if isinstance(additional, dict):
        _check_schema(additional, f"{path}.additionalProperties")
    for key in ("items", "not"):
        if key in schema:
            _check_schema(schema[key], f"{path}.{key}")
    for key in ("allOf", "anyOf", "oneOf"):
        if key in schema:
            if not isinstance(schema[key], list) or not schema[key]:
                raise ValueError(f"{path}.{key} must be a nonempty schema array")
            for child in schema[key]:
                _check_schema(child, f"{path}.{key}")
    for key in ("minItems", "maxItems", "minLength", "maxLength", "minProperties", "maxProperties"):
        if key in schema and (type(schema[key]) is not int or schema[key] < 0):
            raise ValueError(f"{path}.{key} must be a nonnegative integer")
    for key in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf"):
        if key in schema and (isinstance(schema[key], bool) or not isinstance(schema[key], (int, float)) or not math.isfinite(schema[key]) or (key == "multipleOf" and schema[key] <= 0)):
            raise ValueError(f"{path}.{key} must be finite and valid")
    if "pattern" in schema:
        if not isinstance(schema["pattern"], str):
            raise ValueError(f"{path}.pattern must be a string")
        re.compile(schema["pattern"])
    if "enum" in schema and (not isinstance(schema["enum"], list) or not schema["enum"]):
        raise ValueError(f"{path}.enum must be nonempty")
    if "uniqueItems" in schema and not isinstance(schema["uniqueItems"], bool):
        raise ValueError(f"{path}.uniqueItems must be boolean")
    for lower, upper in (("minItems", "maxItems"), ("minLength", "maxLength"), ("minProperties", "maxProperties"), ("minimum", "maximum"), ("exclusiveMinimum", "exclusiveMaximum")):
        if lower in schema and upper in schema and schema[lower] > schema[upper]:
            raise ValueError(f"{path} has contradictory {lower}/{upper}")
    json_bytes(schema)


def _has_pattern(schema: Any) -> bool:
    if isinstance(schema, dict):
        return "pattern" in schema or any(_has_pattern(child) for child in schema.values())
    if isinstance(schema, list):
        return any(_has_pattern(child) for child in schema)
    return False


def _json_equal(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return left == right
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_json_equal(value, right[key]) for key, value in left.items())
    if isinstance(left, list):
        return len(left) == len(right) and all(_json_equal(a, b) for a, b in zip(left, right))
    return left == right


def _validate_schema(value: Any, schema: dict[str, Any], path: str = "arguments") -> None:
    expected = schema.get("type")
    if isinstance(expected, list):
        if not any(_type_matches(value, alternative) for alternative in expected):
            raise ValueError(f"{path} has an invalid type")
    elif expected is not None and not _type_matches(value, expected):
        raise ValueError(f"{path} must be {expected}")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} must be finite")
    if "const" in schema and not _json_equal(value, schema["const"]):
        raise ValueError(f"{path} does not match const")
    if "enum" in schema and not any(_json_equal(value, item) for item in schema["enum"]):
        raise ValueError(f"{path} is not an allowed enum value")
    for key in ("allOf", "anyOf", "oneOf"):
        if key in schema:
            matches = 0
            for child in schema[key]:
                try:
                    _validate_schema(value, child, path)
                    matches += 1
                except ValueError:
                    pass
            valid = matches == len(schema[key]) if key == "allOf" else matches >= 1 if key == "anyOf" else matches == 1
            if not valid:
                raise ValueError(f"{path} violates {key}")
    if "not" in schema:
        try:
            _validate_schema(value, schema["not"], path)
        except ValueError:
            pass
        else:
            raise ValueError(f"{path} violates not")
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                raise ValueError(f"{path}.{key} is required")
        unknown = set(value) - set(properties)
        additional = schema.get("additionalProperties", True)
        if additional is False and unknown:
            raise ValueError(f"{path} contains unknown field(s): {', '.join(sorted(unknown))}")
        for key, child in value.items():
            if key in properties:
                _validate_schema(child, properties[key], f"{path}.{key}")
            elif isinstance(additional, dict):
                _validate_schema(child, additional, f"{path}.{key}")
        _length_limits(len(value), schema, "Properties", path)
    elif isinstance(value, list):
        _length_limits(len(value), schema, "Items", path)
        for index, child in enumerate(value):
            if "items" in schema:
                _validate_schema(child, schema["items"], f"{path}[{index}]")
        if schema.get("uniqueItems") and any(_json_equal(item, other) for index, item in enumerate(value) for other in value[:index]):
            raise ValueError(f"{path} items must be unique")
    elif isinstance(value, str):
        _length_limits(len(value), schema, "Length", path)
        if "pattern" in schema and re.search(schema["pattern"], value) is None:
            raise ValueError(f"{path} does not match pattern")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        for key, violation in (("minimum", lambda bound: value < bound), ("maximum", lambda bound: value > bound),
                               ("exclusiveMinimum", lambda bound: value <= bound), ("exclusiveMaximum", lambda bound: value >= bound)):
            if key in schema and violation(schema[key]):
                raise ValueError(f"{path} violates {key}")
        if "multipleOf" in schema and not math.isclose(value / schema["multipleOf"], round(value / schema["multipleOf"]), abs_tol=1e-9):
            raise ValueError(f"{path} violates multipleOf")


def _type_matches(value: Any, expected: str) -> bool:
    return {
        "object": isinstance(value, dict), "array": isinstance(value, list),
        "string": isinstance(value, str), "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "boolean": isinstance(value, bool), "null": value is None,
    }.get(expected, False)


def _length_limits(length: int, schema: dict[str, Any], suffix: str, path: str) -> None:
    if length < schema.get(f"min{suffix}", 0) or length > schema.get(f"max{suffix}", math.inf):
        raise ValueError(f"{path} violates length bounds")


class ToolRegistry:
    def __init__(self, specs: Iterable[ToolSpec] = ()):
        self._tools: dict[str, ToolSpec] = {}
        for spec in specs:
            self.register(spec)

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise KeyError(f"Tool already registered: {spec.name}")
        if not spec.name or not callable(spec.fn):
            raise ValueError("Tool requires a name and trusted callable")
        _check_schema(spec.input_schema)
        if not math.isfinite(spec.timeout_seconds) or spec.timeout_seconds <= 0:
            raise ValueError("tool timeout must be positive and finite")
        if type(spec.max_output_bytes) is not int or spec.max_output_bytes < 2:
            raise ValueError("tool output limit must be an integer >= 2")
        if spec.approval_policy not in {"never", "always"}:
            raise ValueError("unsupported approval policy")
        if spec.max_memory_bytes is not None and (type(spec.max_memory_bytes) is not int or spec.max_memory_bytes <= 0):
            raise ValueError("tool memory limit must be positive")
        self._tools[spec.name] = replace(spec, input_schema=copy.deepcopy(spec.input_schema))

    def get(self, name: str) -> ToolSpec:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"Unknown tool: {name}") from exc

    def schema(self) -> list[dict[str, Any]]:
        return [self._tools[name].schema() for name in sorted(self._tools)]


class ToolExecutor:
    def __init__(self, registry: ToolRegistry, *, policy: CapabilityPolicy | None = None, approval: ApprovalProvider | None = None, budget: ExecutionBudget | None = None, cancellation: CancellationToken | None = None, trace: InMemoryTrace | None = None, capabilities: set[str] | None = None):
        self.registry = registry
        self.policy = policy or AllowReadOnlyPolicy()
        self.approval = approval or DenyApproval()
        self.budget = budget or ExecutionBudget()
        self.cancellation = cancellation or CancellationToken()
        self.trace = trace or InMemoryTrace()
        self.capabilities = {"workspace.read"} if capabilities is None else set(capabilities)

    def _approve(self, call: ToolCall, spec: ToolSpec) -> bool:
        # UI approvals stay in the parent; a late response can never dispatch a tool.
        responses: queue.Queue[Any] = queue.Queue(maxsize=1)
        def ask() -> None:
            try:
                responses.put(self.approval.approve(call, spec))
            except BaseException as exc:
                responses.put(exc)
        threading.Thread(target=ask, daemon=True).start()
        while True:
            self.cancellation.raise_if_cancelled()
            self.budget.check_wall()
            try:
                result = responses.get(timeout=min(.02, self.budget.remaining_seconds()))
            except queue.Empty:
                continue
            if isinstance(result, BaseException):
                raise result
            self.cancellation.raise_if_cancelled()
            self.budget.check_wall()
            return result is True

    def execute(self, call: ToolCall | dict[str, Any] | str) -> ToolResult:
        started = time.perf_counter()
        parsed = ToolCall(name="unknown")
        worker = None
        limit = self.budget.max_output_bytes
        status, output, error, truncated = "error", None, None, False
        try:
            self.cancellation.raise_if_cancelled()
            self.budget.check_tool_call()
            parsed = ToolCall.from_value(call)
            spec = self.registry.get(parsed.name)
            limit = min(limit, spec.max_output_bytes)
            if _has_pattern(spec.input_schema):
                # Arbitrary model text can trigger catastrophic Python regexp backtracking.
                # Validate patterned schemas in a separately cancellable process first.
                validation_deadline = min(self.budget.deadline, time.monotonic() + spec.timeout_seconds)
                validation_worker = IsolatedWorker(_validate_schema, deadline=validation_deadline, cancellation=self.cancellation, max_memory_bytes=self.budget.max_memory_bytes)
                try:
                    validation_worker.call(parsed.arguments, spec.input_schema, deadline=validation_deadline, max_message_bytes=4096)
                finally:
                    validation_worker.close()
            else:
                _validate_schema(parsed.arguments, spec.input_schema)
            self.trace.emit("validation", "accepted", parsed.name, {"call_id": parsed.call_id})
            decision = self.policy.decide(spec)
            missing = sorted(spec.capabilities - self.capabilities)
            self.trace.emit("policy", "allowed" if decision.allowed and not missing else "denied", parsed.name, {"reason": decision.reason, "missing_capabilities": missing})
            if missing:
                status, error = "denied", f"missing capability: {', '.join(missing)}"
            elif not decision.allowed and not decision.requires_approval:
                status, error = "denied", decision.reason
            elif (decision.requires_approval or spec.approval_policy == "always") and not self._approve(parsed, spec):
                status, error = "denied", "approval denied"
                self.trace.emit("approval", "denied", parsed.name, {"call_id": parsed.call_id})
            else:
                if decision.requires_approval or spec.approval_policy == "always":
                    self.trace.emit("approval", "approved", parsed.name, {"call_id": parsed.call_id})
                self.cancellation.raise_if_cancelled()
                deadline = min(self.budget.deadline, time.monotonic() + spec.timeout_seconds)
                memory_limits = [value for value in (self.budget.max_memory_bytes, spec.max_memory_bytes) if value is not None]
                self.trace.emit("tool", "started", parsed.name, {"call_id": parsed.call_id, "arguments": parsed.arguments})
                worker = IsolatedWorker(spec.fn, deadline=deadline, cancellation=self.cancellation, max_memory_bytes=min(memory_limits) if memory_limits else None)
                context = ToolContext(self.cancellation, deadline, limit) if spec.inject_context else None
                response = worker.call(deadline=deadline, kwargs=parsed.arguments, context=context, bound_json=True, max_output_bytes=limit, max_message_bytes=max(4096, limit * 2))
                self.cancellation.raise_if_cancelled()
                self.budget.check_wall()
                status, output, truncated = "ok", response["output"], response["truncated"]
        except TimeoutError as exc:
            status, error = "timeout", str(exc)
        except ExecutionCancelled as exc:
            status, error = "cancelled", str(exc)
        except Exception as exc:
            status = "cancelled" if self.cancellation.cancelled else "error"
            error = f"{type(exc).__name__}: {exc}"
        finally:
            if worker is not None:
                worker.close()
        result = ToolResult(parsed.call_id, parsed.name, status, output, error, truncated, (time.perf_counter() - started) * 1000, effects_settled=worker is None or worker.effects_settled).bounded(limit)
        self.budget.output_bytes_used += len(json_bytes(result.output)) if result.output is not None else 0
        self.trace.emit("tool", status, parsed.name, {"call_id": result.call_id, "observation": {"call_id": result.call_id, "tool_name": result.tool_name, "status": result.status, "output": result.output, "error": result.error, "truncated": result.truncated, "effects_settled": result.effects_settled}})
        return result
