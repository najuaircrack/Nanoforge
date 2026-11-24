"""Strict, versioned wire actions for model-to-tool boundaries."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any

from nanoforge.agent.protocol import ToolCall

ACTION_SCHEMA_VERSION = 1
DEFAULT_MAX_ACTION_BYTES = 16 * 1024
DEFAULT_MAX_ACTION_DEPTH = 16


@dataclass(frozen=True)
class FinalAction:
    content: str


@dataclass(frozen=True)
class ToolAction:
    call: ToolCall


AgentAction = FinalAction | ToolAction


class ActionDecodeError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _duplicate_key_guard(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ActionDecodeError("duplicate_key", f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ActionDecodeError("invalid_constant", f"JSON constant is not allowed: {value}")


def _check_depth(value: Any, depth: int, max_depth: int) -> None:
    if depth > max_depth:
        raise ActionDecodeError("depth_exceeded", "action JSON nesting depth exceeded")
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ActionDecodeError("invalid_type", "action object keys must be strings")
            try:
                key.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise ActionDecodeError("invalid_unicode", "action contains invalid Unicode") from exc
            _check_depth(child, depth + 1, max_depth)
    elif isinstance(value, list):
        for child in value:
            _check_depth(child, depth + 1, max_depth)
    elif isinstance(value, str):
        try:
            value.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ActionDecodeError("invalid_unicode", "action contains invalid Unicode") from exc


def decode_action(
    raw: str,
    *,
    schema_version: int = ACTION_SCHEMA_VERSION,
    max_bytes: int = DEFAULT_MAX_ACTION_BYTES,
    max_depth: int = DEFAULT_MAX_ACTION_DEPTH,
) -> AgentAction:
    """Decode exactly one strict action; never repair or strip model output."""

    if not isinstance(raw, str):
        raise ActionDecodeError("invalid_type", "action output must be a string")
    try:
        encoded = raw.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ActionDecodeError("invalid_unicode", "action contains invalid Unicode") from exc
    if len(encoded) > max_bytes:
        raise ActionDecodeError("oversize", f"action output exceeds {max_bytes} bytes")
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_duplicate_key_guard,
            parse_constant=_reject_constant,
        )
    except ActionDecodeError:
        raise
    except json.JSONDecodeError as exc:
        raise ActionDecodeError("invalid_json", f"invalid action JSON at character {exc.pos}") from exc
    if not isinstance(value, dict):
        raise ActionDecodeError("root_not_object", "action root must be a JSON object")
    _check_depth(value, 1, max_depth)
    if type(value.get("schema_version")) is not int or value.get("schema_version") != schema_version:
        raise ActionDecodeError("unsupported_version", f"action schema_version must be {schema_version}")
    action_type = value.get("type")
    if action_type == "final":
        expected = {"schema_version", "type", "content"}
        unknown = sorted(set(value) - expected)
        if unknown:
            raise ActionDecodeError("unknown_field", f"unknown action field(s): {', '.join(unknown)}")
        content = value.get("content")
        if not isinstance(content, str):
            raise ActionDecodeError("invalid_type", "final.content must be a string")
        return FinalAction(content)
    if action_type == "tool_call":
        expected = {"schema_version", "type", "name", "arguments"}
        unknown = sorted(set(value) - expected)
        if unknown:
            raise ActionDecodeError("unknown_field", f"unknown action field(s): {', '.join(unknown)}")
        name = value.get("name")
        arguments = value.get("arguments")
        if not isinstance(name, str) or not name:
            raise ActionDecodeError("invalid_tool_name", "tool_call.name must be a non-empty string")
        if not isinstance(arguments, dict):
            raise ActionDecodeError("invalid_type", "tool_call.arguments must be an object")
        return ToolAction(ToolCall(name=name, arguments=arguments, call_id=str(uuid.uuid4())))
    raise ActionDecodeError("invalid_type", "action.type must be 'final' or 'tool_call'")


def encode_action(action: AgentAction, *, schema_version: int = ACTION_SCHEMA_VERSION) -> str:
    """Encode an internal action into its canonical wire representation."""

    if isinstance(action, FinalAction):
        value = {"schema_version": schema_version, "type": "final", "content": action.content}
    elif isinstance(action, ToolAction):
        value = {
            "schema_version": schema_version,
            "type": "tool_call",
            "name": action.call.name,
            "arguments": action.call.arguments,
        }
    else:
        raise TypeError(f"unsupported action type: {type(action).__name__}")
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
