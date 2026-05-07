from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., Any]


class ToolRegistry:
    """Minimal function-calling registry for instruction-tuned models."""

    def __init__(self):
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def schema(self) -> list[dict[str, Any]]:
        return [
            {"name": t.name, "description": t.description, "parameters": t.parameters}
            for t in self._tools.values()
        ]

    def call_json(self, payload: str) -> Any:
        request = json.loads(payload)
        name = request["name"]
        args = request.get("arguments", {})
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        return self._tools[name].fn(**args)

