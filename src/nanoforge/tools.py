"""Compatibility facade over the bounded agent executor.

Legacy functions require explicit capability and human approval. Model output
never bypasses schema validation or dispatches an unrestricted Python call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from nanoforge.agent.executor import ToolExecutor
from nanoforge.agent.executor import ToolRegistry as BoundedToolRegistry
from nanoforge.agent.protocol import ToolSpec as BoundedToolSpec


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., Any]
    capabilities: frozenset[str] = frozenset({"legacy.execute"})
    approval_policy: str = "always"
    side_effect: str = "write"
    timeout_seconds: float = 30.0
    max_output_bytes: int = 65536


class ToolRegistry:
    def __init__(self, *, executor_kwargs: dict[str, Any] | None = None):
        self._registry = BoundedToolRegistry()
        self.executor_kwargs = dict(executor_kwargs or {})

    def register(self, spec: ToolSpec) -> None:
        self._registry.register(BoundedToolSpec(
            spec.name, spec.description, spec.parameters, spec.fn,
            capabilities=spec.capabilities, approval_policy=spec.approval_policy,
            side_effect=spec.side_effect, timeout_seconds=spec.timeout_seconds,
            max_output_bytes=spec.max_output_bytes,
        ))

    def schema(self) -> list[dict[str, Any]]:
        return self._registry.schema()

    def call_json(self, payload: str) -> Any:
        result = ToolExecutor(self._registry, **self.executor_kwargs).execute(payload)
        if result.status == "denied":
            raise PermissionError(result.error or "legacy tool execution denied")
        if result.status != "ok":
            raise RuntimeError(result.error or result.status)
        return result.output
