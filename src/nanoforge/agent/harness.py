"""Bounded action loop for coding-agent workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from nanoforge.agent.executor import ToolExecutor, ToolRegistry
from nanoforge.agent.protocol import CancellationToken, ExecutionBudget, InMemoryTrace, ToolCall


class ActionProvider(Protocol):
    def __call__(self, request: str, observations: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]: ...


@dataclass
class AgentRunResult:
    run_id: str
    status: str
    answer: str | None
    observations: list[dict[str, Any]] = field(default_factory=list)
    trace: list[dict[str, Any]] = field(default_factory=list)


class AgentHarness:
    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def run(
        self,
        request: str,
        provider: ActionProvider,
        *,
        budget: ExecutionBudget | None = None,
        cancellation: CancellationToken | None = None,
        trace: InMemoryTrace | None = None,
        executor_kwargs: dict[str, Any] | None = None,
    ) -> AgentRunResult:
        trace = trace or InMemoryTrace()
        cancellation = cancellation or CancellationToken()
        executor = ToolExecutor(
            self.registry,
            budget=budget,
            cancellation=cancellation,
            trace=trace,
            **(executor_kwargs or {}),
        )
        observations: list[dict[str, Any]] = []
        while True:
            try:
                executor.budget.check_step()
                cancellation.raise_if_cancelled()
                trace.emit("model", "started", "next_action", {"observation_count": len(observations)})
                action = provider(request, observations, self.registry.schema())
                if not isinstance(action, dict):
                    raise ValueError("model action must be an object")
                if action.get("type") == "final":
                    answer = action.get("content", "")
                    if not isinstance(answer, str):
                        raise ValueError("final content must be a string")
                    trace.emit("model", "completed", "final", {"answer": answer[:4000]})
                    return AgentRunResult(trace.run_id, "completed", answer, observations, trace.to_list())
                if action.get("type") != "tool_call":
                    raise ValueError("action type must be 'tool_call' or 'final'")
                result = executor.execute(ToolCall.from_value(action))
                observation = {"call_id": result.call_id, "tool_name": result.tool_name, "status": result.status, "output": result.output, "error": result.error}
                observations.append(observation)
                if result.status in {"denied", "timeout", "error"}:
                    trace.emit("harness", "observation", "tool_failure", {"status": result.status})
            except Exception as exc:
                status = "cancelled" if cancellation.cancelled else "blocked"
                trace.emit("harness", status, "run", {"error": f"{type(exc).__name__}: {exc}"})
                return AgentRunResult(trace.run_id, status, None, observations, trace.to_list())


class ReplayRunner:
    """Replay traces as dry-run records; it never applies writes or reruns tests."""

    def replay(self, trace: list[dict[str, Any]]) -> dict[str, Any]:
        return {"status": "dry_run", "events": len(trace), "trace": trace, "side_effects_applied": False}
