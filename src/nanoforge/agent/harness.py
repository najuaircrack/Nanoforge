"""Bounded action loop and deterministic, side-effect-free trace reconstruction."""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence

from nanoforge.agent.actions import ActionDecodeError, FinalAction, decode_action
from nanoforge.agent.executor import ToolExecutor, ToolRegistry
from nanoforge.agent.isolation import ExecutionCancelled, IsolatedWorker
from nanoforge.agent.protocol import (
    CancellationToken,
    ExecutionBudget,
    InMemoryTrace,
    ToolCall,
    _bounded,
    json_bytes,
)
from nanoforge.agent.provider import (
    ProviderIdentity,
    ProviderRequest,
    ProviderResponse,
    ScriptedProvider,
)


class ActionProvider(Protocol):
    def __call__(self, request: str, observations: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]: ...


class ManualActionProvider:
    """Deterministic operator actions; no model or arbitrary callback is invoked."""

    def __init__(self, actions: Sequence[dict[str, Any]]):
        self._actions = list(actions)
        self._index = 0

    def __call__(self, request: str, observations: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        if self._index >= len(self._actions):
            raise ValueError("manual action sequence ended before a final action")
        action = self._actions[self._index]
        self._index += 1
        if not isinstance(action, dict):
            raise ValueError("manual actions must be JSON objects")
        return action


@dataclass
class AgentRunResult:
    run_id: str
    status: str
    answer: str | None
    observations: list[dict[str, Any]] = field(default_factory=list)
    trace: list[dict[str, Any]] = field(default_factory=list)
    error_code: str | None = None
    error: str | None = None
    provider: ProviderIdentity | None = None
    resources: dict[str, Any] = field(default_factory=dict)
    answer_truncated: bool = False


class AgentHarness:
    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def run(self, request: str, provider: ActionProvider, *, budget: ExecutionBudget | None = None, cancellation: CancellationToken | None = None, trace: InMemoryTrace | None = None, executor_kwargs: dict[str, Any] | None = None, trace_path: str | Path | None = None, context_items: Sequence[dict[str, Any]] | None = None) -> AgentRunResult:
        trace = trace or InMemoryTrace(path=trace_path)
        if trace_path is not None:
            trace.path = Path(trace_path)
        cancellation = cancellation or CancellationToken()
        executor = ToolExecutor(self.registry, budget=budget, cancellation=cancellation, trace=trace, **(executor_kwargs or {}))
        budget = executor.budget
        observations: list[dict[str, Any]] = []
        provider_identity = getattr(provider, "identity", None)
        worker: IsolatedWorker | None = None
        trace.emit("request", "accepted", "run", {"request": request, "budgets": budget.snapshot()})

        def finish(status: str, answer: str | None = None, error_code: str | None = None, error: str | None = None, truncated: bool = False) -> AgentRunResult:
            if worker is not None:
                worker.close()
            trace.emit("harness", status, "final_state", {"status": status, "answer": answer, "error_code": error_code, "error": error, "resources": budget.snapshot(), "answer_truncated": truncated})
            return AgentRunResult(trace.run_id, status, answer, observations, trace.to_list(), error_code, error, provider_identity, budget.snapshot(), truncated)

        try:
            if context_items is not None and not isinstance(context_items, (list, tuple)):
                raise ValueError("context_items must be a list or tuple")
            selected_context = []
            for item in context_items or ():
                if not isinstance(item, dict) or set(item) - {"id", "text", "provenance", "owner", "scope"}:
                    raise ValueError("context item must contain only id/text/provenance/owner/scope")
                if not isinstance(item.get("id"), str) or not item["id"] or not isinstance(item.get("text"), str) or not isinstance(item.get("provenance"), (str, dict)):
                    raise ValueError("context item requires string id/text and declared provenance")
                selected_context.append(copy.deepcopy(item))
            if selected_context:
                trace.emit("context", "selected", "explicit_context", {"items": selected_context, "count": len(selected_context)})
            while True:
                budget.check_step()
                cancellation.raise_if_cancelled()
                tools = self.registry.schema()
                context_size = len(json_bytes({"request": request, "observations": observations, "tools": tools, "context_items": selected_context}))
                if context_size > budget.max_context_bytes:
                    raise RuntimeError("agent context byte budget exceeded")
                trace.emit("provider", "started", "next_action", {"step": budget.steps_used, "observation_count": len(observations), "context_bytes": context_size})
                complete = hasattr(provider, "complete")
                # These exact types are deterministic, finite library implementations.
                safe_inline = type(provider) in {ManualActionProvider, ScriptedProvider}
                if not safe_inline and worker is None:
                    factory = getattr(provider, "isolation_factory", None)
                    worker = IsolatedWorker(factory if callable(factory) else provider, factory=callable(factory), deadline=budget.deadline, cancellation=cancellation, max_memory_bytes=budget.max_memory_bytes)
                if complete:
                    remaining_tokens = budget.max_tokens - budget.tokens_used
                    if remaining_tokens <= 0:
                        raise RuntimeError("agent token budget exceeded")
                    provider_request = ProviderRequest(
                        user_request=request, observations=tuple(observations), tools=tuple(tools),
                        step=budget.steps_used, max_new_tokens=min(256, remaining_tokens),
                        context_items=tuple(selected_context),
                    )
                    response = provider.complete(provider_request, cancellation=cancellation) if safe_inline else worker.call(provider_request, mode="provider", deadline=budget.deadline, max_message_bytes=65536)
                    cancellation.raise_if_cancelled()
                    budget.check_wall()
                    if not isinstance(response, ProviderResponse) or not isinstance(response.text, str):
                        raise TypeError("model provider must return ProviderResponse with text")
                    provider_identity = response.identity
                    raw_text = response.text
                    prompt_usage = response.prompt_tokens if response.prompt_tokens is not None else context_size
                    completion_usage = response.completion_tokens if response.completion_tokens is not None else len(raw_text.encode("utf-8"))
                    budget.charge_tokens(prompt_usage)
                    budget.charge_tokens(completion_usage)
                    usage_source = getattr(response, "usage_source", None)
                    if usage_source:
                        budget.token_accounting = usage_source
                    trace.emit("provider", "completed", "text_response", {"provider_id": response.identity.provider_id, "provider_kind": response.identity.kind, "model_fingerprint": getattr(response.identity, "model_fingerprint", None), "output_bytes": len(raw_text.encode("utf-8")), "output_sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(), "prompt_usage": prompt_usage, "completion_usage": completion_usage, "usage_source": usage_source or "conservative-utf8-bytes"})
                    decoded = decode_action(raw_text)
                    if isinstance(decoded, FinalAction):
                        action = {"type": "final", "content": decoded.content}
                    else:
                        action = {"type": "tool_call", "name": decoded.call.name, "arguments": decoded.call.arguments, "call_id": decoded.call.call_id}
                else:
                    action = provider(request, observations, tools) if safe_inline else worker.call(request, observations, tools, deadline=budget.deadline, max_message_bytes=65536)
                    cancellation.raise_if_cancelled()
                    budget.check_wall()
                    if not isinstance(action, dict):
                        raise ValueError("model action must be an object")
                    if len(json_bytes(action)) > 16 * 1024:
                        raise ValueError("action output exceeds 16384 bytes")
                    if not safe_inline:
                        budget.charge_tokens(context_size + len(json_bytes(action)))
                if action.get("type") == "final":
                    if set(action) - {"type", "content", "schema_version"}:
                        raise ValueError("final action contains unknown fields")
                    answer = action.get("content")
                    if not isinstance(answer, str):
                        raise ValueError("final content must be a string")
                    cancellation.raise_if_cancelled()
                    budget.check_wall()
                    bounded_answer = _bounded(answer, budget.max_output_bytes)
                    truncated = bounded_answer != answer
                    answer = bounded_answer if isinstance(bounded_answer, str) else bounded_answer["text"]
                    budget.output_bytes_used += len(json_bytes(answer))
                    trace.emit("action", "decoded", "final", {"content": answer})
                    return finish("completed", answer=answer, truncated=truncated)
                if action.get("type") != "tool_call":
                    raise ValueError("action type must be 'tool_call' or 'final'")
                call = ToolCall.from_value(action)
                trace.emit("action", "decoded", "tool_call", {"call_id": call.call_id, "name": call.name, "arguments": call.arguments})
                result = executor.execute(call)
                observation = {"call_id": result.call_id, "tool_name": result.tool_name, "status": result.status, "output": result.output, "error": result.error, "truncated": result.truncated, "effects_settled": result.effects_settled}
                observations.append(observation)
                trace.emit("state", "updated", "observation", {"observation": observation, "resources": budget.snapshot()})
                if result.status == "cancelled":
                    return finish("cancelled", error_code="cancelled", error=result.error)
                if result.status == "timeout" and budget.remaining_seconds() <= 0:
                    return finish("timeout", error_code="deadline_exceeded", error=result.error)
        except ActionDecodeError as exc:
            trace.emit("action", "rejected", exc.code, {"error": str(exc)})
            return finish("blocked", error_code=exc.code, error=str(exc))
        except TimeoutError as exc:
            return finish("timeout", error_code="deadline_exceeded", error=str(exc))
        except ExecutionCancelled as exc:
            return finish("cancelled", error_code="cancelled", error=str(exc))
        except Exception as exc:
            if cancellation.cancelled:
                status, code = "cancelled", "cancelled"
            elif isinstance(exc, RuntimeError) and "budget exceeded" in str(exc):
                status, code = "blocked", "budget_exceeded"
            elif isinstance(exc, ValueError) and str(exc).startswith("context"):
                status, code = "blocked", "context_error"
            else:
                status, code = ("provider_error", "provider_error") if hasattr(provider, "complete") else ("blocked", "harness_error")
            return finish(status, error_code=code, error=f"{type(exc).__name__}: {exc}")
        finally:
            if worker is not None:
                worker.close()
            if trace.path is not None:
                trace.save(trace.path)


class ReplayRunner:
    """Reconstruct recorded control state without invoking any model or tool."""

    def replay(self, trace: list[dict[str, Any]] | str | Path | InMemoryTrace) -> dict[str, Any]:
        if isinstance(trace, (str, Path)):
            trace = InMemoryTrace.load(trace)
        dropped = trace.dropped_events if isinstance(trace, InMemoryTrace) else 0
        events = trace.to_list() if isinstance(trace, InMemoryTrace) else trace
        if not isinstance(events, list):
            raise ValueError("replay trace must be an event list")
        state: dict[str, Any] = {"request": None, "context_items": [], "actions": [], "observations": [], "status": "incomplete", "answer": None}
        run_id = None
        complete = not dropped
        for sequence, event in enumerate(events):
            if not isinstance(event, dict) or event.get("sequence") != sequence:
                raise ValueError("trace event order is invalid")
            if sequence == 0:
                run_id = event.get("run_id")
            if event.get("run_id") != run_id:
                raise ValueError("trace combines different runs")
            details = event.get("details", {})
            if details.get("truncated"):
                complete = False
            if event.get("phase") == "request":
                state["request"] = details.get("request")
            elif event.get("phase") == "context":
                state["context_items"] = details.get("items", [])
            elif event.get("phase") == "action" and event.get("status") == "decoded":
                state["actions"].append(details)
            elif event.get("phase") == "state" and event.get("action") == "observation":
                state["observations"].append(details.get("observation"))
            elif event.get("phase") == "harness" and event.get("action") == "final_state":
                state.update({key: details.get(key) for key in ("status", "answer", "error_code", "error", "resources")})
        complete = complete and state["status"] != "incomplete"
        return {"status": "dry_run", "events": len(events), "trace": events, "state": state, "trace_complete": complete, "side_effects_applied": False, "payloads_may_be_redacted": True}
