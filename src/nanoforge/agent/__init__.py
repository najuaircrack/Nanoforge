"""Bounded, replayable coding-agent harness primitives."""

from nanoforge.agent.actions import (
    ACTION_SCHEMA_VERSION,
    ActionDecodeError,
    FinalAction,
    ToolAction,
    decode_action,
    encode_action,
)
from nanoforge.agent.executor import ToolExecutor, ToolRegistry
from nanoforge.agent.harness import AgentHarness, AgentRunResult, ManualActionProvider, ReplayRunner
from nanoforge.agent.project import (
    ProjectManifest,
    build_project_provider,
    build_project_runtime,
    evaluate_project_model,
    load_project_manifest,
    project_context_items,
    project_policy,
    project_workspace,
)
from nanoforge.agent.protocol import (
    ApprovalProvider,
    CancellationToken,
    ExecutionBudget,
    InMemoryTrace,
    StaticApproval,
    ToolCall,
    ToolContext,
    ToolResult,
    ToolSpec,
)
from nanoforge.agent.provider import (
    ModelProvider,
    ProviderIdentity,
    ProviderRequest,
    ProviderResponse,
    ScriptedProvider,
    TextCompletionProvider,
)
from nanoforge.agent.repository import RepositoryWorkspace, register_repository_tools

__all__ = [
    "AgentHarness",
    "AgentRunResult",
    "ACTION_SCHEMA_VERSION",
    "ActionDecodeError",
    "FinalAction",
    "ToolAction",
    "decode_action",
    "encode_action",
    "ProjectManifest",
    "build_project_runtime",
    "build_project_provider",
    "project_workspace",
    "project_context_items",
    "evaluate_project_model",
    "load_project_manifest",
    "project_policy",
    "ModelProvider",
    "ProviderIdentity",
    "ProviderRequest",
    "ProviderResponse",
    "ScriptedProvider",
    "TextCompletionProvider",
    "ManualActionProvider",
    "ApprovalProvider",
    "CancellationToken",
    "ExecutionBudget",
    "InMemoryTrace",
    "ReplayRunner",
    "RepositoryWorkspace",
    "StaticApproval",
    "ToolCall",
    "ToolExecutor",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
    "ToolContext",
    "register_repository_tools",
]
