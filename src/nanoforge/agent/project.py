"""Versioned project manifests and local model selection for the agent harness."""
from __future__ import annotations

import hashlib
import json
import math
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import yaml

from nanoforge.agent.executor import ToolRegistry
from nanoforge.agent.protocol import AllowReadOnlyPolicy, DenyApproval, ExecutionBudget
from nanoforge.agent.repository import RepositoryWorkspace, register_repository_tools

_SCHEMA_VERSION = 2
_KNOWN_KEYS = {"schema_version", "name", "workspace_root", "provider_kind", "enabled_tools",
               "capabilities", "approval_mode", "budgets", "model", "resources", "trace_dir", "eval_suite"}
_KNOWN_TOOLS = {"repo.list_files", "repo.read_file", "repo.search", "repo.apply_patch", "repo.run_tests"}
_DEFAULT_TOOLS = ("repo.list_files", "repo.read_file", "repo.search")
_DEFAULT_BUDGETS = {"max_steps": 12, "max_tool_calls": 12, "max_wall_seconds": 120.0,
                    "max_output_bytes": 65536, "max_tokens": 4096, "max_context_bytes": 262144,
                    "max_memory_bytes": None}
_DEFAULT_RESOURCES = {"max_file_bytes": 262144, "max_patch_bytes": 524288,
                      "max_scan_entries": 10000, "max_input_file_bytes": 33554432, "context_paths": []}
_MODEL_KEYS = {"source", "device", "chat_format", "max_new_tokens", "max_prompt_bytes", "allow_network"}


@dataclass(frozen=True)
class ProjectManifest:
    schema_version: int
    name: str
    workspace_root: str
    provider_kind: str
    enabled_tools: tuple[str, ...]
    capabilities: frozenset[str]
    approval_mode: str
    budgets: dict[str, int | float | None]
    model: dict[str, Any] = field(default_factory=dict)
    resources: dict[str, Any] = field(default_factory=lambda: dict(_DEFAULT_RESOURCES))
    trace_dir: str = ".latticekin/traces"
    eval_suite: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["enabled_tools"] = list(self.enabled_tools)
        result["capabilities"] = sorted(self.capabilities)
        return result

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _as_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be an object with string keys")
    return value


def _strict_int(value: Any, label: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _strict_number(value: Any, label: str, *, minimum: float = 0.0) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < minimum:
        raise ValueError(f"{label} must be a finite number >= {minimum}")
    return value


def _relative(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must be relative and cannot contain '..'")
    return path.as_posix()


def project_workspace(manifest: ProjectManifest, manifest_path: str | Path) -> Path:
    parent = Path(manifest_path).resolve().parent
    root = (parent / _relative(manifest.workspace_root, "workspace_root")).resolve()
    try:
        root.relative_to(parent)
    except ValueError as exc:
        raise ValueError("workspace_root symlink escapes manifest directory") from exc
    if not root.is_dir():
        raise NotADirectoryError(root)
    return root


def load_project_manifest(path: str | Path) -> ProjectManifest:
    manifest_path = Path(path).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if manifest_path.stat().st_size > 65536:
        raise ValueError("project manifest exceeds 64 KiB")
    try:
        values = _as_mapping(yaml.safe_load(manifest_path.read_text(encoding="utf-8")), "project manifest")
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid project manifest YAML: {exc}") from exc
    unknown = sorted(set(values) - _KNOWN_KEYS)
    if unknown:
        raise ValueError(f"project manifest contains unknown key(s): {', '.join(unknown)}")
    version = _strict_int(values.get("schema_version", 1), "schema_version")
    if version > _SCHEMA_VERSION:
        raise ValueError(f"unsupported project manifest schema_version={version}")
    name = values.get("name", manifest_path.stem)
    if not isinstance(name, str) or not name.strip():
        raise ValueError("name must be a non-empty string")
    workspace = _relative(values.get("workspace_root", "."), "workspace_root")
    kind = values.get("provider_kind", "manual")
    if kind not in {"manual", "native", "imported"}:
        raise ValueError("provider_kind must be manual, native or imported")
    tools = values.get("enabled_tools", list(_DEFAULT_TOOLS))
    if not isinstance(tools, list) or not tools or any(not isinstance(item, str) for item in tools):
        raise ValueError("enabled_tools must be a non-empty list of strings")
    if len(tools) != len(set(tools)) or set(tools) - _KNOWN_TOOLS:
        raise ValueError("enabled_tools contains duplicate or unknown tools")
    capabilities = values.get("capabilities", ["workspace.read"])
    known_capabilities = {"workspace.read", "workspace.write", "process.test", "workspace.read_sensitive"}
    if not isinstance(capabilities, list) or any(not isinstance(item, str) for item in capabilities) or set(capabilities) - known_capabilities:
        raise ValueError("capabilities contains an unknown capability")
    approval = values.get("approval_mode", "read_only")
    if approval not in {"read_only", "interactive"}:
        raise ValueError("approval_mode must be read_only or interactive")
    budget_values = _as_mapping(values.get("budgets", {}), "budgets")
    if set(budget_values) - set(_DEFAULT_BUDGETS):
        raise ValueError("budgets contains unknown keys")
    budgets = dict(_DEFAULT_BUDGETS)
    for key, value in budget_values.items():
        budgets[key] = (_strict_number(value, "budgets." + key, minimum=0.01) if key == "max_wall_seconds"
                        else None if key == "max_memory_bytes" and value is None
                        else _strict_int(value, "budgets." + key, minimum=2 if key == "max_output_bytes" else 1))
    resources_values = _as_mapping(values.get("resources", {}), "resources")
    if set(resources_values) - set(_DEFAULT_RESOURCES):
        raise ValueError("resources contains unknown keys")
    resources = dict(_DEFAULT_RESOURCES)
    for key, value in resources_values.items():
        if key == "context_paths":
            if not isinstance(value, list) or len(value) > 32:
                raise ValueError("resources.context_paths must be a list of at most 32 paths")
            resources[key] = [_relative(item, "context path") for item in value]
        else:
            resources[key] = _strict_int(value, "resources." + key)
    model = _as_mapping(values.get("model", {}), "model")
    if set(model) - _MODEL_KEYS:
        raise ValueError("model contains unknown keys")
    if kind == "manual" and model:
        raise ValueError("manual providers cannot contain an unused model configuration")
    if kind != "manual":
        source = model.get("source")
        if not isinstance(source, str) or not source.strip():
            raise ValueError("native/imported providers require model.source")
        if not isinstance(model.get("allow_network", False), bool):
            raise ValueError("model.allow_network must be boolean")
        if not model.get("allow_network", False):
            _relative(source, "model.source")
        if kind == "native" and model.get("allow_network", False):
            raise ValueError("native checkpoints are local files")
    if model.get("chat_format", "auto") not in {"auto", "native", "hf", "chatml", "raw_json"}:
        raise ValueError("unsupported model.chat_format")
    if not isinstance(model.get("device", "cpu"), str):
        raise ValueError("model.device must be a string")
    if "max_new_tokens" in model:
        _strict_int(model["max_new_tokens"], "model.max_new_tokens")
    if "max_prompt_bytes" in model:
        _strict_int(model["max_prompt_bytes"], "model.max_prompt_bytes")
    trace_dir = _relative(values.get("trace_dir", ".latticekin/traces"), "trace_dir")
    eval_suite = values.get("eval_suite")
    if eval_suite is not None:
        eval_suite = _relative(eval_suite, "eval_suite")
    manifest = ProjectManifest(version, name.strip(), workspace, kind, tuple(tools), frozenset(capabilities),
                               approval, budgets, model, resources, trace_dir, eval_suite)
    project_workspace(manifest, manifest_path)
    return manifest


def build_project_runtime(manifest: ProjectManifest, manifest_path: str | Path) -> tuple[RepositoryWorkspace, ToolRegistry, ExecutionBudget]:
    root = project_workspace(manifest, manifest_path)
    kwargs = {key: value for key, value in manifest.resources.items() if key != "context_paths"}
    workspace = RepositoryWorkspace(root, allow_sensitive="workspace.read_sensitive" in manifest.capabilities, **kwargs)
    specs = {spec.name: spec for spec in register_repository_tools(workspace)}
    registry = ToolRegistry(specs[name] for name in manifest.enabled_tools)
    return workspace, registry, ExecutionBudget(**manifest.budgets)


def project_context_items(manifest: ProjectManifest, workspace: RepositoryWorkspace) -> list[dict[str, Any]]:
    """Read only explicitly selected, policy-allowed context with content provenance."""
    items = []
    for path in manifest.resources.get("context_paths", []):
        content = workspace.read_file(path)
        items.append({"id": path, "text": content["text"], "owner": manifest.name, "scope": "project",
                      "provenance": {"path": path, "sha256": content["sha256"],
                                     "size_bytes": content["size_bytes"], "truncated": content["truncated"]}})
    if len(json.dumps(items, ensure_ascii=False).encode("utf-8")) > manifest.budgets["max_context_bytes"]:
        raise ValueError("selected context exceeds project context byte budget")
    return items


@dataclass(frozen=True)
class ProjectProviderFactory:
    """Serializable configuration; a persistent worker loads its own model."""
    manifest: ProjectManifest
    manifest_path: str

    def __call__(self):
        return _load_project_model_provider(self.manifest, self.manifest_path)


class ProjectModelProvider:
    def __init__(self, factory: ProjectProviderFactory):
        from nanoforge.agent.provider import ProviderIdentity
        self.isolation_factory = factory
        self.identity = ProviderIdentity(factory.manifest.provider_kind, factory.manifest.name,
                                         factory.manifest.model["source"], "project-selected-local-model")
        self._loaded = None

    def _load(self):
        if self._loaded is None:
            self._loaded = self.isolation_factory()
        return self._loaded

    @property
    def engine(self):
        return self._load().engine

    def complete(self, request, *, cancellation):
        cancellation.raise_if_cancelled()
        return self._load().complete(request, cancellation=cancellation)


def build_project_provider(manifest: ProjectManifest, manifest_path: str | Path, manual_actions: Sequence[dict[str, Any]] | None = None):
    from nanoforge.agent.provider import ScriptedProvider
    if manifest.provider_kind == "manual":
        if manual_actions is None:
            raise ValueError("manual provider requires an explicit operator action sequence")
        return ScriptedProvider([json.dumps(action, ensure_ascii=False, allow_nan=False) for action in manual_actions], provider_id=manifest.name)
    root = project_workspace(manifest, manifest_path)
    if not manifest.model.get("allow_network", False):
        source = (root / manifest.model["source"]).resolve()
        try:
            source.relative_to(root)
        except ValueError as exc:
            raise ValueError("model source symlink escapes workspace") from exc
        if not source.exists():
            raise FileNotFoundError(source)
    return ProjectModelProvider(ProjectProviderFactory(manifest, str(Path(manifest_path).resolve())))


def _load_project_model_provider(manifest: ProjectManifest, manifest_path: str | Path):
    from nanoforge.agent.provider import ProviderIdentity, TextCompletionProvider
    root = project_workspace(manifest, manifest_path)
    source = manifest.model["source"]
    local_source = (root / source).resolve()
    allow_network = manifest.model.get("allow_network", False)
    if not allow_network:
        try:
            local_source.relative_to(root)
        except ValueError as exc:
            raise ValueError("model source symlink escapes workspace") from exc
        if not local_source.exists():
            raise FileNotFoundError(local_source)
    else:
        source = str(local_source) if local_source.exists() else source
    if manifest.provider_kind == "native":
        from nanoforge.generation.engine import GenerationEngine
        from nanoforge.training.checkpoint import load_checkpoint
        payload = load_checkpoint(local_source, map_location="cpu")
        tokenizer_path = getattr(payload["config"].data, "tokenizer_path", None)
        if tokenizer_path is not None:
            tokenizer_path = (root / _relative(tokenizer_path, "checkpoint tokenizer_path")).resolve()
            try:
                tokenizer_path.relative_to(root)
            except ValueError as exc:
                raise ValueError("checkpoint tokenizer symlink escapes project") from exc
        del payload
        engine = GenerationEngine.from_checkpoint(str(local_source), tokenizer_path=tokenizer_path,
                                                  device=manifest.model.get("device", "cpu"))
    else:
        from nanoforge.imports import load_imported_engine
        engine = load_imported_engine(str(local_source) if not allow_network else source,
                                      device=manifest.model.get("device", "cpu"), allow_network=allow_network)
    provider = TextCompletionProvider(engine, provider_id=manifest.name, model_id=manifest.model["source"],
                                     chat_format=manifest.model.get("chat_format", "auto"),
                                     max_prompt_bytes=manifest.model.get("max_prompt_bytes", manifest.budgets.get("max_context_bytes", 262144)),
                                     max_new_tokens=manifest.model.get("max_new_tokens", 256))
    provenance = "local-checkpoint" if manifest.provider_kind == "native" else "imported-local-model" if not allow_network else "explicit-network-model-import"
    provider.identity = ProviderIdentity(manifest.provider_kind, manifest.name, manifest.model["source"], provenance,
                                         _model_fingerprint(local_source) if local_source.exists() else None)
    return provider


def _model_fingerprint(source: Path) -> str:
    digest = hashlib.sha256()
    files = [source] if source.is_file() else sorted(
        file for file in source.rglob("*") if file.is_file()
        and file.suffix.lower() in {".json", ".txt", ".bin", ".pt", ".safetensors", ".gguf", ".onnx", ".model"}
    )
    if len(files) > 10000:
        raise ValueError("model artifact exceeds fingerprint file-count limit")
    for file in files:
        if source.is_dir():
            try:
                file.resolve().relative_to(source.resolve())
            except ValueError as exc:
                raise ValueError("model artifact symlink escapes selected directory") from exc
            digest.update(file.relative_to(source).as_posix().encode("utf-8"))
        with file.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def project_policy(manifest: ProjectManifest, approval: Any = None) -> tuple[AllowReadOnlyPolicy, Any]:
    return AllowReadOnlyPolicy(), approval if manifest.approval_mode == "interactive" and approval is not None else DenyApproval()


@dataclass(frozen=True)
class ProjectEvaluationFactory:
    manifest: ProjectManifest
    manifest_path: str

    def __call__(self):
        return ProjectEvaluationRunner(_load_project_model_provider(self.manifest, self.manifest_path))


class ProjectEvaluationRunner:
    def __init__(self, provider):
        self.provider = provider

    def __call__(self, suite_path, output_path):
        from nanoforge.evaluation.suite import evaluate_task_suite, load_task_suite, write_evaluation_report
        report = evaluate_task_suite(load_task_suite(suite_path), self.provider.engine,
            model_metadata=asdict(self.provider.identity))
        write_evaluation_report(report, output_path)
        return report.to_dict()


def evaluate_project_model(manifest: ProjectManifest, manifest_path: str | Path, *, cancellation=None):
    """Run the configured held-out suite in a bounded model worker and save local results."""
    from nanoforge.agent.isolation import IsolatedWorker
    from nanoforge.agent.protocol import CancellationToken
    if manifest.provider_kind == "manual" or not manifest.eval_suite:
        raise ValueError("project evaluation requires a selected model and eval_suite")
    workspace, _, budget = build_project_runtime(manifest, manifest_path)
    suite_path = workspace._allowed_file(manifest.eval_suite)
    report_path = workspace.resolve(f".latticekin/evaluations/{uuid.uuid4().hex}/report.json")
    token = cancellation if cancellation is not None else CancellationToken()
    factory = ProjectEvaluationFactory(manifest, str(Path(manifest_path).resolve()))
    worker = IsolatedWorker(factory, factory=True, deadline=budget.deadline,
                            cancellation=token, max_memory_bytes=budget.max_memory_bytes)
    try:
        report = worker.call(str(suite_path), str(report_path), deadline=budget.deadline,
                             max_message_bytes=8 * 1024 * 1024)
        return {"report": report, "report_path": report_path.relative_to(workspace.root).as_posix()}
    finally:
        worker.close()

