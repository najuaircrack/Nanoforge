from __future__ import annotations

import json

import pytest

from nanoforge.agent import (
    AgentHarness,
    ManualActionProvider,
    build_project_runtime,
    load_project_manifest,
)


def _write_manifest(tmp_path, **overrides):
    values = {
        "schema_version": 1,
        "name": "test-project",
        "workspace_root": ".",
        "provider_kind": "manual",
        "enabled_tools": ["repo.list_files", "repo.read_file", "repo.apply_patch"],
        "capabilities": ["workspace.read"],
        "approval_mode": "read_only",
        "budgets": {"max_steps": 4, "max_tool_calls": 4, "max_wall_seconds": 20, "max_output_bytes": 4096},
    }
    values.update(overrides)
    path = tmp_path / "project.yaml"
    lines = [f"{key}: {json.dumps(value)}" for key, value in values.items() if key != "budgets"]
    lines.append("budgets:")
    for key, value in values["budgets"].items():
        lines.append(f"  {key}: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_project_manifest_builds_allowlisted_read_only_runtime(tmp_path):
    (tmp_path / "note.txt").write_text("hello\n", encoding="utf-8")
    manifest_path = _write_manifest(tmp_path)
    manifest = load_project_manifest(manifest_path)
    _workspace, registry, budget = build_project_runtime(manifest, manifest_path)

    assert manifest.name == "test-project"
    assert [item["name"] for item in registry.schema()] == ["repo.apply_patch", "repo.list_files", "repo.read_file"]
    assert budget.max_steps == 4
    result = AgentHarness(registry).run(
        "inspect",
        ManualActionProvider([
            {"type": "tool_call", "name": "repo.read_file", "arguments": {"path": "note.txt"}},
            {"type": "final", "content": "done"},
        ]),
        budget=budget,
        executor_kwargs={"capabilities": set(manifest.capabilities)},
    )
    assert result.status == "completed"
    assert result.observations[0]["output"]["text"].replace("\r\n", "\n") == "hello\n"


def test_project_manifest_denies_write_without_capability(tmp_path):
    manifest_path = _write_manifest(tmp_path)
    manifest = load_project_manifest(manifest_path)
    _workspace, registry, budget = build_project_runtime(manifest, manifest_path)
    result = AgentHarness(registry).run(
        "patch",
        ManualActionProvider([
            {
                "type": "tool_call",
                "name": "repo.apply_patch",
                "arguments": {"path": "new.txt", "expected_sha256": None, "replacement_text": "nope"},
            },
            {"type": "final", "content": "done"},
        ]),
        budget=budget,
        executor_kwargs={"capabilities": set(manifest.capabilities)},
    )
    assert result.status == "completed"
    assert result.observations[0]["status"] == "denied"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", 3),
        ("provider_kind", "checkpoint"),
        ("unknown", "value"),
        ("workspace_root", "../outside"),
    ],
)
def test_project_manifest_rejects_unsupported_values(tmp_path, field, value):
    manifest_path = _write_manifest(tmp_path, **{field: value})
    with pytest.raises((ValueError, NotADirectoryError)):
        load_project_manifest(manifest_path)


def test_project_manifest_rejects_absolute_workspace(tmp_path):
    manifest_path = _write_manifest(tmp_path, workspace_root=str(tmp_path))
    with pytest.raises(ValueError, match="relative"):
        load_project_manifest(manifest_path)
