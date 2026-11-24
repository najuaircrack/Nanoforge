from __future__ import annotations

import asyncio
import hashlib
import json
import os
import threading
import time
from types import SimpleNamespace
from pathlib import Path

import pytest
import yaml

from nanoforge.agent.project import build_project_runtime, load_project_manifest
from nanoforge.agent.protocol import CancellationToken
from nanoforge.agent.repository import FilePatch, RepositoryWorkspace


def test_patch_does_not_follow_predictable_staging_hardlink(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    target = root / "main.py"
    target.write_bytes(b"old")
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"unrelated")
    os.link(outside, root / ".main.py.nanoforge-patch.tmp")
    result = RepositoryWorkspace(root).apply_patch(
        FilePatch("main.py", hashlib.sha256(b"old").hexdigest(), "new")
    )
    assert result["sha256"] == hashlib.sha256(b"new").hexdigest()
    assert target.read_bytes() == b"new"
    assert outside.read_bytes() == b"unrelated"


def test_read_hash_streams_instead_of_materializing_entire_file(tmp_path, monkeypatch):
    payload = b"abcdef" * 10000
    (tmp_path / "file.txt").write_bytes(payload)

    def forbidden(*args, **kwargs):
        raise AssertionError("read_bytes materializes the complete file")

    monkeypatch.setattr(Path, "read_bytes", forbidden)
    result = RepositoryWorkspace(tmp_path).read_file("file.txt", 8)
    assert result["text"] == payload[:8].decode()
    assert result["sha256"] == hashlib.sha256(payload).hexdigest()
    assert result["truncated"] is True


def test_repository_filters_ignored_and_sensitive_files(tmp_path):
    (tmp_path / ".gitignore").write_text("ignored/\n*.log\n!keep.log\n", encoding="utf-8")
    (tmp_path / "ignored").mkdir()
    (tmp_path / "ignored" / "file.txt").write_text("fixture marker")
    (tmp_path / ".env").write_text("FIXTURE_TOKEN=fixture marker")
    (tmp_path / "notes.log").write_text("fixture marker")
    (tmp_path / "keep.log").write_text("fixture marker")
    (tmp_path / "code.py").write_text("fixture marker")
    workspace = RepositoryWorkspace(tmp_path)
    assert workspace.list_files()["files"] == [".gitignore", "code.py", "keep.log"]
    assert {m["path"] for m in workspace.search("fixture marker")["matches"]} == {
        "code.py", "keep.log"
    }
    with pytest.raises(PermissionError):
        workspace.read_file(".env")
    with pytest.raises(PermissionError):
        workspace.read_file("ignored/file.txt")


def test_sensitive_read_requires_explicit_workspace_opt_in(tmp_path):
    (tmp_path / ".env").write_text("fixture-only")
    result = RepositoryWorkspace(tmp_path, allow_sensitive=True).read_file(".env")
    assert result["text"] == "fixture-only"


def test_listing_scan_limit_is_observable(tmp_path):
    for index in range(10):
        (tmp_path / f"{index}.txt").write_text("fixture")
    workspace = RepositoryWorkspace(tmp_path, max_scan_entries=4)
    listing = workspace.list_files(max_entries=2)
    assert len(listing["files"]) == 2
    assert listing["truncated"]
    assert listing["scanned_entries"] <= 4


def test_precancelled_test_never_starts(tmp_path):
    (tmp_path / "test_side_effect.py").write_text(
        "from pathlib import Path\ndef test_effect():\n    Path('executed.txt').write_text('ran')\n"
    )
    token = CancellationToken()
    token.cancel()
    result = RepositoryWorkspace(tmp_path).run_tests(cancellation=token)
    assert result["status"] == "cancelled"
    assert not (tmp_path / "executed.txt").exists()


def test_tests_cancel_during_execution_and_bound_utf8_output(tmp_path):
    (tmp_path / "test_slow.py").write_text(
        "import time\nfrom pathlib import Path\ndef test_slow():\n"
        "    print('\\u00e9' * 10000, flush=True)\n"
        "    Path('started.txt').write_text('started')\n"
        "    time.sleep(10)\n    Path('completed.txt').write_text('finished')\n"
    )
    token = CancellationToken()
    result = {}
    workspace = RepositoryWorkspace(tmp_path)
    thread = threading.Thread(
        target=lambda: result.update(workspace.run_tests(
            cancellation=token, timeout_seconds=15, max_output_bytes=128
        ))
    )
    started = time.monotonic()
    thread.start()
    while not (tmp_path / "started.txt").exists() and time.monotonic() - started < 8:
        time.sleep(0.02)
    token.cancel()
    thread.join(timeout=3)
    assert not thread.is_alive()
    assert result["status"] == "cancelled"
    assert len(result["output"].encode("utf-8")) <= 128
    assert not (tmp_path / "completed.txt").exists()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, True])
def test_project_rejects_nonfinite_and_invalid_wall_budget(tmp_path, value):
    path = tmp_path / "project.yaml"
    path.write_text(yaml.safe_dump({"budgets": {"max_wall_seconds": value}}))
    with pytest.raises(ValueError):
        load_project_manifest(path)


def test_project_resource_policy_and_identity(tmp_path):
    path = tmp_path / "project.yaml"
    path.write_text(yaml.safe_dump({
        "schema_version": 2, "provider_kind": "manual", "approval_mode": "interactive",
        "resources": {"max_file_bytes": 1024, "max_scan_entries": 20, "context_paths": []},
        "capabilities": ["workspace.read", "workspace.write"],
    }))
    manifest = load_project_manifest(path)
    workspace, _, _ = build_project_runtime(manifest, path)
    assert workspace.max_file_bytes == 1024
    assert workspace.max_scan_entries == 20
    assert manifest.fingerprint == load_project_manifest(path).fingerprint


def test_tui_harness_runs_real_button_and_shows_observation(tmp_path):
    pytest.importorskip("textual")
    from nanoforge.tui import NanoforgeApp

    project = tmp_path / "project.yaml"
    project.write_text("workspace_root: .\n")
    (tmp_path / "note.txt").write_text("visible synthetic observation")

    async def exercise():
        app = NanoforgeApp(run_dir=str(tmp_path / "empty-run"), project=str(project))
        async with app.run_test(size=(160, 55)) as pilot:
            app.action_switch_screen("harness")
            await pilot.pause()
            app.screen.query_one("#harness-actions").text = json.dumps([
                {"schema_version": 1, "type": "tool_call", "name": "repo.read_file", "arguments": {"path": "note.txt"}},
                {"schema_version": 1, "type": "final", "content": "complete"},
            ])
            await pilot.click("#harness-run-btn")
            for _ in range(100):
                await pilot.pause(0.05)
                if app.screen._last_result is not None:
                    break
            assert app.screen._last_result.status == "completed"
            assert app.screen._last_result.observations[0]["output"]["text"] == "visible synthetic observation"
            assert "visible synthetic observation" in app.screen.query_one("#harness-observations").text
            assert app.screen._running is True  # Textual lifecycle flag remains untouched.

    asyncio.run(exercise())


def test_tui_launch_button_updates_on_ui_thread(tmp_path, monkeypatch):
    pytest.importorskip("textual")
    from nanoforge.tui import NanoforgeApp

    class NoProcessThread:
        def __init__(self, *args, **kwargs):
            pass
        def start(self):
            pass

    async def exercise():
        app = NanoforgeApp(run_dir=str(tmp_path / "empty-run"))
        async with app.run_test(size=(140, 40)) as pilot:
            app.action_switch_screen("launch")
            await pilot.pause()
            monkeypatch.setattr("nanoforge.tui.threading", SimpleNamespace(Thread=NoProcessThread))
            app.screen._launch("train")
            assert "Starting train" in str(app.screen.query_one("#launch-status").render())

    asyncio.run(exercise())


@pytest.mark.parametrize("kind", ["native", "imported"])
def test_real_local_model_factory_loads_only_in_isolated_worker(tmp_path, kind):
    torch = pytest.importorskip("torch")
    from nanoforge.agent.harness import AgentHarness
    from nanoforge.agent.project import build_project_provider
    from nanoforge.config import ModelConfig, NanoforgeConfig
    from nanoforge.model.transformer import build_model
    from nanoforge.training.checkpoint import save_checkpoint

    torch.manual_seed(17)
    config = NanoforgeConfig(model=ModelConfig(
        vocab_size=260, max_seq_len=2048, d_model=16, n_layers=1, n_heads=2, n_kv_heads=1
    ))
    model = build_model(config)
    source = "model.pt" if kind == "native" else "model.gguf"
    if kind == "native":
        save_checkpoint(tmp_path / source, model, None, config, 0)
    else:
        from nanoforge.export.gguf import export_gguf
        export_gguf(model, config, tmp_path / source)
    path = tmp_path / "project.yaml"
    path.write_text(yaml.safe_dump({
        "schema_version": 2, "provider_kind": kind,
        "model": {"source": source, "device": "cpu", "chat_format": "native", "max_new_tokens": 2},
        "budgets": {"max_wall_seconds": 30, "max_tokens": 16384},
    }))
    manifest = load_project_manifest(path)
    _, registry, budget = build_project_runtime(manifest, path)
    provider = build_project_provider(manifest, path)
    assert provider._loaded is None
    result = AgentHarness(registry).run("inspect", provider, budget=budget)
    # A random smoke model is explicitly rejected, not advertised as a capable agent.
    assert result.status == "blocked", result.error
    assert result.provider.kind == kind
    assert result.provider.model_fingerprint
    assert any(event["phase"] == "provider" and event["status"] == "completed" for event in result.trace)
    assert provider._loaded is None


@pytest.mark.parametrize("approve", [True, False])
def test_tui_interactive_approval_and_saved_replay_have_real_effects(tmp_path, approve):
    pytest.importorskip("textual")
    from nanoforge.tui import NanoforgeApp, ToolApprovalModal

    project = tmp_path / "project.yaml"
    project.write_text(yaml.safe_dump({
        "schema_version": 2, "approval_mode": "interactive",
        "enabled_tools": ["repo.apply_patch"], "capabilities": ["workspace.write"],
        "budgets": {"max_wall_seconds": 30, "max_tokens": 16384},
    }))

    async def exercise():
        app = NanoforgeApp(run_dir=str(tmp_path / "empty-run"), project=str(project))
        async with app.run_test(size=(160, 55)) as pilot:
            app.action_switch_screen("harness")
            await pilot.pause()
            screen = app.screen
            screen.query_one("#harness-actions").text = json.dumps([
                {"schema_version": 1, "type": "tool_call", "name": "repo.apply_patch", "arguments": {
                    "path": "new.txt", "expected_sha256": None, "replacement_text": "approved fixture"}},
                {"schema_version": 1, "type": "final", "content": "complete"},
            ])
            await pilot.click("#harness-run-btn")
            for _ in range(100):
                await pilot.pause(0.05)
                if isinstance(app.screen, ToolApprovalModal):
                    break
            assert isinstance(app.screen, ToolApprovalModal)
            assert not (tmp_path / "new.txt").exists()
            await pilot.click("#harness-approve" if approve else "#harness-deny")
            for _ in range(100):
                await pilot.pause(0.05)
                if screen._last_result is not None:
                    break
            assert screen._last_result is not None, (str(screen.query_one("#harness-status").render()), screen._harness_job_running)
            assert screen._last_result.status == "completed"
            assert screen._last_result.observations[0]["status"] == ("ok" if approve else "denied")
            assert (tmp_path / "new.txt").exists() == approve
            trace_relative = screen.query_one("#harness-trace-path").value
            saved = tmp_path / trace_relative
            assert saved.is_file()
            assert "approved fixture" not in saved.read_text(encoding="utf-8")
            before = (tmp_path / "new.txt").read_bytes() if approve else None
            await pilot.click("#harness-replay-btn")
            replay = json.loads(screen.query_one("#harness-result").text)
            assert replay["side_effects_applied"] is False
            assert ((tmp_path / "new.txt").read_bytes() if approve else None) == before

    asyncio.run(exercise())


def test_tui_cancel_while_reviewing_action_prevents_patch(tmp_path):
    pytest.importorskip("textual")
    from nanoforge.tui import NanoforgeApp, ToolApprovalModal

    project = tmp_path / "project.yaml"
    project.write_text(yaml.safe_dump({"approval_mode": "interactive",
        "enabled_tools": ["repo.apply_patch"], "capabilities": ["workspace.write"]}))

    async def exercise():
        app = NanoforgeApp(run_dir=str(tmp_path / "empty-run"), project=str(project))
        async with app.run_test(size=(160, 55)) as pilot:
            app.action_switch_screen("harness")
            await pilot.pause()
            screen = app.screen
            screen.query_one("#harness-actions").text = json.dumps([
                {"schema_version": 1, "type": "tool_call", "name": "repo.apply_patch", "arguments": {
                    "path": "cancelled.txt", "expected_sha256": None, "replacement_text": "never"}},
                {"schema_version": 1, "type": "final", "content": "done"},
            ])
            await pilot.click("#harness-run-btn")
            for _ in range(100):
                await pilot.pause(0.05)
                if isinstance(app.screen, ToolApprovalModal):
                    break
            await pilot.click("#harness-cancel-approval")
            for _ in range(100):
                await pilot.pause(0.05)
                if screen._last_result is not None:
                    break
            assert screen._last_result is not None, (str(screen.query_one("#harness-status").render()), screen._harness_job_running)
            assert screen._last_result.status == "cancelled"
            assert not (tmp_path / "cancelled.txt").exists()

    asyncio.run(exercise())


def test_project_context_is_explicit_bounded_and_provenanced(tmp_path):
    from nanoforge.agent.project import project_context_items
    (tmp_path / "note.txt").write_text("context fixture")
    project = tmp_path / "project.yaml"
    project.write_text(yaml.safe_dump({"resources": {"context_paths": ["note.txt"]}}))
    manifest = load_project_manifest(project)
    workspace, _, _ = build_project_runtime(manifest, project)
    items = project_context_items(manifest, workspace)
    assert items[0]["id"] == "note.txt"
    assert items[0]["text"] == "context fixture"
    assert items[0]["scope"] == "project"
    assert items[0]["provenance"]["sha256"] == hashlib.sha256(b"context fixture").hexdigest()


def test_project_rejects_unused_model_configuration(tmp_path):
    path = tmp_path / "project.yaml"
    path.write_text("provider_kind: manual\nmodel: {source: ignored.pt}\n")
    with pytest.raises(ValueError, match="unused"):
        load_project_manifest(path)
