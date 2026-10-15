from __future__ import annotations

import hashlib

from nanoforge.agent import (
    RepositoryWorkspace,
    StaticApproval,
    ToolExecutor,
    ToolRegistry,
    register_repository_tools,
)


def _executor(root, *, approve=False):
    workspace = RepositoryWorkspace(root)
    registry = ToolRegistry(register_repository_tools(workspace))
    return ToolExecutor(
        registry,
        approval=StaticApproval({"repo.apply_patch", "repo.run_tests"}) if approve else None,
        capabilities={"workspace.read", "workspace.write", "process.test"} if approve else {"workspace.read"},
    )


def test_repository_paths_reads_searches_and_rejects_escape(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("value = 1\n", encoding="utf-8")
    executor = _executor(tmp_path)
    listing = executor.execute({"name": "repo.list_files", "arguments": {"pattern": "**/*.py"}})
    assert listing.status == "ok"
    assert listing.output["files"] == ["src/main.py"]
    read = executor.execute({"name": "repo.read_file", "arguments": {"path": "src/main.py"}})
    assert read.output["text"] == (tmp_path / "src" / "main.py").read_bytes().decode("utf-8")
    search = executor.execute({"name": "repo.search", "arguments": {"pattern": "value"}})
    assert search.output["matches"][0]["path"] == "src/main.py"
    escaped = executor.execute({"name": "repo.read_file", "arguments": {"path": "../outside.txt"}})
    assert escaped.status == "error"


def test_patch_requires_expected_hash_and_approval(tmp_path):
    target = tmp_path / "main.py"
    old = "value = 1\n"
    target.write_text(old, encoding="utf-8")
    expected_hash = hashlib.sha256(target.read_bytes()).hexdigest()
    denied = _executor(tmp_path).execute({"name": "repo.apply_patch", "arguments": {"path": "main.py", "expected_sha256": expected_hash, "replacement_text": "value = 2\n"}})
    assert denied.status == "denied"
    executor = _executor(tmp_path, approve=True)
    stale = executor.execute({"name": "repo.apply_patch", "arguments": {"path": "main.py", "expected_sha256": "stale", "replacement_text": "value = 2\n"}})
    assert stale.status == "error"
    updated = executor.execute({"name": "repo.apply_patch", "arguments": {"path": "main.py", "expected_sha256": expected_hash, "replacement_text": "value = 2\n"}})
    assert updated.status == "ok"
    assert target.read_text(encoding="utf-8") == "value = 2\n"


def test_fixed_test_profile_is_allowlisted(tmp_path):
    (tmp_path / "test_ok.py").write_text("def test_ok():\n    assert 2 + 2 == 4\n", encoding="utf-8")
    executor = _executor(tmp_path, approve=True)
    result = executor.execute({"name": "repo.run_tests", "arguments": {"profile": "pytest", "timeout_seconds": 20}})
    assert result.status == "ok"
    assert result.output["status"] == "passed"
    unknown = executor.execute({"name": "repo.run_tests", "arguments": {"profile": "shell"}})
    assert unknown.status == "error"
