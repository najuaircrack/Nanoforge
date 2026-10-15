"""Bounded repository tools for coding-agent workflows."""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nanoforge.agent.protocol import CancellationToken, ToolSpec


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class FilePatch:
    path: str
    expected_sha256: str | None
    replacement_text: str


class RepositoryWorkspace:
    def __init__(self, root: str | Path, *, max_file_bytes: int = 256 * 1024, max_patch_bytes: int = 512 * 1024):
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise NotADirectoryError(self.root)
        self.max_file_bytes = max_file_bytes
        self.max_patch_bytes = max_patch_bytes

    def resolve(self, relative_path: str) -> Path:
        candidate = Path(relative_path)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError("repository paths must be relative and cannot contain '..'")
        resolved = (self.root / candidate).resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("repository path escapes workspace") from exc
        return resolved

    def list_files(self, pattern: str = "**/*", max_entries: int = 200) -> dict[str, Any]:
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        files = [path for path in self.root.glob(pattern) if path.is_file()]
        names = sorted(path.relative_to(self.root).as_posix() for path in files)
        return {"files": names[:max_entries], "truncated": len(names) > max_entries}

    def read_file(self, path: str, max_bytes: int | None = None) -> dict[str, Any]:
        resolved = self.resolve(path)
        limit = max_bytes or self.max_file_bytes
        if limit <= 0 or limit > self.max_file_bytes:
            raise ValueError(f"max_bytes must be between 1 and {self.max_file_bytes}")
        if not resolved.is_file():
            raise FileNotFoundError(path)
        raw = resolved.read_bytes()
        truncated = len(raw) > limit
        text = raw[:limit].decode("utf-8", errors="replace")
        return {"path": Path(path).as_posix(), "text": text, "truncated": truncated, "sha256": _sha256_bytes(raw)}

    def search(self, pattern: str, include: str = "**/*", max_matches: int = 100) -> dict[str, Any]:
        if max_matches <= 0:
            raise ValueError("max_matches must be positive")
        regex = re.compile(pattern)
        matches: list[dict[str, Any]] = []
        for path in sorted(self.root.glob(include)):
            if not path.is_file() or path.stat().st_size > self.max_file_bytes:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for line_number, line in enumerate(text.splitlines(), 1):
                if regex.search(line):
                    matches.append({"path": path.relative_to(self.root).as_posix(), "line": line_number, "text": line[:1000]})
                    if len(matches) >= max_matches:
                        return {"matches": matches, "truncated": True}
        return {"matches": matches, "truncated": False}

    def apply_patch(self, patch: FilePatch) -> dict[str, Any]:
        if len(patch.replacement_text.encode("utf-8")) > self.max_patch_bytes:
            raise ValueError("replacement exceeds patch byte limit")
        target = self.resolve(patch.path)
        exists = target.exists()
        current_bytes = target.read_bytes() if exists else None
        if exists and patch.expected_sha256 != _sha256_bytes(current_bytes or b""):
            raise ValueError("patch expected_sha256 does not match current file")
        if not exists and patch.expected_sha256 is not None:
            raise ValueError("new files require expected_sha256=null")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.nanoforge-patch.tmp")
        temporary.write_bytes(patch.replacement_text.encode("utf-8"))
        os.replace(temporary, target)
        return {"path": Path(patch.path).as_posix(), "sha256": _sha256_text(patch.replacement_text), "created": not exists}

    def run_tests(self, profile: str = "pytest", timeout_seconds: float = 60.0, max_output_bytes: int = 64 * 1024, cancellation: CancellationToken | None = None) -> dict[str, Any]:
        profiles = {"pytest": [sys.executable, "-m", "pytest", "-q"]}
        if profile not in profiles:
            raise ValueError(f"unknown test profile: {profile}")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        process = subprocess.Popen(
            profiles[profile],
            cwd=self.root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env={
                key: value
                for key, value in os.environ.items()
                if not any(marker in key.upper() for marker in ("TOKEN", "SECRET", "PASSWORD", "API_KEY"))
            },
        )
        try:
            output, _ = process.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            process.kill()
            output, _ = process.communicate()
            return {"status": "timeout", "returncode": None, "output": output[:max_output_bytes], "truncated": len(output) > max_output_bytes}
        if cancellation and cancellation.cancelled:
            process.kill()
            return {"status": "cancelled", "returncode": None, "output": output[:max_output_bytes], "truncated": len(output) > max_output_bytes}
        return {
            "status": "passed" if process.returncode == 0 else "failed",
            "returncode": process.returncode,
            "output": output[:max_output_bytes],
            "truncated": len(output) > max_output_bytes,
        }


def register_repository_tools(workspace: RepositoryWorkspace) -> list[ToolSpec]:
    return [
        ToolSpec(
            name="repo.list_files",
            description="List repository-relative files.",
            input_schema={"type": "object", "properties": {"pattern": {"type": "string"}, "max_entries": {"type": "integer"}}, "additionalProperties": False},
            fn=workspace.list_files,
            capabilities=frozenset({"workspace.read"}),
        ),
        ToolSpec(
            name="repo.read_file",
            description="Read a bounded UTF-8 repository file.",
            input_schema={"type": "object", "properties": {"path": {"type": "string"}, "max_bytes": {"type": "integer"}}, "required": ["path"], "additionalProperties": False},
            fn=workspace.read_file,
            capabilities=frozenset({"workspace.read"}),
        ),
        ToolSpec(
            name="repo.search",
            description="Search repository text with a regular expression.",
            input_schema={"type": "object", "properties": {"pattern": {"type": "string"}, "include": {"type": "string"}, "max_matches": {"type": "integer"}}, "required": ["pattern"], "additionalProperties": False},
            fn=workspace.search,
            capabilities=frozenset({"workspace.read"}),
        ),
        ToolSpec(
            name="repo.apply_patch",
            description="Atomically replace one repository-relative file after an expected hash check.",
            input_schema={"type": "object", "properties": {"path": {"type": "string"}, "expected_sha256": {"type": ["string", "null"]}, "replacement_text": {"type": "string"}}, "required": ["path", "expected_sha256", "replacement_text"], "additionalProperties": False},
            fn=lambda **kwargs: workspace.apply_patch(FilePatch(**kwargs)),
            capabilities=frozenset({"workspace.write"}),
            approval_policy="always",
            side_effect="write",
        ),
        ToolSpec(
            name="repo.run_tests",
            description="Run the fixed repository pytest profile with a timeout.",
            input_schema={"type": "object", "properties": {"profile": {"type": "string"}, "timeout_seconds": {"type": "number"}}, "additionalProperties": False},
            fn=lambda profile="pytest", timeout_seconds=60.0: workspace.run_tests(profile, timeout_seconds),
            capabilities=frozenset({"process.test"}),
            approval_policy="always",
            side_effect="test",
        ),
    ]
