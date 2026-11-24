"""Bounded, ignore-aware repository tools for coding-agent workflows."""
from __future__ import annotations

import fnmatch
import hashlib
import math
import os
import re
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from nanoforge.agent.protocol import CancellationToken, ToolSpec

_SENSITIVE = {".git", ".aws", ".codex", ".agents", ".ssh", ".venv", "venv",
              ".latticekin", "node_modules", "data", "models", "runs", "reports", "checkpoints", "__pycache__"}


def _decode_bounded(data: bytes, limit: int) -> str:
    return data.decode("utf-8", errors="replace").encode("utf-8")[:limit].decode("utf-8", errors="ignore")


@dataclass(frozen=True)
class FilePatch:
    path: str
    expected_sha256: str | None
    replacement_text: str


class RepositoryWorkspace:
    def __init__(self, root: str | Path, *, max_file_bytes: int = 256 * 1024,
                 max_patch_bytes: int = 512 * 1024, max_scan_entries: int = 10000,
                 max_input_file_bytes: int = 32 * 1024 * 1024, allow_sensitive: bool = False):
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise NotADirectoryError(self.root)
        for name, value in {"max_file_bytes": max_file_bytes, "max_patch_bytes": max_patch_bytes,
                            "max_scan_entries": max_scan_entries, "max_input_file_bytes": max_input_file_bytes}.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
            setattr(self, name, value)
        self.allow_sensitive = allow_sensitive

    def _confined_resolved(self, path: Path) -> Path:
        resolved = path.resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("workspace path or symlink escapes workspace") from exc
        return resolved

    def resolve(self, relative_path: str) -> Path:
        candidate = Path(relative_path)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError("repository paths must be relative and cannot contain '..'")
        return self._confined_resolved(self.root / candidate)

    @staticmethod
    def _matches(relative: str, pattern: str) -> bool:
        return (Path(relative).match(pattern) or fnmatch.fnmatchcase(relative, pattern)
                or pattern.startswith("**/") and Path(relative).match(pattern[3:]))

    @staticmethod
    def _ignore_path_matches(relative: str, pattern: str) -> bool:
        pieces = []
        index = 0
        while index < len(pattern):
            char = pattern[index]
            if pattern[index:index + 3] == "**/":
                pieces.append("(?:.*/)?")
                index += 3
                continue
            if pattern[index:index + 2] == "**":
                pieces.append(".*")
                index += 2
                continue
            if char == "*":
                pieces.append("[^/]*")
            elif char == "?":
                pieces.append("[^/]")
            elif char == "[" and "]" in pattern[index + 1:]:
                end = pattern.index("]", index + 1)
                group = pattern[index + 1:end]
                pieces.append("[" + ("^" + group[1:] if group.startswith("!") else group) + "]")
                index = end
            else:
                pieces.append(re.escape(char))
            index += 1
        return re.fullmatch("".join(pieces), relative) is not None

    def _excluded(self, relative: str, *, is_dir: bool = False) -> bool:
        if self.allow_sensitive:
            return False
        parts = Path(relative).parts
        if any(part.lower() in _SENSITIVE or part.lower().startswith(".env") for part in parts):
            return True
        ignored = False
        parent = self.root
        for depth in range(len(parts)):
            ignore_file = parent / ".gitignore"
            if ignore_file.is_file():
                self._confined_resolved(ignore_file)
                if ignore_file.stat().st_size > 65536:
                    raise ValueError(".gitignore exceeds policy byte limit")
                local = "/".join(parts[depth:])
                for raw in ignore_file.read_text(encoding="utf-8", errors="replace").splitlines():
                    rule = raw.rstrip()
                    if not rule or rule.startswith("#"):
                        continue
                    negate = rule.startswith("!")
                    if negate:
                        rule = rule[1:]
                    if rule.startswith("\\#") or rule.startswith("\\!"):
                        rule = rule[1:]
                    directory = rule.endswith("/")
                    anchored = rule.startswith("/")
                    rule = rule.strip("/")
                    candidates = [local]
                    if directory:
                        candidates = ["/".join(parts[depth:depth+i]) for i in range(1, len(parts)-depth+int(is_dir))]
                    matched = any(self._ignore_path_matches(item, rule) if "/" in rule or anchored
                                  else any(fnmatch.fnmatchcase(part, rule) for part in item.split("/"))
                                  for item in candidates)
                    if rule and matched:
                        ignored = not negate
            parent /= parts[depth]
        return ignored

    def _allowed_file(self, path: str) -> Path:
        resolved = self.resolve(path)
        relative = resolved.relative_to(self.root).as_posix()
        parent_excluded = any(self._excluded("/".join(Path(relative).parts[:index]), is_dir=True)
                              for index in range(1, len(Path(relative).parts)))
        if parent_excluded or self._excluded(Path(path).as_posix()) or self._excluded(relative):
            raise PermissionError("path is excluded by workspace context policy")
        return resolved

    def _walk(self, pattern: str, state: dict[str, Any]) -> Iterator[Path]:
        if Path(pattern).is_absolute() or ".." in Path(pattern).parts:
            raise ValueError("include patterns must remain repository-relative")
        for parent, dirs, files in os.walk(self.root, followlinks=False):
            allowed = []
            for name in sorted(dirs):
                if state["scanned_entries"] >= self.max_scan_entries:
                    state["truncated"] = True
                    return
                state["scanned_entries"] += 1
                child = Path(parent) / name
                relative = child.relative_to(self.root).as_posix()
                if self._excluded(relative, is_dir=True):
                    continue
                self._confined_resolved(child)
                if not child.is_symlink():
                    allowed.append(name)
            dirs[:] = allowed
            for name in sorted(files):
                if state["scanned_entries"] >= self.max_scan_entries:
                    state["truncated"] = True
                    return
                state["scanned_entries"] += 1
                path = Path(parent) / name
                relative = path.relative_to(self.root).as_posix()
                if self._excluded(relative):
                    continue
                resolved = self._confined_resolved(path)
                if self._excluded(resolved.relative_to(self.root).as_posix()):
                    continue
                if path.is_file() and self._matches(relative, pattern):
                    yield path

    def list_files(self, pattern: str = "**/*", max_entries: int = 200) -> dict[str, Any]:
        if isinstance(max_entries, bool) or not isinstance(max_entries, int) or max_entries <= 0:
            raise ValueError("max_entries must be positive")
        state = {"scanned_entries": 0, "truncated": False}
        names = []
        for path in self._walk(pattern, state):
            if len(names) == max_entries:
                state["truncated"] = True
                break
            names.append(path.relative_to(self.root).as_posix())
        return {"files": sorted(names), **state}

    def _hash_file(self, path: Path, display_limit: int = 0) -> tuple[str, bytes, int]:
        if path.stat().st_size > self.max_input_file_bytes:
            raise ValueError("file exceeds maximum input-file byte limit")
        digest, prefix, total = hashlib.sha256(), bytearray(), 0
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(65536), b""):
                total += len(chunk)
                if total > self.max_input_file_bytes:
                    raise ValueError("file grew beyond maximum input-file byte limit")
                digest.update(chunk)
                prefix.extend(chunk[:max(0, display_limit - len(prefix))])
        return digest.hexdigest(), bytes(prefix), total

    def read_file(self, path: str, max_bytes: int | None = None) -> dict[str, Any]:
        resolved = self._allowed_file(path)
        limit = self.max_file_bytes if max_bytes is None else max_bytes
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= self.max_file_bytes:
            raise ValueError(f"max_bytes must be between 1 and {self.max_file_bytes}")
        if not resolved.is_file():
            raise FileNotFoundError(path)
        digest, prefix, total = self._hash_file(resolved, limit)
        return {"path": Path(path).as_posix(), "text": _decode_bounded(prefix, limit),
                "truncated": total > limit, "sha256": digest, "size_bytes": total}

    def search(self, pattern: str, include: str = "**/*", max_matches: int = 100) -> dict[str, Any]:
        if isinstance(max_matches, bool) or not isinstance(max_matches, int) or max_matches <= 0:
            raise ValueError("max_matches must be positive")
        if len(pattern.encode("utf-8")) > 4096:
            raise ValueError("search pattern exceeds 4096 bytes")
        regex, matches, skipped = re.compile(pattern), [], 0
        state = {"scanned_entries": 0, "truncated": False}
        for path in self._walk(include, state):
            if path.stat().st_size > self.max_file_bytes:
                skipped += 1
                continue
            with path.open("rb") as stream:
                raw = stream.read(self.max_file_bytes + 1)
            if len(raw) > self.max_file_bytes:
                skipped += 1
                continue
            for number, line in enumerate(raw.decode("utf-8", errors="replace").splitlines(), 1):
                if regex.search(line):
                    matches.append({"path": path.relative_to(self.root).as_posix(), "line": number,
                                    "text": _decode_bounded(line.encode("utf-8"), 1000)})
                    if len(matches) >= max_matches:
                        state["truncated"] = True
                        return {"matches": matches, "skipped_files": skipped, **state}
        return {"matches": matches, "skipped_files": skipped, **state}

    def apply_patch(self, patch: FilePatch) -> dict[str, Any]:
        payload = patch.replacement_text.encode("utf-8")
        if len(payload) > self.max_patch_bytes:
            raise ValueError("replacement exceeds patch byte limit")
        target = self._allowed_file(patch.path)
        exists = target.exists()
        if exists and not target.is_file():
            raise ValueError("patch target must be a regular file")
        current_hash = self._hash_file(target)[0] if exists else None
        if exists and patch.expected_sha256 != current_hash:
            raise ValueError("patch expected_sha256 does not match current file")
        if not exists and patch.expected_sha256 is not None:
            raise ValueError("new files require expected_sha256=null")
        target.parent.mkdir(parents=True, exist_ok=True)
        self._confined_resolved(target.parent)
        descriptor, name = tempfile.mkstemp(prefix=f".{target.name}.patch-", suffix=".tmp", dir=target.parent)
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            if exists:
                os.chmod(temporary, stat.S_IMODE(target.stat().st_mode))
            self._confined_resolved(target.parent)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        return {"path": Path(patch.path).as_posix(), "sha256": hashlib.sha256(payload).hexdigest(), "created": not exists}

    @staticmethod
    def _stop_process_tree(process: subprocess.Popen) -> None:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5, check=False)
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if process.poll() is None:
            process.kill()

    def run_tests(self, profile: str = "pytest", timeout_seconds: float = 60.0,
                  max_output_bytes: int = 65536, cancellation: CancellationToken | None = None,
                  deadline: float | None = None, context: Any = None) -> dict[str, Any]:
        if context is not None:
            cancellation, deadline = context.cancellation, context.deadline
            max_output_bytes = min(max_output_bytes, context.max_output_bytes)
        profiles = {"pytest": [sys.executable, "-m", "pytest", "-q", "-s"]}
        if profile not in profiles:
            raise ValueError(f"unknown test profile: {profile}")
        if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive")
        if isinstance(max_output_bytes, bool) or not isinstance(max_output_bytes, int) or max_output_bytes < 1:
            raise ValueError("max_output_bytes must be positive")
        if deadline is not None and not math.isfinite(deadline):
            raise ValueError("deadline must be finite")
        stop_at = min(time.monotonic() + timeout_seconds, deadline if deadline is not None else math.inf)
        empty = {"returncode": None, "output": "", "truncated": False}
        if cancellation and cancellation.cancelled:
            return {"status": "cancelled", **empty}
        if time.monotonic() >= stop_at:
            return {"status": "timeout", **empty}
        process = subprocess.Popen(profiles[profile], cwd=self.root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env={key: value for key, value in os.environ.items() if not any(marker in key.upper() for marker in ("TOKEN", "SECRET", "PASSWORD", "API_KEY"))},
            start_new_session=os.name != "nt" and context is None,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0)
        captured, total = bytearray(), [0]
        def drain():
            assert process.stdout is not None
            try:
                while chunk := process.stdout.read(8192):
                    total[0] += len(chunk)
                    captured.extend(chunk[:max(0, max_output_bytes - len(captured))])
            except (OSError, ValueError):
                pass
        reader = threading.Thread(target=drain, daemon=True)
        reader.start()
        status = None
        try:
            while process.poll() is None:
                if cancellation and cancellation.cancelled:
                    status = "cancelled"
                    break
                if time.monotonic() >= stop_at:
                    status = "timeout"
                    break
                time.sleep(0.02)
            if status:
                self._stop_process_tree(process)
            process.wait(timeout=5)
        finally:
            if process.poll() is None:
                self._stop_process_tree(process)
                process.wait(timeout=5)
            reader.join(timeout=2)
            if process.stdout is not None and not reader.is_alive():
                process.stdout.close()
        return {"status": status or ("passed" if process.returncode == 0 else "failed"),
                "returncode": None if status else process.returncode,
                "output": _decode_bounded(bytes(captured), max_output_bytes),
                "truncated": total[0] > max_output_bytes, "output_bytes": total[0]}

    def patch_tool(self, path: str, expected_sha256: str | None, replacement_text: str) -> dict[str, Any]:
        return self.apply_patch(FilePatch(path, expected_sha256, replacement_text))


def register_repository_tools(workspace: RepositoryWorkspace) -> list[ToolSpec]:
    specs = [
        ToolSpec(name="repo.list_files", description="List ignore-aware repository files with a bounded scan.",
            input_schema={"type": "object", "properties": {"pattern": {"type": "string"}, "max_entries": {"type": "integer", "minimum": 1}}, "additionalProperties": False},
            fn=workspace.list_files, capabilities=frozenset({"workspace.read"})),
        ToolSpec(name="repo.read_file", description="Read a bounded UTF-8 repository file and stream its content hash.",
            input_schema={"type": "object", "properties": {"path": {"type": "string"}, "max_bytes": {"type": "integer", "minimum": 1}}, "required": ["path"], "additionalProperties": False},
            fn=workspace.read_file, capabilities=frozenset({"workspace.read"})),
        ToolSpec(name="repo.search", description="Search non-excluded repository text with a regular expression.",
            input_schema={"type": "object", "properties": {"pattern": {"type": "string"}, "include": {"type": "string"}, "max_matches": {"type": "integer", "minimum": 1}}, "required": ["pattern"], "additionalProperties": False},
            fn=workspace.search, capabilities=frozenset({"workspace.read"})),
        ToolSpec(name="repo.apply_patch", description="Atomically replace one repository file after an expected hash check.",
            input_schema={"type": "object", "properties": {"path": {"type": "string"}, "expected_sha256": {"type": ["string", "null"]}, "replacement_text": {"type": "string"}}, "required": ["path", "expected_sha256", "replacement_text"], "additionalProperties": False},
            fn=workspace.patch_tool, capabilities=frozenset({"workspace.write"}), approval_policy="always", side_effect="write"),
    ]
    specs.append(ToolSpec(name="repo.run_tests", description="Run only the fixed pytest profile with cancellation and bounded output.",
        input_schema={"type": "object", "properties": {"profile": {"type": "string", "enum": ["pytest"]}, "timeout_seconds": {"type": "number", "exclusiveMinimum": 0}}, "additionalProperties": False},
        fn=workspace.run_tests, capabilities=frozenset({"process.test"}), approval_policy="always", side_effect="test", inject_context=True))
    return specs

