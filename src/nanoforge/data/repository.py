"""Bounded, repository-relative source snapshots and dependency-aware training input."""
from __future__ import annotations

import ast
import fnmatch
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from nanoforge.data.formats import DatasetRecord, bounded_source_files
from nanoforge.data.preprocess import CODE_EXTENSIONS

_DEFAULT_EXCLUDES = (
    ".git", ".venv", "venv", "node_modules", "__pycache__", "target", "dist", "build",
    "data", "models", "runs", "reports", ".aws", ".ssh", ".codex", ".agents",
)
_SECRET_GLOBS = (".env*", "*.pem", "*.key", "*.p12", "*.pfx", "*credentials*", "*secrets*")


@dataclass
class RepositoryDataStats:
    files_seen: int = 0
    files_kept: int = 0
    excluded_files: int = 0
    oversized_files: int = 0
    bytes_read: int = 0
    unresolved_imports: int = 0


def _dependencies(path: str, text: str, available: set[str]) -> tuple[list[str], int]:
    candidates: set[str] = set()
    missing = 0
    suffix = Path(path).suffix.lower()
    if suffix == ".py":
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return [], 0
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = (node.module or "").replace(".", "/")
                if node.level:
                    parent = Path(path).parent
                    for _ in range(node.level - 1):
                        parent = parent.parent
                    base = (parent / base).as_posix()
                modules = [base] + [base + "/" + alias.name for alias in node.names]
            for module in modules:
                base = module.replace(".", "/")
                choices = {base + ".py", base + "/__init__.py"}
                found = choices & available
                if found:
                    candidates.update(found)
                elif module and module.split("/")[0] in {Path(item).parts[0] for item in available}:
                    missing += 1
    else:
        imports = re.findall(r"(?:from\s*|import\s*|require\s*\(|#include\s*)[\"']([^\"']+)[\"']", text)
        for module in imports:
            if not module.startswith(".") and suffix not in {".c", ".cpp", ".h", ".hpp"}:
                continue
            base = (Path(path).parent / module).as_posix()
            # Resolve dot segments without permitting an escape outside the snapshot.
            pieces = []
            escaped = False
            for piece in base.split("/"):
                if piece == "..":
                    if pieces:
                        pieces.pop()
                    else:
                        escaped = True
                elif piece not in {"", "."}:
                    pieces.append(piece)
            if escaped:
                continue
            base = "/".join(pieces)
            choices = {base} | {base + extension for extension in (".js", ".jsx", ".ts", ".tsx", ".rs", ".py", ".h", ".hpp")}
            choices |= {base + "/index" + extension for extension in (".js", ".ts", ".tsx")}
            found = choices & available
            candidates.update(found)
            missing += int(not found)
    candidates.discard(path)
    return sorted(candidates), missing


def iter_repository_records(root: str | Path, *, max_file_bytes: int = 1_048_576,
                            max_total_bytes: int = 67_108_864, max_files: int = 10_000,
                            excludes: tuple[str, ...] = _DEFAULT_EXCLUDES,
                            stats: RepositoryDataStats | None = None) -> Iterator[DatasetRecord]:
    root = Path(root).resolve()
    if not root.is_dir() or min(max_file_bytes, max_total_bytes, max_files) < 1:
        raise ValueError("Repository root and positive snapshot limits are required")
    stats = stats if stats is not None else RepositoryDataStats()
    # Metadata and dependency grouping are bounded by max_files; source text is
    # reread one file at a time rather than retaining the complete corpus.
    sources: dict[str, tuple[Path, str, list[str], str]] = {}
    license_name = ""
    for license_file in sorted(root.glob("LICENSE*")):
        if license_file.is_file() and not license_file.is_symlink() and license_file.stat().st_size <= 65536:
            body = license_file.read_text(encoding="utf-8", errors="replace")
            if "MIT License" in body or "Permission is hereby granted, free of charge" in body:
                license_name = "MIT"
                break
    for file in bounded_source_files(root, max_files=max_files, excluded_directories=excludes):
        if not file.is_file():
            continue
        stats.files_seen += 1
        relative = file.relative_to(root).as_posix()
        parts = file.relative_to(root).parts
        if file.is_symlink() or any(part in excludes for part in parts) or any(
            fnmatch.fnmatch(file.name.casefold(), pattern) for pattern in _SECRET_GLOBS
        ) or file.suffix.lower() not in CODE_EXTENSIONS:
            stats.excluded_files += 1
            continue
        if len(sources) >= max_files:
            raise ValueError("Repository snapshot file-count limit exceeded")
        size = file.stat().st_size
        if size > max_file_bytes:
            stats.oversized_files += 1
            continue
        if stats.bytes_read + size > max_total_bytes:
            raise ValueError("Repository snapshot byte limit exceeded")
        data = file.read_bytes()
        stats.bytes_read += len(data)
        digest = hashlib.sha256(data).hexdigest()
        text = data.decode("utf-8", errors="replace")
        spdx = re.search(r"SPDX-License-Identifier:\s*([^\s*]+)", text[:4096])
        sources[relative] = (file, digest, [], spdx.group(1) if spdx else license_name)
    available = set(sources)
    parents = {name: name for name in sources}

    def find(name):
        while parents[name] != name:
            parents[name] = parents[parents[name]]
            name = parents[name]
        return name

    for name, (file, digest, _, license_id) in sources.items():
        dependencies, missing = _dependencies(name, file.read_text(encoding="utf-8", errors="replace"), available)
        stats.unresolved_imports += missing
        sources[name] = (file, digest, dependencies, license_id)
        for dependency in dependencies:
            first, second = find(name), find(dependency)
            parents[max(first, second)] = min(first, second)
    snapshot = hashlib.sha256()
    for name, (_, digest, _, _) in sources.items():
        snapshot.update(name.encode())
        snapshot.update(bytes.fromhex(digest))
    snapshot_id = snapshot.hexdigest()
    for name, (file, digest, dependencies, license_id) in sources.items():
        data = file.read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f"Repository changed during snapshot: {name}")
        stats.files_kept += 1
        yield DatasetRecord(data.decode("utf-8", errors="replace"), name, {
            "format": "code", "mode": "code", "suffix": file.suffix,
            "license": license_id, "dependencies": dependencies,
            "group_id": f"{snapshot_id}:{find(name)}", "snapshot_fingerprint": snapshot_id,
            "content_fingerprint": digest,
        })


def prepare_repository_dataset(root, out_dir, tokenizer, **kwargs):
    from nanoforge.data.packing import build_packed_dataset_streaming

    snapshot_keys = {"max_file_bytes", "max_total_bytes", "max_files", "excludes"}
    snapshot_options = {key: kwargs.pop(key) for key in list(kwargs) if key in snapshot_keys}
    stats = RepositoryDataStats()
    records = iter_repository_records(root, stats=stats, **snapshot_options)
    result = build_packed_dataset_streaming([], out_dir, tokenizer, records=records, **kwargs)
    return result, stats

