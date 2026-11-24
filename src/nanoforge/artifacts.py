"""Versioned provenance manifests for Nanoforge artifacts and runs.

Manifests are additive sidecars. They do not replace checkpoint payload schemas,
checkpoint hashes, or existing dataset/tokenizer file formats. The module keeps
paths portable, fingerprints content, and writes JSON atomically so interrupted
runs cannot leave a half-written provenance record.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from nanoforge.progress import json_safe

MANIFEST_SCHEMA_VERSION = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def canonical_json(value: Any) -> str:
    """Serialize JSON-compatible data deterministically for fingerprints."""

    if is_dataclass(value):
        value = asdict(value)
    return json.dumps(json_safe(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def content_fingerprint(value: Any) -> str:
    digest = hashlib.blake2b(digest_size=32)
    digest.update(canonical_json(value).encode("utf-8"))
    return f"blake2b-256:{digest.hexdigest()}"


def fingerprint_file(path: str | Path) -> str:
    digest = hashlib.blake2b(digest_size=32)
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"blake2b-256:{digest.hexdigest()}"


def _portable_path(path: str | Path) -> str:
    """Return a repository-relative path or a redacted external label."""

    candidate = Path(path)
    try:
        relative = candidate.resolve().relative_to(Path.cwd().resolve())
        return relative.as_posix()
    except (OSError, ValueError):
        return f"external/{candidate.name or 'artifact'}"


def fingerprint_files(paths: Iterable[str | Path]) -> str:
    files: list[dict[str, Any]] = []
    for raw_path in sorted((Path(path) for path in paths), key=lambda item: _portable_path(item)):
        entry: dict[str, Any] = {"path": _portable_path(raw_path), "exists": raw_path.exists()}
        if raw_path.exists() and raw_path.is_file():
            entry["size_bytes"] = raw_path.stat().st_size
            entry["fingerprint"] = fingerprint_file(raw_path)
        files.append(entry)
    return content_fingerprint(files)


def _as_dict(value: Any) -> dict[str, Any]:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return dict(value)
    return {"value": value}


def _manifest_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.name.endswith(".manifest.json") else path.with_suffix(path.suffix + ".manifest.json")


def atomic_write_json(path: str | Path, value: Any, *, overwrite: bool = True) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(canonical_json(value) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


@dataclass
class ArtifactManifest:
    schema_version: int = MANIFEST_SCHEMA_VERSION
    manifest_type: str = "artifact"
    artifact_id: str = ""
    artifact_kind: str = "unknown"
    created_at: str = field(default_factory=utc_now)
    content_fingerprint: str | None = None
    files: list[dict[str, Any]] = field(default_factory=list)
    parent_artifact_ids: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    legacy: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ArtifactManifest":
        migrated = migrate_manifest(payload)
        return cls(**{field: migrated[field] for field in cls.__dataclass_fields__ if field in migrated})


@dataclass
class RunManifest:
    schema_version: int = MANIFEST_SCHEMA_VERSION
    manifest_type: str = "run"
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    run_name: str = "nanoforge"
    status: str = "running"
    started_at: str = field(default_factory=utc_now)
    completed_at: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    config_fingerprint: str = ""
    artifacts: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    tokenizer: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    resources: dict[str, Any] = field(default_factory=dict)
    evaluations: list[dict[str, Any]] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RunManifest":
        migrated = migrate_manifest(payload, expected_type="run")
        return cls(**{field: migrated[field] for field in cls.__dataclass_fields__ if field in migrated})


def migrate_manifest(payload: dict[str, Any], expected_type: str | None = None) -> dict[str, Any]:
    """Migrate a manifest payload without rewriting the source file."""

    if not isinstance(payload, dict):
        raise ValueError("Manifest must be a JSON object.")
    version = payload.get("schema_version", 0)
    if isinstance(version, bool) or not isinstance(version, int) or version < 0:
        raise ValueError("Manifest schema_version must be a nonnegative integer.")
    if version > MANIFEST_SCHEMA_VERSION:
        raise RuntimeError(
            f"Manifest schema {version} is newer than supported {MANIFEST_SCHEMA_VERSION}."
        )
    migrated = dict(payload)
    manifest_type = expected_type or migrated.get("manifest_type", "artifact")
    migrated["schema_version"] = MANIFEST_SCHEMA_VERSION
    migrated.setdefault("manifest_type", manifest_type)
    if migrated["manifest_type"] != manifest_type:
        raise ValueError(f"Expected {manifest_type} manifest, found {migrated['manifest_type']}.")
    if manifest_type == "artifact":
        migrated.setdefault("artifact_id", "legacy:unknown")
        migrated.setdefault("artifact_kind", "unknown")
        migrated.setdefault("created_at", "")
        migrated.setdefault("content_fingerprint", None)
        migrated.setdefault("files", [])
        migrated.setdefault("parent_artifact_ids", [])
        migrated.setdefault("provenance", {})
        migrated.setdefault("metadata", {})
        migrated.setdefault("legacy", version == 0)
    elif manifest_type == "run":
        migrated.setdefault("run_id", "legacy:unknown")
        migrated.setdefault("run_name", "legacy")
        migrated.setdefault("status", "unknown")
        migrated.setdefault("started_at", "")
        migrated.setdefault("completed_at", None)
        for key in ("config", "artifacts", "data", "tokenizer", "model", "environment", "resources", "provenance"):
            migrated.setdefault(key, {})
        migrated.setdefault("config_fingerprint", "")
        migrated.setdefault("evaluations", [])
    else:
        raise ValueError(f"Unsupported manifest_type={manifest_type!r}.")
    return migrated


def build_artifact_manifest(
    artifact_path: str | Path,
    *,
    artifact_kind: str,
    metadata: dict[str, Any] | None = None,
    parent_artifact_ids: Iterable[str] = (),
    provenance: dict[str, Any] | None = None,
) -> ArtifactManifest:
    path = Path(artifact_path)
    file_entry: dict[str, Any] = {"path": _portable_path(path), "exists": path.exists()}
    file_fingerprint = None
    if path.exists() and path.is_file():
        file_fingerprint = fingerprint_file(path)
        file_entry.update({"size_bytes": path.stat().st_size, "fingerprint": file_fingerprint})
    manifest = ArtifactManifest(
        artifact_kind=artifact_kind,
        content_fingerprint=file_fingerprint,
        files=[file_entry],
        parent_artifact_ids=list(parent_artifact_ids),
        provenance=provenance or {},
        metadata=metadata or {},
    )
    manifest.artifact_id = content_fingerprint(
        {
            "artifact_kind": manifest.artifact_kind,
            "content_fingerprint": manifest.content_fingerprint,
            "files": manifest.files,
            "parent_artifact_ids": manifest.parent_artifact_ids,
            "provenance": manifest.provenance,
            "metadata": manifest.metadata,
        }
    )
    return manifest


def write_artifact_manifest(
    artifact_path: str | Path,
    manifest: ArtifactManifest | None = None,
    **kwargs: Any,
) -> Path:
    manifest = manifest or build_artifact_manifest(artifact_path, **kwargs)
    return atomic_write_json(_manifest_path(artifact_path), manifest.to_dict())


def load_artifact_manifest(artifact_path: str | Path) -> ArtifactManifest:
    path = _manifest_path(artifact_path)
    if path.exists():
        return ArtifactManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))
    artifact = Path(artifact_path)
    if artifact.exists():
        return ArtifactManifest(
            artifact_id="legacy:" + fingerprint_file(artifact),
            artifact_kind="checkpoint" if artifact.suffix == ".pt" else "unknown",
            content_fingerprint=fingerprint_file(artifact),
            files=[{"path": _portable_path(artifact), "size_bytes": artifact.stat().st_size}],
            provenance={"source": "legacy_artifact_without_manifest"},
            legacy=True,
        )
    raise FileNotFoundError(path)


def write_run_manifest(path: str | Path, manifest: RunManifest) -> Path:
    return atomic_write_json(path, manifest.to_dict())


def load_run_manifest(path: str | Path) -> RunManifest:
    return RunManifest.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def environment_snapshot(device: str | None = None) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "device": device,
        "code_revision": _git_revision(),
    }
    try:
        import torch

        snapshot["torch"] = torch.__version__
        snapshot["cuda"] = torch.version.cuda
        snapshot["cuda_available"] = torch.cuda.is_available()
    except Exception:
        snapshot["torch"] = None
        snapshot["cuda"] = None
        snapshot["cuda_available"] = False
    return snapshot


def resource_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {"process_rss_mb": None, "peak_process_rss_mb": None, "peak_cuda_allocated_mb": None}
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in (
                    "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                    "PagefileUsage", "PeakPagefileUsage",
                )
            ]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        current = ctypes.windll.kernel32.GetCurrentProcess
        current.restype = wintypes.HANDLE
        query = ctypes.windll.psapi.GetProcessMemoryInfo
        query.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if query(current(), ctypes.byref(counters), counters.cb):
            snapshot["process_rss_mb"] = counters.WorkingSetSize / (1024 * 1024)
            snapshot["peak_process_rss_mb"] = counters.PeakWorkingSetSize / (1024 * 1024)
    elif sys.platform.startswith("linux"):
        try:
            resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
            snapshot["process_rss_mb"] = resident_pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
        except (OSError, ValueError, IndexError):
            pass
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
        snapshot["peak_process_rss_mb"] = usage.ru_maxrss / (1024 if sys.platform != "darwin" else 1024 * 1024)
    except (ImportError, OSError):
        pass
    try:
        import torch

        if torch.cuda.is_available():
            snapshot["peak_cuda_allocated_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
    except Exception:
        pass
    return snapshot


class ResourceMonitor:
    """Sample this process during a run; samples are not an exact allocator peak."""

    def __init__(self, interval: float = 0.05):
        if interval <= 0:
            raise ValueError("resource sampling interval must be positive")
        self.interval = interval
        self.result: dict[str, Any] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._peak = 0.0
        self._samples = 0

    def _sample(self) -> None:
        value = resource_snapshot().get("process_rss_mb")
        if value is not None:
            self._peak = max(self._peak, value)
            self._samples += 1

    def __enter__(self):
        self._started = time.perf_counter()
        self._cpu_started = time.process_time()
        self._sample()

        def observe():
            while not self._stop.wait(self.interval):
                self._sample()

        self._thread = threading.Thread(target=observe, name="nanoforge-resources", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_args):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._sample()
        self.result = {
            **resource_snapshot(),
            "elapsed_seconds": time.perf_counter() - self._started,
            "cpu_seconds": time.process_time() - self._cpu_started,
            "run_peak_sampled_rss_mb": self._peak if self._samples else None,
            "sample_count": self._samples,
            "sample_interval_seconds": self.interval,
            "peak_is_sampled": True,
            "scope": "process-during-run",
            "concurrent_process_work_included": True,
        }


def list_artifact_manifests(root: str | Path, *, max_files: int = 1000) -> list[dict[str, Any]]:
    """Bounded manifest catalog; callers authorize the root, not model content."""
    root = Path(root).resolve()
    if max_files < 1:
        raise ValueError("max_files must be positive")
    result = []
    for directory, subdirs, files in os.walk(root, followlinks=False):
        subdirs[:] = sorted(name for name in subdirs if name not in {".git", "node_modules", ".venv", ".unlazy"})
        for name in sorted(files):
            if not name.endswith(".manifest.json"):
                continue
            path = Path(directory) / name
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                continue
            if path.stat().st_size > 4 * 1024 * 1024:
                raise ValueError("manifest exceeds catalog byte limit")
            raw = json.loads(path.read_text(encoding="utf-8"))
            if raw.get("manifest_type", "artifact") != "artifact":
                continue
            manifest = ArtifactManifest.from_dict(raw)
            result.append({"path": path.relative_to(root).as_posix(), "manifest": manifest.to_dict()})
            if len(result) >= max_files:
                return result
    return result


class ArtifactRegistry:
    """Opt-in immutable, content-addressed copies; no training-data auto-import."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _directory(self, artifact_id: str) -> Path:
        prefix = "blake2b-256:"
        digest = artifact_id.removeprefix(prefix)
        if not artifact_id.startswith(prefix) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("invalid artifact identity")
        return self.root / digest

    def register(self, source: str | Path, *, artifact_kind: str, owner: str,
                 parent_artifact_ids: Iterable[str] = (), metadata: dict[str, Any] | None = None) -> ArtifactManifest:
        source = Path(source)
        if not owner.strip() or not artifact_kind.strip() or not source.is_file() or source.is_symlink():
            raise ValueError("registration requires a regular file, owner and kind")
        digest = fingerprint_file(source)
        identity = content_fingerprint({"content": digest, "kind": artifact_kind, "owner": owner,
                                        "parents": list(parent_artifact_ids), "metadata": metadata or {}})
        directory = self._directory(identity)
        directory.mkdir(parents=True, exist_ok=True)
        payload = directory / "payload"
        if payload.exists():
            self.verify(identity)
            return load_artifact_manifest(payload)
        temporary = directory / f".{uuid.uuid4().hex}.tmp"
        try:
            shutil.copyfile(source, temporary)
            if fingerprint_file(temporary) != digest:
                raise ValueError("source changed during artifact registration")
            os.link(temporary, payload)
        finally:
            temporary.unlink(missing_ok=True)
        manifest = build_artifact_manifest(payload, artifact_kind=artifact_kind,
                                           parent_artifact_ids=parent_artifact_ids,
                                           metadata={**(metadata or {}), "owner": owner, "immutable": True})
        manifest.artifact_id = identity
        atomic_write_json(_manifest_path(payload), manifest.to_dict(), overwrite=False)
        return manifest

    def path_for(self, artifact_id: str) -> Path:
        return self._directory(artifact_id) / "payload"

    def verify(self, artifact_id: str) -> bool:
        path = self.path_for(artifact_id)
        manifest = load_artifact_manifest(path)
        if manifest.artifact_id != artifact_id or fingerprint_file(path) != manifest.content_fingerprint:
            raise ValueError("artifact integrity check failed")
        return True


class RunManifestWriter:
    """Small lifecycle writer used by training and future job runners."""

    def __init__(self, output_dir: str | Path, run_name: str, config: Any, *, device: str | None = None):
        self.output_dir = Path(output_dir)
        self.path = self.output_dir / "run.manifest.json"
        self.manifest = RunManifest(
            run_name=run_name,
            config=_as_dict(config),
            config_fingerprint=content_fingerprint(_as_dict(config)),
            environment=environment_snapshot(device),
            provenance={"paths_are_relative": True},
        )
        self.write()

    def write(self) -> None:
        write_run_manifest(self.path, self.manifest)

    def update(self, **fields: Any) -> None:
        for key, value in fields.items():
            if not hasattr(self.manifest, key):
                raise AttributeError(f"Unknown run manifest field: {key}")
            current = getattr(self.manifest, key)
            if isinstance(current, dict) and isinstance(value, dict):
                setattr(self.manifest, key, {**current, **value})
            else:
                setattr(self.manifest, key, value)
        self.write()

    def add_checkpoint(self, path: str | Path) -> None:
        manifest = load_artifact_manifest(path)
        checkpoints = self.manifest.artifacts.setdefault("checkpoints", [])
        checkpoints.append({"path": _portable_path(path), "artifact_id": manifest.artifact_id})
        self.write()

    def add_evaluation(self, report: dict[str, Any]) -> None:
        self.manifest.evaluations.append(report)
        self.write()

    def finish(self, status: str = "completed", **fields: Any) -> None:
        self.manifest.status = status
        self.manifest.completed_at = utc_now()
        self.manifest.resources = {**self.manifest.resources, **resource_snapshot()}
        for key, value in fields.items():
            if not hasattr(self.manifest, key):
                raise AttributeError(f"Unknown run manifest field: {key}")
            if key == "resources" and isinstance(value, dict):
                self.manifest.resources = {**self.manifest.resources, **value}
            else:
                setattr(self.manifest, key, value)
        self.write()
