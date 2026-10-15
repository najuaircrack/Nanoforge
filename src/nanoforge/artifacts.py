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
import subprocess
import sys
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


def atomic_write_json(path: str | Path, value: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(canonical_json(value) + "\n", encoding="utf-8")
    os.replace(temporary, path)
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
    version = int(payload.get("schema_version", 0))
    if version > MANIFEST_SCHEMA_VERSION:
        raise RuntimeError(
            f"Manifest schema {version} is newer than supported {MANIFEST_SCHEMA_VERSION}."
        )
    migrated = dict(payload)
    manifest_type = expected_type or migrated.get("manifest_type", "artifact")
    migrated.setdefault("schema_version", MANIFEST_SCHEMA_VERSION)
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
    snapshot: dict[str, Any] = {"process_rss_mb": None, "peak_cuda_allocated_mb": None}
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
        snapshot["process_rss_mb"] = usage.ru_maxrss / (1024 if sys.platform != "darwin" else 1024 * 1024)
    except (ImportError, OSError):
        pass
    try:
        import torch

        if torch.cuda.is_available():
            snapshot["peak_cuda_allocated_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
    except Exception:
        pass
    return snapshot


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
