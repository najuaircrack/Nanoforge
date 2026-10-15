from __future__ import annotations

import json

import pytest

from nanoforge.artifacts import (
    ArtifactManifest,
    RunManifest,
    atomic_write_json,
    content_fingerprint,
    load_artifact_manifest,
    load_run_manifest,
    write_artifact_manifest,
    write_run_manifest,
)


def test_canonical_fingerprint_ignores_mapping_order():
    assert content_fingerprint({"a": 1, "b": 2}) == content_fingerprint({"b": 2, "a": 1})


def test_artifact_manifest_roundtrip_and_content_fingerprint(tmp_path):
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"nanoforge")
    path = write_artifact_manifest(artifact, artifact_kind="dataset", metadata={"rows": 1})

    loaded = load_artifact_manifest(artifact)
    assert path.name == "artifact.bin.manifest.json"
    assert loaded.artifact_kind == "dataset"
    assert loaded.content_fingerprint is not None
    assert loaded.files[0]["path"] == "external/artifact.bin"


def test_run_manifest_roundtrip(tmp_path):
    path = tmp_path / "run.manifest.json"
    manifest = RunManifest(run_name="smoke", config={"seed": 7}, config_fingerprint="x")
    write_run_manifest(path, manifest)
    loaded = load_run_manifest(path)
    assert loaded.run_name == "smoke"
    assert loaded.config == {"seed": 7}


def test_future_manifest_version_is_rejected(tmp_path):
    path = tmp_path / "future.manifest.json"
    atomic_write_json(path, {"schema_version": 99, "manifest_type": "artifact"})
    with pytest.raises(RuntimeError, match="newer than supported"):
        load_artifact_manifest(path)


def test_legacy_artifact_without_manifest_is_read_only(tmp_path):
    artifact = tmp_path / "legacy.pt"
    artifact.write_bytes(b"legacy")
    manifest = load_artifact_manifest(artifact)
    assert isinstance(manifest, ArtifactManifest)
    assert manifest.legacy is True
    assert manifest.artifact_kind == "checkpoint"


def test_atomic_json_is_valid(tmp_path):
    path = tmp_path / "value.json"
    atomic_write_json(path, {"b": 2, "a": 1})
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 1, "b": 2}
