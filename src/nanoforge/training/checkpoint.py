from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import tempfile
import threading
import warnings
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nanoforge.artifacts import content_fingerprint, write_artifact_manifest
from nanoforge.config import (
    DataConfig, InferenceConfig, ModelConfig, MoEConfig, NanoforgeConfig, TrainConfig, parse_config, save_config,
)
from nanoforge.registry import registry_snapshot

CHECKPOINT_SCHEMA_VERSION = 3
_LOAD_LOCK = threading.RLock()


def checkpoint_hash(path: str | Path) -> str:
    digest = hashlib.blake2b(digest_size=32)
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.random.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    config: NanoforgeConfig,
    step: int,
    val_loss: float | None = None,
    ema_state: dict[str, torch.Tensor] | None = None,
    *,
    training_state: dict[str, Any] | None = None,
) -> None:
    payload = make_checkpoint_payload(
        model, optimizer, config, step, val_loss, ema_state, training_state=training_state
    )
    write_checkpoint_payload(path, payload, config, step, val_loss)


def make_checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    config: NanoforgeConfig,
    step: int,
    val_loss: float | None = None,
    ema_state: dict[str, torch.Tensor] | None = None,
    *,
    clone_tensors: bool = False,
    training_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if step < 0:
        raise ValueError("Completed update count cannot be negative.")
    model_state = model.state_dict()
    optimizer_state = optimizer.state_dict() if optimizer is not None else None
    ema = ema_state
    if clone_tensors:
        model_state = _clone_state(model_state)
        optimizer_state = _clone_state(optimizer_state)
        ema = _clone_state(ema)
        training_state = _clone_state(training_state)
    return _primitive_state({
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model": model_state,
        "optimizer": optimizer_state,
        "config": asdict(config),
        "config_schema_version": 1,
        "state_encoding_version": 1,
        "step": step,
        "next_step": step,
        "step_semantics": "completed_updates",
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "training_state": training_state or {},
        "val_loss": val_loss,
        "ema": ema,
        "rng": rng_state(),
        "registry": registry_snapshot(),
    })


def write_checkpoint_payload(
    path: str | Path,
    payload: dict[str, Any],
    config: NanoforgeConfig,
    step: int,
    val_loss: float | None,
) -> None:
    path = Path(path)
    if isinstance(config, dict):
        config = _restore_config(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    digest = checkpoint_hash(path)
    path.with_suffix(path.suffix + ".hash").write_text(digest + "\n", encoding="utf-8")
    path.with_suffix(path.suffix + ".meta.json").write_text(
        json.dumps(
            {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "step": step,
                "val_loss": val_loss,
                "hash": digest,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    save_config(config, path.with_suffix(".yaml"))
    write_artifact_manifest(
        path,
        artifact_kind="checkpoint",
        metadata={
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "step": step,
            "val_loss": val_loss,
            "config_fingerprint": content_fingerprint(config),
            "parameter_count": payload["parameter_count"],
            "next_step": payload["next_step"],
        },
        provenance={"checkpoint_step_semantics": "completed_updates"},
    )


def _clone_state(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _clone_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_state(item) for item in value)
    return value


def verify_checkpoint(path: str | Path) -> bool:
    path = Path(path)
    hash_path = path.with_suffix(path.suffix + ".hash")
    if not hash_path.exists():
        return True
    expected = hash_path.read_text(encoding="utf-8").strip()
    return checkpoint_hash(path) == expected


def _primitive_state(value: Any) -> Any:
    """New payloads contain tensors and primitive containers, never arbitrary Python objects."""
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject or value.dtype.kind == "c":
            raise TypeError("Object/complex NumPy arrays are not supported checkpoint state.")
        return {"__nanoforge_array__": 1, "dtype": str(value.dtype), "values": value.tolist(), "shape": list(value.shape)}
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value) or value is None or type(value) in (str, bytes, int, float, bool):
        return value
    if isinstance(value, dict):
        if "__nanoforge_array__" in value:
            raise ValueError("Checkpoint state uses a reserved NumPy encoding key.")
        if any(type(key) not in (str, int) for key in value):
            raise TypeError("Checkpoint mapping keys must be strings or integers.")
        return {key: _primitive_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_primitive_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_primitive_state(item) for item in value)
    raise TypeError(f"Unsupported checkpoint state type: {type(value).__name__}")


def _restore_state(value: Any) -> Any:
    if isinstance(value, dict):
        if set(value) == {"__nanoforge_array__", "dtype", "values", "shape"} and value["__nanoforge_array__"] == 1:
            dtype = np.dtype(value["dtype"])
            if dtype.hasobject:
                raise ValueError("Object-dtype NumPy state is not supported.")
            return np.asarray(value["values"], dtype=dtype).reshape(value["shape"])
        return {key: _restore_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_restore_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_restore_state(item) for item in value)
    return value


def _restore_config(value: dict[str, Any]) -> NanoforgeConfig:
    return parse_config(json.dumps(value), strict=True)


def _legacy_config_values(value):
    if isinstance(value, (NanoforgeConfig, ModelConfig, MoEConfig, TrainConfig, DataConfig, InferenceConfig)):
        return {key: _legacy_config_values(item) for key, item in vars(value).items()}
    if isinstance(value, dict):
        return {key: _legacy_config_values(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_legacy_config_values(item) for item in value]
    return value


def _legacy_safe_globals():
    # Explicit constructors used by historical Nanoforge configs and NumPy RNG snapshots.
    core = np._core if hasattr(np, "_core") else np.core
    reconstruct = core.multiarray._reconstruct
    return [
        NanoforgeConfig, ModelConfig, MoEConfig, TrainConfig, DataConfig, InferenceConfig,
        np.ndarray, np.dtype, reconstruct,
        (reconstruct, "numpy.core.multiarray._reconstruct"),
        type(np.dtype("uint32")), type(np.dtype("int64")), type(np.dtype("float64")),
    ]


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    version = str(torch.__version__).split("+", 1)[0].split(".")
    if tuple(int(item) for item in version[:2]) < (2, 6) or not hasattr(torch.serialization, "safe_globals"):
        raise RuntimeError("Safe checkpoint loading requires PyTorch >= 2.6; unsafe pickle fallback is disabled.")
    path = Path(path)
    if not verify_checkpoint(path):
        raise RuntimeError(f"Checkpoint integrity check failed: {path}")
    with _LOAD_LOCK, torch.serialization.safe_globals(_legacy_safe_globals()):
        payload = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise ValueError("Checkpoint must contain a model state dictionary.")
    if payload.get("state_encoding_version", 1) != 1:
        raise ValueError("Unsupported checkpoint state encoding version.")
    payload = _restore_state(payload)
    if isinstance(payload.get("config"), dict):
        payload["config"] = _restore_config(payload["config"])
    elif isinstance(payload.get("config"), NanoforgeConfig):
        payload["config"] = _restore_config(_legacy_config_values(payload["config"]))
    return migrate_checkpoint(payload, checkpoint_path=path)


def migrate_checkpoint(
    payload: dict[str, Any], checkpoint_path: str | Path | None = None
) -> dict[str, Any]:
    version = int(payload.get("schema_version", 1))
    if version < 1 or version > CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError(
            f"Checkpoint schema {version} is newer than supported {CHECKPOINT_SCHEMA_VERSION}."
        )
    if version < 3:
        payload.setdefault("rng", None)
        payload.setdefault("registry", {})
        name = Path(checkpoint_path).name if checkpoint_path is not None else ""
        stored_step = int(payload.get("step", 0))
        completed = stored_step if name in {"last.pt", "interrupt.pt"} else stored_step + 1
        warnings.warn(
            "Migrating legacy update counters; old early-stopped last.pt counts may "
            "overstate work and cannot be recovered exactly.",
            RuntimeWarning,
            stacklevel=2,
        )
        payload["step"] = completed
        payload["next_step"] = completed
        payload["step_semantics"] = "completed_updates"
        payload["legacy_schema_version"] = version
        payload["schema_version"] = CHECKPOINT_SCHEMA_VERSION
    if int(payload.get("next_step", -1)) < 0:
        raise ValueError("Checkpoint next_step must be a nonnegative completed update count.")
    payload.setdefault("training_state", {})
    return payload


class AsyncCheckpointSaver:
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self._executor = ThreadPoolExecutor(max_workers=1) if enabled else None
        self._future: Future[None] | None = None
        self.closed = False

    def save(
        self,
        path: str | Path,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None,
        config: NanoforgeConfig,
        step: int,
        val_loss: float | None = None,
        ema_state: dict[str, torch.Tensor] | None = None,
        *,
        training_state: dict[str, Any] | None = None,
    ) -> None:
        if self.closed:
            raise RuntimeError("Checkpoint saver is closed.")
        self.wait()
        if self._executor is None:
            save_checkpoint(
                path,
                model,
                optimizer,
                config,
                step,
                val_loss,
                ema_state,
                training_state=training_state,
            )
            return
        payload = make_checkpoint_payload(
            model,
            optimizer,
            config,
            step,
            val_loss,
            ema_state,
            clone_tensors=True,
            training_state=training_state,
        )
        self._future = self._executor.submit(
            write_checkpoint_payload,
            path,
            payload,
            payload["config"],
            step,
            val_loss,
        )

    def wait(self) -> None:
        if self._future is not None:
            future, self._future = self._future, None
            future.result()

    def save_payload(self, path: str | Path, payload: dict[str, Any]) -> None:
        """Persist a collectively gathered FSDP payload only on the designated writer."""
        if self.closed:
            raise RuntimeError("Checkpoint saver is closed.")
        self.wait()
        snapshot = _clone_state(payload) if self.enabled else payload
        arguments = (path, snapshot, snapshot["config"], snapshot["step"], snapshot.get("val_loss"))
        if self._executor is None:
            write_checkpoint_payload(*arguments)
        else:
            self._future = self._executor.submit(write_checkpoint_payload, *arguments)

    def close(self) -> None:
        if self.closed:
            return
        try:
            self.wait()
        finally:
            if self._executor is not None:
                self._executor.shutdown(wait=True)
            self.closed = True
