from __future__ import annotations

import hashlib
import json
import os
import random
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nanoforge.config import NanoforgeConfig, save_config
from nanoforge.registry import registry_snapshot


CHECKPOINT_SCHEMA_VERSION = 2


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
        torch.random.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    config: NanoforgeConfig,
    step: int,
    val_loss: float | None = None,
    ema_state: dict[str, torch.Tensor] | None = None,
) -> None:
    payload = make_checkpoint_payload(model, optimizer, config, step, val_loss, ema_state)
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
) -> dict[str, Any]:
    model_state = model.state_dict()
    optimizer_state = optimizer.state_dict() if optimizer is not None else None
    ema = ema_state
    if clone_tensors:
        model_state = _clone_state(model_state)
        optimizer_state = _clone_state(optimizer_state)
        ema = _clone_state(ema)
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model": model_state,
        "optimizer": optimizer_state,
        "config": config,
        "step": step,
        "val_loss": val_loss,
        "ema": ema,
        "rng": rng_state(),
        "registry": registry_snapshot(),
    }


def write_checkpoint_payload(
    path: str | Path,
    payload: dict[str, Any],
    config: NanoforgeConfig,
    step: int,
    val_loss: float | None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)
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


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    path = Path(path)
    if not verify_checkpoint(path):
        raise RuntimeError(f"Checkpoint integrity check failed: {path}")
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=map_location)
    if "schema_version" not in payload:
        payload["schema_version"] = 1
    return migrate_checkpoint(payload)


def migrate_checkpoint(payload: dict[str, Any]) -> dict[str, Any]:
    version = int(payload.get("schema_version", 1))
    if version > CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError(
            f"Checkpoint schema {version} is newer than supported {CHECKPOINT_SCHEMA_VERSION}."
        )
    if version < 2:
        payload.setdefault("rng", None)
        payload.setdefault("registry", {})
        payload["schema_version"] = 2
    return payload


class AsyncCheckpointSaver:
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self._executor = ThreadPoolExecutor(max_workers=1) if enabled else None
        self._future: Future[None] | None = None

    def save(
        self,
        path: str | Path,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None,
        config: NanoforgeConfig,
        step: int,
        val_loss: float | None = None,
        ema_state: dict[str, torch.Tensor] | None = None,
    ) -> None:
        self.wait()
        if self._executor is None:
            save_checkpoint(path, model, optimizer, config, step, val_loss, ema_state)
            return
        payload = make_checkpoint_payload(
            model,
            optimizer,
            config,
            step,
            val_loss,
            ema_state,
            clone_tensors=True,
        )
        self._future = self._executor.submit(
            write_checkpoint_payload,
            path,
            payload,
            config,
            step,
            val_loss,
        )

    def wait(self) -> None:
        if self._future is not None:
            self._future.result()
            self._future = None

    def close(self) -> None:
        self.wait()
        if self._executor is not None:
            self._executor.shutdown(wait=True)
