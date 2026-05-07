from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from nanoforge.config import NanoforgeConfig, save_config


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    config: NanoforgeConfig,
    step: int,
    val_loss: float | None = None,
    ema_state: dict[str, torch.Tensor] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "config": config,
        "step": step,
        "val_loss": val_loss,
        "ema": ema_state,
    }
    torch.save(payload, path)
    save_config(config, path.with_suffix(".yaml"))


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    try:
        return torch.load(Path(path), map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(Path(path), map_location=map_location)
