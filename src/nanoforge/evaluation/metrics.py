from __future__ import annotations

import math
from pathlib import Path

import torch

from nanoforge.data.dataset import PackedMemmapDataset, make_torch_batch
from nanoforge.model.transformer import NanoforgeForCausalLM
from nanoforge.training.checkpoint import load_checkpoint
from nanoforge.training.utils import resolve_device


@torch.inference_mode()
def evaluate_checkpoint(
    checkpoint: str | Path,
    data_path: str | Path,
    seq_len: int,
    batches: int = 20,
    device: str = "auto",
) -> dict[str, float]:
    payload = load_checkpoint(checkpoint, map_location="cpu")
    cfg = payload["config"]
    model = NanoforgeForCausalLM(cfg.model)
    model.load_state_dict(payload["model"], strict=True)
    dev = resolve_device(device)
    model.to(dev).eval()
    data = PackedMemmapDataset(data_path, seq_len)
    losses: list[float] = []
    correct = 0
    total = 0
    for _ in range(batches):
        x, y = make_torch_batch(data, cfg.training.micro_batch_size, str(dev), False)
        out = model(x, labels=y)
        losses.append(float(out.loss.detach().cpu()))
        pred = out.logits.argmax(dim=-1)
        correct += int((pred == y).sum().detach().cpu())
        total += y.numel()
    loss = sum(losses) / max(len(losses), 1)
    return {
        "loss": loss,
        "perplexity": math.exp(min(20.0, loss)),
        "token_accuracy": correct / max(total, 1),
    }

