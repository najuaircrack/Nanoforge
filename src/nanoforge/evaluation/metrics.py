from __future__ import annotations

import math
from pathlib import Path

import torch

from nanoforge.data.dataset import PackedMemmapDataset, make_torch_batch
from nanoforge.model.transformer import NanoforgeForCausalLM
from nanoforge.training.checkpoint import load_checkpoint
from nanoforge.training.utils import resolve_device
from nanoforge.training.utils import seed_everything


@torch.inference_mode()
def evaluate_checkpoint(
    checkpoint: str | Path,
    data_path: str | Path,
    seq_len: int,
    batches: int = 20,
    device: str = "auto",
    seed: int = 1337,
) -> dict[str, float]:
    if isinstance(batches, bool) or not isinstance(batches, int) or batches < 1:
        raise ValueError("batches must be positive")
    seed_everything(seed)
    payload = load_checkpoint(checkpoint, map_location="cpu")
    cfg = payload["config"]
    model = NanoforgeForCausalLM(cfg.model)
    model.load_state_dict(payload["model"], strict=True)
    dev = resolve_device(device)
    model.to(dev).eval()
    data = PackedMemmapDataset(data_path, seq_len)
    loss_sum = 0.0
    correct = 0
    total = 0
    for _ in range(batches):
        x, y = make_torch_batch(data, cfg.training.micro_batch_size, str(dev), False)
        out = model(x, labels=y)
        valid = y != -100
        count = int(valid.sum().detach().cpu())
        if not count:
            continue
        loss_sum += float(out.loss.detach().cpu()) * count
        pred = out.logits.argmax(dim=-1)
        correct += int(((pred == y) & valid).sum().detach().cpu())
        total += count
    if not total:
        raise ValueError("evaluation contains no supervised tokens")
    loss = loss_sum / total
    return {
        "loss": loss,
        "perplexity": math.exp(min(20.0, loss)),
        "token_accuracy": correct / max(total, 1),
        "evaluated_tokens": total,
        "seed": seed,
    }

