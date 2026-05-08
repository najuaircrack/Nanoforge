from __future__ import annotations

import time
from pathlib import Path

import torch

from nanoforge.config import load_config
from nanoforge.model.transformer import NanoforgeForCausalLM
from nanoforge.profiling import estimate_model_profile
from nanoforge.training.utils import resolve_device


@torch.inference_mode()
def benchmark_forward(config_path: str | Path, batch_size: int = 1, steps: int = 20, device: str = "auto") -> dict[str, float]:
    cfg = load_config(config_path)
    dev = resolve_device(device)
    model = NanoforgeForCausalLM(cfg.model).to(dev).eval()
    x = torch.randint(0, cfg.model.vocab_size, (batch_size, cfg.data.seq_len), device=dev)
    for _ in range(3):
        model(x)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(steps):
        model(x)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    return {
        "tokens_per_sec": batch_size * cfg.data.seq_len * steps / max(elapsed, 1e-9),
        "ms_per_step": elapsed * 1000 / steps,
        "params": model.estimate_num_params(),
        **estimate_model_profile(
            cfg.model,
            batch_size=batch_size,
            seq_len=cfg.data.seq_len,
        ).to_dict(),
    }
