from __future__ import annotations

import torch


def dynamic_quantize_cpu(model: torch.nn.Module) -> torch.nn.Module:
    """Apply dynamic int8 quantization to Linear layers for CPU inference."""

    return torch.quantization.quantize_dynamic(model.cpu().eval(), {torch.nn.Linear}, dtype=torch.qint8)


def prepare_qat(model: torch.nn.Module) -> torch.nn.Module:
    """Prepare a model for quantization-aware experiments on supported modules."""

    model.train()
    model.qconfig = torch.quantization.get_default_qat_qconfig("fbgemm")
    return torch.quantization.prepare_qat(model, inplace=False)

