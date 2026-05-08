from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class QuantizationConfig:
    backend: str = "none"
    dtype: str = "int8"
    quantize_kv_cache: bool = False
    group_size: int = 128


def apply_quantization(model: torch.nn.Module, backend: str = "none", **kwargs) -> torch.nn.Module:
    key = backend.strip().lower().replace("-", "_")
    if key in {"none", "fp32"}:
        return model
    if key in {"int8", "dynamic_int8", "cpu_int8"}:
        return dynamic_quantize_cpu(model)
    if key in {"int4", "gptq", "awq"}:
        return prepare_weight_only_quantization(model, bits=4)
    if key == "gguf":
        return mark_gguf_export_compatible(model)
    raise ValueError(f"Unknown quantization backend '{backend}'.")


def dynamic_quantize_cpu(model: torch.nn.Module) -> torch.nn.Module:
    """Apply dynamic int8 quantization to Linear layers for CPU inference."""

    quant = getattr(torch, "ao", torch).quantization
    return quant.quantize_dynamic(model.cpu().eval(), {torch.nn.Linear}, dtype=torch.qint8)


def prepare_weight_only_quantization(model: torch.nn.Module, bits: int = 4) -> torch.nn.Module:
    """Attach metadata for external GPTQ/AWQ style weight-only passes.

    PyTorch does not ship a universal int4 transformer quantizer. Nanoforge keeps this hook
    explicit so low-memory runtimes can detect intent and route to external kernels.
    """

    if bits not in {4, 8}:
        raise ValueError("Weight-only quantization supports 4 or 8 bits.")
    model.eval()
    setattr(model, "nanoforge_quantization", QuantizationConfig(backend=f"int{bits}", dtype=f"int{bits}"))
    return model


def mark_gguf_export_compatible(model: torch.nn.Module) -> torch.nn.Module:
    model.eval()
    setattr(model, "nanoforge_quantization", QuantizationConfig(backend="gguf"))
    return model


def prepare_qat(model: torch.nn.Module) -> torch.nn.Module:
    """Prepare a model for quantization-aware experiments on supported modules."""

    model.train()
    quant = getattr(torch, "ao", torch).quantization
    model.qconfig = quant.get_default_qat_qconfig("fbgemm")
    return quant.prepare_qat(model, inplace=False)


def quantize_kv_tensor(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = tensor.detach().abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 127.0
    return torch.clamp((tensor / scale).round(), -128, 127).to(torch.int8), scale


def dequantize_kv_tensor(qtensor: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return qtensor.float() * scale
