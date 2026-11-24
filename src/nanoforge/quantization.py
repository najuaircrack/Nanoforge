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
        raise NotImplementedError(
            f"Quantization backend '{backend}' is not implemented natively. "
            "Nanoforge does not ship int4/GPTQ/AWQ kernels. Use a supported external "
            "pretrained quantized runtime, or backend='int8' for "
            "built-in dynamic CPU quantization."
        )
    if key == "gguf":
        return mark_gguf_export_compatible(model)
    raise ValueError(f"Unknown quantization backend '{backend}'.")


def dynamic_quantize_cpu(model: torch.nn.Module) -> torch.nn.Module:
    """Apply dynamic int8 quantization to Linear layers for CPU inference."""

    from nanoforge.model.lora import merge_lora

    dense = merge_lora(model).cpu().eval()
    quant = getattr(torch, "ao", torch).quantization
    converted = quant.quantize_dynamic(dense, {torch.nn.Linear}, dtype=torch.qint8)
    converted.nanoforge_quantization = QuantizationConfig(backend="int8")
    return converted


def prepare_inference_model(
    model: torch.nn.Module, backend: str = "none", *, device: str = "auto"
) -> torch.nn.Module:
    """Apply quantization after loading floating-point weights; never during training."""
    from nanoforge.training.utils import resolve_device

    key = backend.strip().lower().replace("-", "_")
    if key in {"int8", "dynamic_int8", "cpu_int8"}:
        if device != "auto" and torch.device(device).type != "cpu":
            raise ValueError("Dynamic int8 inference requires a CPU device")
        return dynamic_quantize_cpu(model)
    result = apply_quantization(model, backend)
    return result.to(resolve_device(device)).eval()


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
