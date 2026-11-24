from __future__ import annotations

import inspect
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

import torch

from nanoforge.artifacts import atomic_write_json
from nanoforge.export.gguf import _tokenizer_payload
from nanoforge.model.transformer import NanoforgeForCausalLM


def export_onnx(
    model: NanoforgeForCausalLM, out_path: str | Path, opset: int = 17, *, config=None
) -> None:
    if model.config.attention_backend not in {"sdpa", "manual"}:
        raise ValueError(
            "ONNX export currently requires dense manual or SDPA attention; reference tiled/paged algorithms need a dynamic export implementation"
        )
    model.eval()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.ones(1, min(16, model.config.max_seq_len), dtype=torch.long, device=model.device)
    wrapper = _LogitsOnly(model)
    descriptor, temporary = tempfile.mkstemp(
        prefix=out_path.name + ".", suffix=".onnx", dir=out_path.parent
    )
    os.close(descriptor)
    options = (
        {"dynamo": False} if "dynamo" in inspect.signature(torch.onnx.export).parameters else {}
    )
    try:
        torch.onnx.export(
            wrapper,
            (dummy,),
            temporary,
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "logits": {0: "batch", 1: "sequence"},
            },
            opset_version=opset,
            **options,
        )
        import onnx

        onnx.checker.check_model(temporary)
        os.replace(temporary, out_path)
        if config is not None:
            atomic_write_json(
                out_path.with_suffix(out_path.suffix + ".nanoforge.json"),
                {
                    "schema_version": 1,
                    "config": asdict(config),
                    "tokenizer": _tokenizer_payload(config),
                    "context_length": model.config.max_seq_len,
                },
            )
    finally:
        Path(temporary).unlink(missing_ok=True)


class _LogitsOnly(torch.nn.Module):
    def __init__(self, model: NanoforgeForCausalLM):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids).logits
