from __future__ import annotations

from pathlib import Path

import torch

from nanoforge.model.transformer import NanoforgeForCausalLM


def export_onnx(model: NanoforgeForCausalLM, out_path: str | Path, opset: int = 17) -> None:
    model.eval()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.ones(1, min(16, model.config.max_seq_len), dtype=torch.long, device=model.device)
    wrapper = _LogitsOnly(model)
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(out_path),
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {1: "sequence"}, "logits": {1: "sequence"}},
        opset_version=opset,
    )


class _LogitsOnly(torch.nn.Module):
    def __init__(self, model: NanoforgeForCausalLM):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids).logits
