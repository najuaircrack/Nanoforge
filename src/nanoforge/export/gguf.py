from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from nanoforge.config import NanoforgeConfig


def write_gguf_manifest(config: NanoforgeConfig, checkpoint_path: str | Path, out_path: str | Path) -> None:
    """Write converter metadata for GGUF workflows.

    GGUF is a binary format with model-family-specific tensor naming rules. Nanoforge writes
    a manifest that external converters can consume without guessing architecture fields.
    """

    payload = {
        "format": "nanoforge-gguf-manifest-v1",
        "checkpoint": str(checkpoint_path),
        "architecture": "nanoforge.decoder_only",
        "config": asdict(config),
        "tensor_mapping": {
            "token_embd.weight": "embed.weight",
            "output.weight": "lm_head.weight",
            "blk.*.attn_q.weight": "blocks.*.attn.q_proj.weight",
            "blk.*.attn_k.weight": "blocks.*.attn.k_proj.weight",
            "blk.*.attn_v.weight": "blocks.*.attn.v_proj.weight",
            "blk.*.attn_output.weight": "blocks.*.attn.o_proj.weight",
        },
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

