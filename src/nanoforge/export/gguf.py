from __future__ import annotations

import json
import struct
from dataclasses import asdict
from enum import IntEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nanoforge.config import NanoforgeConfig

if TYPE_CHECKING:
    import torch

GGUF_MAGIC = 0x46554747  # "GGUF" little-endian
GGUF_VERSION = 3
GGUF_DEFAULT_ALIGNMENT = 32


class GGUFValueType(IntEnum):
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


class GGMLQuantType(IntEnum):
    F32 = 0
    F16 = 1



def _pack_str(s: str) -> bytes:
    encoded = s.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _pack_value(vtype: GGUFValueType, value: Any) -> bytes:
    if vtype == GGUFValueType.UINT8:
        return struct.pack("<B", value)
    if vtype == GGUFValueType.INT8:
        return struct.pack("<b", value)
    if vtype == GGUFValueType.UINT16:
        return struct.pack("<H", value)
    if vtype == GGUFValueType.INT16:
        return struct.pack("<h", value)
    if vtype == GGUFValueType.UINT32:
        return struct.pack("<I", value)
    if vtype == GGUFValueType.INT32:
        return struct.pack("<i", value)
    if vtype == GGUFValueType.FLOAT32:
        return struct.pack("<f", value)
    if vtype == GGUFValueType.BOOL:
        return struct.pack("<B", int(bool(value)))
    if vtype == GGUFValueType.STRING:
        return _pack_str(str(value))
    if vtype == GGUFValueType.UINT64:
        return struct.pack("<Q", value)
    if vtype == GGUFValueType.INT64:
        return struct.pack("<q", value)
    if vtype == GGUFValueType.FLOAT64:
        return struct.pack("<d", value)
    raise ValueError(f"Unsupported GGUF value type: {vtype}")


def _pack_kv(key: str, vtype: GGUFValueType, value: Any) -> bytes:
    return _pack_str(key) + struct.pack("<I", int(vtype)) + _pack_value(vtype, value)


def _pack_kv_array(key: str, elem_type: GGUFValueType, values: list[Any]) -> bytes:
    header = _pack_str(key) + struct.pack("<I", int(GGUFValueType.ARRAY))
    array_header = struct.pack("<I", int(elem_type)) + struct.pack("<Q", len(values))
    body = b"".join(_pack_value(elem_type, v) for v in values)
    return header + array_header + body


# Nanoforge state-dict key → GGUF tensor name
_TENSOR_MAP = {
    "backbone.embedding.embed.weight": "token_embd.weight",
    "backbone.norm.weight": "output_norm.weight",
    "backbone.norm.bias": "output_norm.bias",
    "head.lm_head.weight": "output.weight",
}


def _gguf_tensor_name(key: str) -> str | None:
    """Map a Nanoforge state-dict key to a GGUF tensor name, or None to skip."""
    if key in _TENSOR_MAP:
        return _TENSOR_MAP[key]
    # backbone.blocks.{i}.attn_norm.weight  → blk.{i}.attn_norm.weight
    # backbone.blocks.{i}.ffn_norm.weight   → blk.{i}.ffn_norm.weight
    # backbone.blocks.{i}.attn.q_proj.weight → blk.{i}.attn_q.weight
    # backbone.blocks.{i}.attn.k_proj.weight → blk.{i}.attn_k.weight
    # backbone.blocks.{i}.attn.v_proj.weight → blk.{i}.attn_v.weight
    # backbone.blocks.{i}.attn.o_proj.weight → blk.{i}.attn_output.weight
    # backbone.blocks.{i}.ffn.w1.weight      → blk.{i}.ffn_gate.weight
    # backbone.blocks.{i}.ffn.w2.weight      → blk.{i}.ffn_down.weight
    # backbone.blocks.{i}.ffn.w3.weight      → blk.{i}.ffn_up.weight
    import re
    m = re.match(r"backbone\.blocks\.(\d+)\.(.*)", key)
    if not m:
        return None
    idx, rest = m.group(1), m.group(2)
    sub = {
        "attn_norm.weight": "attn_norm.weight",
        "attn_norm.bias": "attn_norm.bias",
        "ffn_norm.weight": "ffn_norm.weight",
        "ffn_norm.bias": "ffn_norm.bias",
        "attn.q_proj.weight": "attn_q.weight",
        "attn.k_proj.weight": "attn_k.weight",
        "attn.v_proj.weight": "attn_v.weight",
        "attn.o_proj.weight": "attn_output.weight",
        "ffn.w1.weight": "ffn_gate.weight",
        "ffn.w2.weight": "ffn_down.weight",
        "ffn.w3.weight": "ffn_up.weight",
        "ffn.gate.weight": "ffn_gate.weight",
        "ffn.up.weight": "ffn_up.weight",
        "ffn.down.weight": "ffn_down.weight",
    }
    mapped = sub.get(rest)
    return f"blk.{idx}.{mapped}" if mapped else None


def export_gguf(
    model: "torch.nn.Module",
    config: NanoforgeConfig,
    out_path: str | Path,
    *,
    dtype: str = "f32",
) -> None:
    """Write a llama.cpp-compatible binary GGUF file from a Nanoforge model.

    Supports f32 and f16 tensor dtypes. The resulting file can be loaded by
    llama.cpp and any tool that reads GGUF v3.
    """
    import torch

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mc = config.model
    quant_type = GGMLQuantType.F16 if dtype == "f16" else GGMLQuantType.F32
    torch_dtype = torch.float16 if dtype == "f16" else torch.float32

    # Collect tensors
    state = model.state_dict()
    tensors: list[tuple[str, "torch.Tensor"]] = []
    for sd_key, tensor in state.items():
        gguf_name = _gguf_tensor_name(sd_key)
        if gguf_name is None:
            continue
        tensors.append((gguf_name, tensor.to(torch_dtype).contiguous().cpu()))

    # Build metadata KV pairs
    kv_bytes = b""
    kv_count = 0

    def kv(key: str, vtype: GGUFValueType, value: Any) -> None:
        nonlocal kv_bytes, kv_count
        kv_bytes += _pack_kv(key, vtype, value)
        kv_count += 1

    def kv_arr(key: str, etype: GGUFValueType, values: list[Any]) -> None:
        nonlocal kv_bytes, kv_count
        kv_bytes += _pack_kv_array(key, etype, values)
        kv_count += 1

    kv("general.architecture", GGUFValueType.STRING, "nanoforge")
    kv("general.name", GGUFValueType.STRING, mc.task_type)
    kv("nanoforge.context_length", GGUFValueType.UINT32, mc.max_seq_len)
    kv("nanoforge.embedding_length", GGUFValueType.UINT32, mc.d_model)
    kv("nanoforge.block_count", GGUFValueType.UINT32, mc.n_layers)
    kv("nanoforge.feed_forward_length", GGUFValueType.UINT32, int(mc.d_model * mc.ffn_hidden_mult))
    kv("nanoforge.attention.head_count", GGUFValueType.UINT32, mc.n_heads)
    kv("nanoforge.attention.head_count_kv", GGUFValueType.UINT32, mc.n_kv_heads or mc.n_heads)
    kv("nanoforge.attention.layer_norm_rms_epsilon", GGUFValueType.FLOAT32, mc.norm_eps)
    kv("nanoforge.rope.freq_base", GGUFValueType.FLOAT32, mc.rope_theta)
    kv("tokenizer.ggml.model", GGUFValueType.STRING, mc.task_type)
    kv("tokenizer.ggml.bos_token_id", GGUFValueType.UINT32, 1)
    kv("tokenizer.ggml.eos_token_id", GGUFValueType.UINT32, 2)
    kv("general.file_type", GGUFValueType.UINT32, int(quant_type))

    # Build tensor info block
    tensor_info_bytes = b""
    tensor_data_parts: list[bytes] = []
    current_offset = 0

    for name, tensor in tensors:
        raw = tensor.numpy().tobytes()
        # Align to GGUF_DEFAULT_ALIGNMENT
        pad = (GGUF_DEFAULT_ALIGNMENT - (current_offset % GGUF_DEFAULT_ALIGNMENT)) % GGUF_DEFAULT_ALIGNMENT
        tensor_data_parts.append(b"\x00" * pad)
        current_offset += pad

        shape = list(tensor.shape)
        n_dims = len(shape)
        tensor_info_bytes += _pack_str(name)
        tensor_info_bytes += struct.pack("<I", n_dims)
        for dim in reversed(shape):  # GGUF stores dims in reverse (innermost first)
            tensor_info_bytes += struct.pack("<Q", dim)
        tensor_info_bytes += struct.pack("<I", int(quant_type))
        tensor_info_bytes += struct.pack("<Q", current_offset)

        tensor_data_parts.append(raw)
        current_offset += len(raw)

    tensor_data = b"".join(tensor_data_parts)

    # Write file
    with out_path.open("wb") as f:
        # Header
        f.write(struct.pack("<I", GGUF_MAGIC))
        f.write(struct.pack("<I", GGUF_VERSION))
        f.write(struct.pack("<Q", len(tensors)))   # tensor count
        f.write(struct.pack("<Q", kv_count))        # metadata kv count
        # Metadata KV
        f.write(kv_bytes)
        # Tensor info
        f.write(tensor_info_bytes)
        # Align before tensor data
        pos = f.tell()
        pad = (GGUF_DEFAULT_ALIGNMENT - (pos % GGUF_DEFAULT_ALIGNMENT)) % GGUF_DEFAULT_ALIGNMENT
        f.write(b"\x00" * pad)
        # Tensor data
        f.write(tensor_data)


def write_gguf_manifest(config: NanoforgeConfig, checkpoint_path: str | Path, out_path: str | Path) -> None:
    """Write converter metadata for GGUF workflows (legacy back-compat).

    For a real binary GGUF file use ``export_gguf`` instead.
    """
    payload = {
        "format": "nanoforge-gguf-manifest-v1",
        "checkpoint": str(checkpoint_path),
        "architecture": "nanoforge.decoder_only",
        "config": asdict(config),
        "tensor_mapping": {
            "token_embd.weight": "backbone.embedding.embed.weight",
            "output.weight": "head.lm_head.weight",
            "blk.*.attn_q.weight": "backbone.blocks.*.attn.q_proj.weight",
            "blk.*.attn_k.weight": "backbone.blocks.*.attn.k_proj.weight",
            "blk.*.attn_v.weight": "backbone.blocks.*.attn.v_proj.weight",
            "blk.*.attn_output.weight": "backbone.blocks.*.attn.o_proj.weight",
        },
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
