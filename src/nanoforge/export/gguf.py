from __future__ import annotations

import base64
import copy
import json
import os
import struct
import tempfile
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
    "head.proj.weight": "output.weight",
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
    """Write a faithful Nanoforge GGUF v3 container, reloadable by Nanoforge.

    This custom architecture is not a llama.cpp-compatible architecture.
    Unsupported learned tensors fail explicitly rather than disappearing.
    """
    import torch

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mc = config.model
    quant_type = GGMLQuantType.F16 if dtype == "f16" else GGMLQuantType.F32
    torch_dtype = torch.float16 if dtype == "f16" else torch.float32

    # Collect tensors
    if dtype not in {"f32", "f16"}:
        raise ValueError("GGUF tensor dtype must be f32 or f16")
    state = collect_export_tensors(model, config)
    tensors: list[tuple[str, "torch.Tensor"]] = []
    for gguf_name, tensor in state.items():
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
    saved = copy.deepcopy(config)
    saved.model.lora_rank = 0
    saved.model.quantization_backend = "none"
    kv("nanoforge.schema_version", GGUFValueType.UINT32, 1)
    kv("nanoforge.config", GGUFValueType.STRING, json.dumps(asdict(saved)))
    kv("nanoforge.tokenizer", GGUFValueType.STRING, json.dumps(_tokenizer_payload(config)))
    kv("general.name", GGUFValueType.STRING, mc.task_type)
    kv("nanoforge.context_length", GGUFValueType.UINT32, mc.max_seq_len)
    kv("nanoforge.embedding_length", GGUFValueType.UINT32, mc.d_model)
    kv("nanoforge.block_count", GGUFValueType.UINT32, mc.n_layers)
    kv("nanoforge.feed_forward_length", GGUFValueType.UINT32, int(mc.d_model * mc.ffn_hidden_mult))
    kv("nanoforge.attention.head_count", GGUFValueType.UINT32, mc.n_heads)
    kv("nanoforge.attention.head_count_kv", GGUFValueType.UINT32, mc.n_kv_heads or mc.n_heads)
    kv("nanoforge.attention.layer_norm_rms_epsilon", GGUFValueType.FLOAT32, mc.norm_eps)
    kv("nanoforge.rope.freq_base", GGUFValueType.FLOAT32, mc.rope_theta)
    kv("general.file_type", GGUFValueType.UINT32, int(quant_type))

    # Build tensor info block
    tensor_info_bytes = b""
    current_offset = 0

    for name, tensor in tensors:
        # Align to GGUF_DEFAULT_ALIGNMENT
        pad = (
            GGUF_DEFAULT_ALIGNMENT - (current_offset % GGUF_DEFAULT_ALIGNMENT)
        ) % GGUF_DEFAULT_ALIGNMENT
        current_offset += pad

        shape = list(tensor.shape)
        n_dims = len(shape)
        tensor_info_bytes += _pack_str(name)
        tensor_info_bytes += struct.pack("<I", n_dims)
        for dim in reversed(shape):  # GGUF stores dims in reverse (innermost first)
            tensor_info_bytes += struct.pack("<Q", dim)
        tensor_info_bytes += struct.pack("<I", int(quant_type))
        tensor_info_bytes += struct.pack("<Q", current_offset)

        current_offset += tensor.numel() * tensor.element_size()

    # Write file
    descriptor, name = tempfile.mkstemp(
        prefix=out_path.name + ".", suffix=".tmp", dir=out_path.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        with temporary.open("wb") as f:
            # Header
            f.write(struct.pack("<I", GGUF_MAGIC))
            f.write(struct.pack("<I", GGUF_VERSION))
            f.write(struct.pack("<Q", len(tensors)))  # tensor count
            f.write(struct.pack("<Q", kv_count))  # metadata kv count
            # Metadata KV
            f.write(kv_bytes)
            # Tensor info
            f.write(tensor_info_bytes)
            # Align before tensor data
            pos = f.tell()
            pad = (GGUF_DEFAULT_ALIGNMENT - (pos % GGUF_DEFAULT_ALIGNMENT)) % GGUF_DEFAULT_ALIGNMENT
            f.write(b"\x00" * pad)
            # Tensor data
            for _, tensor in tensors:
                pad = (
                    GGUF_DEFAULT_ALIGNMENT - (f.tell() % GGUF_DEFAULT_ALIGNMENT)
                ) % GGUF_DEFAULT_ALIGNMENT
                f.write(b"\x00" * pad)
                f.write(tensor.numpy().tobytes())
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, out_path)
    finally:
        temporary.unlink(missing_ok=True)


def collect_export_tensors(model, config: NanoforgeConfig) -> dict[str, Any]:
    from nanoforge.model.lora import merge_lora

    if config.model.task_type != "causal_lm":
        raise ValueError("unsupported GGUF task head; only causal_lm is representable")
    merged = merge_lora(model)
    result = {}
    for name, tensor in merged.state_dict().items():
        mapped = _gguf_tensor_name(name)
        if mapped is None or not hasattr(tensor, "shape"):
            raise ValueError(f"unsupported learned GGUF tensor: {name}")
        if mapped in result:
            raise ValueError(f"duplicate GGUF tensor: {mapped}")
        result[mapped] = tensor.detach().cpu()
    return result


def _tokenizer_payload(config):
    payload = {"type": config.data.tokenizer_type}
    if config.data.tokenizer_path:
        path = Path(config.data.tokenizer_path)
        payload.update(data=base64.b64encode(path.read_bytes()).decode("ascii"), suffix=path.suffix)
    elif config.data.tokenizer_type not in {"byte", "byte-native"}:
        raise ValueError("GGUF export requires the actual tokenizer artifact")
    return payload


def _read_string(stream):
    raw = stream.read(8)
    if len(raw) != 8:
        raise ValueError("truncated GGUF string")
    length = struct.unpack("<Q", raw)[0]
    if length > 16 * 1024 * 1024:
        raise ValueError("GGUF metadata string exceeds limit")
    data = stream.read(length)
    if len(data) != length:
        raise ValueError("truncated GGUF string data")
    return data.decode("utf-8")


def _read_value(stream, kind):
    formats = {
        0: "B",
        1: "b",
        2: "H",
        3: "h",
        4: "I",
        5: "i",
        6: "f",
        7: "?",
        10: "Q",
        11: "q",
        12: "d",
    }
    if kind == 8:
        return _read_string(stream)
    if kind == 9:
        subtype, count = struct.unpack("<IQ", stream.read(12))
        if count > 1000000:
            raise ValueError("GGUF array exceeds limit")
        return [_read_value(stream, subtype) for _ in range(count)]
    if kind not in formats:
        raise ValueError("unsupported GGUF metadata type")
    fmt = "<" + formats[kind]
    return struct.unpack(fmt, stream.read(struct.calcsize(fmt)))[0]


def read_gguf_metadata(path):
    with Path(path).open("rb") as stream:
        magic, version, count, kv_count = struct.unpack("<IIQQ", stream.read(24))
        if magic != GGUF_MAGIC or version != 3 or kv_count > 100000 or count > 100000:
            raise ValueError("unsupported GGUF header")
        metadata = {}
        for _ in range(kv_count):
            key = _read_string(stream)
            kind = struct.unpack("<I", stream.read(4))[0]
            if key in metadata:
                raise ValueError("duplicate GGUF metadata key")
            metadata[key] = _read_value(stream, kind)
        return metadata


def load_nanoforge_gguf(path):
    """Reload the explicit Nanoforge architecture; reject foreign/legacy containers."""
    import numpy as np
    import torch

    from nanoforge.config import DataConfig, InferenceConfig, ModelConfig, TrainConfig
    from nanoforge.model.transformer import build_model

    metadata = read_gguf_metadata(path)
    if (
        metadata.get("general.architecture") != "nanoforge"
        or metadata.get("nanoforge.schema_version") != 1
    ):
        raise ValueError("GGUF is not a supported versioned Nanoforge container")
    raw = json.loads(metadata["nanoforge.config"])
    cfg = NanoforgeConfig(
        ModelConfig(**raw["model"]),
        TrainConfig(**raw["training"]),
        DataConfig(**raw["data"]),
        InferenceConfig(**raw["inference"]),
    )
    model = build_model(cfg)
    reverse = {_gguf_tensor_name(key): key for key in model.state_dict()}
    state = {}
    path = Path(path)
    with path.open("rb") as stream:
        _, _, tensor_count, kv_count = struct.unpack("<IIQQ", stream.read(24))
        for _ in range(kv_count):
            _read_string(stream)
            _read_value(stream, struct.unpack("<I", stream.read(4))[0])
        infos = []
        for _ in range(tensor_count):
            name = _read_string(stream)
            ndim = struct.unpack("<I", stream.read(4))[0]
            if ndim > 8:
                raise ValueError("unsupported GGUF tensor rank")
            shape = tuple(reversed([struct.unpack("<Q", stream.read(8))[0] for _ in range(ndim)]))
            kind, offset = struct.unpack("<IQ", stream.read(12))
            if name not in reverse or kind not in {0, 1}:
                raise ValueError(f"unsupported GGUF tensor: {name}")
            infos.append((name, shape, kind, offset))
        start = stream.tell() + (-stream.tell() % GGUF_DEFAULT_ALIGNMENT)
        expected = model.state_dict()
        for name, shape, kind, offset in infos:
            key = reverse[name]
            if shape != tuple(expected[key].shape) or key in state:
                raise ValueError(f"GGUF tensor shape/identity mismatch: {name}")
            dtype = np.dtype("<f4" if kind == 0 else "<f2")
            size = expected[key].numel() * dtype.itemsize
            if offset % 32 or start + offset + size > path.stat().st_size:
                raise ValueError("GGUF tensor data exceeds file bounds")
            stream.seek(start + offset)
            state[key] = torch.from_numpy(
                np.frombuffer(stream.read(size), dtype=dtype).copy()
            ).reshape(shape)
    model.load_state_dict(state, strict=True)
    tokenizer = restore_tokenizer_payload(json.loads(metadata["nanoforge.tokenizer"]))
    cfg.data.tokenizer_path = None
    return model.eval(), cfg, tokenizer


def restore_tokenizer_payload(token):
    from nanoforge.data.tokenizer import load_tokenizer

    suffix = token.get("suffix", ".json")
    if not isinstance(suffix, str) or not suffix.startswith(".") or not suffix[1:].isalnum() or len(suffix) > 16:
        raise ValueError("invalid embedded tokenizer suffix")

    if "data" in token:
        with tempfile.TemporaryDirectory() as temporary:
            token_path = Path(temporary) / ("tokenizer" + suffix)
            token_path.write_bytes(base64.b64decode(token["data"], validate=True))
            tokenizer = load_tokenizer(token["type"], token_path)
    else:
        tokenizer = load_tokenizer(token["type"], None)
    return tokenizer


def write_gguf_manifest(
    config: NanoforgeConfig, checkpoint_path: str | Path, out_path: str | Path
) -> None:
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
            "output.weight": "head.proj.weight",
            "blk.*.attn_q.weight": "backbone.blocks.*.attn.q_proj.weight",
            "blk.*.attn_k.weight": "backbone.blocks.*.attn.k_proj.weight",
            "blk.*.attn_v.weight": "backbone.blocks.*.attn.v_proj.weight",
            "blk.*.attn_output.weight": "backbone.blocks.*.attn.o_proj.weight",
        },
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
