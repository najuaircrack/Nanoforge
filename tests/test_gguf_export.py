from __future__ import annotations

import struct

import pytest

torch = pytest.importorskip("torch")

from nanoforge.config import (  # noqa: E402
    DataConfig,
    InferenceConfig,
    ModelConfig,
    NanoforgeConfig,
    TrainConfig,
)
from nanoforge.export.gguf import (  # noqa: E402
    GGUF_DEFAULT_ALIGNMENT,
    GGUF_MAGIC,
    GGUF_VERSION,
    GGUFValueType,
    export_gguf,
)
from nanoforge.model.transformer import build_model  # noqa: E402


def _tiny_config() -> NanoforgeConfig:
    return NanoforgeConfig(
        model=ModelConfig(
            vocab_size=64, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2,
            max_seq_len=64, ffn_hidden_mult=2.0,
        ),
        training=TrainConfig(),
        data=DataConfig(),
        inference=InferenceConfig(),
    )


def _read_str(f) -> str:
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n).decode("utf-8")


def _skip_value(f, vtype: int) -> None:
    sizes = {
        GGUFValueType.UINT8: 1, GGUFValueType.INT8: 1, GGUFValueType.BOOL: 1,
        GGUFValueType.UINT16: 2, GGUFValueType.INT16: 2,
        GGUFValueType.UINT32: 4, GGUFValueType.INT32: 4, GGUFValueType.FLOAT32: 4,
        GGUFValueType.UINT64: 8, GGUFValueType.INT64: 8, GGUFValueType.FLOAT64: 8,
    }
    if vtype == GGUFValueType.STRING:
        _read_str(f)
    elif vtype == GGUFValueType.ARRAY:
        (etype,) = struct.unpack("<I", f.read(4))
        (count,) = struct.unpack("<Q", f.read(8))
        for _ in range(count):
            _skip_value(f, etype)
    else:
        f.read(sizes[vtype])


def test_gguf_export_is_valid_binary(tmp_path):
    cfg = _tiny_config()
    model = build_model(cfg).eval()
    out = tmp_path / "model.gguf"
    export_gguf(model, cfg, out, dtype="f32")

    assert out.exists() and out.stat().st_size > 0

    with out.open("rb") as f:
        (magic,) = struct.unpack("<I", f.read(4))
        (version,) = struct.unpack("<I", f.read(4))
        (n_tensors,) = struct.unpack("<Q", f.read(8))
        (n_kv,) = struct.unpack("<Q", f.read(8))

        assert magic == GGUF_MAGIC
        assert version == GGUF_VERSION
        assert n_tensors > 0
        assert n_kv > 0

        # Parse all metadata KV pairs (proves the KV section is well-formed)
        keys = []
        for _ in range(n_kv):
            key = _read_str(f)
            (vtype,) = struct.unpack("<I", f.read(4))
            _skip_value(f, vtype)
            keys.append(key)

        assert "general.architecture" in keys
        assert "nanoforge.block_count" in keys

        # Parse tensor info blocks
        tensor_names = []
        max_offset = 0
        for _ in range(n_tensors):
            name = _read_str(f)
            (n_dims,) = struct.unpack("<I", f.read(4))
            dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
            (_qtype,) = struct.unpack("<I", f.read(4))
            (offset,) = struct.unpack("<Q", f.read(8))
            assert all(d > 0 for d in dims)
            assert offset % GGUF_DEFAULT_ALIGNMENT == 0  # aligned
            max_offset = max(max_offset, offset)
            tensor_names.append(name)

    # Standard llama.cpp tensor names must be present
    assert "token_embd.weight" in tensor_names
    assert "output_norm.weight" in tensor_names
    assert any(n.startswith("blk.0.attn_q") for n in tensor_names)


def test_gguf_export_f16(tmp_path):
    cfg = _tiny_config()
    model = build_model(cfg).eval()
    out = tmp_path / "model_f16.gguf"
    export_gguf(model, cfg, out, dtype="f16")
    with out.open("rb") as f:
        (magic,) = struct.unpack("<I", f.read(4))
    assert magic == GGUF_MAGIC
    assert out.stat().st_size > 0
