from __future__ import annotations

import struct
from pathlib import Path


def _write_bin(path: Path, n: int = 4096) -> None:
    path.write_bytes(struct.pack(f"{n}H", *([1] * n)))


class TestDataMixture:
    def test_sample_shape(self, tmp_path):
        import torch

        from nanoforge.data.mixture import DataMixture, MixtureSource

        for name in ("a.bin", "b.bin"):
            _write_bin(tmp_path / name)

        mix = DataMixture(
            [MixtureSource(path=str(tmp_path / "a.bin"), weight=3.0, name="a"),
             MixtureSource(path=str(tmp_path / "b.bin"), weight=1.0, name="b")],
            seq_len=32,
        )
        x, y = mix.sample(8)
        assert isinstance(x, torch.Tensor)
        assert x.shape == (8, 32)
        assert y.shape == (8, 32)

    def test_describe_ratios(self, tmp_path):
        from nanoforge.data.mixture import DataMixture, MixtureSource

        for name in ("a.bin", "b.bin"):
            _write_bin(tmp_path / name)

        mix = DataMixture(
            [MixtureSource(path=str(tmp_path / "a.bin"), weight=3.0, name="a"),
             MixtureSource(path=str(tmp_path / "b.bin"), weight=1.0, name="b")],
            seq_len=32,
        )
        desc = mix.describe()
        assert abs(desc["a"] - 0.75) < 1e-9
        assert abs(desc["b"] - 0.25) < 1e-9

    def test_backward_compat_no_sources(self, tmp_path):
        from nanoforge.config import DataConfig
        from nanoforge.data.mixture import build_mixture_from_config

        cfg = DataConfig(train_path=str(tmp_path / "train.bin"), sources=None)
        assert build_mixture_from_config(cfg, seq_len=32) is None

    def test_build_mixture_from_config(self, tmp_path):
        from nanoforge.config import DataConfig
        from nanoforge.data.mixture import build_mixture_from_config

        p = tmp_path / "shard.bin"
        _write_bin(p)
        cfg = DataConfig(sources=[{"path": str(p), "weight": 1.0, "name": "test"}])
        mix = build_mixture_from_config(cfg, seq_len=32)
        assert mix is not None
        assert len(mix.datasets) == 1


class TestPipeline:
    def _base_yaml(self, tmp: Path) -> Path:
        for name in ("train.bin", "val.bin"):
            _write_bin(tmp / name)
        p = tmp / "base.yaml"
        p.write_text(f"""
model:
  vocab_size: 260
  max_seq_len: 32
  d_model: 64
  n_layers: 2
  n_heads: 4
  n_kv_heads: 2
  tie_embeddings: true
  use_flash: false
  attention_backend: sdpa
  normalization: rmsnorm
  position_embedding: rope
  activation: swiglu
  block_type: transformer
  quantization_backend: none
  task_type: causal_lm
training:
  mode: pretrain
  run_name: test
  output_dir: {tmp}/runs
  seed: 1337
  max_steps: 2
  batch_size: 2
  micro_batch_size: 2
  grad_accum_steps: 1
  learning_rate: 1.0e-3
  min_learning_rate: 1.0e-4
  warmup_ratio: 0.0
  optimizer: adamw
  scheduler: cosine
  weight_decay: 0.1
  grad_clip: 1.0
  precision: fp32
  gradient_checkpointing: false
  eval_interval: 9999
  eval_steps: 1
  save_interval: 9999
  early_stopping_patience: 9999
  log_interval: 1
  health_interval: 1
  tensorboard: false
  wandb: false
  distributed_backend: none
data:
  train_path: {tmp}/train.bin
  val_path: {tmp}/val.bin
  tokenizer_type: byte
  seq_len: 32
  mode: pretrain
  loss_masking: none
  assistant_only_loss: false
  num_workers: 0
  pin_memory: false
inference:
  mode: balanced
""", encoding="utf-8")
        return p

    def test_apply_overrides_deep_copy(self, tmp_path):
        from nanoforge.config import load_config
        from nanoforge.training.pipeline import _apply_overrides

        cfg = load_config(self._base_yaml(tmp_path))
        out = _apply_overrides(cfg, {"training": {"max_steps": 99, "learning_rate": 5e-5}})
        assert out.training.max_steps == 99
        assert abs(out.training.learning_rate - 5e-5) < 1e-12
        assert cfg.training.max_steps == 2  # original unchanged

    def test_load_recipe(self, tmp_path):
        from nanoforge.training.pipeline import load_recipe

        base = self._base_yaml(tmp_path)
        recipe_path = tmp_path / "recipe.yaml"
        recipe_path.write_text(f"""
base_config: {base}
output_dir: {tmp_path}/out
stages:
  - name: pretrain
    overrides:
      training:
        max_steps: 2
  - name: sft
    overrides:
      training:
        max_steps: 2
        mode: chat
""", encoding="utf-8")
        recipe = load_recipe(recipe_path)
        assert len(recipe.stages) == 2
        assert recipe.stages[0].name == "pretrain"
        assert recipe.stages[1].overrides["training"]["mode"] == "chat"

    def test_weight_handoff_path(self, tmp_path):
        """_apply_overrides correctly sets resume_from_checkpoint."""
        import torch

        from nanoforge.config import load_config
        from nanoforge.training.pipeline import _apply_overrides

        cfg = load_config(self._base_yaml(tmp_path))
        fake_ckpt = tmp_path / "best.pt"
        torch.save({"step": 2}, fake_ckpt)
        out = _apply_overrides(cfg, {})
        out.training.resume_from_checkpoint = str(fake_ckpt)
        assert out.training.resume_from_checkpoint == str(fake_ckpt)
