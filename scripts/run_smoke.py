"""Run a tiny authored-data CPU pipeline in a new output directory."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from make_public_fixtures import write_fixtures  # noqa: E402

from nanoforge.config import (  # noqa: E402
    DataConfig,
    ModelConfig,
    NanoforgeConfig,
    TrainConfig,
    save_config,
)
from nanoforge.data.dataset import PackedMemmapDataset, build_packed_dataset  # noqa: E402
from nanoforge.data.tokenizer import ByteTokenizer  # noqa: E402
from nanoforge.evaluation.metrics import evaluate_checkpoint  # noqa: E402
from nanoforge.training.trainer import Trainer  # noqa: E402


def run_smoke(out_dir: str | Path, *, steps: int = 3) -> dict:
    if steps < 1:
        raise ValueError("steps must be positive")
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = out / "authored-data"
    write_fixtures(source, kind="c")
    packed = out / "packed"
    build_packed_dataset([source], packed, ByteTokenizer(), val_fraction=0.25,
                         seq_len=64, mode="fim", loss_masking="none", seed=42)
    cfg = NanoforgeConfig(
        model=ModelConfig(vocab_size=260, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2,
                          max_seq_len=64, ffn_hidden_mult=2, use_flash=False),
        training=TrainConfig(run_name="public-smoke", output_dir=str(out / "run"), seed=42,
                             max_steps=steps, batch_size=1, micro_batch_size=1, grad_accum_steps=1,
                             warmup_steps=0, eval_interval=1, eval_steps=1, save_interval=1,
                             log_interval=1, health_interval=1, gradient_checkpointing=False,
                             tensorboard=False, device="cpu"),
        data=DataConfig(train_path=str(packed / "train.bin"), val_path=str(packed / "val.bin"),
                        tokenizer_type="byte", seq_len=64, mode="fim", loss_masking="none",
                        num_workers=0, pin_memory=False),
    )
    save_config(cfg, out / "smoke.yaml")
    with PackedMemmapDataset(packed / "train.bin", 64) as train:
        train_tokens = sum(len(tokens) for tokens in train.tokens)
    with PackedMemmapDataset(packed / "val.bin", 64) as val:
        val_tokens = sum(len(tokens) for tokens in val.tokens)
    trainer = Trainer(cfg)
    trainer.train()
    checkpoint = out / "run" / "last.pt"
    metrics = evaluate_checkpoint(checkpoint, packed / "val.bin", seq_len=64, batches=1, device="cpu")
    result = {"train_tokens": train_tokens, "val_tokens": val_tokens,
              "checkpoint": checkpoint, "eval_loss": metrics["loss"]}
    (out / "smoke-result.json").write_text(
        json.dumps({**result, "checkpoint": str(checkpoint.relative_to(out)),
                    "scope": "pipeline correctness; no coding quality claim"}, indent=2) + "\n",
        encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("runs/public-smoke"))
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()
    result = run_smoke(args.out, steps=args.steps)
    print(f"Smoke passed: held-out loss={result['eval_loss']:.4f}; output={args.out}")
    print("This verifies the pipeline, not useful general coding ability.")


if __name__ == "__main__":
    main()

