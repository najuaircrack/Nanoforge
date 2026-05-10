"""
check_data.py  —  nanoforge data pipeline diagnostic
Run from your project root:
    python check_data.py --config configs/ultrachat-18m.yaml

Checks everything that can cause loss=70:
  1. Files exist and are readable
  2. Tokenizer loads and round-trips correctly
  3. Binary dataset shape and dtype
  4. Label file presence and alignment
  5. Unmasked token rate per batch (the main culprit)
  6. Initial model loss  (should be near log(vocab_size) ≈ 8.99)
  7. Logit scale / variance at init
  8. Sample decode of real batches so you can eyeball them
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

# ── allow running from project root without install ──────────────────────────
sys.path.insert(0, str(Path(__file__).parent / "src"))

# ─────────────────────────────────────────────────────────────────────────────
SEP = "─" * 70

def ok(msg):   print(f"  ✓  {msg}")
def warn(msg): print(f"  ⚠  {msg}")
def fail(msg): print(f"  ✗  {msg}")
def hdr(msg):  print(f"\n{SEP}\n  {msg}\n{SEP}")

# ─────────────────────────────────────────────────────────────────────────────
def check_files(cfg):
    hdr("1 · File existence")
    paths = {
        "train tokens": cfg.data.train_path,
        "val tokens":   cfg.data.val_path,
        "tokenizer":    cfg.data.tokenizer_path,
    }
    all_ok = True
    for name, p in paths.items():
        p = Path(p)
        if p.exists():
            size_mb = p.stat().st_size / 1e6
            ok(f"{name}: {p}  ({size_mb:.1f} MB)")
        else:
            fail(f"{name} NOT FOUND: {p}")
            all_ok = False

    # label files (optional but critical for assistant_only)
    train_labels = Path(cfg.data.train_path).with_name(
        Path(cfg.data.train_path).stem + ".labels.manifest.json"
    )
    if train_labels.exists():
        ok(f"label manifest found: {train_labels}")
    else:
        warn(
            "No label manifest found. If loss_masking=assistant_only, "
            "all tokens will be treated as targets → loss will be wrong."
        )
    return all_ok


# ─────────────────────────────────────────────────────────────────────────────
def check_tokenizer(cfg):
    hdr("2 · Tokenizer")
    from nanoforge.data.tokenizer import load_tokenizer

    tok = load_tokenizer(cfg.data.tokenizer_type, cfg.data.tokenizer_path)
    ok(f"type={cfg.data.tokenizer_type}  vocab_size={tok.vocab_size}")
    ok(f"special ids  bos={tok.bos_id}  eos={tok.eos_id}  pad={tok.pad_id}")

    # round-trip test
    sample = "<|user|>\nHello, how are you?\n<|assistant|>\nI'm doing great!"
    ids = tok.encode(sample, add_bos=True, add_eos=True)
    decoded = tok.decode(ids)
    ok(f"round-trip encode→decode  ({len(ids)} tokens)")
    print(f"     original : {sample!r}")
    print(f"     decoded  : {decoded!r}")
    if sample not in decoded and decoded not in sample:
        warn("decoded text does not match original — tokenizer may be broken")

    # check role markers survive tokenization
    for marker in ["<|user|>", "<|assistant|>"]:
        ids_m = tok.encode(marker, add_bos=False, add_eos=False)
        back  = tok.decode(ids_m)
        if marker in back:
            ok(f"role marker survives tokenization: {marker!r}")
        else:
            warn(f"role marker lost after tokenization: {marker!r} → {back!r}  "
                 "(assistant_only masking will break)")

    expected_initial_loss = math.log(tok.vocab_size)
    print(f"\n  Expected initial loss ≈ log({tok.vocab_size}) = {expected_initial_loss:.4f}")
    return tok


# ─────────────────────────────────────────────────────────────────────────────
def check_dataset(cfg, tok, split="train", n_samples=200):
    hdr(f"3 · Dataset binary — {split}")
    from nanoforge.data.dataset import PackedMemmapDataset

    path = cfg.data.train_path if split == "train" else cfg.data.val_path
    try:
        ds = PackedMemmapDataset(path, cfg.data.seq_len)
    except Exception as e:
        fail(f"Failed to load dataset: {e}")
        return None

    ok(f"loaded  shards={len(ds.tokens)}  total_len={ds.__len__()}")
    for i, t in enumerate(ds.tokens):
        ok(f"  shard {i}: {len(t)} tokens  dtype={t.dtype}")

    if ds.labels is not None:
        ok(f"label arrays found ({len(ds.labels)} shards)")
        for i, (t, l) in enumerate(zip(ds.tokens, ds.labels)):
            if len(t) != len(l):
                fail(f"  shard {i}: token len {len(t)} ≠ label len {len(l)}")
            else:
                ok(f"  shard {i}: token/label lengths match ({len(t)})")
    else:
        warn("No label arrays — labels will be auto-shifted input_ids (no masking).")

    # sample analysis
    total_tokens = 0
    total_unmasked = 0
    all_masked_batches = 0

    for _ in range(n_samples):
        x, y = ds.sample(1)
        total_tokens  += y.numel()
        unmasked       = (y != -100).sum().item()
        total_unmasked += unmasked
        if unmasked == 0:
            all_masked_batches += 1

    mask_rate = total_unmasked / max(total_tokens, 1)
    print(f"\n  Sampled {n_samples} batches  (seq_len={cfg.data.seq_len})")

    if mask_rate == 0.0:
        fail(
            f"Unmasked token rate = 0%  ← THIS IS YOUR BUG.\n"
            f"     Every label is -100. The model computes zero loss, "
            f"gradients are garbage.\n"
            f"     Your packed data has no assistant tokens or the label\n"
            f"     file is all -100. Re-run `nanoforge prepare` with\n"
            f"     --mode chat --loss-masking assistant_only."
        )
    elif mask_rate < 0.05:
        warn(
            f"Unmasked rate = {mask_rate*100:.1f}%  (very low).\n"
            f"     Only {mask_rate*100:.1f}% of tokens contribute to loss.\n"
            f"     Training will be very noisy and slow."
        )
    else:
        ok(f"Unmasked token rate = {mask_rate*100:.1f}%  ({total_unmasked}/{total_tokens})")

    if all_masked_batches > 0:
        pct = all_masked_batches / n_samples * 100
        warn(f"{all_masked_batches}/{n_samples} batches ({pct:.1f}%) had zero unmasked tokens")
    else:
        ok("No all-masked batches found")

    return ds


# ─────────────────────────────────────────────────────────────────────────────
def decode_samples(cfg, ds, tok, n=3):
    hdr("4 · Decoded batch samples (eyeball check)")
    for i in range(n):
        x, y = ds.sample(1)
        x_ids = x[0].tolist()
        y_ids = y[0].tolist()

        input_text  = tok.decode([t for t in x_ids if t not in (tok.pad_id,)])
        target_ids  = [t for t in y_ids if t != -100]
        target_text = tok.decode(target_ids) if target_ids else "(all masked)"

        unmasked = len(target_ids)
        total    = len(y_ids)
        pct      = unmasked / total * 100

        print(f"\n  ── sample {i+1}  unmasked={unmasked}/{total} ({pct:.0f}%) ──")
        print(f"  INPUT  (first 120 chars): {input_text[:120]!r}")
        print(f"  TARGET (first 120 chars): {target_text[:120]!r}")


# ─────────────────────────────────────────────────────────────────────────────
def check_model_loss(cfg, ds):
    hdr("5 · Model initial loss and logit scale")
    from nanoforge.model.transformer import NanoforgeForCausalLM

    device = torch.device("cpu")
    model  = NanoforgeForCausalLM(cfg.model).to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    ok(f"model params = {n_params:,}  ({n_params/1e6:.2f}M)")
    ok(f"vocab_size={cfg.model.vocab_size}  d_model={cfg.model.d_model}  "
       f"n_layers={cfg.model.n_layers}  tie_embeddings={cfg.model.tie_embeddings}")

    expected = math.log(cfg.model.vocab_size)
    print(f"  Expected initial loss ≈ {expected:.4f}")

    losses, logit_stds, unmasked_counts = [], [], []
    with torch.no_grad():
        for _ in range(10):
            x, y = ds.sample(1)
            out  = model(x, labels=y)
            loss = out.loss

            n_unmasked = (y != -100).sum().item()
            unmasked_counts.append(n_unmasked)

            logit_std = out.logits.std().item()
            logit_stds.append(logit_std)

            if loss is not None and torch.isfinite(loss):
                losses.append(loss.item())

    mean_loss    = sum(losses)  / max(len(losses), 1)
    mean_std     = sum(logit_stds) / len(logit_stds)
    mean_unmasked = sum(unmasked_counts) / len(unmasked_counts)

    print(f"\n  Results over 10 forward passes:")
    print(f"    mean loss        = {mean_loss:.4f}  (expected ≈ {expected:.4f})")
    print(f"    mean logit std   = {mean_std:.4f}  (expected ≈ 1–3; >10 is bad)")
    print(f"    mean unmasked/batch = {mean_unmasked:.1f} / {cfg.data.seq_len}")

    # verdict
    if mean_unmasked < 1.0:
        fail("All batches had zero unmasked tokens → loss is meaningless (0 or nan)")
    elif abs(mean_loss - expected) < 2.0:
        ok(f"Initial loss {mean_loss:.4f} is close to expected {expected:.4f}  ✓")
    elif mean_loss > expected * 3:
        fail(
            f"Initial loss {mean_loss:.4f} is {mean_loss/expected:.1f}× expected.\n"
            f"     Causes: huge logit variance (logit_std={mean_std:.2f}), "
            f"bad weight init, or near-zero unmasked tokens."
        )
    else:
        warn(f"Initial loss {mean_loss:.4f} is somewhat above expected {expected:.4f}")

    if mean_std > 10:
        warn(
            f"Logit std = {mean_std:.2f} is very large.\n"
            f"     The embedding scale + tied lm_head is amplifying variance.\n"
            f"     Try: logit temperature scaling or lower embed init std."
        )
    elif mean_std > 5:
        warn(f"Logit std = {mean_std:.2f} is moderately large (ideal: 1–3)")
    else:
        ok(f"Logit std = {mean_std:.4f}  (looks healthy)")


# ─────────────────────────────────────────────────────────────────────────────
def check_residual_scale(cfg):
    hdr("6 · Residual scale sanity")
    rs = cfg.model.residual_scale
    computed = 1.0 / math.sqrt(2 * cfg.model.n_layers)
    if rs:
        print(f"  residual_scale = {rs} (explicit)")
    else:
        print(f"  residual_scale = None → computed = 1/√(2×{cfg.model.n_layers}) = {computed:.4f}")
        rs = computed

    if rs < 0.2:
        warn(
            f"residual_scale={rs:.4f} is very small. "
            f"Residual stream is heavily dampened → large initial logit variance."
        )
    else:
        ok(f"residual_scale={rs:.4f} looks fine")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="nanoforge data diagnostic")
    parser.add_argument("--config", default="configs/ultrachat-18m.yaml",
                        help="Path to your .yaml config file")
    parser.add_argument("--samples", type=int, default=200,
                        help="Number of batches to sample for mask rate check")
    args = parser.parse_args()

    print(f"\n{'═'*70}")
    print(f"  nanoforge data diagnostic")
    print(f"  config: {args.config}")
    print(f"{'═'*70}")

    from nanoforge.config import load_config
    cfg = load_config(args.config)

    files_ok = check_files(cfg)
    if not files_ok:
        print("\n  Cannot continue — fix missing files first.\n")
        sys.exit(1)

    tok = check_tokenizer(cfg)
    ds  = check_dataset(cfg, tok, split="train", n_samples=args.samples)

    if ds is None:
        print("\n  Cannot continue — dataset failed to load.\n")
        sys.exit(1)

    decode_samples(cfg, ds, tok, n=3)
    check_residual_scale(cfg)
    check_model_loss(cfg, ds)

    print(f"\n{'═'*70}")
    print("  Diagnostic complete.")
    print("  Fix all  ✗  items first, then  ⚠  warnings.")
    print(f"{'═'*70}\n")


if __name__ == "__main__":
    main()