# Training Guide

Nanoforge supports three main local training styles:

- `generative`: plain next-token text or story continuation.
- `chat`: role-based conversations with assistant-only labels.
- `instruct`: instruction/response pairs with completion-only labels.

For Windows 11 CPU-only training with 8GB RAM, start with `native-bpe`, `vocab_size=8000`,
`seq_len=512` for chat/instruct, and `seq_len=1024` for plain generative text.

## Fast Path

Use the automatic pipeline:

```powershell
nanoforge auto-train ^
  --input data/raw ^
  --name ultrachat-18m ^
  --mode chat ^
  --tokenizer native-bpe ^
  --vocab-size 8000 ^
  --text-column messages ^
  --seq-len 512
```

Add `--no-train` to stop after tokenizer training, data packing, and config generation.

## Manual Path

Train a tokenizer:

```powershell
nanoforge train-tokenizer ^
  --input data/raw ^
  --type native-bpe ^
  --vocab-size 8000 ^
  --text-column messages ^
  --out data/tokenizers/tiny-bpe.json
```

Prepare chat data:

```powershell
nanoforge prepare ^
  --input data/raw ^
  --tokenizer native-bpe ^
  --tokenizer-path data/tokenizers/tiny-bpe.json ^
  --mode chat ^
  --loss-masking assistant_only ^
  --text-column messages ^
  --seq-len 512 ^
  --out data/packed/ultrachat
```

Train:

```powershell
nanoforge train --config configs/ultrachat.yaml
```

## Boundary-Aware Chat Packing

Chat and instruct packing writes fixed sequences instead of sampling arbitrary windows from one
long stream. Each sequence starts from a conversation boundary, includes user/context tokens,
and labels assistant/completion tokens only. Long assistant responses are split into multiple
sequences with the user/context prefix repeated so batches do not become 100% unmasked.

Healthy chat/instruct packing:

- 40-70% masked labels,
- 30-60% unmasked labels,
- no all-masked sampled batches,
- no all-unmasked sampled batches.

## Healthy Training Signs

- `vocab_size=8000` should start near loss `8-10`.
- Loss should decrease over the first few hundred steps.
- Gradient norm before clipping should usually stay below `5.0`.
- `skip=no` should appear on most steps.
- Stop and inspect data if loss is `0.000`, `NaN`, or starts around `30-60`.

## CPU Presets

- 4GB RAM: `d_model=192`, `n_layers=3`, `n_heads=3`, `seq_len=384`.
- 8GB RAM fast chat: `d_model=256`, `n_layers=4`, `n_heads=4`, `seq_len=512`.
- 8GB RAM balanced: `d_model=384`, `n_layers=6`, `n_heads=6`, `seq_len=1024`.
- 16GB RAM: `d_model=512`, `n_layers=8`, `n_heads=8`, `seq_len=1024`.

Use `nanoforge new-config` for an interactive config generator.
