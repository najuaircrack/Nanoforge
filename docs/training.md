# Training Guide

This is the single source of truth for training workflows. For the full flag list of any
command see [cli.md](cli.md); for config fields see [configuration.md](configuration.md).

Nanoforge trains three main styles:

- **generative** — plain next-token text/story continuation (no label masking).
- **chat** — role-based conversations, assistant-only labels.
- **instruct** — instruction/response pairs, completion-only labels.

For Windows 11 CPU-only training with ~8 GB RAM, a good starting point is `native-bpe`,
`vocab_size=8000`, `seq_len=512` for chat/instruct, `seq_len=1024` for generative text —
or the dependency-free `byte` tokenizer for the smallest experiments.

Examples use `^` (Windows) for line continuation; use `\` on Linux/macOS.

---

## Fast path: `auto-train`

One command inspects the dataset, infers/applies the mode, trains the tokenizer, packs data
with the right masking, writes a CPU-friendly config, and starts training:

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

This writes:

- tokenizer → `data/tokenizers/ultrachat-18m-bpe.json`
- packed data → `data/packed/ultrachat-18m/`
- config → `configs/ultrachat-18m.yaml`
- run outputs → `runs/ultrachat-18m/`

Add `--no-train` to stop after setup (tokenizer + packing + config), then train later:

```powershell
nanoforge train --config configs/ultrachat-18m.yaml
```

### Auto mode detection

With `--mode auto`, dataset fields decide the mode:

| Fields present | Mode |
| --- | --- |
| `messages` or `conversations` | `chat` |
| `instruction` + (`output` or `response`) | `instruct` |
| `code` | `code` |
| otherwise | `generative` |

For chat data, `auto-train` also defaults `--text-column messages` when that field exists.
Loss masking auto-resolves to `assistant_only` (chat), `completion_only` (instruct), or
`none` (generative).

---

## Manual path

Full control in three steps: train a tokenizer → pack data → train.

```powershell
:: 1. Train a tokenizer (skip for the byte tokenizer)
nanoforge train-tokenizer ^
  --input data/raw ^
  --type native-bpe ^
  --vocab-size 8000 ^
  --text-column messages ^
  --out data/tokenizers/chat-bpe.json

:: 2. Pack into token/label memmaps
nanoforge prepare ^
  --input data/raw ^
  --tokenizer native-bpe ^
  --tokenizer-path data/tokenizers/chat-bpe.json ^
  --mode chat ^
  --loss-masking assistant_only ^
  --text-column messages ^
  --seq-len 512 ^
  --out data/packed/chat

:: 3. Train
nanoforge train --config configs/my-chat.yaml
```

The byte-tokenizer smoke path skips step 1 entirely:

```powershell
nanoforge prepare --input data/raw --tokenizer byte --out data/packed/tiny
nanoforge train --config configs/tiny.yaml
```

---

## Recipes

### Generative (stories, articles, plain text)

```powershell
nanoforge prepare ^
  --input data/raw ^
  --tokenizer native-bpe --tokenizer-path data/tokenizers/story-bpe.json ^
  --mode generative --loss-masking none --seq-len 1024 --out data/packed/story

nanoforge train --config configs/stories-18m.yaml
nanoforge generate --checkpoint runs/stories-18m/best.pt --mode creative --prompt "The old city was quiet"
```

### Chat (UltraChat / ShareGPT / `messages` rows)

```powershell
nanoforge prepare ^
  --input data/raw ^
  --tokenizer native-bpe --tokenizer-path data/tokenizers/chat-bpe.json ^
  --mode chat --loss-masking assistant_only --text-column messages ^
  --seq-len 512 --out data/packed/chat

nanoforge train --config configs/ultrachat-18m.yaml
nanoforge chat --checkpoint runs/ultrachat-18m/best.pt --mode chat ^
  --temperature 0.75 --top-p 0.92 --repetition-penalty 1.1 --no-repeat-ngram-size 4
```

### Instruct (Alpaca-style `instruction`/`output`)

```powershell
nanoforge prepare ^
  --input data/alpaca.jsonl ^
  --tokenizer native-bpe --tokenizer-path data/tokenizers/chat-bpe.json ^
  --mode instruct --loss-masking completion_only --seq-len 512 --out data/packed/instruct

nanoforge train --config configs/my-instruct.yaml
```

### Import an external model (no training)

```powershell
nanoforge import --model path\to\model.gguf --name my-gguf
nanoforge chat --model my-gguf --mode chat
```

---

## Boundary-aware chat/instruct packing

Chat and instruct packing writes fixed-length sequences that start at conversation
boundaries rather than sampling arbitrary windows from one long stream. Each sequence
includes the user/system context and labels assistant/completion tokens only. Long assistant
responses are split across sequences with the context prefix repeated, so batches never
become 100% masked or 100% unmasked.

> Note: the boundary packer truncates each training prefix to roughly its last `seq_len/2`
> tokens. Keep system prompts short so the assistant marker lands at a position the model saw
> during training (see [nano_chat_quickstart.md](nano_chat_quickstart.md) for why this matters).

---

## Check your data before training

Verify the label mask balance on packed chat/instruct data:

```powershell
python -c "import numpy as np; labels=np.fromfile('data/packed/chat/train.labels.bin', dtype=np.int32); masked=(labels==-100).sum(); total=len(labels); print(f'Masked: {masked/total*100:.1f}%'); print(f'Unmasked: {(total-masked)/total*100:.1f}%')"
```

Healthy targets:

- **Chat/instruct:** ~40–70% masked, ~30–60% unmasked.
- **Generative:** no label file, or 0% masked if one exists.
- No sampled batch should be entirely masked or entirely unmasked.

---

## Healthy training signs

- For `vocab_size=8000`, step-1 loss should start around `8–10`, not `30–60`.
- Loss declines steadily over the first few hundred steps.
- Gradient norm before clipping usually stays below `5.0`.
- `skip=no` appears on most steps.
- Stop and inspect if you see persistent `loss=0.000`, `loss=NaN`, or all-masked batches.

If loss explodes: cut `learning_rate` by 2–4×, keep `grad_clip: 1.0`, increase warmup, and
check for corrupt/duplicated data before touching the architecture.

---

## CPU presets

| RAM | Suggested model |
| --- | --- |
| 4 GB | `d_model=192`, `n_layers=3`, `n_heads=3`, `seq_len=384` |
| 8 GB (fast chat) | `d_model=256`, `n_layers=4`, `n_heads=4`, `seq_len=512` |
| 8 GB (balanced) | `d_model=384`, `n_layers=6`, `n_heads=6`, `seq_len=1024` |
| 16 GB | `d_model=512`, `n_layers=8`, `n_heads=8`, `seq_len=1024` |

On CPU, prefer `precision: fp32`, `micro_batch_size: 1` with higher `grad_accum_steps`,
`num_workers: 0`, and `pin_memory: false`. Use `nanoforge new-config` for an interactive
generator.

---

## Monitoring

Add `--dashboard` to `train` (or run `nanoforge dashboard --run runs/<name>`) for a live
browser view of train/val loss, perplexity, gradient norm, learning rate, and throughput at
`http://127.0.0.1:7860`.

Evaluate a finished checkpoint:

```powershell
nanoforge evaluate --checkpoint runs/tiny/best.pt --data data/packed/tiny/val.bin --batches 20
```
