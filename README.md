# Nanoforge

Nanoforge is a local-first Python/PyTorch toolkit evolving into a platform for training,
adapting, evaluating, and safely running small agentic AI systems on CPU, one GPU, or a small
distributed job.

**Important:** the current repository is a compact transformer toolkit plus a partial web studio.
It is not yet a complete agent harness or a replacement for a modern pretrained model. Read
[`docs/platform-audit.md`](docs/platform-audit.md) for the honest capability map, research-backed
CPU strategy, limitations, and staged platform plan.

## What exists

- Streaming dataset inspection, validation, cleaning, deduplication, tokenizer reporting, and
  fixed-sequence packing with chat/instruction label masking.
- A configurable PyTorch transformer with GQA/MQA, RoPE, tied embeddings, SwiGLU/GEGLU, LoRA,
  optional MoE, gradient checkpointing, checkpoint integrity, and CPU/GPU training.
- BPE/WordPiece/SentencePiece adapters, a dependency-free byte tokenizer for smoke tests, and an
  optional Rust tokenizer extension with Python fallbacks.
- Generation, chat, sampling, KV caching, local FastAPI serving, RAG primitives, DPO/distillation
  experiments, GGUF/ONNX export, and a React local studio.

## Install

Nanoforge supports Python 3.10–3.12; the default is recorded in `.python-version`.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS:        source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Optional capabilities:

```bash
python -m pip install -e ".[tokenizers,data,serve,tui,logging,export]"
```

For CPU-only PyTorch, install the appropriate CPU wheel from the PyTorch index before installing
Nanoforge. Do not commit downloaded data, model weights, caches, or private indexes.

## Verify the checkout

```bash
make test
```

This runs `python -m ruff check .` and `python -m pytest -q`. If the web files changed:

```bash
cd web
npm ci
npm run build
npm run lint
```

If Rust files changed:

```bash
cd native/nanoforge-tokenizers
cargo test --locked
```

## First workflow: inspect → prepare → train → evaluate

Put a small, licensed dataset under an ignored runtime directory such as `data/raw/`.

```bash
nanoforge inspect-dataset --input data/raw
nanoforge validate-dataset --input data/raw
nanoforge profile-dataset --input data/raw --json

# Byte tokenizer is for smoke tests. Use native-bpe/BPE for real text when available.
nanoforge auto-train \
  --input data/raw \
  --name local-specialist \
  --mode auto \
  --tokenizer native-bpe \
  --vocab-size 8000 \
  --seq-len 512 \
  --ram 8GB \
  --no-train

nanoforge train --config configs/local-specialist.yaml
nanoforge evaluate \
  --checkpoint runs/local-specialist/best.pt \
  --data data/packed/local-specialist/val.bin \
  --batches 20
nanoforge chat --checkpoint runs/local-specialist/best.pt --mode chat
```

For a dependency-free smoke path:

```bash
python scripts/make_chat_data.py --out data/raw/smoke.jsonl
nanoforge prepare --input data/raw/smoke.jsonl --tokenizer byte --mode chat \
  --loss-masking assistant_only --seq-len 128 --out data/packed/smoke
nanoforge train --config configs/nano-debug.yaml --no-tui
```

The exact presets and Windows examples are in [`docs/training.md`](docs/training.md). Always
inspect label-mask balance before trusting a chat run.

## Adaptation is the quality path

If the goal is a useful general assistant, do not expect a tiny randomly initialized CPU model to
learn broad knowledge. Prefer:

1. import a capable compatible base model;
2. keep changing/private facts in governed RAG or memory;
3. use carefully curated SFT/LoRA/QLoRA data to teach behavior and domain style;
4. evaluate tool calls, grounding, refusals, multi-turn behavior, latency, and memory;
5. export a quantized model to a dedicated CPU runtime when that is faster and smaller.

The audit explains why this is the practical route and which pieces Nanoforge still needs to make
it first-class.

## Web studio

```bash
cd web
npm ci
npm run dev
```

Build assets into `src/nanoforge/web_dist/` with `npm run build`, then run the local backend with
the `nanoforge web` command. The default should remain localhost. Do not expose the current web
studio to a network: it can inspect data, write configs, load models, and start jobs without a
complete authentication/authorization layer.

## Native tokenizer extension

Python fallbacks are mandatory, but the optional Rust path accelerates tokenizer work:

```bash
python -m pip install -e ".[native]"
cd native/nanoforge-tokenizers
maturin develop --release
nanoforge tokenizer-status
```

## Project map

| Area | Location |
| --- | --- |
| Python runtime | `src/nanoforge/` |
| Model | `src/nanoforge/model/` |
| Data/tokenizers | `src/nanoforge/data/` |
| Training | `src/nanoforge/training/` |
| Generation/server | `src/nanoforge/generation/`, `server.py` |
| RAG | `src/nanoforge/rag/` |
| Web studio | `web/`, `src/nanoforge/webapp.py` |
| Configs/docs/tests | `configs/`, `docs/`, `tests/` |
| Agent instructions | `AGENTS.md` |

## Safety and privacy

Nanoforge is experimental. Model outputs may be wrong or unsafe. Keep servers localhost-only,
never execute arbitrary model-generated shell code, keep personal data outside source control,
and treat memory/tool access as explicit capabilities. See `AGENTS.md` for the full engineering
and self-improvement contract.

## License

MIT — see [`LICENSE.md`](LICENSE.md).
