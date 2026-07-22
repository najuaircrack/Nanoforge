# Nanoforge

Nanoforge is a compact, from-scratch transformer framework for training and running
small-to-medium decoder-only language models on local hardware (CPU or a single GPU).

It is built for **learning, experimentation, and rapid prototyping** — the whole stack
(model, tokenizers, data pipeline, training loop, inference, CLI) is meant to stay small
enough to read and hack. It is *not* a production framework like vLLM, DeepSpeed, or
Hugging Face Transformers.

> [!WARNING]
> Nanoforge is pre-alpha experimental software. Expect incomplete features, unstable APIs,
> rough edges, and evolving checkpoint formats. Parts of the code and docs were
> AI-assisted. Use it to learn and tinker, not for anything you depend on.

---

## Documentation

| Doc | What's in it |
| --- | --- |
| [docs/cli.md](docs/cli.md) | Every CLI command, its flags, and a runnable example. |
| [docs/configuration.md](docs/configuration.md) | Full YAML config schema — every field and default. |
| [docs/training.md](docs/training.md) | Training workflows: auto-train, manual pipeline, recipes, health checks. |
| [docs/architecture.md](docs/architecture.md) | How the model, data pipeline, and training loop work internally. |
| [docs/nano_chat_quickstart.md](docs/nano_chat_quickstart.md) | End-to-end tutorial: train a tiny byte-level chat model from scratch. |
| [docs/roadmap.md](docs/roadmap.md) | Planned/future work (not yet implemented). |

---

## Install

Requires Python 3.10+.

```bash
python -m venv .venv
# Windows:        .venv\Scripts\activate
# Linux/macOS:    source .venv/bin/activate

pip install -e .            # core (torch, numpy, pyyaml, tqdm)
pip install -e ".[all]"     # everything (tokenizers, data, serve, logging, export, dev)
```

**CPU-only note (Windows / Git Bash):** install the CPU torch wheel from PyTorch's index
first, then the rest:

```bash
python -m pip install torch numpy --index-url https://download.pytorch.org/whl/cpu
python -m pip install pyyaml tqdm tokenizers
python -m pip install -e . --no-deps
```

Optional extras (install what you use):

| Extra | Enables |
| --- | --- |
| `tokenizers` | HuggingFace BPE / WordPiece / SentencePiece tokenizers |
| `data` | Parquet / Arrow / `datasets` / WebDataset ingestion |
| `serve` | FastAPI inference server and live dashboard |
| `logging` | TensorBoard and Weights & Biases logging |
| `export` | ONNX export and ONNX Runtime inference |
| `native` | `maturin`, to build the optional Rust tokenizer extension |
| `dev` | `pytest`, `ruff` |

---

## 60-second quickstart

The fastest path uses the dependency-free **byte tokenizer** (vocab 260) — no tokenizer
training needed.

```bash
# 1. Pack some raw text/JSONL into token memmaps
nanoforge prepare --input data/raw --tokenizer byte --out data/packed/tiny

# 2. Train (add --dashboard for a live browser view)
nanoforge train --config configs/tiny.yaml

# 3. Chat or generate
nanoforge chat --checkpoint runs/tiny/best.pt
nanoforge generate --checkpoint runs/tiny/best.pt --prompt "Once upon a time"
```

Inspect the framework itself:

```bash
nanoforge registries                    # list registered components
nanoforge validate-config --config configs/tiny.yaml
nanoforge params --config configs/tiny.yaml
```

For a complete tutorial that trains a tiny chat model which remembers your name and says
"I don't know" for unknown questions, see
[docs/nano_chat_quickstart.md](docs/nano_chat_quickstart.md).

---

## Features

**Model architecture**
- GPT-style decoder-only transformer, pre-norm, tied embeddings, residual scaling
- Grouped-Query / Multi-Query Attention (via `n_kv_heads`)
- RoPE (with linear / dynamic / YaRN-style scaling) and ALiBi position embeddings
- Sliding-window attention
- KV cache for fast incremental decoding
- Attention via PyTorch SDPA (with a manual fallback path)
- RMSNorm and LayerNorm
- SwiGLU / GEGLU feed-forward, plus optional Mixture-of-Experts (with load-balancing loss)
- LoRA-capable projection layers
- `transformer` and `parallel_residual` (GPT-NeoX style) blocks

**Training**
- Mixed precision (bf16/fp16/fp32), gradient accumulation, gradient checkpointing
- Optimizers: AdamW, Lion, Adafactor, SophiaG
- Schedulers: cosine, linear, constant — with warmup
- EMA, early stopping, grad clipping, NaN/Inf skip + rollback, health monitoring
- Robust checkpointing (atomic writes, integrity hash, config sidecar, async saves)
- TensorBoard / W&B logging and a built-in live web dashboard

**Tokenization**
- Byte tokenizer (vocab 260, deterministic, no training)
- HuggingFace BPE / WordPiece / SentencePiece / Unigram
- Pure-Python byte-level BPE fallback + optional native Rust byte/BPE backend
- Streaming, schema-aware ingestion: text, JSON(L), CSV/TSV, YAML, XML, SQLite, Parquet,
  Arrow, ZIP/TAR archives, HTTP, and Hugging Face streaming refs
- Boundary-aware chat/instruct packing with assistant-only / completion-only label masks
- Dataset inspect / validate / clean / deduplicate / convert CLIs

**Inference**
- Streaming generation and an interactive chat CLI (applies the chat template)
- Sampling: temperature, top-k, top-p, min-p, repetition / frequency / presence penalties,
  no-repeat-ngram, Mirostat (v2-style)
- Preset modes: `balanced`, `chat`, `creative`, `coding`, `deterministic`, `low_memory`,
  `high_quality`
- Beam search, prefix caching, EOS/role-boundary and runaway-repetition stopping
- Import & run external models: GGUF (`llama-cpp-python`), HuggingFace / SafeTensors
  (`transformers`), ONNX (`onnxruntime`)
- FastAPI server with OpenAI-style `/v1/completions` and `/v1/chat/completions`

**Export & evaluation**
- ONNX export (logits-only graph)
- GGUF metadata manifest for external converter tooling
- int8 dynamic CPU quantization; QAT preparation
- Checkpoint evaluation (loss, perplexity, token accuracy), forward-pass benchmark, and
  analytical param/FLOP/memory profiling

See [docs/architecture.md](docs/architecture.md) for what each of these does under the hood,
including honest notes on which pieces are experimental or partial.

---

## Project layout

```text
Nanoforge/
├── configs/               # YAML training configs (tiny, cpu-tiny, nano-chat, ...)
├── docs/                  # documentation (this set)
├── scripts/               # helper scripts (data generation, conversion, checks)
├── tests/                 # pytest suite
├── native/                # optional Rust tokenizer extension (nanoforge-tokenizers)
├── src/nanoforge/
│   ├── cli.py             # command-line interface (entry point: `nanoforge`)
│   ├── config.py          # dataclass config system + YAML loader
│   ├── registry.py        # component registries + plugin discovery
│   ├── model/             # transformer, attention, rope, norms, moe, lora, ...
│   ├── data/              # formats, cleaning, tokenizers, packing, dataset, modes
│   ├── training/          # trainer, optimizers, schedulers, checkpoint, health
│   ├── generation/        # inference engine + sampling
│   ├── evaluation/        # checkpoint metrics
│   ├── export/            # onnx + gguf manifest
│   ├── server.py          # FastAPI inference server
│   └── dashboard.py       # live training dashboard
└── runs/                  # training outputs (checkpoints, metrics) — gitignored
```

---

## Limitations

- No distributed training yet (FSDP / DeepSpeed / tensor parallel are roadmap items).
- GGUF "export" writes a JSON manifest, not a binary `.gguf` file.
- int4 / GPTQ / AWQ quantization are metadata hooks only; int8 dynamic (CPU) is real.
- Speculative decoding, paged attention, and NTK RoPE scaling are **not** implemented
  (see [docs/roadmap.md](docs/roadmap.md)).
- Distillation and DPO loss functions exist as standalone utilities but are not wired into
  the training CLI.
- The dashboard is a local single-run viewer, not a multi-user experiment tracker.

## Disclaimer

Nanoforge is an educational project. Model outputs may be wrong, biased, nonsensical, or
unpredictable. Do not rely on them for legal, medical, financial, or safety-critical use.

## License

MIT — see [LICENSE.md](LICENSE.md).
