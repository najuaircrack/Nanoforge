# Nanoforge

Nanoforge is a compact, from-scratch transformer framework for training and running
transformer models on local hardware (CPU or a single GPU). It is built on **PyTorch**
(`torch >= 2.2`) — there is no TensorFlow or JAX anywhere in the stack — with an optional
**Rust** performance layer (via PyO3/maturin) for hot loops like tokenizer training.

The model core is a **task-agnostic transformer backbone with pluggable output heads**, so
the same body can be trained as a causal LM / chatbot, a masked LM, a sequence or token
classifier, or a reward model — it is *not* locked to a decoder-only GPT clone. The Python
layer is the orchestration and config surface; Rust is the performance path.

It is built for **learning, experimentation, and rapid prototyping** — the whole stack
(model, tokenizers, data pipeline, training loop, inference, CLI/TUI) is meant to stay small
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

The fastest path is `auto-train`, which trains a tokenizer, packs your data, writes a
CPU-friendly config, and starts training — in one command. Point it at any text/JSONL/chat
data (a JSONL file with a `messages: [{role, content}]` field is treated as chat):

```bash
# One shot: tokenizer -> pack -> config -> train (chat model, assistant-only masking)
nanoforge auto-train --input data/raw/chat.jsonl --name mychat \
    --mode chat --tokenizer bpe --vocab-size 32000 --seq-len 512 \
    --ram 16GB --max-steps 3000 --dashboard

# Then chat with / generate from the result
nanoforge chat     --checkpoint runs/mychat/best.pt
nanoforge generate --checkpoint runs/mychat/best.pt --prompt "Explain RoPE in one line."
```

Prefer the manual pipeline (or the dependency-free **byte tokenizer**, vocab 260, no training)?

```bash
nanoforge prepare --input data/raw --tokenizer byte --out data/packed/demo
nanoforge train   --config configs/nano-debug.yaml     # or small-base / chat-sft / moe-large
```

Watch a run live in the browser dashboard or the terminal UI:

```bash
nanoforge dashboard --run runs/mychat      # browser
nanoforge tui       --run runs/mychat      # terminal (Textual; pip install -e .[tui])
```

Inspect the framework itself:

```bash
nanoforge registries                       # list registered components (heads, losses, ...)
nanoforge validate-config --config configs/nano-debug.yaml
nanoforge params          --config configs/nano-debug.yaml
```

For a complete tutorial that trains a tiny chat model which remembers your name and says
"I don't know" for unknown questions, see
[docs/nano_chat_quickstart.md](docs/nano_chat_quickstart.md).

---

## Features

**Model architecture**
- Task-agnostic transformer **backbone** + pluggable **task heads** (`task_type` in config):
  - `causal_lm` / `chat` / `code_generation` / `instruction_following` — autoregressive LM
  - `masked_lm` — BERT-style bidirectional (attention becomes non-causal automatically)
  - `sequence_classification`, `token_classification` — pooled / per-token classifiers
  - `reward_modeling` / `value_estimation` — scalar reward head (pairwise or MSE loss)
  - (image/audio/seq2seq heads are registered as stubs — see Limitations)
- GPT-style pre-norm transformer, tied embeddings, residual scaling
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
- **Real turn-based chat, not raw autocompletion**: a model trained in `chat` mode learns
  the `<|system|>` / `<|user|>` / `<|assistant|>` format with **assistant-only loss masking**
  (loss is computed only on the assistant's tokens), and generation **stops at the next role
  boundary** (`<|user|>` / `<|system|>` / `<|endoftext|>`) — so it answers as an assistant and
  hands the turn back instead of continuing the whole transcript. (Response *quality* still
  depends on model size and training budget — a tiny model trained briefly will be coherent in
  shape but weak in content.)
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

## Performance layer (Rust)

Nanoforge keeps orchestration and modelling in Python (PyTorch) but pushes CPU-bound,
embarrassingly-parallel hot loops into an optional **Rust** extension
(`native/nanoforge-tokenizers/`, built with PyO3 + `maturin`, using `rayon` for
multithreading and `lto = "thin"`). The extension is **optional** — every Rust path has a
pure-Python fallback, so the framework runs without a Rust toolchain.

```bash
pip install -e ".[native]"          # installs maturin
cd native/nanoforge-tokenizers
maturin develop --release           # builds the extension into your venv
nanoforge tokenizer-status          # confirms native acceleration is active
```

**In Rust today**
- Byte-level BPE tokenizer *training* (merge-pair heap, parallel word-frequency counting)
- Parallel batch encoding/decoding

**Good candidates to move into Rust next** (currently pure Python, and the CPU bottlenecks):
- **Data packing** (`data/packing.py`) — tokenize + pack into binary shards; the single most
  expensive preprocessing step and trivially parallel across documents.
- **Cleaning & dedup** (`data/cleaning.py`) — UTF-8 normalization, hashing-based dedup,
  language filtering over large corpora.
- **Memmap dataset iteration / shard sampling** (`data/dataset.py`) — hot path during
  training; a Rust iterator would cut Python per-batch overhead.
- **Format streaming** (`data/formats.py`) — JSONL/CSV/Parquet record extraction.

Model math (attention, matmuls) stays in PyTorch — that already dispatches to optimized
C++/CUDA kernels, so rewriting it in Rust would be slower, not faster. Rust is for the
data-and-tokenizer path, not the tensor path.

---

## Project layout

```text
Nanoforge/
├── configs/               # YAML training configs (nano-debug, small-base, chat-sft, moe-large)
├── docs/                  # documentation (this set)
├── scripts/               # helper scripts (data generation, conversion, checks)
├── tests/                 # pytest suite
├── native/                # optional Rust tokenizer extension (nanoforge-tokenizers)
├── src/nanoforge/
│   ├── cli.py             # command-line interface (entry point: `nanoforge`)
│   ├── config.py          # dataclass config system + YAML loader
│   ├── registry.py        # component registries + plugin discovery
│   ├── model/             # backbone, task heads, attention, rope, norms, moe, lora, ...
│   ├── data/              # formats, cleaning, tokenizers, packing, dataset, modes
│   ├── training/          # trainer, optimizers, schedulers, checkpoint, health
│   ├── generation/        # inference engine + sampling
│   ├── evaluation/        # checkpoint metrics
│   ├── export/            # onnx + gguf manifest
│   ├── server.py          # FastAPI inference server
│   ├── dashboard.py       # live training / inference / data web dashboard
│   └── tui.py             # full-scope terminal UI (Textual)
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
