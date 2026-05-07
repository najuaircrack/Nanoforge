
# Nanoforge

Nanoforge is an experimental transformer framework for training small-to-medium decoder-only language models on local hardware.

The project focuses on understanding and experimenting with:
- transformer internals,
- tokenization,
- training pipelines,
- inference systems,
- chat formatting,
- dataset preparation,
- local LLM workflows.

Nanoforge is not intended to compete with production-grade frameworks such as:
- Megatron-LM,
- vLLM,
- DeepSpeed,
- llama.cpp,
- Hugging Face Transformers.

Instead, the goal is:
- education,
- experimentation,
- rapid prototyping,
- learning modern LLM systems from the ground up.

---

> [!WARNING]
> Nanoforge is currently pre-alpha experimental software.
>
> Many parts of the project, documentation, configs, utilities, and workflows were created or accelerated with AI-assisted development tools.
>
> Expect:
>
> - incomplete features,
> - unstable APIs,
> - rough implementations,
> - architectural changes,
> - missing optimizations,
> - experimental training behavior,
> - limited testing.
>
> This repository exists primarily for learning, experimentation, and hobby/research development.

---

## Features

### Model Architecture

- GPT-style decoder-only transformer
- Rotary positional embeddings (RoPE)
- RMSNorm
- SwiGLU / optional GEGLU
- Grouped Query Attention (GQA)
- KV cache support
- Sliding-window attention
- Residual scaling
- Tied embeddings
- Flash Attention via PyTorch SDPA when available
- Optional Mixture-of-Experts feed-forward layers

### Training

- Mixed precision training
- Gradient accumulation
- Gradient checkpointing
- AdamW optimizer
- Cosine LR scheduler
- Warmup support
- EMA support
- Early stopping
- TensorBoard logging
- Lightweight training dashboard

### Tokenization

- Byte tokenizer
- BPE tokenizer
- SentencePiece support
- Special token support
- Packed memmap datasets
- Streaming dataset support
- JSONL and conversation dataset paths

### Inference

- Streaming generation
- Chat CLI
- Batch generation
- Top-k sampling
- Top-p sampling
- Temperature sampling
- Repetition penalty
- Mirostat sampling hooks

### Export

- ONNX export helpers
- GGUF metadata helpers
- LoRA adapter support

---

## Philosophy

Nanoforge intentionally prioritizes:
- readability,
- simplicity,
- experimentation,
- hackability,

over maximum production performance.

The project is designed for:
- hobbyists,
- students,
- indie researchers,
- developers learning transformers,
- local AI experimentation.

The codebase aims to remain relatively approachable compared to large-scale enterprise LLM frameworks.

---

# Quick Start

## 1. Create Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
````

## 2. Install

Minimal install:

```powershell
pip install -e .
```

Full install:

```powershell
pip install -e ".[all]"
```

---

# Byte Tokenizer Smoke Test

## Prepare Dataset

```powershell
nanoforge prepare --input data/raw --tokenizer byte --out data/packed/tiny
```

## Train

```powershell
nanoforge train --config configs/tiny.yaml --dashboard
```

## Chat

```powershell
nanoforge chat --checkpoint runs/tiny/best.pt
```

---

# BPE Training Workflow

## Train Tokenizer

```powershell
nanoforge train-tokenizer ^
  --input data/raw ^
  --type bpe ^
  --vocab-size 4000 ^
  --out data/tokenizers/tiny-bpe.json
```

## Prepare Dataset

```powershell
nanoforge prepare ^
  --input data/raw ^
  --tokenizer bpe ^
  --tokenizer-path data/tokenizers/tiny-bpe.json ^
  --out data/packed/tiny
```

## Train Model

```powershell
nanoforge train --config configs/tiny.yaml --dashboard
```

---

# Dashboard

Start dashboard:

```powershell
nanoforge dashboard --run runs/tiny
```

Default URL:

```text
http://127.0.0.1:7860
```

Dashboard currently tracks:

* train loss,
* validation loss,
* perplexity,
* gradient norm,
* learning rate,
* throughput.

---

# Chat Dataset Format

Nanoforge supports conversational formatting.

Example:

```text
<|system|>
You are a helpful assistant.

<|user|>
Hello

<|assistant|>
Hi! How can I help you?

<|endoftext|>
```

Special tokens currently recommended:

```text
<|user|>
<|assistant|>
<|system|>
<|endoftext|>
```

---

# Recommended Training Sizes

| Target             | Layers |   Width |   Context | Hardware   |
| ------------------ | -----: | ------: | --------: | ---------- |
| Tiny smoke         |    4-6 |     256 |   128-512 | CPU        |
| Small local        |    6-8 | 384-512 |  512-2048 | 4-8 GB GPU |
| Medium local       |     12 |     768 | 2048-4096 | 12 GB GPU  |
| Large experimental |  18-24 |   1024+ | 4096-8192 | 24 GB+ GPU |

---

# Recommended Datasets

Good starter datasets:

* TinyStories
* OpenAssistant
* UltraChat
* Alpaca
* permissive code repositories
* structured QA datasets

For code-focused training:

* prioritize clean permissive repositories,
* high-quality docs,
* tests,
* structured conversations,
* validated examples.

---

# Project Layout

```text
Nanoforge/
│
├── configs/
├── data/
├── docs/
├── scripts/
├── tests/
│
├── src/nanoforge/
│   ├── model/
│   ├── data/
│   ├── generation/
│   ├── training/
│   ├── export/
│   ├── cli.py
│   └── server.py
│
└── runs/
```

---

# Current Limitations

Current known limitations include:

* limited optimization,
* incomplete tokenizer tooling,
* minimal distributed support,
* no production inference backend,
* limited evaluation tooling,
* experimental dataset ingestion,
* evolving checkpoint formats,
* minimal safety/alignment tooling.

---

# Development Goals

Planned long-term areas:

* improved dataset ingestion,
* better tokenizer tooling,
* distributed training,
* quantization,
* inference optimization,
* richer evaluation suites,
* better chat alignment workflows,
* improved memory efficiency,
* advanced sampling strategies,
* tool calling experiments.

---

# Contributing

This is primarily an experimental learning project.

Contributions, experiments, fixes, and ideas are welcome.

Before contributing:

* expect APIs to change,
* expect rapid iteration,
* expect rough edges.

---

# Disclaimer

Nanoforge is an educational and experimental project.

Generated outputs may:

* hallucinate,
* produce incorrect information,
* generate nonsensical text,
* reflect dataset bias,
* behave unpredictably.

Do not rely on outputs for:

* legal advice,
* medical advice,
* financial decisions,
* safety-critical systems.

Use responsibly.

