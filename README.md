
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
- Cosine / linear / constant LR schedulers
- Warmup support
- EMA support
- Early stopping
- NaN/Inf and gradient health checks
- TensorBoard logging
- Lightweight training dashboard

### Tokenization

- Byte tokenizer
- Optional native Rust byte tokenizer backend
- HuggingFace BPE tokenizer
- Built-in pure-Python byte-level BPE fallback
- SentencePiece support
- Special token support
- Packed memmap datasets
- Streaming dataset support
- JSONL and conversation dataset paths
- Schema-aware parquet/Arrow/csv/sqlite/json ingestion paths
- Dataset inspection, validation, cleaning, conversion, and deduplication CLIs
- Automatic datasource mode detection for chat, instruct, completion, code, and generative rows
- Assistant-only and completion-only packed label masks for responder-style training

### Inference

- Streaming generation
- Chat CLI
- Batch generation
- Top-k sampling
- Top-p sampling
- Temperature sampling
- Repetition penalty
- Mirostat sampling hooks
- Frequency and presence penalties
- N-gram repetition blocking
- Deterministic, chat, creative, coding, low-memory, and high-quality inference modes
- Role-boundary stop detection, EOS-aware stopping, and repetition/runaway interruption

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

Inspect registry-backed components:

```powershell
nanoforge registries
nanoforge registries --name attention
nanoforge validate-config --config configs/tiny.yaml
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
  --type python-bpe ^
  --vocab-size 4000 ^
  --out data/tokenizers/tiny-python-bpe.json
```

Use `--type bpe` for the HuggingFace `tokenizers` Rust implementation. Use `--type python-bpe`
when you want a dependency-free Nanoforge tokenizer artifact that can be loaded with
`tokenizer_type: python-bpe` or `native-bpe`.

Check tokenizer acceleration:

```powershell
nanoforge tokenizer-status
```

Benchmark tokenizer throughput and memory:

```powershell
nanoforge benchmark-tokenizer ^
  --input data/raw ^
  --tokenizer byte-native ^
  --limit 10000 ^
  --batch-size 256
```

Build the optional Rust tokenizer extension:

```powershell
pip install maturin
cd native\nanoforge-tokenizers
maturin develop --release
```

If the native extension is not installed, `byte-native` automatically falls back to the compatible
Python byte tokenizer. BPE can use either HuggingFace `tokenizers` or Nanoforge's built-in
byte-level Python BPE fallback; both are fed by streaming structured-data readers.

`native-bpe` uses the Rust `ByteLevelBpeTokenizer` when the extension is built, and falls back to
the same Nanoforge BPE artifact loader in Python when it is not installed:

```powershell
nanoforge train-tokenizer ^
  --input data/raw ^
  --type native-bpe ^
  --vocab-size 32000 ^
  --out data/tokenizers/native-bpe.json

nanoforge prepare ^
  --input data/raw ^
  --tokenizer native-bpe ^
  --tokenizer-path data/tokenizers/native-bpe.json ^
  --mode auto ^
  --loss-masking auto ^
  --out data/packed/native
```

Structured datasets are read through Nanoforge's dataset adapters before tokenizer fitting, so
parquet/Arrow files are streamed by text columns rather than treated as raw bytes. You can dry-run
large or messy corpora before training:

```powershell
nanoforge train-tokenizer ^
  --input data/raw ^
  --type bpe ^
  --text-column text ^
  --dry-run ^
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

For the built-in tokenizer, switch to:

```powershell
nanoforge prepare ^
  --input data/raw ^
  --tokenizer python-bpe ^
  --tokenizer-path data/tokenizers/tiny-python-bpe.json ^
  --out data/packed/tiny
```

---

# Configuration Reference

Nanoforge configs are split into `model`, `training`, `data`, and `inference`. Start from
`configs/tiny.yaml`, then adjust only the fields you need.

## Model Options

| Option | Default | Use | Practical values |
| --- | ---: | --- | --- |
| `vocab_size` | `32000` | Tokenizer vocabulary size. | Match your tokenizer exactly. Byte tokenizer uses `260`. |
| `max_seq_len` | `2048` | Maximum training context. | `128-512` CPU smoke, `1024-2048` small GPU, `4096+` high VRAM. |
| `d_model` | `512` | Hidden width. | Must divide by `n_heads`; raise for quality, lower for memory. |
| `n_layers` | `8` | Transformer depth. | `2-6` smoke, `6-12` local, `18+` experimental. |
| `n_heads` | `8` | Attention heads. | Keep `d_model / n_heads` near `64` when possible. |
| `n_kv_heads` | `n_heads` | GQA KV heads. | Use `1`, `2`, or `4` to reduce KV cache memory. |
| `ffn_hidden_mult` | `2.67` | FFN expansion. | `2.0` low memory, `2.67-4.0` quality. |
| `dropout` | `0.0` | Regularization. | `0.0` small clean data, `0.05-0.1` noisy data. |
| `rope_theta` | `10000` | RoPE frequency base. | Larger values help longer context experiments. |
| `block_type` | `transformer` | Block implementation. | `transformer`, `parallel_residual`. |
| `attention_backend` | `sdpa` | Attention implementation. | `sdpa` best default, `manual` for debugging, `sliding_window` with `sliding_window`. |
| `ffn_type` | auto | FFN backend. | `swiglu`, `geglu`, `moe`. |
| `normalization` | `rmsnorm` | Norm layer. | `rmsnorm` default, `layernorm` compatibility. |
| `position_embedding` | `rope` | Position system. | `rope`, `alibi`, `none`. |
| `quantization_backend` | `none` | Inference quantization hook. | `none`, `int8`, `int4`, `gptq`, `awq`, `gguf`. |
| `sliding_window` | `null` | KV cache/window length. | `512-4096` for low-memory long generation. |
| `gradient_checkpointing` | `false` | Save activation memory. | Enable for limited VRAM training. |
| `moe` | `null` | Mixture-of-experts config. | Use only for experiments; start with dense FFN first. |
| `lora_rank` | `0` | LoRA adapter rank. | `0` off, `4-16` efficient fine-tuning. |

## Training Options

| Option | Default | Use | Practical values |
| --- | ---: | --- | --- |
| `mode` | `generative` | Training intent. | `generative`, `chat`, `instruct`, `completion`, `code`, `reasoning`, `hybrid`. |
| `max_steps` | `1000` | Total optimizer steps. | Use small smoke tests first, then scale by tokens seen. |
| `batch_size` | `8` | Logical batch size target. | Keep consistent with `micro_batch_size * grad_accum_steps`. |
| `micro_batch_size` | `2` | Per-step batch on device. | Lower until it fits memory. |
| `grad_accum_steps` | `4` | Gradient accumulation. | Raise to simulate larger batches. |
| `learning_rate` | `3e-4` | Peak LR. | `1e-4-3e-4` AdamW small models. |
| `min_learning_rate` | `3e-5` | Final LR for decay. | Usually `0.05-0.1x` peak LR. |
| `warmup_steps` / `warmup_ratio` | `100` / `0.03` | LR warmup. | Prefer ratio for changing run lengths. |
| `optimizer` | `adamw` | Optimizer registry key. | `adamw` stable, `lion` experimental. |
| `scheduler` | `cosine` | LR schedule. | `cosine`, `linear`, `constant`. |
| `grad_clip` | `1.0` | Gradient clipping. | `0.5-1.0` for tiny/local models. |
| `precision` | `auto` | Autocast dtype. | `auto`, `bf16`, `fp16`, `fp32`. |
| `low_memory` | `false` | CUDA allocator tuning. | Enable on fragmented/low-VRAM machines. |
| `ema_decay` | `0.0` | EMA weights. | `0.999-0.9999` for smoother checkpoints. |
| `health_interval` | `10` | Diagnostics cadence. | Lower while debugging unstable runs. |
| `distributed_backend` | `none` | Future distributed hook. | `none` today; keep single-process unless you add an external launcher. |

## Data Options

| Option | Default | Use | Practical values |
| --- | ---: | --- | --- |
| `train_path` / `val_path` | packed `.bin` | Packed token files. | Use `nanoforge prepare` outputs. |
| `tokenizer_type` | `byte` | Tokenizer loader. | `byte`, `byte-native`, `bpe`, `python-bpe`, `wordpiece`, `sentencepiece`. |
| `tokenizer_path` | `null` | Tokenizer artifact. | Required for BPE/WordPiece/SentencePiece. |
| `seq_len` | `2048` | Packed training sequence length. | Match or stay below `model.max_seq_len`. |
| `mode` | `auto` | Data formatting intent. | `auto` detects chat/instruct/completion/code/generative rows. |
| `loss_masking` | `auto` | Label masking policy. | `auto`, `none`, `assistant_only`, `completion_only`, `partial`. |
| `assistant_only_loss` | `false` | Chat SFT target masking. | Enable for assistant response training. |
| `dataset_weights` | `null` | Hybrid data mixing weights. | Example: `{chat: 0.7, text: 0.2, code: 0.1}`. |
| `streaming` | `false` | Streaming dataset intent. | Use preprocessing CLIs today; native training streaming is evolving. |
| `tokenizer_batch_size` | `256` | Batched tokenizer throughput during packing. | Raise for fast native tokenizers; lower for memory constrained machines. |

## Inference Options

| Option | Default | Use | Practical values |
| --- | ---: | --- | --- |
| `mode` | `balanced` | Sampling preset. | `chat`, `creative`, `coding`, `deterministic`, `low_memory`, `high_quality`. |
| `max_new_tokens` | `256` | Generation limit. | Lower for chat, higher for creative/code. |
| `temperature` | `0.8` | Randomness. | `0-0.3` deterministic/code, `0.6-0.8` chat, `0.9+` creative. |
| `top_k` | `50` | Candidate cap. | `20-80`; use `1` for deterministic. |
| `top_p` | `0.95` | Nucleus probability. | `0.85-0.92` chat/code, `0.95` creative. |
| `min_p` | `null` | Relative probability floor. | `0.03-0.08` can stabilize tiny models. |
| `repetition_penalty` | `1.0` | Penalize repeated tokens. | `1.05-1.15` chat/tiny models. |
| `frequency_penalty` | `0.0` | Penalize repeated counts. | `0.1-0.5` for loop reduction. |
| `presence_penalty` | `0.0` | Penalize seen tokens. | `0.1-0.3` for diversity. |
| `no_repeat_ngram_size` | `0` | Block repeated n-grams. | `3-5` for chat/code stability. |
| `deterministic` | `false` | Greedy decoding. | Enable for reproducible tests and coding. |
| `stop_on_repetition` | `true` | End repetitive tails. | Keep enabled for tiny/chat models. |
| `repetition_window` | `64` | Repetition detector window. | `32-128` depending on output length. |
| `repetition_threshold` | `0.85` | Tail repetition cutoff. | Lower to stop loops sooner. |
| `mirostat` | `false` | Entropy-targeted sampling. | Useful for long creative outputs. |
| `stop_tokens` | role/EOS tokens | Stop boundaries. | Include `<|user|>` for chat models. |

## Starter Presets

Tiny CPU smoke:

```yaml
model:
  vocab_size: 260
  max_seq_len: 128
  d_model: 128
  n_layers: 2
  n_heads: 4
training:
  max_steps: 50
  micro_batch_size: 1
  grad_accum_steps: 1
  precision: fp32
data:
  tokenizer_type: byte
  seq_len: 128
inference:
  mode: deterministic
```

Small chat SFT:

```yaml
training:
  mode: chat
  learning_rate: 2e-4
  scheduler: cosine
data:
  mode: chat
  tokenizer_type: python-bpe
  tokenizer_path: data/tokenizers/tiny-python-bpe.json
  loss_masking: assistant_only
  assistant_only_loss: true
inference:
  mode: chat
  repetition_penalty: 1.1
  no_repeat_ngram_size: 4
```

Low-memory long-context inference:

```yaml
model:
  n_kv_heads: 2
  sliding_window: 1024
  quantization_backend: int8
inference:
  mode: low_memory
  top_k: 40
  min_p: 0.05
```

---

# CPU Training Workflows

These commands are tuned for Windows 11 CPU-only training with about 8GB RAM.

## 0. One-Command Auto Training

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

This trains the tokenizer, prepares boundary-aware packed data, writes `configs/ultrachat-18m.yaml`,
and starts training. Add `--no-train` to stop after setup.

## 1. Generative Model

Use this for stories, Wikipedia-style text, notes, books, or plain continuation models.

```powershell
nanoforge train-tokenizer ^
  --input data/raw ^
  --type native-bpe ^
  --vocab-size 8000 ^
  --out data/tokenizers/tiny-bpe.json

nanoforge prepare ^
  --input data/raw ^
  --tokenizer native-bpe ^
  --tokenizer-path data/tokenizers/tiny-bpe.json ^
  --mode generative ^
  --loss-masking none ^
  --seq-len 1024 ^
  --out data/packed/story

nanoforge train --config configs/my-story.yaml
nanoforge generate --checkpoint runs/my-story/best.pt --mode creative --prompt "Once upon a time"
```

## 2. Chat Model

Use this for UltraChat, ShareGPT, or rows containing `messages`/`conversations`.

```powershell
nanoforge train-tokenizer ^
  --input data/raw ^
  --type native-bpe ^
  --vocab-size 8000 ^
  --text-column messages ^
  --out data/tokenizers/tiny-bpe.json

nanoforge prepare ^
  --input data/raw ^
  --tokenizer native-bpe ^
  --tokenizer-path data/tokenizers/tiny-bpe.json ^
  --mode chat ^
  --loss-masking assistant_only ^
  --text-column messages ^
  --seq-len 512 ^
  --out data/packed/ultrachat

nanoforge train --config configs/ultrachat.yaml
nanoforge chat --checkpoint runs/ultrachat-18m/best.pt --mode chat
```

Chat packing is conversation-boundary aware. Each fixed sequence starts from a user/system/assistant boundary, long assistant responses are split with the user prefix repeated, and labels target assistant tokens only.

## 3. Instruct Model

Use this for Alpaca-style rows with `instruction`, optional `input`, and `output`/`response`.

```powershell
nanoforge prepare ^
  --input data/alpaca.jsonl ^
  --tokenizer native-bpe ^
  --tokenizer-path data/tokenizers/tiny-bpe.json ^
  --mode instruct ^
  --loss-masking completion_only ^
  --seq-len 512 ^
  --out data/packed/instruct

nanoforge train --config configs/my-instruct.yaml
```

## 4. Check Data Before Training

```powershell
python -c "import numpy as np; labels=np.fromfile('data/packed/ultrachat/train.labels.bin', dtype=np.int32); masked=(labels==-100).sum(); total=len(labels); print(f'Masked: {masked/total*100:.1f}%'); print(f'Unmasked: {(total-masked)/total*100:.1f}%')"
```

Healthy targets:

- Chat/instruct: about 40-70% masked and 30-60% unmasked.
- Generative: no label file, or 0% masked if a label file exists.
- No sampled batch should be entirely masked or entirely unmasked for chat data.

## 5. Healthy Training Signs

- For `vocab_size=8000`, step-1 loss should usually start around `8-10`, not `30-60`.
- Loss should decline steadily over the first few hundred steps.
- Gradient norm before clipping should usually stay below `5.0`.
- `skip=no` should appear on most steps.
- Stop immediately if you see persistent `loss=0.000`, `loss=NaN`, or all-masked batches.

## 6. New Config Wizard

```powershell
nanoforge new-config --out configs/my-model.yaml
```

The wizard asks what you are training, available RAM, speed/size preference, and dataset format, then writes a complete CPU-friendly YAML.

## 7. Import External Models

```powershell
nanoforge import --model path\to\model.gguf --name my-gguf
nanoforge import --model mistralai/Mistral-7B-v0.1 --name mistral
nanoforge import --model path\to\model.onnx --name my-onnx --tokenizer path\to\tokenizer-dir

nanoforge chat --model my-gguf --mode chat
nanoforge generate --model mistral --prompt "Hello"
```

Backends:

- GGUF uses `llama-cpp-python`.
- HuggingFace directories and Hub IDs use `transformers`.
- SafeTensors runs through `transformers` when a compatible `config.json` and tokenizer are present.
- ONNX uses `onnxruntime` with an adjacent or specified HuggingFace tokenizer.

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

# Dataset Tooling

Inspect and validate structured datasets before packing:

```powershell
nanoforge inspect-dataset --input data/raw --limit 1000
nanoforge validate-dataset --input data/raw --limit 1000
```

Clean, deduplicate, or convert inputs to normalized text records:

```powershell
nanoforge clean-dataset --input data/raw --out data/clean/train.jsonl
nanoforge deduplicate-dataset --input data/raw --out data/clean/deduped.jsonl
nanoforge convert-dataset --input data/raw --format txt --out data/clean/train.txt
```

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
