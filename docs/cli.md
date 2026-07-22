# CLI Reference

Every Nanoforge command. Invoke as `nanoforge <command>` after `pip install -e .`, or as
`python -m nanoforge.cli <command>` without installing.

```bash
nanoforge --help            # list all commands
nanoforge <command> --help  # flags for one command
```

Windows uses `^` for line continuation in the examples below; Linux/macOS use `\`.

**Commands by group**

- [Setup & inspection](#setup--inspection): `params`, `registries`, `validate-config`, `new-config`
- [Dataset tooling](#dataset-tooling): `inspect-dataset`, `validate-dataset`, `clean-dataset`, `deduplicate-dataset`, `convert-dataset`
- [Tokenizers](#tokenizers): `train-tokenizer`, `tokenizer-status`, `tokenizer-report`, `benchmark-tokenizer`
- [Data packing](#data-packing): `prepare`
- [Training](#training): `auto-train`, `train`, `dashboard`
- [Inference](#inference): `generate`, `chat`, `serve`
- [Import external models](#import-external-models): `import`
- [Export & evaluation](#export--evaluation): `export`, `evaluate`, `benchmark`, `profile-config`

---

## Setup & inspection

### `params`
Estimate a model's parameter count from a config.

| Flag | Default | Description |
| --- | --- | --- |
| `--config` | required | Path to a YAML config. |

```bash
nanoforge params --config configs/tiny.yaml
```

### `registries`
List registered components (attention backends, FFN types, optimizers, samplers, etc.).

| Flag | Default | Description |
| --- | --- | --- |
| `--name` | all | One of `attention, ffn, activation, position, optimizer, scheduler, tokenizer, sampler, normalization, quantization, block`. |

```bash
nanoforge registries
nanoforge registries --name attention
```

### `validate-config`
Load a config and validate its registry-backed keys (fails fast on bad values).

| Flag | Default | Description |
| --- | --- | --- |
| `--config` | required | Path to a YAML config. |

```bash
nanoforge validate-config --config configs/nano-chat.yaml
```

### `new-config`
Interactive wizard that writes a CPU-friendly config (asks intent, RAM, speed, data format).

| Flag | Default | Description |
| --- | --- | --- |
| `--out` | `configs/my-model.yaml` | Where to write the config. |

```bash
nanoforge new-config --out configs/my-model.yaml
```

---

## Dataset tooling

All of these read the streaming, schema-aware dataset readers (text, JSON(L), CSV/TSV,
YAML, XML, SQLite, Parquet, Arrow, archives, HTTP, HF refs).

### `inspect-dataset`
Report formats, fields, schemas, detected text columns, and warnings.

| Flag | Default | Description |
| --- | --- | --- |
| `--input` | required | One or more files/dirs. |
| `--text-key` | `text` | JSON key holding text. |
| `--limit` | `1000` | Max records to scan. |

```bash
nanoforge inspect-dataset --input data/raw --limit 2000
```

### `validate-dataset`
Validate that readable text records exist; exits non-zero if there are errors or no records.

| Flag | Default | Description |
| --- | --- | --- |
| `--input` | required | Files/dirs to validate. |
| `--text-key` | `text` | JSON key holding text. |
| `--limit` | `1000` | Max records to check. |

```bash
nanoforge validate-dataset --input data/raw
```

### `clean-dataset`
Clean, normalize, and (by default) deduplicate inputs into a JSONL file.

| Flag | Default | Description |
| --- | --- | --- |
| `--input` | required | Files/dirs to clean. |
| `--out` | required | Output JSONL path. |
| `--text-key` | `text` | JSON key holding text. |
| `--text-column` | — | Structured text column(s); repeatable. |
| `--min-chars` | `16` | Drop records shorter than this. |
| `--max-chars` | — | Drop records longer than this. |
| `--collapse-whitespace` | off | Collapse runs of whitespace. |
| `--no-deduplicate` | off | Disable exact deduplication. |
| `--near-deduplicate` | off | Enable near-duplicate removal. |
| `--language` | — | Keep only this language. |

```bash
nanoforge clean-dataset --input data/raw --out data/clean/train.jsonl --collapse-whitespace
```

### `deduplicate-dataset`
Same as `clean-dataset` but with deduplication forced on (convenience alias).

```bash
nanoforge deduplicate-dataset --input data/raw --out data/clean/deduped.jsonl
```

### `convert-dataset`
Convert structured inputs into flat `txt` or `jsonl` text records.

| Flag | Default | Description |
| --- | --- | --- |
| `--input` | required | Files/dirs to convert. |
| `--out` | required | Output path. |
| `--format` | `jsonl` | `txt` or `jsonl`. |
| `--text-key` | `text` | JSON key holding text. |
| `--text-column` | — | Structured text column(s); repeatable. |

```bash
nanoforge convert-dataset --input data/raw --format txt --out data/clean/train.txt
```

---

## Tokenizers

### `train-tokenizer`
Train a BPE / WordPiece / SentencePiece / Unigram tokenizer from streaming records.

| Flag | Default | Description |
| --- | --- | --- |
| `--input` | required | Corpus files/dirs. |
| `--out` | required | Output tokenizer artifact. |
| `--type` | `bpe` | `bpe, native-bpe, python-bpe, wordpiece, sentencepiece, unigram`. |
| `--vocab-size` | `32000` | Target vocabulary size. |
| `--min-frequency` | `2` | Minimum merge frequency. |
| `--text-key` | `text` | JSON key holding text. |
| `--text-column` | — | Structured text column(s); repeatable. |
| `--max-records` | — | Cap records used for fitting. |
| `--dry-run` | off | Scan and report corpus health without fitting. |
| `--no-progress` | off | Disable the merge progress bar. |

- `bpe` uses the HuggingFace `tokenizers` Rust library.
- `python-bpe` writes a dependency-free Nanoforge byte-level BPE artifact.
- `native-bpe` uses the Rust extension when built, else falls back to the Python artifact.

```bash
nanoforge train-tokenizer --input data/raw --type native-bpe \
  --vocab-size 8000 --text-column messages --out data/tokenizers/chat-bpe.json
```

### `tokenizer-status`
Show whether the native Rust tokenizer acceleration is available.

```bash
nanoforge tokenizer-status
```

### `tokenizer-report`
Measure tokenizer compression (chars/token), vocabulary usage, and unknown-token rate.

| Flag | Default | Description |
| --- | --- | --- |
| `--input` | required | Corpus to measure on. |
| `--tokenizer` | `byte` | `byte, byte-native, bpe, python-bpe, native-bpe, wordpiece, sentencepiece`. |
| `--tokenizer-path` | — | Artifact path (required for BPE/WordPiece/SentencePiece). |
| `--text-key` | `text` | JSON key holding text. |
| `--limit` | `1000` | Max records. |
| `--out` | — | Optional path to save the JSON report. |

```bash
nanoforge tokenizer-report --input data/raw --tokenizer bpe \
  --tokenizer-path data/tokenizers/chat-bpe.json
```

### `benchmark-tokenizer`
Benchmark tokenizer throughput and memory use.

| Flag | Default | Description |
| --- | --- | --- |
| `--input` | required | Corpus. |
| `--tokenizer` | `byte-native` | Tokenizer type (same choices as above). |
| `--tokenizer-path` | — | Artifact path if needed. |
| `--text-key` | `text` | JSON key holding text. |
| `--text-column` | — | Structured text column(s); repeatable. |
| `--limit` | `1000` | Max records. |
| `--batch-size` | `64` | Batch size for encoding. |
| `--add-bos` / `--add-eos` | off | Add special tokens while benchmarking. |

```bash
nanoforge benchmark-tokenizer --input data/raw --tokenizer byte-native --limit 10000 --batch-size 256
```

---

## Data packing

### `prepare`
Tokenize and pack raw data into `train.bin` / `val.bin` (plus `.labels.bin` and manifests)
for training. This is the step the trainer actually consumes.

| Flag | Default | Description |
| --- | --- | --- |
| `--input` | required | Files/dirs to pack. |
| `--out` | required | Output packed directory. |
| `--tokenizer` | `byte` | `byte, byte-native, bpe, python-bpe, native-bpe, wordpiece, sentencepiece`. |
| `--tokenizer-path` | — | Artifact path (required for non-byte tokenizers). |
| `--val-fraction` | `0.01` | Fraction held out for validation. |
| `--code-only` | off | Keep only code records. |
| `--jsonl` | off | Force JSONL parsing. |
| `--jsonl-text-key` | `text` | JSON key holding text. |
| `--mode` | `auto` | `auto, generative, chat, instruct, completion, code, reasoning, hybrid`. |
| `--loss-masking` | `auto` | `auto, none, assistant_only, completion_only, partial`. |
| `--tokenizer-batch-size` | `256` | Batched tokenization throughput. |
| `--seq-len` | `512` | Packed sequence length (≤ `model.max_seq_len`). |
| `--min-chars` | `16` | Drop shorter records. |
| `--no-progress` | off | Disable the progress bar. |
| `--text-column` | — | Structured text column(s); repeatable. |

```bash
# Byte-tokenizer smoke pack
nanoforge prepare --input data/raw --tokenizer byte --out data/packed/tiny

# Chat pack with assistant-only labels
nanoforge prepare --input data/raw --tokenizer native-bpe \
  --tokenizer-path data/tokenizers/chat-bpe.json \
  --mode chat --loss-masking assistant_only --text-column messages \
  --seq-len 512 --out data/packed/chat
```

---

## Training

### `auto-train`
One command that inspects data, (optionally) trains a tokenizer, packs data with the right
masking, writes a CPU-friendly config, and starts training. See
[docs/training.md](training.md) for the full workflow.

Key flags (run `nanoforge auto-train --help` for all): `--input` (required), `--name`
(required), `--mode` (`auto` by default), `--tokenizer` (`native-bpe` by default),
`--vocab-size`, `--text-column`, `--seq-len`, `--max-steps`, `--ram`, `--speed`,
`--no-train` (setup only), `--dashboard`. Outputs go to `data/tokenizers/<name>*`,
`data/packed/<name>`, `configs/<name>.yaml`, and `runs/<name>`.

```bash
nanoforge auto-train --input data/raw --name ultrachat-18m --mode chat \
  --tokenizer native-bpe --vocab-size 8000 --text-column messages --seq-len 512
```

### `train`
Train from an existing YAML config.

| Flag | Default | Description |
| --- | --- | --- |
| `--config` | required | Path to a YAML config. |
| `--dashboard` | off | Launch the live web dashboard alongside training. |
| `--dashboard-host` | `127.0.0.1` | Dashboard bind host. |
| `--dashboard-port` | `7860` | Dashboard port. |

```bash
nanoforge train --config configs/tiny.yaml --dashboard
```

### `dashboard`
Serve the live dashboard for an existing run directory (reads `metrics.jsonl`).

| Flag | Default | Description |
| --- | --- | --- |
| `--run` | `runs/tiny` | Run directory to visualize. |
| `--host` | `127.0.0.1` | Bind host. |
| `--port` | `7860` | Port. |

```bash
nanoforge dashboard --run runs/tiny
```

Tracks train/val loss, perplexity, gradient norm (pre/post clip), learning rate, and
throughput at `http://127.0.0.1:7860`.

---

## Inference

`generate` and `chat` share the same sampling flags.

**Shared flags**

| Flag | Default | Description |
| --- | --- | --- |
| `--checkpoint` | — | Nanoforge checkpoint (`.pt`). |
| `--model` | — | Imported model name or external path/HF id (instead of `--checkpoint`). |
| `--prompt` | `""` | Prompt text. |
| `--device` | `auto` | `auto, cpu, cuda`. |
| `--max-new-tokens` | `256` | Generation length. |
| `--mode` | `balanced` | `balanced, chat, creative, coding, deterministic, low_memory, high_quality`. |
| `--temperature` | `0.8` | Sampling temperature. |
| `--top-k` | `50` | Top-k cutoff. |
| `--top-p` | `0.95` | Nucleus cutoff. |
| `--min-p` | — | Relative probability floor. |
| `--repetition-penalty` | `1.0` | Penalize repeated tokens. |
| `--frequency-penalty` | `0.0` | Penalize by frequency. |
| `--presence-penalty` | `0.0` | Penalize seen tokens. |
| `--no-repeat-ngram-size` | `0` | Block repeated n-grams. |
| `--deterministic` | off | Greedy decoding. |
| `--no-repetition-stop` | off | Disable runaway-repetition stopping. |
| `--repetition-window` | `64` | Repetition detector window. |
| `--repetition-threshold` | `0.85` | Tail repetition cutoff. |
| `--stop-token` | — | Extra stop string; repeatable. |
| `--system` | — | System prompt for chat (overrides the default). |
| `--mirostat` | off | Entropy-targeted sampling. |
| `--beams` | `1` | Beam count (`generate` only; >1 uses beam search). |

### `generate`
Generate a single completion. With `--beams > 1`, uses beam search (native checkpoints only).

```bash
nanoforge generate --checkpoint runs/tiny/best.pt --mode creative --prompt "Once upon a time"
```

### `chat`
Interactive REPL chat. Applies the chat template and (for chat-mode checkpoints) injects the
trained system prompt. Submit an empty line or Ctrl+C to exit.

```bash
nanoforge chat --checkpoint runs/nano-chat/best.pt
nanoforge chat --checkpoint runs/nano-chat/best.pt --deterministic --system "You are Nano."
```

### `serve`
Run the FastAPI inference server (OpenAI-style endpoints: `/v1/completions`,
`/v1/chat/completions`, plus `/health` and `/metrics`).

| Flag | Default | Description |
| --- | --- | --- |
| `--checkpoint` | required | Checkpoint to serve. |
| `--host` | `127.0.0.1` | Bind host. |
| `--port` | `8000` | Port. |
| `--device` | `auto` | Inference device. |

```bash
nanoforge serve --checkpoint runs/nano-chat/best.pt --port 8000
```

---

## Import external models

### `import`
Register an external model so `generate` / `chat` can run it via `--model <name>`.

| Flag | Default | Description |
| --- | --- | --- |
| `--model` | required | Path or HuggingFace id. |
| `--name` | required | Local name to register it under. |
| `--tokenizer` | — | Tokenizer path/name (for ONNX or custom imports). |
| `--backend` | auto | `llama_cpp, transformers, onnxruntime, safetensors`. |

Backends: GGUF → `llama-cpp-python`; HuggingFace dirs/Hub ids and SafeTensors →
`transformers`; ONNX → `onnxruntime` (needs an adjacent/specified HF tokenizer).

```bash
nanoforge import --model path/to/model.gguf --name my-gguf
nanoforge chat --model my-gguf --mode chat
```

---

## Export & evaluation

### `export`
Export a checkpoint to ONNX or a GGUF metadata manifest.

| Flag | Default | Description |
| --- | --- | --- |
| `--checkpoint` | required | Checkpoint to export. |
| `--out` | required | Output path. |
| `--format` | required | `onnx` or `gguf`. |
| `--device` | `cpu` | Export device. |

> `onnx` writes a logits-only graph. `gguf` writes a JSON manifest for converter tooling,
> not a binary `.gguf` file.

```bash
nanoforge export --checkpoint runs/tiny/best.pt --format onnx --out exports/tiny.onnx
```

### `evaluate`
Evaluate a checkpoint on packed validation data: loss, perplexity, token accuracy.

| Flag | Default | Description |
| --- | --- | --- |
| `--checkpoint` | required | Checkpoint. |
| `--data` | required | Packed `val.bin`. |
| `--seq-len` | `512` | Evaluation sequence length. |
| `--batches` | `20` | Number of batches. |
| `--device` | `auto` | Device. |

```bash
nanoforge evaluate --checkpoint runs/tiny/best.pt --data data/packed/tiny/val.bin --batches 20
```

### `benchmark`
Benchmark forward-pass throughput for a config.

| Flag | Default | Description |
| --- | --- | --- |
| `--config` | required | Config to benchmark. |
| `--batch-size` | `1` | Batch size. |
| `--steps` | `20` | Iterations. |
| `--device` | `auto` | Device. |

```bash
nanoforge benchmark --config configs/tiny.yaml --batch-size 4 --steps 50
```

### `profile-config`
Estimate parameters, FLOPs, and memory analytically from a config (no training run).

| Flag | Default | Description |
| --- | --- | --- |
| `--config` | required | Config to profile. |
| `--batch-size` | `1` | Batch size for estimates. |
| `--seq-len` | config value | Sequence length override. |
| `--bytes-per-param` | `2` | Bytes/param for the memory estimate. |

```bash
nanoforge profile-config --config configs/code-110m.yaml --batch-size 8 --seq-len 2048
```
