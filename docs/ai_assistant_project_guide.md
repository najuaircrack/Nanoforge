# Nanoforge AI Assistant Project Guide

This document explains how Nanoforge works so another AI assistant can help with the project
without guessing. It is specific to this repository.

## Project Purpose

Nanoforge is a local-first transformer framework for training and running small to medium
decoder-only language models. The goal is not to clone a frontier model. The goal is to make a
clean, inspectable, trainable, and extensible LLM stack that works on consumer CPUs/GPUs and can
grow toward distributed training later.

The codebase currently supports:

- GPT-style decoder-only transformer training
- RoPE and ALiBi positional options
- RMSNorm
- SwiGLU and GEGLU feed-forward blocks
- Grouped-query attention and multi-query attention through `n_kv_heads`
- KV cache inference
- optional MoE feed-forward layers
- LoRA-ready linear layers
- streaming dataset ingestion and packed memmap datasets
- byte, BPE, WordPiece, SentencePiece, and Unigram tokenizer paths
- AdamW, Lion, Adafactor, and SophiaG optimizer options
- mixed precision training with bf16/fp16 logic
- gradient accumulation, clipping, warmup, cosine LR decay, EMA, checkpointing
- live dashboard backed by `metrics.jsonl`
- generation CLI, API server, ONNX export, and GGUF manifest export

## Important Commands

Install:

```powershell
cd C:\Users\USER\Documents\Nanoforge
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[all]"
```

Inspect data:

```powershell
nanoforge inspect-dataset --input data/raw --limit 2000
```

Pack data with the byte tokenizer:

```powershell
nanoforge prepare --input data/raw --tokenizer byte --out data/packed/tiny
```

Train with dashboard:

```powershell
nanoforge train --config configs/tiny.yaml --dashboard
```

Open dashboard for an existing run:

```powershell
nanoforge dashboard --run runs/tiny
```

Generate:

```powershell
nanoforge generate --checkpoint runs/tiny/best.pt --prompt "def hello_world():"
```

Evaluate:

```powershell
nanoforge evaluate --checkpoint runs/tiny/best.pt --data data/packed/tiny/val.bin --seq-len 256 --batches 20
```

## Repository Map

Core files:

- `src/nanoforge/config.py`: dataclass config system and YAML loading.
- `src/nanoforge/cli.py`: command-line interface.
- `src/nanoforge/model/transformer.py`: full causal LM model and transformer block.
- `src/nanoforge/model/attention.py`: causal self-attention, GQA/MQA, RoPE/ALiBi use, KV cache.
- `src/nanoforge/model/rope.py`: rotary embedding cache and scaling.
- `src/nanoforge/model/norms.py`: RMSNorm.
- `src/nanoforge/model/moe.py`: SwiGLU/GEGLU feed-forward and optional MoE feed-forward.
- `src/nanoforge/model/lora.py`: LoRA-capable linear layers.
- `src/nanoforge/data/formats.py`: streaming dataset readers and automatic format detection.
- `src/nanoforge/data/cleaning.py`: UTF-8 cleanup, normalization, exact and near deduplication.
- `src/nanoforge/data/packing.py`: streaming tokenization and sharded binary packing.
- `src/nanoforge/data/dataset.py`: memory-mapped token dataset used during training.
- `src/nanoforge/data/tokenizer.py`: tokenizer adapters and tokenizer training helpers.
- `src/nanoforge/training/trainer.py`: training loop.
- `src/nanoforge/training/utils.py`: device selection, LR schedule, optimizer groups, grad checks.
- `src/nanoforge/training/optimizers.py`: AdamW factory plus Lion, Adafactor, SophiaG.
- `src/nanoforge/generation/engine.py`: inference engine, streaming generation, beam search.
- `src/nanoforge/generation/sampling.py`: temperature, top-k, top-p, repetition penalty, Mirostat.
- `src/nanoforge/dashboard.py`: local FastAPI dashboard UI.
- `src/nanoforge/progress.py`: JSONL metric logging and sanitization.
- `src/nanoforge/evaluation/metrics.py`: checkpoint evaluation.
- `src/nanoforge/export/onnx.py`: ONNX export.
- `src/nanoforge/export/gguf.py`: GGUF converter manifest.
- `configs/tiny.yaml`: current tiny training config.
- `configs/cpu-tiny.yaml`: safer CPU tiny config.
- `configs/code-110m.yaml`: larger code-model preset.

Docs:

- `README.md`: quick start and project summary.
- `docs/train_model_steps.md`: practical training guide.
- `docs/architecture.md`: short architecture overview.
- `docs/training_guide.md`: scaling and dataset advice.
- `docs/ai_assistant_project_guide.md`: this file.

## Configuration Flow

Nanoforge loads YAML into dataclasses in `config.py`.

The top-level config is `NanoforgeConfig`, which contains:

- `ModelConfig`
- `TrainConfig`
- `DataConfig`
- `InferenceConfig`

Important `ModelConfig` fields:

- `vocab_size`: must match tokenizer vocabulary.
- `max_seq_len`: maximum context length used by the model.
- `d_model`: residual stream width.
- `n_layers`: transformer block count.
- `n_heads`: query attention heads.
- `n_kv_heads`: key/value heads. If `1`, the model is MQA. If less than `n_heads`, it is GQA.
- `ffn_hidden_mult`: feed-forward expansion.
- `rope_theta`: RoPE base.
- `position_embedding`: `rope` or `alibi`.
- `sliding_window`: optional local attention window.
- `moe`: optional MoE config.
- `lora_rank`, `lora_alpha`, `lora_dropout`: LoRA adapter settings.

Important `TrainConfig` fields:

- `learning_rate`: peak LR.
- `min_learning_rate`: final cosine-decayed LR.
- `warmup_steps`: explicit warmup steps.
- `warmup_ratio`: used when `warmup_steps <= 0`.
- `optimizer`: usually `adamw`.
- `betas`: stable default is `[0.9, 0.95]`.
- `eps`: AdamW epsilon, default `1e-8`.
- `weight_decay`: default `0.1`.
- `grad_clip`: default `1.0`.
- `precision`: `auto`, `bf16`, `fp16`, or `fp32`.
- `micro_batch_size`: actual per-step batch before accumulation.
- `grad_accum_steps`: accumulation steps before optimizer step.
- `eval_interval`, `eval_steps`, `save_interval`: evaluation/checkpoint schedule.

Important `DataConfig` fields:

- `train_path`: usually `data/packed/<name>/train.bin`.
- `val_path`: usually `data/packed/<name>/val.bin`.
- `tokenizer_type`: `byte`, `bpe`, `wordpiece`, or `sentencepiece`.
- `tokenizer_path`: path to tokenizer file, or `null` for byte tokenizer.
- `seq_len`: training sequence length.

## Data Pipeline

The data pipeline is designed to avoid loading large corpora into RAM.

The path is:

1. User runs `nanoforge prepare`.
2. `cli.py` calls `build_packed_dataset`.
3. `dataset.py` delegates to `build_packed_dataset_streaming`.
4. `formats.py` detects and streams records from files/directories/archives/HTTP/HF refs.
5. `cleaning.py` normalizes text, removes bad records, and deduplicates.
6. `packing.py` tokenizes each record and writes binary shards incrementally.
7. `PackedMemmapDataset` reads the packed shards with NumPy memmap during training.

Supported raw formats:

- text, markdown, source files
- JSON and JSONL
- CSV and TSV
- YAML
- XML
- SQLite
- ZIP and TAR archives
- HTTP text/JSONL
- Hugging Face streaming references
- Parquet and Arrow if `pyarrow` is installed

Packed dataset outputs:

- `train.bin`
- `val.bin`
- optional `train.00001.bin` and `val.00001.bin` shards
- `train.manifest.json`
- `val.manifest.json`
- `packing.manifest.json`

The trainer reads only packed binary files. It does not train directly from raw text.

## Tokenizer System

Tokenizer code lives in `src/nanoforge/data/tokenizer.py`.

Available tokenizer paths:

- `ByteTokenizer`: no dependency, 260 vocab IDs. It is best for smoke tests.
- `TokenizersBPE`: uses Hugging Face `tokenizers`.
- `WordPieceTokenizer`: uses Hugging Face `tokenizers`.
- `SentencePieceTokenizer`: supports SentencePiece BPE or Unigram.

Important rule:

`model.vocab_size` must be at least the tokenizer vocab size. For byte tokenizer, use `260`, not `256`.

Tokenizer quality checks:

```powershell
nanoforge tokenizer-report --input data/raw --tokenizer byte
nanoforge tokenizer-report --input data/raw --tokenizer bpe --tokenizer-path data/tokenizers/code-bpe.json
```

Good signs:

- `unk_rate` near zero
- reasonable `chars_per_token`
- no strange over-fragmentation of code symbols

## Transformer Forward Pass

The model class is `NanoforgeForCausalLM`.

Forward pass:

1. `input_ids` enter `self.embed`.
2. Embeddings pass through dropout.
3. Each `TransformerBlock` runs attention and feed-forward sublayers.
4. Final RMSNorm normalizes the residual stream.
5. `lm_head` projects hidden states to vocabulary logits.
6. If `labels` are passed, cross-entropy loss is computed.

Shape convention:

- `input_ids`: `[batch, sequence]`
- hidden state: `[batch, sequence, d_model]`
- logits: `[batch, sequence, vocab_size]`

The model is causal. Token `t` can attend only to tokens at positions `<= t`.

## Transformer Block

Each block in `transformer.py` follows:

1. RMSNorm before attention.
2. Causal self-attention.
3. Residual add with residual scaling.
4. RMSNorm before feed-forward.
5. Feed-forward or MoE feed-forward.
6. Residual add with residual scaling.

This is a pre-norm transformer. Pre-norm is more stable for training than post-norm in many
small and medium decoder models.

## Attention

Attention code lives in `model/attention.py`.

The attention module creates:

- `q_proj`: query projection with `n_heads`
- `k_proj`: key projection with `n_kv_heads`
- `v_proj`: value projection with `n_kv_heads`
- `o_proj`: output projection

GQA/MQA behavior:

- If `n_kv_heads == n_heads`, it is standard multi-head attention.
- If `n_kv_heads < n_heads`, it is grouped-query attention.
- If `n_kv_heads == 1`, it is multi-query attention.

Why GQA matters:

GQA reduces KV cache size during inference. This is important for long contexts and low VRAM.

Position handling:

- Default is RoPE.
- ALiBi is available via `position_embedding: alibi`.

Flash attention:

The implementation uses PyTorch `scaled_dot_product_attention` when available and when the
attention mode is compatible. This lets PyTorch choose efficient kernels on supported hardware.

KV cache:

During generation, each layer stores previous keys and values in `KVCache`. New tokens append
to that cache so generation does not recompute the entire prompt every step.

## RoPE

RoPE code lives in `model/rope.py`.

RoPE rotates query and key vectors based on token position. It is applied before attention
scores are computed. Nanoforge caches cosine/sine tensors by device, dtype, and sequence length.

Supported scaling knobs:

- linear scaling
- dynamic NTK-style scaling
- YaRN-style scaling

Use RoPE for most code and general language models. Use ALiBi mostly for experiments or simple
long-context extrapolation tests.

## Feed-Forward

Feed-forward code lives in `model/moe.py`.

Default FFN:

```text
SwiGLU: w2(silu(w1(x)) * w3(x))
```

GEGLU is also available by setting `activation: geglu`.

Why SwiGLU:

SwiGLU usually gives better quality per parameter than plain GELU MLPs. It is common in modern
LLM architectures.

MoE:

If `model.moe` is configured, `MoEFeedForward` routes tokens through top-k experts and returns
an auxiliary router loss. MoE increases capacity but can destabilize small experiments, so it
should stay off until dense training is healthy.

## Normalization

RMSNorm is implemented in `model/norms.py`.

RMSNorm normalizes by root mean square without subtracting the mean. It is cheaper than
LayerNorm and widely used in modern decoder-only LLMs.

## LoRA

LoRA support is in `model/lora.py`.

`LoRALinear` wraps projection layers with optional low-rank adapters. If `lora_rank > 0`, each
linear projection adds a trainable low-rank update:

```text
output = base_linear(x) + lora_b(lora_a(dropout(x))) * scaling
```

Use LoRA for instruction tuning or style tuning when the base model is already trained.

## Training Loop

Training code lives in `training/trainer.py`.

Training loop:

1. Load config.
2. Seed randomness.
3. Resolve device.
4. Build model.
5. Build optimizer with decay/no-decay parameter groups.
6. Load packed train and validation datasets.
7. Reset `metrics.jsonl` for a fresh dashboard run.
8. For each step, compute LR with warmup and cosine decay.
9. Accumulate gradients for `grad_accum_steps`.
10. Unscale gradients if fp16 GradScaler is active.
11. Measure grad norm before clipping.
12. Skip optimizer step if loss or gradients are NaN/Inf.
13. Clip gradients to `grad_clip`.
14. Measure grad norm after clipping.
15. Optimizer step.
16. GradScaler update if fp16.
17. EMA update if enabled.
18. Log metrics.
19. Run validation at `eval_interval`.
20. Save best and periodic checkpoints.

Mixed precision:

- `precision: auto` chooses bf16 on CUDA if supported, otherwise fp16.
- CPU uses fp32.
- fp16 uses GradScaler.
- bf16 does not need GradScaler.

Gradient stability:

- `grad_clip: 1.0` prevents runaway update magnitudes.
- non-finite loss or gradients cause the optimizer step to be skipped.
- dashboard logs pre-clip and post-clip grad norm.

Optimizer defaults:

- AdamW
- betas `(0.9, 0.95)`
- eps `1e-8`
- weight decay `0.1`
- cosine scheduler
- warmup ratio `0.03`

## Learning Rate Schedule

LR schedule is in `training/utils.py`.

Behavior:

1. During warmup, LR linearly increases from near zero to `learning_rate`.
2. After warmup, LR follows cosine decay to `min_learning_rate`.

If `warmup_steps <= 0`, trainer computes:

```text
warmup_steps = max(1, int(max_steps * warmup_ratio))
```

For unstable runs, reduce `learning_rate` first. A 2x to 4x reduction is often more effective
than changing architecture.

## Checkpoints

Checkpoint code lives in `training/checkpoint.py`.

Checkpoint payload includes:

- model state dict
- optimizer state dict
- config object
- step
- validation loss
- EMA state if enabled

Common files:

- `runs/<name>/best.pt`
- `runs/<name>/last.pt`
- `runs/<name>/step-<n>.pt`

Each checkpoint also writes a YAML copy of the config beside it.

## Metrics And Dashboard

Metric logging lives in `progress.py`.

The trainer writes:

```text
runs/<name>/metrics.jsonl
```

Every new run resets this file. The old file is backed up with a timestamped name so the
dashboard starts clean after canceling and restarting training.

Dashboard code lives in `dashboard.py`.

Dashboard shows:

- progress bar
- train loss
- validation loss
- perplexity
- tokens/sec
- learning rate
- grad norm before and after clipping
- recent events

Run:

```powershell
nanoforge train --config configs/tiny.yaml --dashboard
```

or:

```powershell
nanoforge dashboard --run runs/tiny
```

## Inference

Inference code lives in `generation/engine.py`.

Generation flow:

1. Encode prompt.
2. Run model on prompt with `use_cache=True`.
3. Sample or select next token from final logits.
4. Feed only the newest token back into the model with layer KV caches.
5. Decode generated token IDs to text.

Sampling code lives in `generation/sampling.py`.

Supported sampling:

- temperature
- top-k
- top-p
- repetition penalty
- Mirostat

Beam search is available in `GenerationEngine.beam_search`.

Important limitation:

Speculative decoding currently exists as a compatibility hook but falls back to target-model
generation. It is not yet a full verified speculative decoder.

## Evaluation

Evaluation code lives in `evaluation/metrics.py`.

It loads a checkpoint and packed validation data, then reports:

- loss
- perplexity
- token accuracy

Run:

```powershell
nanoforge evaluate --checkpoint runs/tiny/best.pt --data data/packed/tiny/val.bin --seq-len 256 --batches 20
```

Use evaluation metrics together with generated samples. Loss alone can be misleading if the
dataset is tiny, duplicated, or not representative.

## Export

ONNX export:

- file: `export/onnx.py`
- command: `nanoforge export --format onnx`

GGUF manifest:

- file: `export/gguf.py`
- command: `nanoforge export --format gguf`

The GGUF path currently writes a manifest for converter tooling. It is not a complete direct
binary GGUF writer.

## Why The Transformer Works

Nanoforge is an autoregressive language model. During training, it receives a sequence of token
IDs and learns to predict the next token at every position.

Example:

```text
input:  [BOS, "def", " ", "add", "("]
target: ["def", " ", "add", "(", "a"]
```

The model does this by:

1. Converting token IDs to vectors.
2. Mixing information across previous tokens with causal attention.
3. Transforming each position independently with feed-forward layers.
4. Repeating this for several transformer blocks.
5. Projecting final hidden states into vocabulary logits.
6. Using cross-entropy loss to make the correct next token more likely.

Causal masking is critical. Without it, the model could cheat by seeing future target tokens.

RoPE or ALiBi gives the model information about token positions. Without positional information,
attention would not know token order.

RMSNorm and pre-norm blocks improve optimization stability.

Residual connections let information and gradients move through many layers.

SwiGLU increases useful nonlinear capacity.

GQA reduces inference memory while keeping attention quality close to full multi-head attention.

## How To Debug Training

If loss explodes:

- reduce LR by 2x to 4x
- keep `grad_clip: 1.0`
- inspect `train/grad_norm_before_clip`
- confirm `train/grad_norm_after_clip` is near `1.0`
- increase warmup
- check for corrupt or repeated data

If validation loss is much lower than training loss:

- validation set may be too small
- validation set may duplicate training data
- training data may contain harder/noisier samples

If validation loss is much higher than training loss:

- model is overfitting
- validation distribution differs from train
- dataset may be too small
- dropout or weight decay may need tuning

If generated text repeats:

- increase data diversity
- deduplicate more aggressively
- use repetition penalty at inference
- lower temperature if output is chaotic
- increase temperature if output is stuck

If training is slow on CPU:

- use `configs/cpu-tiny.yaml`
- keep `seq_len` at 128 or 256
- use byte tokenizer first
- keep `micro_batch_size: 1`
- increase `grad_accum_steps` instead of batch size

## Safe Modification Points

Good places for another AI assistant to modify:

- Add dataset formats in `data/formats.py`.
- Add cleaning filters in `data/cleaning.py`.
- Add tokenizer metrics in `data/tokenizer_metrics.py`.
- Add optimizer options in `training/optimizers.py`.
- Add metrics in `training/trainer.py` and `dashboard.py`.
- Add sampling methods in `generation/sampling.py`.
- Add evaluation tasks in `evaluation/`.
- Add config presets in `configs/`.

Be careful when modifying:

- `model/attention.py`, because shape logic for GQA and KV cache is easy to break.
- `model/transformer.py`, because checkpoint compatibility depends on module names.
- `training/checkpoint.py`, because old checkpoints must remain loadable.
- `data/dataset.py`, because memmap dtype and shard manifest logic must stay aligned with
  `data/packing.py`.

## Current Known Limitations

Nanoforge is still an experimental local framework. Important limitations:

- FSDP/DeepSpeed/tensor parallelism are not fully implemented yet.
- GGUF export is a manifest path, not a full direct writer.
- Speculative decoding is a placeholder hook.
- Dashboard is local and reads JSONL metrics; it is not a multi-user experiment tracker.
- MoE is available but should be treated as experimental.
- No full benchmark harness for HumanEval, MMLU, or GSM8K yet.

## Best Next Improvements

Highest-impact next work:

- Add resume training from checkpoint.
- Add checkpoint metadata page to dashboard.
- Add dataset split by repository instead of random record split.
- Add sample-generation callback during validation.
- Add HumanEval-style code evaluation harness.
- Add true speculative decoding.
- Add FSDP wrapper for multi-GPU training.
- Add better BPE tokenizer presets for code.
- Add config validation warnings before training starts.

## Quick Context Prompt For Another AI

Use this prompt when asking another AI to help:

```text
You are assisting with Nanoforge, a local-first decoder-only transformer framework in Python/PyTorch.
Read docs/ai_assistant_project_guide.md first. The core model is in src/nanoforge/model/transformer.py
and attention is in src/nanoforge/model/attention.py. Training is in src/nanoforge/training/trainer.py.
Data ingestion and packing are streaming and live in src/nanoforge/data/. The dashboard reads
runs/<name>/metrics.jsonl via src/nanoforge/dashboard.py. Keep changes minimal, preserve checkpoint
compatibility where possible, and run pytest after edits.
```

