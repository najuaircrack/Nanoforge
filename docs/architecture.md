# Architecture

How Nanoforge works internally. This is the map for reading and modifying the code; for
commands see [cli.md](cli.md), for config fields see [configuration.md](configuration.md),
and for training workflows see [training.md](training.md).

Nanoforge is a from-scratch decoder-only (GPT-style) transformer in PyTorch. The whole stack
— model, tokenizers, data pipeline, training loop, inference, CLI — is meant to be small
enough to read end to end.

---

## Repository map

```text
src/nanoforge/
├── cli.py              # argparse CLI, entry point `nanoforge.cli:main`
├── config.py           # dataclass config system + YAML loader
├── registry.py         # typed component registries + plugin discovery
├── progress.py         # JSONL metric logging + sanitization
├── server.py           # FastAPI inference server
├── dashboard.py        # live training dashboard (reads metrics.jsonl)
├── model/
│   ├── transformer.py  # NanoforgeForCausalLM + TransformerBlock
│   ├── attention.py    # CausalSelfAttention, GQA/MQA, KV cache
│   ├── rope.py         # rotary embedding cache + scaling
│   ├── norms.py        # RMSNorm, LayerNorm
│   ├── moe.py          # SwiGLU/GEGLU FFN + MoE FFN
│   ├── lora.py         # LoRA-capable linear layers
│   └── registry.py     # model-component registrations
├── data/
│   ├── formats.py      # streaming, schema-aware dataset readers
│   ├── cleaning.py     # UTF-8 cleanup, normalization, dedup
│   ├── packing.py      # streaming tokenization → binary shards
│   ├── dataset.py      # PackedMemmapDataset (memmap reader)
│   ├── tokenizer.py    # tokenizer adapters + training helpers
│   └── modes.py        # mode/masking resolution
├── training/
│   ├── trainer.py      # training loop
│   ├── optimizers.py   # AdamW, Lion, Adafactor, SophiaG
│   ├── schedulers.py   # cosine / linear / constant + warmup
│   ├── checkpoint.py   # atomic checkpoint save/load
│   └── health.py       # grad-spike / NaN health monitor
├── generation/
│   ├── engine.py       # inference engine, streaming, beam search
│   └── sampling.py     # temperature/top-k/top-p/penalties/mirostat
├── evaluation/metrics.py  # loss, perplexity, token accuracy
└── export/
    ├── onnx.py         # ONNX (logits-only graph)
    └── gguf.py         # GGUF metadata manifest (not a binary .gguf)
```

Configs live in `configs/`, helper scripts in `scripts/`, the pytest suite in `tests/`, and
the optional Rust tokenizer extension in `native/nanoforge-tokenizers`.

---

## The transformer block

The default block (`model/transformer.py`) is pre-norm:

1. RMSNorm
2. Causal self-attention (GQA + RoPE, optional sliding window)
3. Residual add with residual scaling
4. RMSNorm
5. SwiGLU feed-forward (or GEGLU / MoE)
6. Residual add with residual scaling

Pre-norm is more stable to train than post-norm for small/medium decoders. `block_type` can
also be `parallel_residual` (GPT-NeoX style: attention and FFN read the same normalized
input and their outputs are summed into the residual once).

Residual scaling defaults to `1/sqrt(2·n_layers)` (`residual_scale: null`), which keeps the
residual stream variance controlled as depth grows.

### Forward pass

The model class is `NanoforgeForCausalLM`:

1. `input_ids` `[batch, seq]` → embedding → dropout.
2. Each `TransformerBlock` runs attention then feed-forward.
3. Final normalization over the residual stream.
4. `lm_head` projects hidden states `[batch, seq, d_model]` → logits `[batch, seq, vocab]`.
   Output weights are tied to the input embedding by default (`tie_embeddings: true`).
5. If `labels` are passed, cross-entropy loss is computed (positions labeled `-100` are
   ignored — this is how assistant-only / completion-only masking works).

The model is causal: token `t` attends only to positions `≤ t`.

---

## Attention (`model/attention.py`)

Projections: `q_proj` (`n_heads`), `k_proj`/`v_proj` (`n_kv_heads`), `o_proj`.

- `n_kv_heads == n_heads` → standard multi-head attention.
- `1 < n_kv_heads < n_heads` → grouped-query attention (GQA).
- `n_kv_heads == 1` → multi-query attention (MQA).

GQA/MQA shrink the KV cache during inference, which is what matters for long context and low
memory. `n_heads` must be divisible by `n_kv_heads`, and `d_model` by `n_heads` (enforced in
`ModelConfig.__post_init__`).

Attention runs through PyTorch `scaled_dot_product_attention` (SDPA) when available, with a
manual softmax fallback path. During generation, each layer stores previous keys/values in a
`KVCache` so new tokens don't recompute the whole prefix.

> **Honest note on attention backends.** The registry lists `sdpa, flash, flash_v2, flash_v3,
> manual, sliding_window, chunked, sparse, hybrid_local_global, paged`, but they are **not**
> distinct kernels. `sdpa`/`flash*` all use the base `CausalSelfAttention` via SDPA; `manual`
> uses the explicit softmax path; `sliding_window` sets a local window; and `chunked`,
> `sparse`, `hybrid_local_global`, and `paged` are thin subclasses that just **preset a
> sliding window**. There is no FlashAttention-2/3 kernel and no real sparse/paged attention.

---

## Position embeddings (`model/rope.py`)

Default is RoPE: query/key vectors are rotated by position before attention scores are
computed. Cosine/sine tensors are cached by device, dtype, and sequence length.

RoPE scaling (`rope_scaling`) supports `linear`, `dynamic`, and `yarn`. **NTK scaling is not
implemented** — passing it raises. `alibi` and `none` are the other `position_embedding`
options; ALiBi is mostly for long-context extrapolation experiments.

---

## Feed-forward and MoE (`model/moe.py`)

Default FFN is SwiGLU: `w2(silu(w1(x)) * w3(x))`. `GEGLU` is available via
`activation: geglu`. SwiGLU typically gives better quality per parameter than a plain
GELU MLP.

If `model.moe` is set, `MoEFeedForward` routes each token through top-k experts and adds a
load-balancing auxiliary loss (`router_aux_loss_coef`). MoE adds capacity but destabilizes
small runs — get dense training healthy first.

> `mamba_selective_state` and `rwkv_mixing` are also registered as `ffn_type` values, but
> they are **experimental FFN mixers**, not full SSM/RWKV implementations.

---

## Normalization and LoRA

`RMSNorm` (default) normalizes by root-mean-square without subtracting the mean — cheaper
than LayerNorm. `LayerNorm` is available via `normalization: layernorm`.

`LoRALinear` (`model/lora.py`) wraps projections with an optional low-rank adapter. When
`lora_rank > 0`: `output = base_linear(x) + lora_b(lora_a(dropout(x))) * scaling`. Use it for
instruction/style tuning on top of an already-trained base.

---

## Data pipeline

Nanoforge treats every dataset as a stream of `DatasetRecord` objects and never loads a full
corpus into RAM. The `prepare` path is:

1. `cli.py` calls `build_packed_dataset`.
2. `dataset.py` delegates to the streaming builder.
3. `formats.py` detects and streams records from files/dirs/archives/HTTP/HF refs.
4. `cleaning.py` normalizes text, drops bad records, deduplicates.
5. `packing.py` tokenizes each record and writes binary shards incrementally.
6. `PackedMemmapDataset` reads the shards with NumPy memmap during training.

Supported raw inputs: text/markdown/source, JSON(L), CSV/TSV, YAML, XML, SQLite, ZIP/TAR
archives, HTTP text/JSONL, Hugging Face streaming refs, and Parquet/Arrow (if `pyarrow` is
installed).

Packed outputs: `train.bin` / `val.bin` (+ numbered shards), matching `.labels.bin` for
masked modes, and `train.manifest.json` / `val.manifest.json` / `packing.manifest.json`. The
trainer only ever reads packed binaries, never raw text.

### Boundary-aware chat/instruct packing

Chat and instruct packing writes fixed-length sequences that start at conversation
boundaries. Each sequence carries the user/system context but labels only the
assistant/completion tokens (everything else is `-100`). Long assistant responses are split
across sequences with the context prefix repeated, so no batch is 100% masked or unmasked.
Tokenizer training reads the same record stream as packing, so BPE/WordPiece/SentencePiece
fitting stays aligned with preprocessing and never tokenizes raw container bytes.

---

## Tokenizers (`data/tokenizer.py`)

- `ByteTokenizer` — no dependencies, vocab 260 (not 256 — includes special IDs). Best for
  smoke tests. `model.vocab_size` must be ≥ the tokenizer vocab.
- HuggingFace `tokenizers` — BPE, WordPiece, SentencePiece/Unigram.
- Pure-Python byte-level BPE fallback (`python-bpe`), plus `byte-native` / `native-bpe` that
  use the optional Rust extension when built and fall back to Python otherwise. Both keep
  packed data and checkpoints compatible regardless of which backend fitted them.

Check quality with `nanoforge tokenizer-report`: look for near-zero unknown rate, reasonable
chars/token, and no over-fragmentation.

---

## Training loop (`training/trainer.py`)

1. Load config, seed RNG, resolve device, build model.
2. Build optimizer with decay / no-decay parameter groups.
3. Load packed train/val datasets; reset `metrics.jsonl` (old file backed up timestamped).
4. Per step: compute LR (warmup → schedule), accumulate over `grad_accum_steps`.
5. Unscale (fp16 GradScaler), measure pre-clip grad norm.
6. Skip/rollback the step if loss or grads are non-finite (`rollback_on_nonfinite`).
7. Clip to `grad_clip`, measure post-clip grad norm, optimizer step.
8. GradScaler update (fp16), EMA update if enabled, log metrics.
9. Validate at `eval_interval`; save best + periodic checkpoints.

**Mixed precision:** `auto` → bf16 on capable CUDA, else fp16; CPU forces fp32. fp16 uses a
GradScaler, bf16 does not. **Stability:** `grad_clip: 1.0`, non-finite skip/rollback, and an
adaptive-clip health monitor (`training/health.py`) that watches grad-norm spikes.

**LR schedule** (`training/schedulers.py`): linear warmup to `learning_rate`, then cosine
(or linear/constant) decay to `min_learning_rate`. When `warmup_steps <= 0`,
`warmup_steps = max(1, int(max_steps * warmup_ratio))`. For unstable runs, cut LR 2–4× before
touching architecture.

**Optimizers** (`training/optimizers.py`): `adamw` (default), `lion`, `adafactor`,
`sophia`/`sophiag` — all real implementations.

---

## Checkpoints (`training/checkpoint.py`)

Payload: model state dict, optimizer state, config object, step, validation loss, and EMA
state if enabled. Written atomically with an integrity hash and a YAML config sidecar beside
each `.pt`. Common files: `runs/<name>/best.pt`, `last.pt`, `step-<n>.pt`. `async_checkpoint`
saves on a background thread.

---

## Inference (`generation/engine.py`, `generation/sampling.py`)

Generation: encode prompt → run with `use_cache=True` → sample next token → feed only the
newest token back with per-layer KV caches → decode. `build_chat_prompt` / `chat_stream`
apply the chat template and stream tokens; `beam_search` is real but runs without a KV cache
(slow); prefix caching is supported.

Sampling knobs: temperature, top-k, top-p, min-p, repetition/frequency/presence penalties,
no-repeat-ngram, and Mirostat (v2-style; `mirostat_version` is stored but not branched on).
Preset `mode`s (`balanced, chat, creative, coding, deterministic, low_memory, high_quality`)
apply bundles of these; explicit flags override the preset. Runaway-repetition and
EOS/role-boundary stopping guard against loops.

> **Speculative decoding is not a feature** — it exists only as unreachable hook code that
> falls back to normal generation. Do not rely on it.

---

## Registries and plugins (`registry.py`, `model/registry.py`)

Components are resolved through typed registries: transformer blocks, attention backends, FFN
backends, position embeddings, normalization, optimizers, schedulers, tokenizers, samplers,
and quantization backends. Registries support lazy `module:attribute` targets, aliases,
metadata, and validation. `ModelConfig`/`TrainConfig`/`DataConfig`/`InferenceConfig` validate
registry-backed keys on load, so a bad key fails before training starts. Run
`nanoforge registries` to list everything.

External packages can register components without editing core code via an entry point:

```toml
[project.entry-points."nanoforge.plugins"]
my_plugin = "my_package.nanoforge_plugin:register"
```

The callable receives `ALL_REGISTRIES` and registers new components into it.

---

## Export and quantization

- **ONNX** (`export/onnx.py`): real, logits-only graph (no KV cache).
- **GGUF** (`export/gguf.py`): writes a JSON **manifest** for external converter tooling —
  not a binary `.gguf` file.
- **Quantization** (`quantization.py`): `int8` dynamic CPU quantization is real; `int4`,
  `gptq`, `awq`, and `gguf` are metadata hooks only (no actual weight quantization).

`distillation.py` and `preferences.py` (DPO) contain correct standalone functions but are
**not wired** into any CLI or trainer path.

---

## Native extension

`native/nanoforge-tokenizers` is an optional Rust/PyO3 extension (import name
`nanoforge_tokenizers`, built with maturin via the `native` extra). When installed, the
`byte-native` and `native-bpe` tokenizer paths use it for faster parallel batch tokenization
and native byte-level BPE; otherwise they fall back to the Python implementations. Native
modules are always optional — the Python fallbacks keep the framework usable without a
compiler toolchain.

---

## Safe modification points

Low-risk places to extend:

- Dataset formats in `data/formats.py`; cleaning filters in `data/cleaning.py`.
- Optimizers in `training/optimizers.py`; metrics in `training/trainer.py` + `dashboard.py`.
- Sampling methods in `generation/sampling.py`; evaluation tasks in `evaluation/`.
- Config presets in `configs/`. New components via a registry (or an external plugin).

Modify with care (compatibility-sensitive):

- `model/attention.py` — GQA and KV-cache shape logic is easy to break.
- `model/transformer.py` — checkpoint compatibility depends on module names.
- `training/checkpoint.py` — old checkpoints must stay loadable.
- `data/dataset.py` — memmap dtype and shard manifest logic must match `data/packing.py`.
