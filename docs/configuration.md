# Configuration Reference

Nanoforge configs are YAML files with four sections — `model`, `training`, `data`, and
`inference` — each mapping to a dataclass in
[`src/nanoforge/config.py`](../src/nanoforge/config.py). The tables below list **every field
and its exact default**.

Loading (`load_config`): PyYAML is used if installed, otherwise a small built-in parser
handles simple configs. **Unknown keys are silently dropped**, so a typo'd field name is
ignored rather than raising — validate with `nanoforge validate-config --config <file>`.

Validation happens on load: registry-backed keys (`block_type`, `attention_backend`,
`ffn_type`, `normalization`, `position_embedding`, `quantization_backend`, `optimizer`,
`scheduler`, `tokenizer_type`, `sampler`) are checked against the registries, and the model
enforces `n_heads % n_kv_heads == 0` and `d_model % n_heads == 0`.

Only override what you need — every field has a default. A minimal config is valid:

```yaml
model:
  vocab_size: 260
  d_model: 128
  n_layers: 2
  n_heads: 4
data:
  tokenizer_type: byte
  seq_len: 128
```

---

## `model` — `ModelConfig`

| Field | Default | Notes |
| --- | --- | --- |
| `vocab_size` | `32000` | Must match your tokenizer. Byte tokenizer uses `260`. |
| `max_seq_len` | `2048` | Maximum context length. |
| `d_model` | `512` | Hidden width. Must be divisible by `n_heads`. |
| `n_layers` | `8` | Number of transformer blocks. |
| `n_heads` | `8` | Query attention heads. |
| `n_kv_heads` | `null` | KV heads for GQA/MQA. `null` → equals `n_heads` (full MHA); `1` → MQA. Must divide `n_heads`. |
| `ffn_hidden_mult` | `2.67` | Feed-forward expansion factor. |
| `dropout` | `0.0` | Dropout probability. |
| `rope_theta` | `10000.0` | RoPE frequency base. |
| `rope_scaling` | `null` | Dict, e.g. `{type: linear, factor: 2.0}`. Supports `linear`, `dynamic`, `yarn`. |
| `norm_eps` | `1e-5` | Normalization epsilon. |
| `tie_embeddings` | `true` | Tie input embedding and output projection weights. |
| `use_flash` | `true` | Use PyTorch SDPA (falls back to a manual path if unavailable). |
| `block_type` | `transformer` | `transformer` or `parallel_residual`. |
| `attention_backend` | `sdpa` | See [attention backends](#attention-backends). |
| `ffn_type` | `null` | `null` → `moe` if `moe` set, else the `activation` value. See [FFN types](#ffn-types). |
| `normalization` | `rmsnorm` | `rmsnorm` or `layernorm`. |
| `position_embedding` | `rope` | `rope`, `alibi`, or `none`. |
| `activation` | `swiglu` | FFN activation; used as the default `ffn_type`. |
| `quantization_backend` | `none` | See [quantization](#quantization-backends). |
| `sliding_window` | `null` | Local attention window length. |
| `residual_scale` | `null` | Residual scaling factor. `null` → `1/sqrt(2·n_layers)`. |
| `gradient_checkpointing` | `false` | Trade compute for activation memory. |
| `moe` | `null` | MoE config (see [`MoEConfig`](#moe--moeconfig)); enables MoE FFN. |
| `lora_rank` | `0` | LoRA adapter rank. `0` disables LoRA. |
| `lora_alpha` | `16.0` | LoRA scaling. |
| `lora_dropout` | `0.0` | LoRA dropout. |

---

## `training` — `TrainConfig`

| Field | Default | Notes |
| --- | --- | --- |
| `mode` | `auto` | `auto, generative, chat, instruct, completion, code, reasoning`. |
| `run_name` | `nanoforge` | Run label. |
| `output_dir` | `runs/nanoforge` | Where checkpoints and metrics are written. |
| `seed` | `1337` | RNG seed. |
| `max_steps` | `1000` | Total optimizer steps. |
| `batch_size` | `8` | Logical batch target (= `micro_batch_size × grad_accum_steps`). |
| `micro_batch_size` | `2` | Per-step batch on device. |
| `grad_accum_steps` | `4` | Gradient accumulation steps. |
| `learning_rate` | `3e-4` | Peak LR. |
| `min_learning_rate` | `3e-5` | Final LR after decay. |
| `warmup_steps` | `100` | Explicit warmup steps. |
| `warmup_ratio` | `0.03` | Used when `warmup_steps <= 0`. |
| `optimizer` | `adamw` | `adamw, lion, adafactor, sophia/sophiag`. |
| `scheduler` | `cosine` | `cosine, linear, constant`. |
| `weight_decay` | `0.1` | AdamW weight decay. |
| `betas` | `(0.9, 0.95)` | Optimizer betas. |
| `eps` | `1e-8` | Optimizer epsilon. |
| `grad_clip` | `1.0` | Gradient clipping norm. |
| `precision` | `auto` | `auto, bf16, fp16, fp32`. CPU forces fp32. |
| `gradient_checkpointing` | `true` | Activation checkpointing during training. |
| `activation_offload` | `false` | Offload activations (experimental). |
| `low_memory` | `false` | CUDA allocator / low-memory tuning. |
| `eval_interval` | `100` | Steps between validations. |
| `eval_steps` | `20` | Batches per validation. |
| `save_interval` | `250` | Steps between periodic checkpoints. |
| `resume_from_checkpoint` | `null` | Resume from this checkpoint. |
| `async_checkpoint` | `false` | Save checkpoints on a background thread. |
| `early_stopping_patience` | `10` | Evals without improvement before stopping. |
| `ema_decay` | `0.0` | EMA decay (`0` disables). |
| `sample_interval` | `0` | Steps between sample generations (`0` disables). |
| `sample_prompt` | `"The model"` | Prompt for periodic samples. |
| `sample_max_new_tokens` | `128` | Tokens per sample. |
| `health_interval` | `10` | Steps between health-monitor checks. |
| `grad_explosion_factor` | `10.0` | Grad-norm spike factor for the health monitor. |
| `adaptive_grad_clip` | `true` | Adapt clip threshold from grad-norm history. |
| `rollback_on_nonfinite` | `true` | Skip/rollback steps with NaN/Inf loss or grads. |
| `distributed_backend` | `none` | Distributed hook (single-process today). |
| `log_interval` | `10` | Steps between metric logs. |
| `tensorboard` | `true` | TensorBoard logging (needs `logging` extra). |
| `wandb` | `false` | Weights & Biases logging (needs `logging` extra). |
| `compile` | `false` | `torch.compile` the model. |
| `device` | `auto` | `auto, cpu, cuda, mps`. |

---

## `data` — `DataConfig`

| Field | Default | Notes |
| --- | --- | --- |
| `train_path` | `data/packed/train.bin` | Packed training tokens (from `prepare`). |
| `val_path` | `data/packed/val.bin` | Packed validation tokens. |
| `tokenizer_path` | `null` | Tokenizer artifact; `null` for the byte tokenizer. |
| `tokenizer_type` | `byte` | `byte, byte-native, bpe, python-bpe, native-bpe, wordpiece, sentencepiece`. |
| `seq_len` | `2048` | Training sequence length (≤ `model.max_seq_len`). |
| `mode` | `generative` | Data formatting intent (mirrors `prepare --mode`). |
| `loss_masking` | `auto` | `auto, none, assistant_only, completion_only, partial`. |
| `assistant_only_loss` | `false` | Mask everything but assistant tokens (chat SFT). |
| `prompt_template` | `null` | Optional prompt template string. |
| `dataset_weights` | `null` | Hybrid mixing weights, e.g. `{chat: 0.7, text: 0.3}`. |
| `streaming` | `false` | Streaming dataset intent. |
| `tokenizer_batch_size` | `256` | Batched tokenization during packing. |
| `num_workers` | `2` | DataLoader workers (use `0` on CPU/Windows). |
| `prefetch_factor` | `2` | DataLoader prefetch. |
| `pin_memory` | `true` | Pin memory (set `false` on CPU). |

---

## `inference` — `InferenceConfig`

| Field | Default | Notes |
| --- | --- | --- |
| `mode` | `balanced` | `balanced, chat, creative, coding, deterministic, low_memory, high_quality`. |
| `max_new_tokens` | `256` | Generation length. |
| `temperature` | `0.8` | Sampling temperature. |
| `top_k` | `50` | Top-k cutoff (`null` disables). |
| `top_p` | `0.95` | Nucleus cutoff (`null` disables). |
| `min_p` | `null` | Relative probability floor. |
| `repetition_penalty` | `1.0` | Penalize repeated tokens. |
| `frequency_penalty` | `0.0` | Penalize by token frequency. |
| `presence_penalty` | `0.0` | Penalize already-seen tokens. |
| `no_repeat_ngram_size` | `0` | Block repeated n-grams. |
| `deterministic` | `false` | Greedy decoding. |
| `stop_on_repetition` | `true` | Stop on repetitive tails. |
| `repetition_window` | `64` | Repetition detector window. |
| `repetition_threshold` | `0.85` | Tail repetition cutoff. |
| `mirostat` | `false` | Enable Mirostat (forces `sampler = mirostat`). |
| `mirostat_version` | `2` | Stored but not branched on (implementation is v2-style). |
| `mirostat_tau` | `5.0` | Mirostat target entropy. |
| `mirostat_eta` | `0.1` | Mirostat learning rate. |
| `stop_tokens` | `["<|endoftext|>", "<|user|>"]` | Stop strings. |
| `sampler` | `top_k_top_p` | Sampler registry key. |

The `mode` preset applies a bundle of overrides (e.g. `chat` raises repetition penalty and
sets `no_repeat_ngram_size`); `deterministic` forces greedy. Explicit CLI flags override the
preset.

---

## `moe` — `MoEConfig`

Nested under `model.moe`. Setting it switches the FFN to Mixture-of-Experts.

| Field | Default | Notes |
| --- | --- | --- |
| `num_experts` | `4` | Number of experts. |
| `top_k` | `2` | Experts routed per token. |
| `capacity_factor` | `1.25` | Expert capacity multiplier. |
| `router_aux_loss_coef` | `0.01` | Load-balancing auxiliary loss weight. |

> MoE increases capacity but can destabilize small runs. Get dense training healthy first.

---

## Registry-backed options

Run `nanoforge registries` to see everything registered. Honest notes on the current
implementations:

### Attention backends
`sdpa` (default), `flash`, `flash_v2`, `flash_v3`, `manual`, `sliding_window`, `chunked`,
`sparse`, `hybrid_local_global`, `paged`.

> `sdpa`, `flash`, `flash_v2`, `flash_v3`, `manual`, and `sliding_window` all use the base
> `CausalSelfAttention` (`sdpa`/`flash*` via PyTorch SDPA, `manual` via an explicit softmax
> path). `chunked`, `sparse`, `hybrid_local_global`, and `paged` are thin subclasses that
> **preset a sliding window** — they are not distinct sparse/paged attention kernels.

### Position embeddings
`rope` (with `linear`/`dynamic`/`yarn` scaling), `alibi`, `none`. **NTK scaling is not
implemented.**

### FFN types
`swiglu`, `geglu`, `moe`, `mamba_selective_state`, `rwkv_mixing`.

> `mamba_selective_state` and `rwkv_mixing` are experimental FFN mixers, not full SSM/RWKV
> implementations.

### Normalization
`rmsnorm`, `layernorm`.

### Optimizers
`adamw`, `lion`, `adafactor`, `sophia` / `sophiag` — all implemented.

### Schedulers
`cosine`, `linear`, `constant`.

### Samplers
`top_k_top_p` (default), `mirostat`, `beam`, `contrastive`. `contrastive` currently aliases
standard sampling.

### Quantization backends
`none`, `int8` (real dynamic CPU quantization), `int4`, `gptq`, `awq`, `gguf` (metadata
hooks only — no actual weight quantization for the last four).

---

## Starter presets

**Tiny CPU smoke test**

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

**Small chat SFT**

```yaml
training:
  mode: chat
  learning_rate: 2e-4
  scheduler: cosine
data:
  mode: chat
  tokenizer_type: native-bpe
  tokenizer_path: data/tokenizers/chat-bpe.json
  loss_masking: assistant_only
  assistant_only_loss: true
inference:
  mode: chat
  repetition_penalty: 1.1
  no_repeat_ngram_size: 4
```

**Low-memory long-context inference**

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

See [configs/](../configs) for complete, working example configs.
