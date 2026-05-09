from __future__ import annotations

from pathlib import Path


def interactive_new_config(out_path: str | Path = "configs/my-model.yaml") -> Path:
    purpose = _ask("What are you training?", ["chat assistant", "story generator", "code assistant", "custom"])
    ram = _ask("How much RAM do you have?", ["4GB", "8GB", "16GB", "32GB+"])
    speed = _ask("How fast do you want training?", ["fast/small", "balanced", "slow/large"])
    data_format = _ask("What dataset format?", ["parquet chat", "JSONL", "plain text"])
    cfg = build_template_config(purpose, ram, speed, data_format)
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(cfg, encoding="utf-8")
    return path


def build_template_config(purpose: str, ram: str, speed: str, data_format: str) -> str:
    mode = "chat" if "chat" in purpose.lower() or "parquet chat" in data_format.lower() else "generative"
    if "code" in purpose.lower():
        mode = "code"
    if "story" in purpose.lower():
        mode = "generative"
    return build_cpu_config(
        name="my-model",
        mode=mode,
        ram=ram,
        speed=speed,
        data_format=data_format,
        tokenizer_type="native-bpe",
        tokenizer_path="data/tokenizers/tiny-bpe.json",
        packed_dir="data/packed/my-model",
        vocab_size=8000,
    )


def build_cpu_config(
    *,
    name: str,
    mode: str,
    ram: str,
    speed: str,
    data_format: str,
    tokenizer_type: str,
    tokenizer_path: str,
    packed_dir: str,
    vocab_size: int = 8000,
    max_steps: int = 50000,
    seq_len_override: int | None = None,
    loss_masking: str | None = None,
) -> str:
    purpose = mode
    key = (purpose.lower(), ram.lower(), speed.lower())
    if "4gb" in key[1]:
        d_model, layers, heads, seq_len, micro, accum = 192, 3, 3, 384, 1, 8
    elif "8gb" in key[1] and "fast" in key[2]:
        d_model, layers, heads, seq_len, micro, accum = 256, 4, 4, 512, 1, 8
    elif "8gb" in key[1] and "story" in key[0]:
        d_model, layers, heads, seq_len, micro, accum = 384, 6, 6, 1024, 1, 8
    elif "8gb" in key[1]:
        d_model, layers, heads, seq_len, micro, accum = 384, 6, 6, 1024, 1, 8
    elif "16gb" in key[1]:
        d_model, layers, heads, seq_len, micro, accum = 512, 8, 8, 1024, 1, 12
    else:
        d_model, layers, heads, seq_len, micro, accum = 768, 12, 12, 2048, 1, 16

    if seq_len_override is not None:
        seq_len = seq_len_override
    loss = "assistant_only" if mode == "chat" else "none"
    if mode == "instruct":
        loss = "completion_only"
    if loss_masking is not None:
        loss = loss_masking.strip().replace("-", "_")
    text_column = "messages" if "parquet chat" in data_format else "text"
    top_p = 0.92 if mode == "chat" else 0.95
    temp = 0.75 if mode == "chat" else 0.9
    return f"""model:
  vocab_size: {vocab_size}
  max_seq_len: {seq_len}
  d_model: {d_model}
  n_layers: {layers}
  n_heads: {heads}
  n_kv_heads: 2
  ffn_hidden_mult: 2.67
  dropout: 0.0
  rope_theta: 500000.0
  norm_eps: 1.0e-5
  tie_embeddings: true
  use_flash: false
  activation: swiglu
  block_type: transformer
  attention_backend: sdpa
  normalization: rmsnorm
  position_embedding: rope
  quantization_backend: none

training:
  mode: {mode}
  run_name: {name}
  output_dir: runs/{name}
  seed: 1337
  max_steps: {max_steps}
  batch_size: {micro * accum}
  micro_batch_size: {micro}
  grad_accum_steps: {accum}
  learning_rate: 1.0e-4
  min_learning_rate: 1.0e-5
  warmup_ratio: 0.10
  optimizer: adamw
  scheduler: cosine
  weight_decay: 0.1
  betas: [0.9, 0.95]
  eps: 1.0e-8
  grad_clip: 1.0
  precision: fp32
  gradient_checkpointing: false
  eval_interval: 500
  eval_steps: 20
  save_interval: 500
  early_stopping_patience: 15
  log_interval: 10
  health_interval: 10
  low_memory: false
  tensorboard: true
  wandb: false
  distributed_backend: none

data:
  train_path: {packed_dir}/train.bin
  val_path: {packed_dir}/val.bin
  tokenizer_type: {tokenizer_type}
  tokenizer_path: {tokenizer_path}
  seq_len: {seq_len}
  mode: {mode}
  loss_masking: {loss}
  assistant_only_loss: {"true" if loss == "assistant_only" else "false"}
  tokenizer_batch_size: 128
  num_workers: 0
  pin_memory: false

inference:
  mode: {"chat" if mode == "chat" else "creative"}
  max_new_tokens: 128
  temperature: {temp}
  top_k: 40
  top_p: {top_p}
  min_p: 0.05
  repetition_penalty: 1.1
  frequency_penalty: 0.1
  presence_penalty: 0.0
  no_repeat_ngram_size: 4
  stop_on_repetition: true
  repetition_window: 64
  repetition_threshold: 0.85
  deterministic: false
  mirostat: false

# Prepare example:
# nanoforge prepare --input data/raw --tokenizer {tokenizer_type} --tokenizer-path {tokenizer_path} --mode {mode} --loss-masking {loss} --text-column {text_column} --seq-len {seq_len} --out {packed_dir}
"""


def _ask(prompt: str, options: list[str]) -> str:
    print(prompt)
    for idx, option in enumerate(options, 1):
        print(f"  {idx}. {option}")
    raw = input(f"Choose 1-{len(options)} [1]: ").strip()
    try:
        selected = int(raw or "1") - 1
        return options[selected]
    except Exception:
        return options[0]
