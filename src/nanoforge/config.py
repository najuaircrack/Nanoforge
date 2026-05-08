from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from nanoforge.registry import validate_registry_key


@dataclass
class MoEConfig:
    num_experts: int = 4
    top_k: int = 2
    capacity_factor: float = 1.25
    router_aux_loss_coef: float = 0.01


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    max_seq_len: int = 2048
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int | None = None
    ffn_hidden_mult: float = 2.67
    dropout: float = 0.0
    rope_theta: float = 10000.0
    rope_scaling: dict[str, Any] | None = None
    norm_eps: float = 1e-5
    tie_embeddings: bool = True
    use_flash: bool = True
    block_type: str = "transformer"
    attention_backend: str = "sdpa"
    ffn_type: str | None = None
    normalization: str = "rmsnorm"
    position_embedding: str = "rope"
    activation: str = "swiglu"
    quantization_backend: str = "none"
    sliding_window: int | None = None
    residual_scale: float | None = None
    gradient_checkpointing: bool = False
    moe: MoEConfig | dict[str, Any] | None = None
    lora_rank: int = 0
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads for GQA.")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        if isinstance(self.moe, dict):
            self.moe = MoEConfig(**self.moe)
        if self.ffn_type is None:
            self.ffn_type = "moe" if self.moe is not None else self.activation
        validate_registry_key("block", self.block_type)
        validate_registry_key("attention", self.attention_backend)
        validate_registry_key("ffn", self.ffn_type)
        validate_registry_key("normalization", self.normalization)
        validate_registry_key("position", self.position_embedding)
        validate_registry_key("quantization", self.quantization_backend)


@dataclass
class TrainConfig:
    mode: str = "auto"
    run_name: str = "nanoforge"
    output_dir: str = "runs/nanoforge"
    seed: int = 1337
    max_steps: int = 1000
    batch_size: int = 8
    micro_batch_size: int = 2
    grad_accum_steps: int = 4
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 100
    warmup_ratio: float = 0.03
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    grad_clip: float = 1.0
    precision: str = "auto"
    gradient_checkpointing: bool = True
    activation_offload: bool = False
    low_memory: bool = False
    eval_interval: int = 100
    eval_steps: int = 20
    save_interval: int = 250
    resume_from_checkpoint: str | None = None
    async_checkpoint: bool = False
    early_stopping_patience: int = 10
    ema_decay: float = 0.0
    sample_interval: int = 0
    sample_prompt: str = "The model"
    sample_max_new_tokens: int = 128
    health_interval: int = 10
    grad_explosion_factor: float = 10.0
    adaptive_grad_clip: bool = True
    rollback_on_nonfinite: bool = True
    distributed_backend: str = "none"
    log_interval: int = 10
    tensorboard: bool = True
    wandb: bool = False
    compile: bool = False
    device: str = "auto"

    def __post_init__(self) -> None:
        self.betas = tuple(self.betas)  # type: ignore[assignment]
        validate_registry_key("optimizer", self.optimizer)
        validate_registry_key("scheduler", self.scheduler)


@dataclass
class DataConfig:
    train_path: str = "data/packed/train.bin"
    val_path: str = "data/packed/val.bin"
    tokenizer_path: str | None = None
    tokenizer_type: str = "byte"
    seq_len: int = 2048
    mode: str = "generative"
    loss_masking: str = "auto"
    assistant_only_loss: bool = False
    prompt_template: str | None = None
    dataset_weights: dict[str, float] | None = None
    streaming: bool = False
    tokenizer_batch_size: int = 256
    num_workers: int = 2
    prefetch_factor: int = 2
    pin_memory: bool = True

    def __post_init__(self) -> None:
        validate_registry_key("tokenizer", self.tokenizer_type)


@dataclass
class InferenceConfig:
    mode: str = "balanced"
    max_new_tokens: int = 256
    temperature: float = 0.8
    top_k: int | None = 50
    top_p: float | None = 0.95
    min_p: float | None = None
    repetition_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    no_repeat_ngram_size: int = 0
    deterministic: bool = False
    stop_on_repetition: bool = True
    repetition_window: int = 64
    repetition_threshold: float = 0.85
    mirostat: bool = False
    mirostat_version: int = 2
    mirostat_tau: float = 5.0
    mirostat_eta: float = 0.1
    stop_tokens: list[str] = field(default_factory=lambda: ["<|endoftext|>", "<|user|>"])
    sampler: str = "top_k_top_p"

    def __post_init__(self) -> None:
        if self.mirostat:
            self.sampler = "mirostat"
        validate_registry_key("sampler", self.sampler)


@dataclass
class NanoforgeConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)


def _dataclass_from_dict(cls: type, data: dict[str, Any]):
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in names})


def _coerce_value(raw: str) -> Any:
    raw = raw.strip()
    if raw in {"null", "None", "~"}:
        return None
    if raw in {"true", "True"}:
        return True
    if raw in {"false", "False"}:
        return False
    if raw.startswith("[") and raw.endswith("]"):
        inner = raw[1:-1].strip()
        if not inner:
            return []
        return [_coerce_value(x.strip()) for x in inner.split(",")]
    try:
        if "." in raw or "e" in raw.lower():
            return float(raw)
        return int(raw)
    except ValueError:
        return raw.strip("\"'")


def _tiny_yaml_load(text: str) -> dict[str, Any]:
    """Small YAML subset fallback for simple Nanoforge config files."""

    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, _, value = line.strip().partition(":")
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value.strip() == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _coerce_value(value)
    return root


def load_config(path: str | Path) -> NanoforgeConfig:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    try:
        import yaml

        raw = yaml.safe_load(text)
    except Exception:
        raw = _tiny_yaml_load(text)
    raw = raw or {}
    return NanoforgeConfig(
        model=_dataclass_from_dict(ModelConfig, raw.get("model", {})),
        training=_dataclass_from_dict(TrainConfig, raw.get("training", {})),
        data=_dataclass_from_dict(DataConfig, raw.get("data", {})),
        inference=_dataclass_from_dict(InferenceConfig, raw.get("inference", {})),
    )


def save_config(config: NanoforgeConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml

        path.write_text(yaml.safe_dump(asdict(config), sort_keys=False), encoding="utf-8")
    except Exception:
        path.write_text(str(asdict(config)), encoding="utf-8")
