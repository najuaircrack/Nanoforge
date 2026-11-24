from __future__ import annotations

import math
import os
import tempfile
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from types import UnionType
from typing import Any, Union, get_args, get_origin, get_type_hints

from nanoforge.logging_utils import get_logger
from nanoforge.registry import validate_registry_key

_CAUSAL_TASKS: frozenset[str] = frozenset({
    "causal_lm", "chat", "code_generation", "instruction_following",
    "reward_modeling", "value_estimation", "preference_learning", "dpo",
})


@dataclass
class MoEConfig:
    num_experts: int = 4
    top_k: int = 2
    capacity_factor: float = 1.25
    router_aux_loss_coef: float = 0.01

    def __post_init__(self) -> None:
        _validate_types(self)
        if self.num_experts < 1 or not 1 <= self.top_k <= self.num_experts:
            raise ValueError("MoE top_k must be between 1 and num_experts.")
        if self.capacity_factor <= 0 or self.router_aux_loss_coef < 0:
            raise ValueError("Invalid MoE capacity or auxiliary loss coefficient.")


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
    attention_chunk_size: int = 128
    kv_page_size: int = 64
    hybrid_global_stride: int = 64
    residual_scale: float | None = None
    gradient_checkpointing: bool = False
    moe: MoEConfig | dict[str, Any] | None = None
    lora_rank: int = 0
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    # Task / head configuration
    task_type: str = "causal_lm"
    num_labels: int = 2
    input_type: str = "text"
    is_causal: bool | None = None

    def __post_init__(self) -> None:
        _validate_types(self)
        for name in ("vocab_size", "max_seq_len", "d_model", "n_layers", "n_heads",
                     "attention_chunk_size", "kv_page_size", "hybrid_global_stride", "num_labels"):
            if getattr(self, name) <= 0:
                raise ValueError(f"model.{name} must be positive.")
        if self.n_kv_heads is not None and self.n_kv_heads <= 0:
            raise ValueError("model.n_kv_heads must be positive.")
        if not 0 <= self.dropout < 1 or not 0 <= self.lora_dropout < 1:
            raise ValueError("Model dropout must be in [0, 1).")
        if self.ffn_hidden_mult <= 0 or self.norm_eps <= 0 or self.rope_theta <= 0:
            raise ValueError("Model FFN multiplier, norm epsilon and RoPE theta must be positive.")
        if self.sliding_window is not None and self.sliding_window <= 0:
            raise ValueError("model.sliding_window must be positive.")
        if self.residual_scale is not None and self.residual_scale <= 0:
            raise ValueError("model.residual_scale must be positive.")
        if self.lora_rank < 0 or self.lora_alpha <= 0:
            raise ValueError("Invalid LoRA rank or alpha.")
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
        if self.is_causal is None:
            self.is_causal = self.task_type in _CAUSAL_TASKS
        validate_registry_key("block", self.block_type)
        validate_registry_key("attention", self.attention_backend)
        validate_registry_key("ffn", self.ffn_type)
        validate_registry_key("normalization", self.normalization)
        validate_registry_key("position", self.position_embedding)
        validate_registry_key("quantization", self.quantization_backend)
        validate_registry_key("head", self.task_type)


@dataclass
class TrainConfig:
    mode: str = "auto"
    run_name: str = "nanoforge"
    output_dir: str = "runs/nanoforge"
    seed: int = 1337
    max_steps: int = 1000
    batch_size: int | None = None
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
    low_memory: bool = False
    eval_interval: int = 100
    eval_steps: int = 20
    save_interval: int = 250
    resume_from_checkpoint: str | None = None
    init_from_checkpoint: str | None = None
    async_checkpoint: bool = False
    early_stopping_patience: int = 10
    ema_decay: float = 0.0
    sample_interval: int = 0
    sample_prompt: str = "The model"
    sample_max_new_tokens: int = 128
    health_interval: int = 10
    grad_explosion_factor: float = 10.0
    distributed_backend: str = "none"
    log_interval: int = 10
    tensorboard: bool = True
    wandb: bool = False
    compile: bool = False
    device: str = "auto"
    cpu_threads: int = 0
    ema_eval: bool = False
    swa_start: int = -1
    swa_interval: int = 1
    swa_eval: bool = False
    optimizer_cpu_offload: bool = False
    max_consecutive_skipped_steps: int = 10
    recovery_lr_factor: float = 1.0
    curriculum: list[dict[str, int]] = field(default_factory=list)
    # Knowledge distillation (optional): set teacher_checkpoint to enable.
    teacher_checkpoint: str | None = None
    distill_alpha: float = 0.5
    distill_temperature: float = 2.0

    def __post_init__(self) -> None:
        self.betas = tuple(self.betas)  # type: ignore[assignment]
        _validate_types(self)
        for name in ("max_steps", "micro_batch_size", "grad_accum_steps", "eval_interval",
                     "eval_steps", "save_interval", "health_interval", "log_interval",
                     "swa_interval", "max_consecutive_skipped_steps", "sample_max_new_tokens"):
            if getattr(self, name) <= 0:
                raise ValueError(f"training.{name} must be positive.")
        effective = self.micro_batch_size * self.grad_accum_steps
        if self.batch_size is not None and self.batch_size != effective:
            get_logger(__name__).warning("training.batch_size=%s is replaced by the effective microbatch product %s", self.batch_size, effective)
        self.batch_size = effective
        if self.learning_rate <= 0 or self.min_learning_rate < 0 or self.weight_decay < 0 or self.eps <= 0:
            raise ValueError("Invalid training learning rate, weight decay or epsilon.")
        if self.min_learning_rate > self.learning_rate:
            raise ValueError("Minimum learning rate cannot exceed learning rate.")
        if len(self.betas) != 2 or any(not 0 <= beta < 1 for beta in self.betas):
            raise ValueError("training.betas must contain two values in [0, 1).")
        if self.warmup_steps < 0 or not 0 <= self.warmup_ratio <= 1 or self.grad_clip < 0:
            raise ValueError("Invalid warmup or clipping setting.")
        if not 0 <= self.ema_decay < 1 or not 0 < self.recovery_lr_factor <= 1:
            raise ValueError("Invalid EMA decay or recovery learning-rate factor.")
        if self.cpu_threads < 0 or self.swa_start < -1 or self.sample_interval < 0:
            raise ValueError("Invalid training interval/thread setting.")
        if self.resume_from_checkpoint and self.init_from_checkpoint:
            raise ValueError("Exact resume and weights-only initialization are mutually exclusive.")
        if self.distributed_backend not in {"none", "ddp", "fsdp", "zero"}:
            raise ValueError("Unknown distributed backend.")
        if not 0 <= self.distill_alpha <= 1 or self.distill_temperature <= 0:
            raise ValueError("Invalid distillation alpha or temperature.")
        previous = -1
        for stage in self.curriculum:
            if set(stage) != {"step", "seq_len"} or stage["step"] < 0 or stage["seq_len"] < 2 or stage["step"] <= previous:
                raise ValueError("Curriculum stages require increasing step and positive seq_len.")
            previous = stage["step"]
        validate_registry_key("optimizer", self.optimizer)
        validate_registry_key("scheduler", self.scheduler)


@dataclass
class DataConfig:
    train_path: str = "data/packed/train.bin"
    val_path: str | None = "data/packed/val.bin"
    sources: list[dict[str, Any]] | None = None
    tokenizer_path: str | None = None
    tokenizer_type: str = "byte"
    seq_len: int = 2048
    mode: str = "generative"
    loss_masking: str = "auto"
    assistant_only_loss: bool = False
    prompt_template: str | None = None
    # True for chat models trained from an export that has no system turn. Such
    # models are hurt by an out-of-distribution system prompt at inference, so the
    # CLI/server suppress the default system prompt for them (see cli.py).
    no_system_prompt: bool = False
    dataset_weights: dict[str, float] | None = None
    streaming: bool = False
    tokenizer_batch_size: int = 256
    num_workers: int = 0
    pin_memory: bool = True

    def __post_init__(self) -> None:
        _validate_types(self)
        if self.seq_len < 2 or self.tokenizer_batch_size < 1 or self.num_workers < 0:
            raise ValueError("Invalid data sequence/batch/worker settings.")
        if self.dataset_weights is not None and any(value <= 0 for value in self.dataset_weights.values()):
            raise ValueError("Dataset weights must be positive.")
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
    num_beams: int = 4
    length_penalty: float = 0.8
    contrastive_alpha: float = 0.6
    contrastive_top_k: int = 4
    draft_tokens: int = 4
    grammar: str | None = None
    grammar_file: str | None = None
    max_wall_seconds: float = 120.0
    prefix_cache_max_bytes: int = 64 * 1024 * 1024
    prefix_cache_max_entries: int = 8

    def __post_init__(self) -> None:
        _validate_types(self)
        if self.max_new_tokens < 1 or self.max_wall_seconds <= 0 or self.num_beams < 1 or self.draft_tokens < 1:
            raise ValueError("Invalid inference generation budget.")
        if self.temperature < 0 or (self.top_k is not None and self.top_k < 0):
            raise ValueError("Invalid temperature or top_k.")
        if any(value is not None and not 0 < value <= 1 for value in (self.top_p, self.min_p)):
            raise ValueError("top_p and min_p must be in (0, 1].")
        if self.repetition_penalty <= 0 or self.no_repeat_ngram_size < 0 or self.repetition_window < 1:
            raise ValueError("Invalid inference repetition controls.")
        if not 0 <= self.repetition_threshold <= 1 or not 0 <= self.contrastive_alpha <= 1 or self.contrastive_top_k < 1:
            raise ValueError("Invalid inference repetition or contrastive setting.")
        if self.prefix_cache_max_bytes < 0 or self.prefix_cache_max_entries < 0:
            raise ValueError("Prefix cache bounds cannot be negative.")
        if self.mirostat:
            self.sampler = "mirostat"
        validate_registry_key("sampler", self.sampler)


@dataclass
class NanoforgeConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)


def _validate_value(value: Any, annotation: Any, path: str) -> None:
    origin, arguments = get_origin(annotation), get_args(annotation)
    if annotation is Any:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{path} must be finite.")
        return
    if origin in (UnionType, Union):
        for option in arguments:
            try:
                _validate_value(value, option, path)
                return
            except (ValueError, TypeError):
                continue
        raise ValueError(f"{path} has an invalid type or value.")
    if origin in (list, tuple, dict):
        if origin is dict:
            if not isinstance(value, dict):
                raise TypeError(f"{path} must be an object.")
            for key, item in value.items():
                _validate_value(key, arguments[0], path + ".key")
                _validate_value(item, arguments[1], f"{path}.{key}")
        else:
            if not isinstance(value, (list, tuple)):
                raise TypeError(f"{path} must be an array.")
            if origin is tuple and len(arguments) != len(value):
                raise ValueError(f"{path} has an invalid tuple length.")
            for index, item in enumerate(value):
                _validate_value(item, arguments[index] if origin is tuple else arguments[0], f"{path}[{index}]")
        return
    valid = ((isinstance(value, (float, int)) and not isinstance(value, bool)) if annotation is float
             else type(value) is int if annotation is int
             else isinstance(value, annotation))
    if not valid:
        raise TypeError(f"{path} has an invalid type; expected {annotation}.")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} must be finite.")


def _validate_types(instance: Any) -> None:
    for name, annotation in get_type_hints(type(instance)).items():
        _validate_value(getattr(instance, name), annotation, f"{type(instance).__name__}.{name}")


def _dataclass_from_dict(cls: type, data: dict[str, Any], *, strict: bool = False):
    if not isinstance(data, dict):
        raise TypeError(f"Config section {cls.__name__} must be an object.")
    names = {f.name for f in fields(cls)}
    unknown = [k for k in data if k not in names]
    if unknown:
        if strict:
            raise ValueError(f"Config section {cls.__name__} contains unknown key(s): {', '.join(sorted(unknown))}")
        logger = get_logger(__name__)
        logger.warning(
            "config section %s ignored unknown key(s): %s",
            cls.__name__,
            ", ".join(sorted(unknown)),
        )
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


def parse_config(text: str, *, strict: bool = False) -> NanoforgeConfig:
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("PyYAML is required to load configuration.") from error

    class UniqueSafeLoader(yaml.SafeLoader):
        pass

    def mapping(loader, node):
        loader.flatten_mapping(node)
        result = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node)
            if not isinstance(key, str):
                raise ValueError("YAML configuration keys must be strings.")
            if key in result:
                raise ValueError(f"YAML duplicate key: {key}")
            result[key] = loader.construct_object(value_node)
        return result

    UniqueSafeLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    try:
        raw = yaml.load(text, Loader=UniqueSafeLoader)
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid YAML configuration: {error}") from error
    raw = {} if raw is None else raw
    if not isinstance(raw, dict):
        raise TypeError("Configuration root must be an object.")
    unknown = sorted(set(raw) - {"model", "training", "data", "inference"})
    if unknown:
        message = "Configuration root contains unknown key(s): " + ", ".join(unknown)
        if strict:
            raise ValueError(message)
        get_logger(__name__).warning(message)
    return NanoforgeConfig(
        model=_dataclass_from_dict(ModelConfig, raw.get("model", {}), strict=strict),
        training=_dataclass_from_dict(TrainConfig, raw.get("training", {}), strict=strict),
        data=_dataclass_from_dict(DataConfig, raw.get("data", {}), strict=strict),
        inference=_dataclass_from_dict(InferenceConfig, raw.get("inference", {}), strict=strict),
    )


def load_config(path: str | Path, *, strict: bool = False) -> NanoforgeConfig:
    return parse_config(Path(path).read_text(encoding="utf-8"), strict=strict)


def save_config(config: NanoforgeConfig, path: str | Path) -> None:
    import yaml

    if not is_dataclass(config):
        raise TypeError("Configuration must be a dataclass.")
    text = yaml.safe_dump(asdict(config), sort_keys=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
