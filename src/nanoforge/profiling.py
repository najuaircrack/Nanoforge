from __future__ import annotations

from dataclasses import asdict, dataclass

from nanoforge.config import ModelConfig


@dataclass
class ModelProfile:
    parameters: int
    non_embedding_parameters: int
    forward_flops_per_token: int
    train_flops_per_token: int
    activation_memory_mb: float
    kv_cache_memory_mb: float
    optimizer_state_memory_mb: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def estimate_model_profile(
    config: ModelConfig,
    *,
    batch_size: int = 1,
    seq_len: int | None = None,
    bytes_per_param: int = 2,
    optimizer_multiplier: float = 2.0,
) -> ModelProfile:
    seq_len = seq_len or config.max_seq_len
    head_dim = config.d_model // config.n_heads
    hidden = int(config.d_model * config.ffn_hidden_mult)
    embed_params = config.vocab_size * config.d_model
    attn_params = config.n_layers * (
        config.d_model * config.n_heads * head_dim
        + 2 * config.d_model * (config.n_kv_heads or config.n_heads) * head_dim
        + config.d_model * config.n_heads * head_dim
    )
    ffn_params = config.n_layers * 3 * config.d_model * hidden
    norm_params = config.n_layers * 2 * config.d_model + config.d_model
    lm_head_params = 0 if config.tie_embeddings else embed_params
    params = embed_params + attn_params + ffn_params + norm_params + lm_head_params
    attn_flops = 4 * config.d_model * config.d_model + 2 * seq_len * config.d_model
    ffn_flops = 6 * config.d_model * hidden
    forward_flops_per_token = int(config.n_layers * (attn_flops + ffn_flops))
    train_flops_per_token = int(forward_flops_per_token * 3)
    activation_memory = batch_size * seq_len * config.n_layers * config.d_model * bytes_per_param
    kv_heads = config.n_kv_heads or config.n_heads
    kv_cache = batch_size * seq_len * config.n_layers * 2 * kv_heads * head_dim * bytes_per_param
    optimizer_memory = params * bytes_per_param * optimizer_multiplier
    return ModelProfile(
        parameters=int(params),
        non_embedding_parameters=int(params - embed_params),
        forward_flops_per_token=forward_flops_per_token,
        train_flops_per_token=train_flops_per_token,
        activation_memory_mb=activation_memory / (1024 * 1024),
        kv_cache_memory_mb=kv_cache / (1024 * 1024),
        optimizer_state_memory_mb=optimizer_memory / (1024 * 1024),
    )
