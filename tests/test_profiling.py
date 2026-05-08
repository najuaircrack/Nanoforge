from __future__ import annotations

from nanoforge.config import ModelConfig
from nanoforge.profiling import estimate_model_profile


def test_model_profile_estimates_positive_costs():
    cfg = ModelConfig(vocab_size=128, max_seq_len=32, d_model=32, n_layers=2, n_heads=4)
    profile = estimate_model_profile(cfg, batch_size=2, seq_len=16)
    assert profile.parameters > 0
    assert profile.non_embedding_parameters > 0
    assert profile.forward_flops_per_token > 0
    assert profile.train_flops_per_token > profile.forward_flops_per_token
    assert profile.activation_memory_mb > 0
    assert profile.kv_cache_memory_mb > 0
