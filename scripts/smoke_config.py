from __future__ import annotations

from nanoforge.config import ModelConfig
from nanoforge.model.transformer import NanoforgeForCausalLM


def main() -> None:
    cfg = ModelConfig(vocab_size=260, max_seq_len=64, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2)
    model = NanoforgeForCausalLM(cfg)
    print(model.estimate_num_params())


if __name__ == "__main__":
    main()

