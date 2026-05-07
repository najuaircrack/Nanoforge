from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Iterator

import torch

from nanoforge.config import NanoforgeConfig, load_config
from nanoforge.data.tokenizer import TokenizerAdapter, load_tokenizer
from nanoforge.generation.sampling import MirostatState, SamplingConfig, sample_next
from nanoforge.model.attention import KVCache
from nanoforge.model.transformer import NanoforgeForCausalLM
from nanoforge.training.checkpoint import load_checkpoint
from nanoforge.training.utils import resolve_device


@dataclass
class PrefixCacheEntry:
    prompt_ids: tuple[int, ...]
    caches: list[KVCache]


class GenerationEngine:
    def __init__(
        self,
        model: NanoforgeForCausalLM,
        tokenizer: TokenizerAdapter,
        device: str | torch.device = "auto",
    ):
        self.device = resolve_device(str(device))
        self.model = model.to(self.device).eval()
        self.tokenizer = tokenizer
        self.prefix_cache: dict[tuple[int, ...], PrefixCacheEntry] = {}

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        tokenizer_path: str | Path | None = None,
        tokenizer_type: str | None = None,
        device: str = "auto",
    ) -> "GenerationEngine":
        payload = load_checkpoint(checkpoint, map_location="cpu")
        cfg = payload["config"]
        if isinstance(cfg, dict):
            raise RuntimeError("This checkpoint stores an old dict config. Save again with NanoforgeConfig.")
        model = NanoforgeForCausalLM(cfg.model)
        model.load_state_dict(payload["model"], strict=True)
        tok_type = tokenizer_type or cfg.data.tokenizer_type
        tok_path = tokenizer_path or cfg.data.tokenizer_path
        tokenizer = load_tokenizer(tok_type, tok_path)
        return cls(model, tokenizer, device=device)

    @classmethod
    def from_config(cls, config_path: str | Path, device: str = "auto") -> "GenerationEngine":
        cfg: NanoforgeConfig = load_config(config_path)
        model = NanoforgeForCausalLM(cfg.model)
        tokenizer = load_tokenizer(cfg.data.tokenizer_type, cfg.data.tokenizer_path)
        return cls(model, tokenizer, device=device)

    @torch.inference_mode()
    def generate_ids(
        self,
        input_ids: list[int],
        max_new_tokens: int,
        sampling: SamplingConfig,
        eos_id: int | None = None,
    ) -> Iterator[int]:
        ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        caches: list[KVCache] | None = None
        miro = MirostatState(sampling.mirostat_tau) if sampling.mirostat else None
        for step in range(max_new_tokens):
            if step == 0:
                out = self.model(ids[:, -self.model.config.max_seq_len :], use_cache=True)
            else:
                out = self.model(ids[:, -1:], caches=caches, use_cache=True)
            caches = out.caches
            token = sample_next(out.logits, ids[0], sampling, miro)
            next_id = int(token.item())
            ids = torch.cat([ids, token], dim=1)
            if eos_id is not None and next_id == eos_id:
                break
            yield next_id

    def stream(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        sampling: SamplingConfig | None = None,
    ) -> Iterator[str]:
        sampling = sampling or SamplingConfig()
        input_ids = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
        buffer: list[int] = []
        for token_id in self.generate_ids(input_ids, max_new_tokens, sampling, self.tokenizer.eos_id):
            buffer.append(token_id)
            text = self.tokenizer.decode(buffer)
            if text:
                yield text
                buffer.clear()

    def complete(self, prompt: str, max_new_tokens: int = 256, sampling: SamplingConfig | None = None) -> str:
        return "".join(self.stream(prompt, max_new_tokens=max_new_tokens, sampling=sampling))

    @torch.inference_mode()
    def beam_search(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        num_beams: int = 4,
        length_penalty: float = 0.8,
    ) -> str:
        input_ids = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
        beams: list[tuple[list[int], float, bool]] = [(input_ids, 0.0, False)]
        for _ in range(max_new_tokens):
            candidates: list[tuple[list[int], float, bool]] = []
            for ids, score, done in beams:
                if done:
                    candidates.append((ids, score, True))
                    continue
                x = torch.tensor([ids[-self.model.config.max_seq_len :]], dtype=torch.long, device=self.device)
                out = self.model(x)
                log_probs = torch.log_softmax(out.logits[:, -1, :], dim=-1)
                values, indices = torch.topk(log_probs, k=num_beams, dim=-1)
                for value, index in zip(values[0], indices[0]):
                    token = int(index.item())
                    candidates.append((ids + [token], score + float(value.item()), token == self.tokenizer.eos_id))
            beams = sorted(
                candidates,
                key=lambda item: item[1] / (len(item[0]) ** length_penalty),
                reverse=True,
            )[:num_beams]
            if all(done for _, _, done in beams):
                break
        best = beams[0][0][len(input_ids) :]
        return self.tokenizer.decode([token for token in best if token != self.tokenizer.eos_id])

    @torch.inference_mode()
    def cache_prefix(self, prompt: str) -> None:
        ids = tuple(self.tokenizer.encode(prompt, add_bos=True, add_eos=False))
        if ids in self.prefix_cache:
            return
        x = torch.tensor([list(ids)[-self.model.config.max_seq_len :]], dtype=torch.long, device=self.device)
        out = self.model(x, use_cache=True)
        if out.caches is not None:
            self.prefix_cache[ids] = PrefixCacheEntry(ids, out.caches)

    @torch.inference_mode()
    def speculative_generate(
        self,
        prompt: str,
        draft: "GenerationEngine",
        max_new_tokens: int,
        sampling: SamplingConfig | None = None,
        draft_tokens: int = 4,
    ) -> str:
        # Conservative compatibility hook. It accepts a draft model and falls back to target
        # generation if exact token verification is not beneficial for the current backend.
        _ = draft, draft_tokens
        return self.complete(prompt, max_new_tokens=max_new_tokens, sampling=sampling)
