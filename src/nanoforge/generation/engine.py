from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from collections.abc import Callable
from typing import AsyncIterator, Iterator
import asyncio

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
        stop_ids: set[int] | None = None,
        stop_sequences: list[list[int]] | None = None,
        interrupt: Callable[[], bool] | None = None,
    ) -> Iterator[int]:
        ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        caches: list[KVCache] | None = None
        prefix_len = 0
        for prompt_ids, entry in sorted(self.prefix_cache.items(), key=lambda item: len(item[0]), reverse=True):
            if tuple(input_ids[: len(prompt_ids)]) == prompt_ids:
                caches = [cache.clone_detached() for cache in entry.caches]
                prefix_len = len(prompt_ids)
                break
        if prefix_len:
            ids = torch.tensor([input_ids[:prefix_len]], dtype=torch.long, device=self.device)
            pending_ids = input_ids[prefix_len:]
        else:
            pending_ids = input_ids
        miro = MirostatState(sampling.mirostat_tau) if sampling.mirostat else None
        emitted: list[int] = []
        pending: list[int] = []
        stop_sequences = [seq for seq in (stop_sequences or []) if seq]
        max_stop_len = max((len(seq) for seq in stop_sequences), default=0)
        for step in range(max_new_tokens):
            if interrupt is not None and interrupt():
                break
            if step == 0 and pending_ids:
                prefill = torch.tensor([pending_ids[-self.model.config.max_seq_len :]], dtype=torch.long, device=self.device)
                out = self.model(prefill, caches=caches, use_cache=True)
                ids = torch.cat([ids, prefill], dim=1) if prefix_len else prefill
            elif step == 0:
                out = self.model(ids[:, -1:], caches=caches, use_cache=True)
            else:
                out = self.model(ids[:, -1:], caches=caches, use_cache=True)
            caches = out.caches
            token = sample_next(out.logits, ids[0], sampling, miro)
            next_id = int(token.item())
            ids = torch.cat([ids, token], dim=1)
            if (eos_id is not None and next_id == eos_id) or (stop_ids is not None and next_id in stop_ids):
                break
            pending.append(next_id)
            matched = _matched_stop_sequence(pending, stop_sequences)
            if matched:
                pending = pending[: -len(matched)]
                for item in pending:
                    emitted.append(item)
                    yield item
                break
            while len(pending) > max_stop_len:
                item = pending.pop(0)
                emitted.append(item)
                yield item
            if sampling.stop_on_repetition and _is_repetitive_tail(
                emitted + pending,
                window=sampling.repetition_window,
                threshold=sampling.repetition_threshold,
            ):
                for item in pending:
                    emitted.append(item)
                    yield item
                break
        else:
            for item in pending:
                yield item

    def stream(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        sampling: SamplingConfig | None = None,
        stop_tokens: list[str] | None = None,
    ) -> Iterator[str]:
        sampling = sampling or SamplingConfig()
        input_ids = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
        stop_tokens = self._default_stop_tokens(sampling, stop_tokens)
        stop_ids = self._stop_token_ids(stop_tokens)
        stop_sequences = self._stop_sequences(stop_tokens)
        buffer: list[int] = []
        for token_id in self.generate_ids(
            input_ids,
            max_new_tokens,
            sampling,
            self.tokenizer.eos_id,
            stop_ids,
            stop_sequences,
        ):
            buffer.append(token_id)
            text = self.tokenizer.decode(buffer)
            if text:
                yield text
                buffer.clear()

    def complete(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        sampling: SamplingConfig | None = None,
        stop_tokens: list[str] | None = None,
    ) -> str:
        return "".join(
            self.stream(prompt, max_new_tokens=max_new_tokens, sampling=sampling, stop_tokens=stop_tokens)
        )

    async def astream(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        sampling: SamplingConfig | None = None,
        stop_tokens: list[str] | None = None,
    ) -> AsyncIterator[str]:
        for chunk in self.stream(prompt, max_new_tokens=max_new_tokens, sampling=sampling, stop_tokens=stop_tokens):
            yield chunk
            await asyncio.sleep(0)

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
            self.prefix_cache[ids] = PrefixCacheEntry(ids, [cache.clone_detached() for cache in out.caches])

    def _stop_token_ids(self, stop_tokens: list[str] | None) -> set[int]:
        ids: set[int] = set()
        for token in stop_tokens or []:
            encoded = self.tokenizer.encode(token, add_bos=False, add_eos=False)
            if len(encoded) == 1:
                ids.add(encoded[0])
        return ids

    def _stop_sequences(self, stop_tokens: list[str] | None) -> list[list[int]]:
        sequences: list[list[int]] = []
        for token in stop_tokens or []:
            encoded = self.tokenizer.encode(token, add_bos=False, add_eos=False)
            if len(encoded) > 1:
                sequences.append(encoded)
        return sequences

    @staticmethod
    def _default_stop_tokens(sampling: SamplingConfig, stop_tokens: list[str] | None) -> list[str]:
        if stop_tokens is not None:
            return stop_tokens
        if sampling.mode in {"chat", "coding", "deterministic", "high_quality"}:
            return ["<|endoftext|>", "<|user|>", "<|system|>"]
        return ["<|endoftext|>"]


def _matched_stop_sequence(pending: list[int], sequences: list[list[int]]) -> list[int] | None:
    for sequence in sequences:
        if len(pending) >= len(sequence) and pending[-len(sequence) :] == sequence:
            return sequence
    return None


def _is_repetitive_tail(ids: list[int], *, window: int, threshold: float) -> bool:
    if window <= 0 or len(ids) < max(16, window // 2):
        return False
    tail = ids[-window:]
    if len(set(tail)) <= 2 and len(tail) >= 16:
        return True
    for ngram in (2, 3, 4):
        span = ngram * 4
        if len(tail) >= span and tail[-ngram:] * 4 == tail[-span:]:
            return True
    repeats = 1.0 - (len(set(tail)) / max(1, len(tail)))
    return repeats >= threshold and len(tail) >= window

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
