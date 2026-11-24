from __future__ import annotations

import asyncio
import math
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, AsyncIterator, Iterator

import torch

from nanoforge.config import NanoforgeConfig, load_config
from nanoforge.data.tokenizer import TokenizerAdapter, load_tokenizer
from nanoforge.generation.sampling import MirostatState, SamplingConfig, sample_next
from nanoforge.model.attention import KVCache
from nanoforge.model.transformer import build_model
from nanoforge.training.checkpoint import load_checkpoint

if TYPE_CHECKING:
    from nanoforge.model.transformer import NanoforgeForCausalLM


@dataclass
class PrefixCacheEntry:
    prompt_ids: tuple[int, ...]
    caches: list[KVCache]
    logits: torch.Tensor

    @property
    def storage_bytes(self):
        return sum(cache.storage_bytes for cache in self.caches) + self.logits.numel() * self.logits.element_size()


class GenerationEngine:
    def __init__(
        self,
        model: NanoforgeForCausalLM,
        tokenizer: TokenizerAdapter,
        device: str | torch.device = "auto",
        config: NanoforgeConfig | None = None,
        *,
        prefix_cache_max_bytes: int | None = None,
        prefix_cache_max_entries: int | None = None,
    ):
        from nanoforge.quantization import prepare_inference_model

        backend = getattr(model.config, "quantization_backend", "none")
        self.model = prepare_inference_model(model, backend, device=str(device)).eval()
        self.device = next(self.model.parameters()).device
        self.tokenizer = tokenizer
        self.config = config
        inference = getattr(config, "inference", None)
        if prefix_cache_max_bytes is None:
            prefix_cache_max_bytes = getattr(inference, "prefix_cache_max_bytes", 64 * 1024 * 1024)
        if prefix_cache_max_entries is None:
            prefix_cache_max_entries = getattr(inference, "prefix_cache_max_entries", 8)
        if prefix_cache_max_bytes < 0 or prefix_cache_max_entries < 0:
            raise ValueError("Prefix cache budgets must be nonnegative.")
        self.prefix_cache_max_bytes = prefix_cache_max_bytes
        self.prefix_cache_max_entries = prefix_cache_max_entries
        self.prefix_cache: OrderedDict[tuple[int, ...], PrefixCacheEntry] = OrderedDict()

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
        model = build_model(cfg)
        model.load_state_dict(payload["model"], strict=True)
        tok_type = tokenizer_type or cfg.data.tokenizer_type
        tok_path = tokenizer_path or cfg.data.tokenizer_path
        tokenizer = load_tokenizer(tok_type, tok_path)
        return cls(model, tokenizer, device=device, config=cfg)

    @classmethod
    def from_config(cls, config_path: str | Path, device: str = "auto") -> "GenerationEngine":
        cfg: NanoforgeConfig = load_config(config_path)
        model = build_model(cfg)
        tokenizer = load_tokenizer(cfg.data.tokenizer_type, cfg.data.tokenizer_path)
        return cls(model, tokenizer, device=device, config=cfg)

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
        min_new_tokens: int = 0,
        constraint=None,
        usage: dict | None = None,
        max_wall_seconds: float | None = None,
    ) -> Iterator[int]:
        if not input_ids or any(isinstance(i, bool) or not isinstance(i, int)
                                or not 0 <= i < self.model.config.vocab_size for i in input_ids):
            raise ValueError("Prompt IDs must be nonempty valid integer vocabulary IDs.")
        if max_new_tokens < 0 or not 0 <= min_new_tokens <= max_new_tokens:
            raise ValueError("Invalid generation token budget.")
        wall_budget = max_wall_seconds if max_wall_seconds is not None else getattr(
            getattr(self.config, "inference", None), "max_wall_seconds", 120.)
        if not isinstance(wall_budget, (int, float)) or not math.isfinite(wall_budget) or wall_budget <= 0:
            raise ValueError("Generation wall-clock budget must be positive.")
        deadline = time.monotonic() + wall_budget
        if usage is not None:
            usage.update(prompt_tokens=len(input_ids), context_tokens=min(len(input_ids),
                         self.model.config.max_seq_len), generated_tokens=0,
                         counting="exact_generated_ids")
        if constraint is not None and eos_id is None:
            eos_id = constraint.eos_id
        if max_new_tokens == 0:
            if constraint is not None and not constraint.is_accepting([]):
                raise ValueError("Grammar cannot reach an accepting state with zero token budget.")
            return
        if sampling.sampler == "beam":
            result = self._beam_ids(input_ids, max_new_tokens, sampling.num_beams,
                                    sampling.length_penalty, interrupt=interrupt, deadline=deadline,
                                    constraint=constraint, eos_id=eos_id, stop_ids=stop_ids,
                                    stop_sequences=stop_sequences, min_new_tokens=min_new_tokens)
            if usage is not None:
                usage["generated_tokens"] = len(result)
            visible = []
            for token_id in result:
                if token_id == eos_id or token_id in (stop_ids or set()):
                    break
                visible.append(token_id)
                matched = _matched_stop_sequence(visible, stop_sequences or [])
                if matched:
                    del visible[-len(matched):]
                    break
            if constraint is not None and not constraint.is_accepting(visible):
                raise ValueError("Beam grammar did not reach an accepting state within the token budget.")
            yield from visible
            return
        contrastive = sampling.sampler == "contrastive"
        prompt = input_ids[-self.model.config.max_seq_len:]
        context_hidden = None
        caches = None
        prefix_len = 0
        first_logits = None
        for prompt_ids, entry in sorted(self.prefix_cache.items() if not contrastive else [],
                                       key=lambda item: len(item[0]), reverse=True):
            if tuple(prompt[:len(prompt_ids)]) == prompt_ids:
                caches = [cache.clone_detached() for cache in entry.caches]
                prefix_len = len(prompt_ids)
                if prefix_len == len(prompt):
                    first_logits = entry.logits.clone()
                self.prefix_cache.move_to_end(prompt_ids)
                break
        pending_ids = prompt[prefix_len:]
        generated = []
        pending = []
        miro = MirostatState(sampling.mirostat_tau) if sampling.mirostat else None
        sequences = [sequence for sequence in stop_sequences or [] if sequence]
        max_stop_len = max((len(sequence) for sequence in sequences), default=0)
        for step in range(max_new_tokens):
            if time.monotonic() >= deadline:
                raise TimeoutError("Generation wall-clock budget exceeded.")
            if interrupt is not None and interrupt():
                break
            if step == 0 and first_logits is not None:
                logits = first_logits
            else:
                model_ids = pending_ids if step == 0 else [generated[-1]]
                tensor = torch.tensor([model_ids], dtype=torch.long, device=self.device)
                kwargs = {"return_hidden_states": True} if contrastive else {}
                out = self.model(tensor, caches=caches, use_cache=True, **kwargs)
                logits, caches = out.logits, out.caches
                if contrastive:
                    if out.hidden_states is None:
                        raise ValueError("Contrastive search requires a model hidden-state interface.")
                    context_hidden = (out.hidden_states if context_hidden is None else
                                      torch.cat((context_hidden, out.hidden_states), 1))
                    context_hidden = context_hidden[:, -self.model.config.max_seq_len:]
            logits = logits.clone()
            history = torch.tensor(generated, dtype=torch.long, device=self.device)
            if constraint is not None:
                allowed = constraint.allowed_tokens(generated)
                if not allowed or any(not 0 <= token < logits.shape[-1] for token in allowed):
                    raise ValueError("Token grammar has no valid continuation in this vocabulary.")
                mask = torch.ones(logits.shape[-1], dtype=torch.bool, device=self.device)
                mask[list(allowed)] = False
                logits[..., mask] = -torch.inf
            if step < min_new_tokens:
                forbidden = set(stop_ids or [])
                if eos_id is not None:
                    forbidden.add(eos_id)
                for sequence in sequences:
                    if len(sequence) == 1 or generated[-(len(sequence) - 1):] == sequence[:-1]:
                        forbidden.add(sequence[-1])
                if forbidden:
                    logits[..., -1, list(forbidden)] = -torch.inf
            token = (self._contrastive_token(logits, caches, context_hidden, sampling,
                                             interrupt=interrupt, deadline=deadline)
                     if contrastive else sample_next(logits, history, sampling, miro))
            next_id = int(token.item())
            generated.append(next_id)
            if usage is not None:
                usage["generated_tokens"] = len(generated)
            if next_id == eos_id or next_id in (stop_ids or set()):
                break
            pending.append(next_id)
            matched = _matched_stop_sequence(pending, sequences)
            if matched and len(generated) >= min_new_tokens:
                del pending[-len(matched):]
                break
            while len(pending) > max_stop_len:
                yield pending.pop(0)
            if sampling.stop_on_repetition and _is_repetitive_tail(
                generated, window=sampling.repetition_window, threshold=sampling.repetition_threshold
            ):
                break
        yield from pending
        if constraint is not None and not constraint.is_accepting(
            generated[:-1] if generated and generated[-1] == eos_id else generated
        ):
            raise ValueError("Grammar did not reach an accepting state within the token budget.")

    def _contrastive_token(self, logits, caches, context_hidden, sampling, *, interrupt=None, deadline=None):
        probabilities = torch.softmax(logits[:, -1].float(), -1)
        if not torch.isfinite(probabilities).all():
            raise ValueError("No valid contrastive continuation.")
        count = min(sampling.contrastive_top_k, logits.shape[-1])
        values, indices = probabilities.topk(count, dim=-1)
        normalized_context = torch.nn.functional.normalize(context_hidden.float(), dim=-1)
        scores = []
        for value, index in zip(values[0], indices[0]):
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("Contrastive generation wall-clock budget exceeded.")
            if interrupt is not None and interrupt():
                raise RuntimeError("Contrastive generation cancelled.")
            if value == 0:
                scores.append(value.new_tensor(-torch.inf))
                continue
            candidate = self.model(index.reshape(1, 1),
                    caches=[cache.clone_detached() for cache in caches],
                    use_cache=True, return_hidden_states=True)
            hidden = torch.nn.functional.normalize(candidate.hidden_states[:, -1].float(), dim=-1)
            degeneration = (normalized_context * hidden[:, None]).sum(-1).max()
            scores.append((1 - sampling.contrastive_alpha) * value - sampling.contrastive_alpha * degeneration)
        return indices[:, torch.stack(scores).argmax()].reshape(1, 1)

    def contrastive_search(self, prompt, max_new_tokens=128, *, top_k=4, alpha=0.6):
        return self.complete(prompt, max_new_tokens, SamplingConfig(sampler="contrastive",
                             contrastive_top_k=top_k, contrastive_alpha=alpha, stop_on_repetition=False))

    @torch.inference_mode()
    def _beam_ids(self, input_ids, max_new_tokens, num_beams, length_penalty, *, interrupt=None,
                  deadline=None, constraint=None, eos_id=None, stop_ids=None,
                  stop_sequences=None, min_new_tokens=0):
        if not 1 <= num_beams <= self.model.config.vocab_size or max_new_tokens < 0:
            raise ValueError("Invalid beam count or generation budget.")
        eos_id = self.tokenizer.eos_id if eos_id is None else eos_id
        beams = [(list(input_ids), 0., False)]
        for step in range(max_new_tokens):
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("Beam generation wall-clock budget exceeded.")
            if interrupt is not None and interrupt():
                break
            candidates = []
            for ids, score, done in beams:
                if done:
                    candidates.append((ids, score, done))
                    continue
                output = self.model(torch.tensor([ids[-self.model.config.max_seq_len:]],
                                                 device=self.device, dtype=torch.long))
                logits = output.logits[:, -1].clone()
                history = ids[len(input_ids):]
                if constraint is not None:
                    allowed = constraint.allowed_tokens(history)
                    mask = torch.ones(logits.shape[-1], device=self.device, dtype=torch.bool)
                    mask[list(allowed)] = False
                    logits[:, mask] = -torch.inf
                if step < min_new_tokens:
                    forbid = set(stop_ids or []) | {eos_id}
                    for sequence in stop_sequences or []:
                        if len(sequence) == 1 or history[-(len(sequence) - 1):] == sequence[:-1]:
                            forbid.add(sequence[-1])
                    logits[:, list(forbid)] = -torch.inf
                if not torch.isfinite(logits).any():
                    continue
                values, tokens = torch.log_softmax(logits, -1).topk(num_beams, -1)
                for value, token in zip(values[0], tokens[0]):
                    if not torch.isfinite(value):
                        continue
                    next_id = int(token)
                    generated = history + [next_id]
                    done = (next_id == eos_id or next_id in (stop_ids or set())
                            or bool(_matched_stop_sequence(generated, stop_sequences or [])))
                    candidates.append((ids + [next_id], score + float(value), done))
            if not candidates:
                raise ValueError("Beam search has no valid continuation.")
            beams = sorted(candidates,
                key=lambda item: item[1] / max(1, len(item[0]) - len(input_ids)) ** length_penalty,
                reverse=True)[:num_beams]
            if all(done for _, _, done in beams):
                break
        return beams[0][0][len(input_ids):]

    @torch.inference_mode()
    def speculative_generate_ids(self, input_ids, draft_engine, max_new_tokens=128,
                                 *, draft_tokens=4, interrupt=None):
        """Greedy speculative decoding with batched verification and exact rollback.

        Stochastic rejection sampling is a separate algorithm. This method
        preserves target greedy tokens; it does not claim stochastic equivalence.
        """
        if not input_ids or max_new_tokens < 0 or draft_tokens < 1:
            raise ValueError("Invalid speculative prompt or budgets.")
        if draft_tokens > self.model.config.max_seq_len:
            raise ValueError("Draft block must fit the target context.")
        from nanoforge.generation.speculative import tokenizer_identity

        if tokenizer_identity(self.tokenizer) != tokenizer_identity(draft_engine.tokenizer):
            raise ValueError("Speculation requires identical target/draft tokenizer identities.")
        if max_new_tokens == 0:
            return
        prompt = list(input_ids[-self.model.config.max_seq_len:])
        out = self.model(torch.tensor([prompt], device=self.device), use_cache=True)
        caches, logits = out.caches, out.logits[:, -1:]
        generated = []
        self.speculative_stats = {"drafted": 0, "accepted": 0, "verification_calls": 0,
                                  "target_forward_calls": 1}
        while len(generated) < max_new_tokens:
            if interrupt is not None and interrupt():
                return
            remaining = max_new_tokens - len(generated)
            proposed = list(draft_engine.generate_ids(prompt + generated, min(draft_tokens, remaining),
                SamplingConfig(mode="deterministic", stop_on_repetition=False),
                eos_id=None))
            if not proposed:
                raise ValueError("Draft engine produced no speculative token.")
            before = [cache.clone_detached() for cache in caches]
            verified = self.model(torch.tensor([proposed], device=self.device),
                                  caches=caches, use_cache=True)
            self.speculative_stats["drafted"] += len(proposed)
            self.speculative_stats["verification_calls"] += 1
            self.speculative_stats["target_forward_calls"] += 1
            accepted = []
            mismatch = False
            for position, proposal in enumerate(proposed):
                target = int((logits if position == 0 else
                             verified.logits[:, position - 1:position]).argmax(-1).item())
                if target == self.tokenizer.eos_id:
                    return
                if interrupt is not None and interrupt():
                    return
                accepted.append(target)
                generated.append(target)
                yield target
                if proposal != target:
                    mismatch = True
                    break
                self.speculative_stats["accepted"] += 1
            if mismatch:
                out = self.model(torch.tensor([accepted], device=self.device),
                                 caches=before, use_cache=True)
                self.speculative_stats["target_forward_calls"] += 1
                caches, logits = out.caches, out.logits[:, -1:]
            else:
                caches, logits = verified.caches, verified.logits[:, -1:]
                if len(generated) < max_new_tokens:
                    bonus = int(logits.argmax(-1).item())
                    if bonus == self.tokenizer.eos_id:
                        return
                    if interrupt is not None and interrupt():
                        return
                    generated.append(bonus)
                    yield bonus
                    out = self.model(torch.tensor([[bonus]], device=self.device),
                                     caches=caches, use_cache=True)
                    self.speculative_stats["target_forward_calls"] += 1
                    caches, logits = out.caches, out.logits[:, -1:]

    def stochastic_speculative_generate_ids(self, input_ids, draft_engine, max_new_tokens=128,
                                           *, draft_tokens=4, sampling=None, interrupt=None, constraint=None):
        from nanoforge.generation.speculative import stochastic_speculative_ids

        yield from stochastic_speculative_ids(self, draft_engine, input_ids, max_new_tokens,
                    draft_tokens=draft_tokens, sampling=sampling, interrupt=interrupt, constraint=constraint)

    def stream(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        sampling: SamplingConfig | None = None,
        stop_tokens: list[str] | None = None,
        min_new_tokens: int = 0,
        constraint=None,
        interrupt: Callable[[], bool] | None = None,
        usage: dict | None = None,
        max_wall_seconds: float | None = None,
    ) -> Iterator[str]:
        sampling = sampling or SamplingConfig()
        input_ids = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
        stop_tokens = self._default_stop_tokens(sampling, stop_tokens)
        stop_ids = self._stop_token_ids(stop_tokens)
        stop_sequences = self._stop_sequences(stop_tokens)
        buffer: list[int] = []
        emitted_text = ""
        for token_id in self.generate_ids(
            input_ids,
            max_new_tokens,
            sampling,
            self.tokenizer.eos_id,
            stop_ids,
            stop_sequences,
            min_new_tokens=min_new_tokens,
            constraint=constraint,
            interrupt=interrupt,
            usage=usage,
            max_wall_seconds=max_wall_seconds,
        ):
            buffer.append(token_id)
            if usage is not None:
                usage["output_tokens"] = len(buffer)
            text = self.tokenizer.decode(buffer).rstrip("\uFFFD")
            if not text.startswith(emitted_text):
                raise ValueError("Tokenizer revised emitted text; an incremental decoder is required.")
            if len(text) > len(emitted_text):
                yield text[len(emitted_text):]
                emitted_text = text
        final = self.tokenizer.decode(buffer)
        if not final.startswith(emitted_text):
            raise ValueError("Tokenizer revised emitted text at completion.")
        if len(final) > len(emitted_text):
            yield final[len(emitted_text):]

    def complete(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        sampling: SamplingConfig | None = None,
        stop_tokens: list[str] | None = None,
        constraint=None,
        interrupt: Callable[[], bool] | None = None,
        usage: dict | None = None,
        max_wall_seconds: float | None = None,
    ) -> str:
        return "".join(
            self.stream(prompt, max_new_tokens=max_new_tokens, sampling=sampling,
                        stop_tokens=stop_tokens, constraint=constraint,
                        interrupt=interrupt, usage=usage, max_wall_seconds=max_wall_seconds)
        )

    @torch.inference_mode()
    def classify(self, text: str) -> dict[str, float]:
        """Return per-label probabilities for sequence-classification models."""
        input_ids = self.tokenizer.encode(text, add_bos=True, add_eos=True)
        x = torch.tensor([input_ids[: self.model.config.max_seq_len]], dtype=torch.long, device=self.device)
        out = self.model(x)
        logits = out.logits[0]
        if logits.dim() > 1:  # token-level → mean over sequence
            logits = logits.mean(dim=0)
        probs = torch.softmax(logits, dim=-1)
        return {str(i): float(p) for i, p in enumerate(probs.tolist())}

    @staticmethod
    def build_chat_prompt(
        messages: list[tuple[str, str]],
        system: str | None = None,
    ) -> str:
        """Render a role-marked prompt ending in an open assistant turn.

        Matches the training format produced by the chat data pipeline
        (``<|system|>`` / ``<|user|>`` / ``<|assistant|>`` markers), so a
        chat-trained model continues as the assistant.
        """
        parts: list[str] = []
        if system:
            parts.append(f"<|system|>\n{system}\n")
        for role, content in messages:
            parts.append(f"<|{role}|>\n{content}\n")
        parts.append("<|assistant|>\n")
        return "".join(parts)

    def chat_stream(
        self,
        messages: list[tuple[str, str]],
        max_new_tokens: int = 256,
        sampling: SamplingConfig | None = None,
        system: str | None = None,
        min_new_tokens: int = 1,
        interrupt: Callable[[], bool] | None = None,
        usage: dict | None = None,
    ) -> Iterator[str]:
        """Stream an assistant reply for a role-tagged conversation history.

        ``min_new_tokens`` defaults to 1 so a chat model that has learned very
        short replies cannot return an empty turn by emitting EOS immediately.
        """
        sampling = sampling or SamplingConfig(mode="chat")
        prompt = self.build_chat_prompt(messages, system=system)
        yield from self.stream(
            prompt,
            max_new_tokens=max_new_tokens,
            sampling=sampling,
            min_new_tokens=min_new_tokens,
            interrupt=interrupt,
            usage=usage,
        )

    async def astream(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        sampling: SamplingConfig | None = None,
        stop_tokens: list[str] | None = None,
        interrupt: Callable[[], bool] | None = None,
    ) -> AsyncIterator[str]:
        for chunk in self.stream(prompt, max_new_tokens=max_new_tokens, sampling=sampling,
                                 stop_tokens=stop_tokens, interrupt=interrupt):
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
        if num_beams < 1 or num_beams > self.model.config.vocab_size or max_new_tokens < 0:
            raise ValueError("Beam count and generation budget are invalid.")
        if not 0 <= length_penalty <= 2:
            raise ValueError("Beam length penalty must be in [0, 2].")
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
        if not ids or len(ids) > self.model.config.max_seq_len:
            raise ValueError("Shared prefixes must fit the model context.")
        if ids in self.prefix_cache:
            self.prefix_cache.move_to_end(ids)
            return
        if self.prefix_cache_max_entries == 0 or self.prefix_cache_max_bytes == 0:
            return
        out = self.model(torch.tensor([ids], dtype=torch.long, device=self.device), use_cache=True)
        if out.caches is None:
            raise ValueError("This task model does not support causal prefix caches.")
        entry = PrefixCacheEntry(ids, [cache.clone_detached() for cache in out.caches],
                                 out.logits[:, -1:].detach().clone())
        if entry.storage_bytes > self.prefix_cache_max_bytes:
            raise ValueError("Prefix exceeds the configured cache byte budget.")
        while self.prefix_cache and (len(self.prefix_cache) >= self.prefix_cache_max_entries
                or self.prefix_cache_bytes + entry.storage_bytes > self.prefix_cache_max_bytes):
            self.prefix_cache.popitem(last=False)
        self.prefix_cache[ids] = entry

    @property
    def prefix_cache_bytes(self) -> int:
        return sum(entry.storage_bytes for entry in self.prefix_cache.values())

    def clear_prefix_cache(self) -> None:
        self.prefix_cache.clear()

    def count_tokens(self, text: str) -> int:
        """Retokenized visible-text count; stream usage provides original generated-ID counts."""
        return len(self.tokenizer.encode(text, add_bos=False, add_eos=False))

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
