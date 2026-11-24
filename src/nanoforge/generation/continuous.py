from __future__ import annotations

import math
import time
import uuid
from collections import OrderedDict, deque
from dataclasses import dataclass, field

import torch

from nanoforge.generation.sampling import MirostatState, SamplingConfig, sample_next
from nanoforge.model.attention import KVCache, PagedKVCache


@dataclass
class BatchRequest:
    request_id: str
    input_ids: list[int]
    max_new_tokens: int
    sampling: SamplingConfig
    deadline: float
    reserved_tokens: int
    generated: list[int] = field(default_factory=list)
    caches: list[KVCache] | None = None
    mirostat: MirostatState | None = None
    cancelled: bool = False


@dataclass(frozen=True)
class BatchToken:
    request_id: str
    token_id: int | None
    done: bool
    error: str | None = None


class ContinuousBatcher:
    """Bounded continuous scheduler with real grouped model batch forwards.

    Requests with equal input lengths and cache positions are stacked into one
    forward. New requests can enter as soon as earlier ones finish. This reference
    implementation materializes merged K/V batches and copies per-request caches;
    it is not a zero-copy CUDA serving kernel. Cancellation/timeouts are checked
    between model forwards, not inside a native kernel call.
    """

    def __init__(self, engine, *, max_active_requests=16, max_batch_tokens=4096,
                 max_pending_requests=64, max_generated_tokens=8192):
        if min(max_active_requests, max_batch_tokens, max_pending_requests, max_generated_tokens) < 1:
            raise ValueError("Continuous batch limits must be positive.")
        self.engine = engine
        self.max_active_requests, self.max_batch_tokens = max_active_requests, max_batch_tokens
        self.max_pending_requests, self.max_generated_tokens = max_pending_requests, max_generated_tokens
        self.active: OrderedDict[str, BatchRequest] = OrderedDict()
        self.pending = deque()
        self.stats = {"model_forward_calls": 0, "maximum_batch_size": 0,
                      "admitted_requests": 0, "completed_requests": 0}

    @property
    def reserved_tokens(self):
        return sum(request.reserved_tokens for request in self.active.values())

    @property
    def cache_bytes(self):
        return sum(cache.storage_bytes for request in self.active.values()
                   for cache in request.caches or [])

    def submit(self, input_ids, max_new_tokens=256, sampling=None, *, timeout_s=120., request_id=None):
        if (not input_ids or any(isinstance(i, bool) or not isinstance(i, int)
                or not 0 <= i < self.engine.model.config.vocab_size for i in input_ids)):
            raise ValueError("Batch input contains invalid token IDs.")
        if (not 1 <= max_new_tokens <= self.max_generated_tokens
                or not math.isfinite(timeout_s) or timeout_s <= 0):
            raise ValueError("Invalid batch generation/deadline budget.")
        if len(self.pending) >= self.max_pending_requests:
            raise ValueError("Pending batch request limit exceeded.")
        sampling = sampling or SamplingConfig()
        if sampling.sampler in {"beam", "contrastive"}:
            raise ValueError("Continuous batching supports sampling/Mirostat; advanced search needs a separate engine run.")
        prompt = list(input_ids[-self.engine.model.config.max_seq_len:])
        reserved = min(self.engine.model.config.max_seq_len, len(prompt) + max_new_tokens)
        if reserved > self.max_batch_tokens:
            raise ValueError("Request cannot fit the batch token budget.")
        request_id = request_id or uuid.uuid4().hex
        if request_id in self.active or any(req.request_id == request_id for req in self.pending):
            raise ValueError("Duplicate batch request ID.")
        request = BatchRequest(request_id, prompt, max_new_tokens, sampling,
                               time.monotonic() + timeout_s, reserved,
                               mirostat=MirostatState(sampling.mirostat_tau) if sampling.mirostat else None)
        self.pending.append(request)
        return request_id

    def cancel(self, request_id):
        for request in list(self.active.values()) + list(self.pending):
            if request.request_id == request_id:
                request.cancelled = True
                return True
        return False

    def _admit(self):
        while self.pending and len(self.active) < self.max_active_requests:
            request = self.pending[0]
            if self.reserved_tokens + request.reserved_tokens > self.max_batch_tokens:
                break
            self.pending.popleft()
            self.active[request.request_id] = request
            self.stats["admitted_requests"] += 1

    def _finish(self, request, token_id=None, error=None):
        self.active.pop(request.request_id, None)
        self.stats["completed_requests"] += 1
        return BatchToken(request.request_id, token_id, True, error)

    @torch.inference_mode()
    def step(self):
        self._admit()
        events = []
        groups = OrderedDict()
        for request in tuple(self.active.values()):
            if request.cancelled or time.monotonic() >= request.deadline:
                events.append(self._finish(request, error="cancelled" if request.cancelled else "timeout"))
                continue
            cache = request.caches[0] if request.caches else None
            key = (len(request.input_ids), 0 if cache is None else cache.tokens_seen,
                   0 if cache is None else cache.length, cache is None)
            groups.setdefault(key, []).append(request)
        for requests in groups.values():
            ids = torch.tensor([request.input_ids for request in requests],
                               device=self.engine.device, dtype=torch.long)
            merged = _merge_caches([request.caches for request in requests])
            try:
                out = self.engine.model(ids, caches=merged, use_cache=True)
                self.stats["model_forward_calls"] += 1
                self.stats["maximum_batch_size"] = max(self.stats["maximum_batch_size"], len(requests))
                if out.caches is None:
                    raise ValueError("Continuous batching needs causal model caches.")
            except Exception as error:
                events.extend(self._finish(request, error=f"model error: {type(error).__name__}")
                              for request in requests)
                continue
            for index, request in enumerate(requests):
                if request.cancelled or time.monotonic() >= request.deadline:
                    events.append(self._finish(request, error="cancelled" if request.cancelled else "timeout"))
                    continue
                try:
                    history = torch.tensor(request.generated, device=self.engine.device, dtype=torch.long)
                    token = int(sample_next(out.logits[index:index + 1], history,
                                            request.sampling, request.mirostat).item())
                    request.generated.append(token)
                    if token == self.engine.tokenizer.eos_id:
                        events.append(self._finish(request))
                        continue
                    request.caches = _split_caches(out.caches, index)
                    request.input_ids = [token]
                    if len(request.generated) == request.max_new_tokens:
                        events.append(self._finish(request, token_id=token))
                    else:
                        events.append(BatchToken(request.request_id, token, False))
                except Exception as error:
                    events.append(self._finish(request, error=f"sampling error: {type(error).__name__}"))
        self._admit()
        return events

    def run(self):
        while self.pending or self.active:
            yield from self.step()


def _merge_caches(cache_lists):
    if cache_lists[0] is None:
        if any(caches is not None for caches in cache_lists):
            raise ValueError("Cannot merge initialized and uninitialized caches.")
        return None
    merged = []
    for layer in range(len(cache_lists[0])):
        caches = [cache_list[layer] for cache_list in cache_lists]
        key = torch.cat([cache.key for cache in caches], 0)
        value = torch.cat([cache.value for cache in caches], 0)
        state = None
        if caches[0].ffn_state is not None:
            state = torch.cat([cache.ffn_state for cache in caches], 0)
        merged.append(KVCache(key, value, caches[0].max_length, caches[0].tokens_seen, state))
    return merged


def _split_caches(caches, index):
    result = []
    for cache in caches:
        if isinstance(cache, PagedKVCache):
            item = PagedKVCache(cache.max_length, cache.page_size)
            item.pages = [(k[index:index + 1].detach().clone(), v[index:index + 1].detach().clone())
                          for k, v in cache.iter_blocks()]
            item.tokens_seen = cache.tokens_seen
        else:
            item = KVCache(cache.key[index:index + 1].detach().clone(),
                           cache.value[index:index + 1].detach().clone(),
                           cache.max_length, cache.tokens_seen)
        if cache.ffn_state is not None:
            item.ffn_state = cache.ffn_state[index:index + 1].detach().clone()
        result.append(item)
    return result
