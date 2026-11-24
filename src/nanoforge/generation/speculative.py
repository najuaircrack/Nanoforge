from __future__ import annotations

import hashlib
import json

import torch

from nanoforge.generation.sampling import SamplingConfig, sampling_probabilities


def tokenizer_identity(tokenizer):
    special = (tokenizer.vocab_size, tokenizer.bos_id, tokenizer.eos_id,
               getattr(tokenizer, "pad_id", None), getattr(tokenizer, "unk_id", None))
    native = getattr(tokenizer, "tokenizer", None)
    processor = getattr(tokenizer, "processor", None)
    if hasattr(native, "to_str"):
        payload = native.to_str().encode()
    elif processor is not None and hasattr(processor, "serialized_model_proto"):
        payload = processor.serialized_model_proto()
    elif hasattr(tokenizer, "get_vocab"):
        payload = json.dumps(tokenizer.get_vocab(), sort_keys=True).encode()
    elif getattr(tokenizer, "offset", None) is not None and tokenizer.vocab_size == 260:
        payload = f"byte-offset:{tokenizer.offset}".encode()
    else:
        raise ValueError("This tokenizer has no trustworthy identity for speculative decoding.")
    return hashlib.sha256(repr(special).encode() + b"\0" + payload).hexdigest()


def _probabilities(logits, history, sampling, device, constraint):
    logits = logits.clone()
    if constraint is not None:
        allowed = constraint.allowed_tokens(history)
        if not allowed:
            raise ValueError("Grammar has no speculative continuation.")
        mask = torch.ones(logits.shape[-1], dtype=torch.bool, device=device)
        mask[list(allowed)] = False
        logits[..., mask] = -torch.inf
    return sampling_probabilities(logits, torch.tensor(history, device=device, dtype=torch.long), sampling)[0]


@torch.inference_mode()
def stochastic_speculative_ids(target, draft, input_ids, max_new_tokens=128, *,
                                draft_tokens=4, sampling=None, interrupt=None, constraint=None):
    """Exact rejection-sampling reference algorithm for stateless target sampling."""
    sampling = sampling or SamplingConfig()
    if not input_ids or max_new_tokens < 0 or draft_tokens < 1:
        raise ValueError("Invalid stochastic speculative prompt or budget.")
    if draft_tokens > target.model.config.max_seq_len:
        raise ValueError("Draft block must fit the target context.")
    if tokenizer_identity(target.tokenizer) != tokenizer_identity(draft.tokenizer):
        raise ValueError("Speculation requires identical tokenizer identities.")
    if max_new_tokens == 0:
        return
    prompt = list(input_ids[-target.model.config.max_seq_len:])
    out = target.model(torch.tensor([prompt], device=target.device), use_cache=True)
    caches, logits = out.caches, out.logits[:, -1:]
    generated = []
    target.speculative_stats = {"algorithm": "stochastic_rejection", "drafted": 0, "accepted": 0,
                                "verification_calls": 0, "target_forward_calls": 1}
    while len(generated) < max_new_tokens:
        if interrupt is not None and interrupt():
            return
        draft_prompt = (prompt + generated)[-draft.model.config.max_seq_len:]
        draft_out = draft.model(torch.tensor([draft_prompt], device=draft.device), use_cache=True)
        proposed, distributions = [], []
        for _ in range(min(draft_tokens, max_new_tokens - len(generated))):
            q = _probabilities(draft_out.logits, generated + proposed, sampling, draft.device, constraint)
            token = int(torch.multinomial(q, 1))
            proposed.append(token)
            distributions.append(q.to(target.device))
            if token == target.tokenizer.eos_id:
                break
            draft_out = draft.model(torch.tensor([[token]], device=draft.device),
                                    caches=draft_out.caches, use_cache=True)
        before = [cache.clone_detached() for cache in caches]
        verified = target.model(torch.tensor([proposed], device=target.device),
                                 caches=caches, use_cache=True)
        target.speculative_stats["drafted"] += len(proposed)
        target.speculative_stats["verification_calls"] += 1
        target.speculative_stats["target_forward_calls"] += 1
        accepted = []
        rejected = False
        for index, proposal in enumerate(proposed):
            p = _probabilities(logits if index == 0 else verified.logits[:, index - 1:index],
                                generated, sampling, target.device, constraint)
            q = distributions[index]
            ratio = min(1., float(p[proposal] / q[proposal]))
            if float(torch.rand((), device=target.device)) <= ratio:
                token = proposal
                target.speculative_stats["accepted"] += 1
            else:
                residual = (p - q).clamp_min(0)
                total = residual.sum()
                distribution = residual / total if total > 1e-12 else p
                token = int(torch.multinomial(distribution, 1))
                rejected = True
            if token == target.tokenizer.eos_id:
                if constraint is not None and not constraint.is_accepting(generated):
                    raise ValueError("Grammar ended before an accepting state.")
                return
            if interrupt is not None and interrupt():
                return
            generated.append(token)
            accepted.append(token)
            yield token
            if rejected:
                break
        if rejected:
            out = target.model(torch.tensor([accepted], device=target.device),
                               caches=before, use_cache=True)
            target.speculative_stats["target_forward_calls"] += 1
            caches, logits = out.caches, out.logits[:, -1:]
        else:
            caches, logits = verified.caches, verified.logits[:, -1:]
            if len(generated) < max_new_tokens:
                p = _probabilities(logits, generated, sampling, target.device, constraint)
                bonus = int(torch.multinomial(p, 1))
                if bonus == target.tokenizer.eos_id:
                    return
                if interrupt is not None and interrupt():
                    return
                generated.append(bonus)
                yield bonus
                out = target.model(torch.tensor([[bonus]], device=target.device),
                                   caches=caches, use_cache=True)
                target.speculative_stats["target_forward_calls"] += 1
                caches, logits = out.caches, out.logits[:, -1:]
    if constraint is not None and not constraint.is_accepting(generated):
        raise ValueError("Grammar did not reach an accepting state within the token budget.")
