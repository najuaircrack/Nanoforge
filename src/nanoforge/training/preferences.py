from __future__ import annotations

import torch
from torch.nn import functional as F


def sequence_logprob(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if logits.shape[:2] != labels.shape or logits.ndim != 3:
        raise ValueError("logits and labels must have matching batch/sequence dimensions")
    log_probs = F.log_softmax(logits[:, :-1], dim=-1)
    mask = labels[:, 1:] >= 0
    target = labels[:, 1:].masked_fill(~mask, 0).unsqueeze(-1)
    token_logp = log_probs.gather(-1, target).squeeze(-1)
    return (token_logp * mask).sum(dim=-1)


def dpo_loss(
    policy_chosen_logp: torch.Tensor,
    policy_rejected_logp: torch.Tensor,
    ref_chosen_logp: torch.Tensor,
    ref_rejected_logp: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    if beta <= 0:
        raise ValueError("DPO beta must be positive")
    policy_margin = policy_chosen_logp - policy_rejected_logp
    ref_margin = ref_chosen_logp - ref_rejected_logp
    return -F.logsigmoid(beta * (policy_margin - ref_margin)).mean()
