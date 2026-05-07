from __future__ import annotations

import torch
from torch.nn import functional as F


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.5,
    temperature: float = 2.0,
) -> torch.Tensor:
    ce = F.cross_entropy(student_logits.view(-1, student_logits.size(-1)), labels.reshape(-1))
    s = F.log_softmax(student_logits / temperature, dim=-1)
    t = F.softmax(teacher_logits / temperature, dim=-1)
    kl = F.kl_div(s, t, reduction="batchmean") * (temperature**2)
    return alpha * kl + (1.0 - alpha) * ce

