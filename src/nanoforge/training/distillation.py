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
    if not 0 <= alpha <= 1 or temperature <= 0:
        raise ValueError("alpha must be in [0, 1] and temperature must be positive")
    if student_logits.shape != teacher_logits.shape or student_logits.shape[:2] != labels.shape:
        raise ValueError("teacher/student logits and labels must have matching dimensions")
    valid = labels != -100
    if not valid.any():
        return student_logits.sum() * 0.0
    ce = F.cross_entropy(
        student_logits.reshape(-1, student_logits.size(-1)), labels.reshape(-1), ignore_index=-100
    )
    s = F.log_softmax(student_logits / temperature, dim=-1)
    t = F.softmax(teacher_logits / temperature, dim=-1)
    per_token = F.kl_div(s, t, reduction="none").sum(dim=-1)
    kl = per_token[valid].mean() * (temperature**2)
    return alpha * kl + (1.0 - alpha) * ce
