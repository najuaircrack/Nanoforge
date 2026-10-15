from __future__ import annotations

import copy
from pathlib import Path

import torch

from nanoforge.config import NanoforgeConfig, load_config
from nanoforge.logging_utils import get_logger
from nanoforge.model.transformer import build_model
from nanoforge.training.checkpoint import load_checkpoint
from nanoforge.training.distillation import distillation_loss
from nanoforge.training.trainer import Trainer


class DistillationTrainer(Trainer):
    """Knowledge distillation trainer.

    Extends the base Trainer by adding a frozen teacher model. The loss is a
    weighted combination of cross-entropy (student vs labels) and KL divergence
    (student vs teacher soft targets), controlled by ``alpha`` and ``temperature``
    from ``config.training``.

    Set ``config.training.teacher_checkpoint`` to the teacher checkpoint path.
    If not set, the teacher is initialised as a frozen copy of the student's
    initial weights (useful for self-distillation / label smoothing experiments).
    """

    def __init__(self, config: NanoforgeConfig):
        super().__init__(config)
        self._logger = get_logger("training.distillation")
        teacher_ckpt = config.training.teacher_checkpoint
        if teacher_ckpt:
            self.teacher = build_model(config).to(self.device)
            payload = load_checkpoint(teacher_ckpt, map_location=self.device)
            self.teacher.load_state_dict(payload["model"], strict=True)
            self._logger.info("distillation: loaded teacher from %s", teacher_ckpt)
        else:
            self.teacher = copy.deepcopy(self.model)
            self._logger.info("distillation: using frozen copy of student as teacher (self-distillation)")
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.eval()
        self._alpha = config.training.distill_alpha
        self._temperature = config.training.distill_temperature
        self.run_manifest.update(
            provenance={
                "teacher_checkpoint": teacher_ckpt,
                "teacher_type": "checkpoint" if teacher_ckpt else "student_initial_copy",
            }
        )
        self._logger.info("distillation: alpha=%.2f temperature=%.1f", self._alpha, self._temperature)

    def _forward_loss(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Override base forward to blend CE + KL distillation loss."""
        student_out = self.model(x, labels=y)
        with torch.no_grad():
            teacher_out = self.teacher(x)
        loss = distillation_loss(
            student_out.logits,
            teacher_out.logits,
            y,
            alpha=self._alpha,
            temperature=self._temperature,
        )
        return loss, student_out.logits


def distillation_from_config(config_path: str | Path) -> None:
    cfg = load_config(config_path)
    DistillationTrainer(cfg).train()
