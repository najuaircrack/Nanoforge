from __future__ import annotations

import copy
import hashlib
import json
import random
import time
from pathlib import Path

import torch
from torch import nn

from nanoforge.artifacts import RunManifestWriter
from nanoforge.config import NanoforgeConfig, load_config
from nanoforge.data.tokenizer import load_tokenizer
from nanoforge.logging_utils import get_logger
from nanoforge.model.transformer import build_model
from nanoforge.progress import JsonlMetricLogger
from nanoforge.training.checkpoint import (
    AsyncCheckpointSaver,
    load_checkpoint,
    restore_rng_state,
)
from nanoforge.training.preferences import dpo_loss, sequence_logprob
from nanoforge.training.schedulers import create_scheduler
from nanoforge.training.utils import (
    autocast_dtype,
    configure_named_optimizer,
    ensure_dir,
    grads_are_finite,
    resolve_device,
    seed_everything,
)


def tokenizer_identity(config, tokenizer):
    result = {
        "type": config.data.tokenizer_type,
        "vocab_size": tokenizer.vocab_size,
        "special_ids": [tokenizer.pad_id, tokenizer.bos_id, tokenizer.eos_id, tokenizer.unk_id],
    }
    if config.data.tokenizer_path:
        result["content_sha256"] = hashlib.sha256(
            Path(config.data.tokenizer_path).read_bytes()
        ).hexdigest()
    return result


def validate_tokenizer_model(config, tokenizer):
    if tokenizer.vocab_size != config.model.vocab_size:
        raise ValueError(
            f"tokenizer vocabulary {tokenizer.vocab_size} does not match model vocabulary {config.model.vocab_size}"
        )


def _encode_record(record, tokenizer, seq_len, location):
    if not isinstance(record, dict) or any(
        not isinstance(record.get(key), str) or not record[key].strip()
        for key in ("chosen", "rejected")
    ):
        raise ValueError(f"{location}: chosen/rejected must be nonempty strings")
    if "prompt" in record and not isinstance(record["prompt"], str):
        raise ValueError(f"{location}: prompt must be a string")
    prompt = tokenizer.encode(record.get("prompt", ""), add_bos=True)
    pairs = []
    for key in ("chosen", "rejected"):
        response = tokenizer.encode(record[key], add_eos=True)
        ids = (prompt + response)[:seq_len]
        labels = ([-100] * len(prompt) + response)[:seq_len]
        if not any(value >= 0 for value in labels[1:]):
            raise ValueError(f"{location}: truncation removed all response targets")
        pairs.append((torch.tensor(ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)))
    return pairs


class PreferenceDataset:
    """Validate JSONL once; keep offsets rather than materializing tokenized pairs."""

    def __init__(self, path, tokenizer, seq_len, *, skip_invalid=False):
        self.path, self.tokenizer, self.seq_len = Path(path), tokenizer, seq_len
        if seq_len < 2:
            raise ValueError("DPO sequence length must be at least two")
        self.offsets = []
        self.groups = []
        self.stats = {"accepted": 0, "rejected": 0, "blank": 0}
        digest = hashlib.sha256()
        logger = get_logger("training.dpo")
        with self.path.open("rb") as stream:
            line_number = 0
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                line_number += 1
                digest.update(line)
                if not line.strip():
                    self.stats["blank"] += 1
                    continue
                try:
                    record = json.loads(line.decode("utf-8"))
                    _encode_record(record, tokenizer, seq_len, f"line {line_number}")
                except (ValueError, UnicodeError) as error:
                    self.stats["rejected"] += 1
                    if not skip_invalid:
                        raise ValueError(f"line {line_number}: {error}") from error
                    logger.warning("rejected preference line %s: %s", line_number, error)
                    continue
                self.offsets.append(offset)
                group = (
                    record.get("group_id")
                    or record.get("prompt")
                    or json.dumps(
                        {"chosen": record["chosen"], "rejected": record["rejected"]}, sort_keys=True
                    )
                )
                self.groups.append(hashlib.sha256(str(group).encode("utf-8")).hexdigest())
                self.stats["accepted"] += 1
        self.fingerprint = digest.hexdigest()
        stat = self.path.stat()
        self.source_stat = (stat.st_size, stat.st_mtime_ns)
        if not self.offsets:
            raise ValueError("No valid preference pairs found")

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, index):
        stat = self.path.stat()
        if (stat.st_size, stat.st_mtime_ns) != self.source_stat:
            raise ValueError("preference source changed during training")
        with self.path.open("rb") as stream:
            stream.seek(self.offsets[index])
            record = json.loads(stream.readline())
        return _encode_record(record, self.tokenizer, self.seq_len, f"record {index}")


def _iter_preference_jsonl(path, tokenizer, seq_len):
    dataset = PreferenceDataset(path, tokenizer, seq_len)
    for index in range(len(dataset)):
        chosen, rejected = dataset[index]
        yield chosen[0], rejected[0]


def _pad_batch(seqs, pad_id=0):
    if not seqs:
        raise ValueError("cannot pad an empty batch")
    output = torch.full((len(seqs), max(s.numel() for s in seqs)), pad_id, dtype=torch.long)
    for index, seq in enumerate(seqs):
        output[index, : seq.numel()] = seq
    return output


def _batch_logprob(model: nn.Module, ids, autocast_ctx, labels=None):
    with autocast_ctx:
        output = model(ids)
    return sequence_logprob(output.logits, ids if labels is None else labels)


class DPOTrainer:
    """Bounded preference optimization with explicit initialization and exact resume."""

    def __init__(
        self,
        config: NanoforgeConfig,
        preference_data,
        beta=0.1,
        reference_checkpoint=None,
        *,
        skip_invalid=False,
        validation_data=None,
    ):
        self.config, self.train_cfg, self.data_cfg = config, config.training, config.data
        if beta <= 0:
            raise ValueError("DPO beta must be positive")
        if (
            min(
                self.train_cfg.max_steps,
                self.train_cfg.micro_batch_size,
                self.train_cfg.grad_accum_steps,
                self.train_cfg.save_interval,
            )
            <= 0
        ):
            raise ValueError("DPO step/batch/save budgets must be positive")
        if config.training.distributed_backend != "none":
            raise NotImplementedError("DPO distributed training is not implemented")
        if config.model.quantization_backend != "none":
            raise ValueError("DPO training requires floating-point weights")
        self.beta = beta
        self.preference_data = Path(preference_data)
        self.logger = get_logger("training.dpo")
        self.device = resolve_device(self.train_cfg.device)
        seed_everything(self.train_cfg.seed)
        self.tokenizer = load_tokenizer(self.data_cfg.tokenizer_type, self.data_cfg.tokenizer_path)
        validate_tokenizer_model(config, self.tokenizer)
        self.dataset = PreferenceDataset(
            preference_data, self.tokenizer, self.data_cfg.seq_len, skip_invalid=skip_invalid
        )
        indices = list(range(len(self.dataset)))
        groups = sorted(set(self.dataset.groups))
        random.Random(self.train_cfg.seed).shuffle(groups)
        if validation_data:
            self.validation_dataset = PreferenceDataset(
                validation_data, self.tokenizer, self.data_cfg.seq_len
            )
            if set(self.dataset.groups) & set(self.validation_dataset.groups):
                raise ValueError("preference train/validation groups overlap")
            self.train_indices, self.val_indices = (
                indices,
                list(range(len(self.validation_dataset))),
            )
        elif len(groups) >= 2:
            heldout_groups = set(groups[: max(1, len(groups) // 10)])
            self.train_indices = [
                index for index in indices if self.dataset.groups[index] not in heldout_groups
            ]
            self.val_indices = [
                index for index in indices if self.dataset.groups[index] in heldout_groups
            ]
            self.validation_dataset = self.dataset
        else:
            self.train_indices, self.val_indices, self.validation_dataset = (
                indices,
                [],
                self.dataset,
            )
        config.model.gradient_checkpointing = self.train_cfg.gradient_checkpointing
        self.model = build_model(config).to(self.device)
        self.completed_updates, self.forward_microsteps = 0, 0
        self.best_val_loss = float("inf")
        self._stop_requested = False
        resume_path = self.train_cfg.resume_from_checkpoint
        init_path = getattr(self.train_cfg, "init_from_checkpoint", None)
        if resume_path and init_path:
            raise ValueError("choose weights initialization or exact resume, not both")
        payload = (
            load_checkpoint(resume_path or init_path, map_location="cpu")
            if (resume_path or init_path)
            else None
        )
        if payload:
            self.model.load_state_dict(payload["model"], strict=True)
        self.ref_model = copy.deepcopy(self.model)
        state = payload.get("training_state", {}) if resume_path else {}
        if resume_path:
            if state.get("kind") != "dpo" or "reference_model" not in state:
                raise ValueError(
                    "exact DPO resume requires a DPO checkpoint; use init_from_checkpoint for base weights"
                )
            if (
                state["dataset_fingerprint"] != self.dataset.fingerprint
                or state["tokenizer"] != tokenizer_identity(config, self.tokenizer)
                or state["beta"] != beta
            ):
                raise ValueError("DPO resume data/tokenizer/beta identity mismatch")
            if reference_checkpoint:
                raise ValueError("exact resume uses the saved reference, not an override")
            self.ref_model.load_state_dict(state["reference_model"], strict=True)
            self.completed_updates = int(state["completed_updates"])
            self.best_val_loss = state.get("best_val_loss", float("inf"))
        elif reference_checkpoint:
            reference = load_checkpoint(reference_checkpoint, map_location="cpu")
            if tokenizer_identity(
                reference["config"],
                load_tokenizer(
                    reference["config"].data.tokenizer_type, reference["config"].data.tokenizer_path
                ),
            ) != tokenizer_identity(config, self.tokenizer):
                raise ValueError("reference tokenizer identity mismatch")
            self.ref_model.load_state_dict(reference["model"], strict=True)
        self.ref_model.to(self.device).eval()
        self.ref_model.requires_grad_(False)
        self.optimizer = configure_named_optimizer(
            self.model,
            self.train_cfg.optimizer,
            self.train_cfg.learning_rate,
            self.train_cfg.weight_decay,
            self.train_cfg.betas,
            self.train_cfg.eps,
        )
        enabled = (
            self.device.type == "cuda"
            and autocast_dtype(self.train_cfg.precision, self.device) == torch.float16
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=enabled)
        if resume_path:
            self.optimizer.load_state_dict(payload["optimizer"])
            self.scaler.load_state_dict(state["scaler"])
            restore_rng_state(payload.get("rng"))
        self.output_dir = ensure_dir(self.train_cfg.output_dir)
        self.run_manifest = RunManifestWriter(
            self.output_dir, self.train_cfg.run_name, config, device=str(self.device)
        )
        self.run_manifest.update(
            data={
                "fingerprint": self.dataset.fingerprint,
                "filter_counts": self.dataset.stats,
                "split_seed": self.train_cfg.seed,
                "train_pairs": len(self.train_indices),
                "held_out_pairs": len(self.val_indices),
            },
            tokenizer=tokenizer_identity(config, self.tokenizer),
            model={"training_mode": "dpo"},
            provenance={
                "reference_checkpoint": str(reference_checkpoint) if reference_checkpoint else None
            },
        )
        self.metric_logger = JsonlMetricLogger(
            self.output_dir / "metrics.jsonl", reset=not bool(resume_path)
        )
        self.checkpoints = AsyncCheckpointSaver(self.train_cfg.async_checkpoint)

    def _autocast(self):
        dtype = autocast_dtype(self.train_cfg.precision, self.device)
        return torch.autocast(device_type=self.device.type, dtype=dtype, enabled=dtype is not None)

    def _log(self, metrics, step, event="train"):
        self.metric_logger.log(event, step, metrics)

    def request_stop(self):
        self._stop_requested = True

    def _loss(self, samples):
        chosen = [pair[0] for pair in samples]
        rejected = [pair[1] for pair in samples]

        def logp(model, records):
            ids = _pad_batch([item[0] for item in records], self.tokenizer.pad_id).to(self.device)
            labels = _pad_batch([item[1] for item in records], -100).to(self.device)
            return _batch_logprob(model, ids, self._autocast(), labels)

        policy_c, policy_r = logp(self.model, chosen), logp(self.model, rejected)
        with torch.no_grad():
            reference_c, reference_r = logp(self.ref_model, chosen), logp(self.ref_model, rejected)
        loss = dpo_loss(policy_c, policy_r, reference_c, reference_r, self.beta)
        margin = policy_c - policy_r - reference_c + reference_r
        return loss, float((margin > 0).float().mean().detach().cpu())

    @torch.no_grad()
    def evaluate(self):
        if not self.val_indices:
            return {"held_out_pairs": 0, "held_out_loss": None, "preference_accuracy": None}
        training = self.model.training
        self.model.eval()
        try:
            values = [self._loss([self.validation_dataset[index]]) for index in self.val_indices]
            return {
                "held_out_pairs": len(values),
                "held_out_loss": sum(float(value[0].cpu()) for value in values) / len(values),
                "preference_accuracy": sum(value[1] for value in values) / len(values),
            }
        finally:
            self.model.train(training)

    def _save(self, name):
        state = {
            "kind": "dpo",
            "completed_updates": self.completed_updates,
            "reference_model": self.ref_model.state_dict(),
            "scaler": self.scaler.state_dict(),
            "beta": self.beta,
            "dataset_fingerprint": self.dataset.fingerprint,
            "tokenizer": tokenizer_identity(self.config, self.tokenizer),
            "best_val_loss": self.best_val_loss,
        }
        self.checkpoints.save(
            self.output_dir / name,
            self.model,
            self.optimizer,
            self.config,
            self.completed_updates,
            training_state=state,
        )

    def train(self):
        started = time.monotonic()
        status, error = "failed", None
        self.model.train()
        warmup = self.train_cfg.warmup_steps or int(
            self.train_cfg.max_steps * self.train_cfg.warmup_ratio
        )
        scheduler = create_scheduler(
            self.train_cfg.scheduler,
            max_steps=self.train_cfg.max_steps,
            warmup_steps=warmup,
            learning_rate=self.train_cfg.learning_rate,
            min_learning_rate=self.train_cfg.min_learning_rate,
        )
        try:
            while self.completed_updates < self.train_cfg.max_steps and not self._stop_requested:
                for group in self.optimizer.param_groups:
                    group["lr"] = scheduler(self.completed_updates)
                self.optimizer.zero_grad(set_to_none=True)
                total = 0.0
                for _ in range(self.train_cfg.grad_accum_steps):
                    indices = torch.randint(
                        len(self.train_indices), (self.train_cfg.micro_batch_size,)
                    )
                    samples = [self.dataset[self.train_indices[int(index)]] for index in indices]
                    loss, _ = self._loss(samples)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("non-finite DPO loss; no update applied")
                    self.scaler.scale(loss / self.train_cfg.grad_accum_steps).backward()
                    total += float(loss.detach().cpu()) / self.train_cfg.grad_accum_steps
                    self.forward_microsteps += 1
                self.scaler.unscale_(self.optimizer)
                if not grads_are_finite(self.model.parameters()):
                    raise FloatingPointError("non-finite DPO gradients; no update applied")
                if self.train_cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.train_cfg.grad_clip
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.completed_updates += 1
                self._log(
                    {
                        "dpo/loss": total,
                        "dpo/elapsed": time.monotonic() - started,
                        "dpo/effective_batch": self.train_cfg.micro_batch_size
                        * self.train_cfg.grad_accum_steps,
                    },
                    self.completed_updates,
                )
                if self.completed_updates % self.train_cfg.save_interval == 0:
                    self._save(f"step-{self.completed_updates}.pt")
                if self.completed_updates % self.train_cfg.eval_interval == 0:
                    evaluation = self.evaluate()
                    self._log(evaluation, self.completed_updates, "eval")
                    if (
                        evaluation["held_out_loss"] is not None
                        and evaluation["held_out_loss"] < self.best_val_loss
                    ):
                        self.best_val_loss = evaluation["held_out_loss"]
                        self._save("best.pt")
            status = "interrupted" if self._stop_requested else "completed"
            self._save("interrupt.pt" if self._stop_requested else "last.pt")
            evaluation = self.evaluate()
            self._log(evaluation, self.completed_updates, "eval")
            self.run_manifest.add_evaluation(evaluation)
            self._log(
                {"dpo/completed_updates": self.completed_updates},
                self.completed_updates,
                "interrupted" if self._stop_requested else "done",
            )
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            try:
                self.checkpoints.close()
                for path in sorted(self.output_dir.glob("*.pt")):
                    self.run_manifest.add_checkpoint(path)
            except BaseException as settlement_error:
                status = "failed"
                if error is None:
                    error = f"{type(settlement_error).__name__}: {settlement_error}"
                    raise
                self.logger.error("checkpoint settlement failed after training error: %s", settlement_error)
            finally:
                self.run_manifest.finish(status, resources={"completed_updates": self.completed_updates, "elapsed_seconds": time.monotonic() - started}, provenance={"error": error} if error else {})


def dpo_from_config(config_path, preference_data, beta=0.1):
    DPOTrainer(load_config(config_path), preference_data, beta=beta).train()
