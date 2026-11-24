from __future__ import annotations

import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from nanoforge.artifacts import atomic_write_json, environment_snapshot, fingerprint_file
from nanoforge.training.utils import resolve_device, seed_everything


@dataclass
class PretrainedAdaptationConfig:
    source: str
    data_path: str
    output_dir: str
    method: str = "lora"
    max_steps: int = 100
    batch_size: int = 1
    grad_accum_steps: int = 1
    learning_rate: float = 2e-4
    seq_len: int = 256
    seed: int = 1337
    device: str = "cpu"
    lora_rank: int = 8
    lora_alpha: float = 16
    target_modules: list[str] | None = None
    allow_network: bool = False
    validation_path: str | None = None
    max_seconds: float = 3600
    max_record_bytes: int = 1048576

    def __post_init__(self):
        if self.method not in {"full", "lora", "qlora"}:
            raise ValueError("method must be full, lora or qlora")
        if (
            min(
                self.max_steps, self.batch_size, self.grad_accum_steps, self.seq_len, self.lora_rank
            )
            <= 0
            or self.seq_len < 2
            or self.learning_rate <= 0
            or self.max_seconds <= 0
            or self.max_record_bytes <= 0
        ):
            raise ValueError("adaptation budgets and learning rate must be positive")


def _dependencies():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("Install latticekin[adaptation] for pretrained adaptation") from error
    return AutoModelForCausalLM, AutoTokenizer


def _encode_sft(record, tokenizer, seq_len, line_number, stats=None):
    if (
        not isinstance(record, dict)
        or not isinstance(record.get("prompt"), str)
        or not isinstance(record.get("response"), str)
        or not record["response"].strip()
    ):
        raise ValueError(f"SFT line {line_number}: prompt/response strings are required")
    prompt = tokenizer.encode(record["prompt"], add_special_tokens=False)
    response = tokenizer.encode(record["response"], add_special_tokens=False)
    if tokenizer.eos_token_id is not None:
        response.append(tokenizer.eos_token_id)
    original_tokens = len(prompt) + len(response)
    response = response[: seq_len - 1]
    prompt = prompt[-(seq_len - len(response)) :]
    ids, labels = prompt + response, [-100] * len(prompt) + response
    if len(ids) < 2 or not any(value >= 0 for value in labels[1:]):
        raise ValueError(f"SFT line {line_number}: no causal response targets")
    if stats is not None:
        stats["accepted"] += 1
        stats["truncated_tokens"] += original_tokens - len(ids)
        stats["supervised_tokens"] += sum(value >= 0 for value in labels[1:])
    return ids, labels, hashlib.sha256(record["prompt"].encode()).hexdigest()


@dataclass
class _SFTRecord:
    path: Path
    offset: int
    group: str
    tokenizer: object
    seq_len: int
    source_stat: tuple[int, int]

    def load(self):
        current = self.path.stat()
        if (current.st_size, current.st_mtime_ns) != self.source_stat:
            raise ValueError("SFT source changed during adaptation")
        with self.path.open("rb") as stream:
            stream.seek(self.offset)
            return _encode_sft(
                json.loads(stream.readline()), self.tokenizer, self.seq_len, self.offset
            )

    def __getitem__(self, index):
        return self.group if index == 2 else self.load()[index]

    def __iter__(self):
        return iter(self.load())


class _SFTRecords(list):
    def __init__(self):
        super().__init__()
        self.stats = {"accepted": 0, "blank": 0, "truncated_tokens": 0, "supervised_tokens": 0}


def _records(path, tokenizer, seq_len, max_record_bytes=1048576):
    path = Path(path)
    stat = path.stat()
    identity = (stat.st_size, stat.st_mtime_ns)
    result = _SFTRecords()
    with path.open("rb") as stream:
        line_number = 0
        while True:
            offset = stream.tell()
            line = stream.readline(max_record_bytes + 1)
            if not line:
                break
            line_number += 1
            if len(line) > max_record_bytes:
                raise ValueError(f"SFT line {line_number}: record exceeds byte limit")
            if not line.strip():
                result.stats["blank"] += 1
                continue
            try:
                _, _, group = _encode_sft(
                    json.loads(line), tokenizer, seq_len, line_number, result.stats
                )
            except (ValueError, UnicodeError) as error:
                raise ValueError(f"SFT line {line_number}: {error}") from error
            result.append(_SFTRecord(path, offset, group, tokenizer, seq_len, identity))
    if not result:
        raise ValueError("SFT dataset is empty")
    return result


def _batch(records, tokenizer, device):
    records = [tuple(record) for record in records]
    length = max(len(record[0]) for record in records)
    ids = torch.full((len(records), length), tokenizer.pad_token_id, dtype=torch.long)
    labels = torch.full_like(ids, -100)
    attention = torch.zeros_like(ids)
    for index, (tokens, targets, _) in enumerate(records):
        ids[index, : len(tokens)] = torch.tensor(tokens)
        labels[index, : len(targets)] = torch.tensor(targets)
        attention[index, : len(tokens)] = 1
    return {
        "input_ids": ids.to(device),
        "labels": labels.to(device),
        "attention_mask": attention.to(device),
    }


def _heldout(model, records, tokenizer, device):
    training = model.training
    model.eval()
    total, tokens, correct = 0.0, 0, 0
    started = time.monotonic()
    with torch.inference_mode():
        for record in records:
            batch = _batch([record], tokenizer, device)
            valid = int((batch["labels"][:, 1:] != -100).sum())
            output = model(**batch)
            loss = output.loss
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite held-out loss")
            total += float(loss.cpu()) * valid
            tokens += valid
            mask = batch["labels"][:, 1:] != -100
            correct += int(
                ((output.logits[:, :-1].argmax(-1) == batch["labels"][:, 1:]) & mask).sum()
            )
    model.train(training)
    return {
        "held_out_loss": total / tokens,
        "held_out_tokens": tokens,
        "token_accuracy": correct / tokens,
        "perplexity": math.exp(min(total / tokens, 80)),
        "elapsed_seconds": time.monotonic() - started,
    }


def train_pretrained(config: PretrainedAdaptationConfig, *, interrupt=None) -> dict:
    """Local-only by default: adapt a real imported base, evaluate, save, and verify reload."""
    AutoModel, AutoTokenizer = _dependencies()
    if not Path(config.source).exists() and not config.allow_network:
        raise ValueError("pretrained source is not local; network requires allow_network=True")
    if Path(config.output_dir).exists() and any(Path(config.output_dir).iterdir()):
        raise ValueError("adaptation output directory must be empty")
    seed_everything(config.seed)
    device = resolve_device(config.device)
    tokenizer = AutoTokenizer.from_pretrained(
        config.source, local_files_only=not config.allow_network, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer requires a PAD or EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    options = {"local_files_only": not config.allow_network, "trust_remote_code": False}
    if config.method == "qlora":
        if device.type != "cuda":
            raise ValueError("QLoRA requires a CUDA bitsandbytes backend; CPU QLoRA is unsupported")
        try:
            import bitsandbytes  # noqa: F401
            from peft import prepare_model_for_kbit_training
            from transformers import BitsAndBytesConfig
        except ImportError as error:
            raise RuntimeError("QLoRA requires bitsandbytes and PEFT") from error
        options["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
        )
        options["device_map"] = {"": str(device)}
        model = prepare_model_for_kbit_training(AutoModel.from_pretrained(config.source, **options))
    else:
        model = AutoModel.from_pretrained(config.source, torch_dtype=torch.float32, **options).to(
            device
        )
    if model.get_input_embeddings().num_embeddings != len(tokenizer):
        raise ValueError("base/tokenizer vocabulary mismatch")
    context = getattr(
        model.config, "max_position_embeddings", getattr(model.config, "n_positions", None)
    )
    if context and config.seq_len > context:
        raise ValueError("adaptation sequence exceeds base context")
    if config.method in {"lora", "qlora"}:
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as error:
            raise RuntimeError("Install PEFT for LoRA adaptation") from error
        model = get_peft_model(
            model,
            LoraConfig(
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                target_modules=config.target_modules or "all-linear",
                task_type=TaskType.CAUSAL_LM,
                lora_dropout=0,
            ),
        )
    dataset = _records(config.data_path, tokenizer, config.seq_len, config.max_record_bytes)
    if config.validation_path:
        train, val = (
            dataset,
            _records(config.validation_path, tokenizer, config.seq_len, config.max_record_bytes),
        )
        if {record[2] for record in train} & {record[2] for record in val}:
            raise ValueError("train/validation prompt groups overlap")
    else:
        groups = sorted({record[2] for record in dataset})
        if len(groups) < 2:
            raise ValueError("held-out evaluation needs at least two distinct prompt groups")
        random.Random(config.seed).shuffle(groups)
        heldout = set(groups[: max(1, len(groups) // 10)])
        train = [record for record in dataset if record[2] not in heldout]
        val = [record for record in dataset if record[2] in heldout]
    initial_eval = _heldout(model, val, tokenizer, device)
    trainable = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate)
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    started, cpu_started, completed = time.monotonic(), time.process_time(), 0
    loss_history = []
    status = "running"
    manifest = {
        "schema_version": 1,
        "kind": "pretrained_adaptation",
        "method": config.method,
        "config": asdict(config),
        "source": config.source,
        "base_fingerprint": _directory_identity(Path(config.source))
        if Path(config.source).is_dir()
        else None,
        "data_fingerprint": fingerprint_file(config.data_path),
        "filter_counts": dataset.stats,
        "sequence_length": config.seq_len,
        "masking_policy": "response_only",
        "environment": environment_snapshot(str(device)),
        "evaluation_kind": "held_out_response_likelihood_and_token_accuracy",
        "split_seed": config.seed,
        "train_records": len(train),
        "held_out_records": len(val),
        "initial_evaluation": initial_eval,
        "completed_updates": 0,
        "status": status,
        "trainable_parameters": sum(param.numel() for param in trainable),
        "total_parameters": sum(param.numel() for param in model.parameters()),
    }
    atomic_write_json(output / "adaptation.manifest.json", manifest)
    try:
        model.train()
        while completed < config.max_steps:
            if (interrupt and interrupt()) or time.monotonic() - started >= config.max_seconds:
                status = "interrupted"
                break
            optimizer.zero_grad(set_to_none=True)
            averaged = 0.0
            for _ in range(config.grad_accum_steps):
                indices = torch.randint(len(train), (config.batch_size,)).tolist()
                loss = model(**_batch([train[index] for index in indices], tokenizer, device)).loss
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite adaptation loss")
                (loss / config.grad_accum_steps).backward()
                averaged += float(loss.detach().cpu()) / config.grad_accum_steps
            if any(
                param.grad is not None and not torch.isfinite(param.grad).all()
                for param in trainable
            ):
                raise FloatingPointError("non-finite adaptation gradients")
            torch.nn.utils.clip_grad_norm_(trainable, 1)
            optimizer.step()
            completed += 1
            loss_history.append(averaged)
        else:
            status = "completed"
        evaluation = _heldout(model, val, tokenizer, device)
        model.save_pretrained(output, safe_serialization=True)
        tokenizer.save_pretrained(output)
        manifest.update(
            status=status,
            completed_updates=completed,
            evaluation=evaluation,
            training_losses=loss_history,
            elapsed_seconds=time.monotonic() - started,
            process_cpu_seconds=time.process_time() - cpu_started,
            device=str(device),
            tokenizer_fingerprint=_directory_identity(output, tokenizer_only=True),
            saved_model_fingerprint=_directory_identity(output),
        )
        atomic_write_json(output / "adaptation.manifest.json", manifest)
        reloaded, reload_tokenizer, _ = load_pretrained_adaptation(
            output, device=str(device), allow_network=config.allow_network
        )
        probe = _batch([val[0]], tokenizer, device)
        model.eval()
        with torch.inference_mode():
            expected = model(
                input_ids=probe["input_ids"], attention_mask=probe["attention_mask"]
            ).logits.float()
            actual = reloaded(
                input_ids=probe["input_ids"], attention_mask=probe["attention_mask"]
            ).logits.float()
        if not torch.allclose(expected, actual, atol=1e-5, rtol=1e-4):
            raise RuntimeError("saved adaptation reload changed logits")
        manifest["reload_verified"] = True
        atomic_write_json(output / "adaptation.manifest.json", manifest)
        return manifest
    except BaseException as error:
        manifest.update(
            status="failed", completed_updates=completed, error=f"{type(error).__name__}: {error}"
        )
        atomic_write_json(output / "adaptation.manifest.json", manifest)
        raise


def _directory_identity(path: Path, tokenizer_only=False):
    digest = hashlib.sha256()
    for item in sorted(path.iterdir()):
        if not item.is_file() or item.name == "adaptation.manifest.json":
            continue
        if tokenizer_only and not any(
            key in item.name for key in ("token", "vocab", "merges", "special", "added")
        ):
            continue
        digest.update(item.name.encode())
        digest.update(fingerprint_file(item).encode())
    return digest.hexdigest()


def load_pretrained_adaptation(path, *, device="cpu", allow_network=False):
    AutoModel, AutoTokenizer = _dependencies()
    path = Path(path)
    manifest = json.loads((path / "adaptation.manifest.json").read_text())
    if manifest.get("schema_version") != 1 or manifest.get("kind") != "pretrained_adaptation":
        raise ValueError("unsupported adaptation manifest schema")
    if manifest.get("status") not in {"completed", "interrupted"}:
        raise ValueError("adaptation artifact is not ready for inference")
    if manifest.get("saved_model_fingerprint") != _directory_identity(path):
        raise ValueError("saved adaptation fingerprint changed")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    if manifest["method"] in {"lora", "qlora"}:
        from peft import PeftModel

        source = manifest["source"]
        if not Path(source).exists() and not allow_network:
            raise ValueError("adapter base is unavailable locally; network is opt-in")
        if (
            Path(source).is_dir()
            and _directory_identity(Path(source)) != manifest["base_fingerprint"]
        ):
            raise ValueError("adapter base fingerprint changed")
        options = {"local_files_only": not allow_network, "trust_remote_code": False}
        if manifest["method"] == "qlora":
            selected = resolve_device(device)
            if selected.type != "cuda":
                raise ValueError(
                    "saved QLoRA inference requires CUDA bitsandbytes; export a separately evaluated CPU artifact"
                )
            from transformers import BitsAndBytesConfig

            options["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
            )
            options["device_map"] = {"": str(selected)}
        else:
            options["torch_dtype"] = torch.float32
        base = AutoModel.from_pretrained(source, **options)
        model = PeftModel.from_pretrained(base, path, is_trainable=False)
    else:
        model = AutoModel.from_pretrained(
            path, torch_dtype=torch.float32, local_files_only=True, trust_remote_code=False
        )
    if len(tokenizer) != model.get_input_embeddings().num_embeddings:
        raise ValueError("adaptation tokenizer/model mismatch")
    result = model if manifest["method"] == "qlora" else model.to(resolve_device(device))
    return result.eval(), tokenizer, manifest
