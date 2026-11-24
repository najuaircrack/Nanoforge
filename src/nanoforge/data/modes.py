from __future__ import annotations

import re
from dataclasses import dataclass

from nanoforge.data.fim import transform_fim_text, validate_fim_tokenizer
from nanoforge.data.formats import DatasetRecord
from nanoforge.data.tokenizer import TokenizerLike

ROLE_RE = re.compile(r"(<\|(system|user|assistant|tool)\|>\s*)")
CODE_SUFFIXES = {".py", ".js", ".ts", ".tsx", ".jsx", ".rs", ".c", ".cpp", ".h", ".hpp", ".go", ".java"}


@dataclass(frozen=True)
class EncodedTrainingRecord:
    ids: list[int]
    labels: list[int]
    mode: str
    loss_masking: str


@dataclass(frozen=True)
class EncodedTrainingSequence:
    input_ids: list[int]
    labels: list[int]
    mode: str
    loss_masking: str


def infer_record_mode(record: DatasetRecord, requested_mode: str = "auto") -> str:
    mode = requested_mode.strip().lower().replace("-", "_")
    if mode != "auto":
        return mode
    hinted = str(record.metadata.get("mode", "")).strip().lower().replace("-", "_")
    if hinted:
        return hinted
    text = record.text
    if "<|assistant|>" in text and "<|user|>" in text:
        return "chat"
    suffix = str(record.metadata.get("suffix", "")).lower()
    if suffix in CODE_SUFFIXES or record.metadata.get("format") == "code":
        return "code"
    return "generative"


def resolve_loss_masking(mode: str, requested_policy: str = "auto") -> str:
    policy = requested_policy.strip().lower().replace("-", "_")
    if policy != "auto":
        return policy
    if mode in {"chat", "instruct", "reasoning", "roleplay"}:
        return "assistant_only"
    if mode == "completion":
        return "completion_only"
    return "none"


def encode_training_record(
    record: DatasetRecord,
    tokenizer: TokenizerLike,
    *,
    mode: str = "auto",
    loss_masking: str = "auto",
    add_bos: bool = True,
    add_eos: bool = True,
) -> EncodedTrainingRecord:
    actual_mode = infer_record_mode(record, mode)
    policy = resolve_loss_masking(actual_mode, loss_masking)
    if actual_mode == "fim":
        validate_fim_tokenizer(tokenizer)
        transformed = transform_fim_text(record.text)
        ids = tokenizer.encode(transformed, add_bos=add_bos, add_eos=add_eos)
        labels = list(ids)
        if labels:
            labels[0] = -100
        return EncodedTrainingRecord(ids=ids, labels=labels, mode=actual_mode, loss_masking="none")
    if policy == "assistant_only":
        ids, labels = _encode_role_masked(record.text, tokenizer, add_bos=add_bos, add_eos=add_eos)
    elif policy == "completion_only":
        ids, labels = _encode_completion_masked(record.text, tokenizer, add_bos=add_bos, add_eos=add_eos)
    elif policy == "partial":
        ids = tokenizer.encode(record.text, add_bos=add_bos, add_eos=add_eos)
        midpoint = len(ids) // 2
        labels = [-100 if idx < midpoint else token for idx, token in enumerate(ids)]
    else:
        ids = tokenizer.encode(record.text, add_bos=add_bos, add_eos=add_eos)
        labels = list(ids)
        if labels:
            labels[0] = -100
    return EncodedTrainingRecord(ids=ids, labels=labels, mode=actual_mode, loss_masking=policy)


def encode_training_sequences(
    record: DatasetRecord,
    tokenizer: TokenizerLike,
    *,
    seq_len: int,
    mode: str = "auto",
    loss_masking: str = "auto",
    assistant_target_fraction: float = 0.5,
) -> list[EncodedTrainingSequence]:
    actual_mode = infer_record_mode(record, mode)
    policy = resolve_loss_masking(actual_mode, loss_masking)
    if actual_mode in {"chat", "instruct", "reasoning", "roleplay"} and policy in {
        "assistant_only",
        "completion_only",
    }:
        sequences = _encode_role_boundary_sequences(
            record.text,
            tokenizer,
            seq_len=seq_len,
            mode=actual_mode,
            loss_masking=policy,
            assistant_target_fraction=assistant_target_fraction,
        )
        if sequences:
            return sequences
    encoded = encode_training_record(record, tokenizer, mode=actual_mode, loss_masking=policy)
    return _fixed_shifted_sequences(
        encoded.ids,
        encoded.labels,
        tokenizer.pad_id,
        seq_len=seq_len,
        mode=encoded.mode,
        loss_masking=encoded.loss_masking,
    )


def _encode_role_masked(
    text: str,
    tokenizer: TokenizerLike,
    *,
    add_bos: bool,
    add_eos: bool,
) -> tuple[list[int], list[int]]:
    ids: list[int] = []
    labels: list[int] = []
    current_role = ""
    if add_bos:
        ids.append(tokenizer.bos_id)
        labels.append(-100)
    parts = ROLE_RE.split(text)
    idx = 0
    while idx < len(parts):
        part = parts[idx]
        if not part:
            idx += 1
            continue
        if part.startswith("<|") and idx + 1 < len(parts):
            marker_ids = tokenizer.encode(part, add_bos=False, add_eos=False)
            ids.extend(marker_ids)
            labels.extend([-100] * len(marker_ids))
            current_role = parts[idx + 1]
            idx += 2
            continue
        part_ids = tokenizer.encode(part, add_bos=False, add_eos=False)
        ids.extend(part_ids)
        if current_role == "assistant":
            labels.extend(part_ids)
        else:
            labels.extend([-100] * len(part_ids))
        idx += 1
    if add_eos:
        ids.append(tokenizer.eos_id)
        labels.append(tokenizer.eos_id if current_role == "assistant" else -100)
    return ids, labels


def _encode_role_boundary_sequences(
    text: str,
    tokenizer: TokenizerLike,
    *,
    seq_len: int,
    mode: str,
    loss_masking: str,
    assistant_target_fraction: float,
) -> list[EncodedTrainingSequence]:
    if seq_len < 2:
        raise ValueError("seq_len must be at least two")
    blocks = _role_blocks(text)
    if not any(role == "assistant" for role, _ in blocks):
        return []
    history_ids: list[int] = []
    sequences: list[EncodedTrainingSequence] = []
    max_target = max(1, min(seq_len - 1, int(seq_len * max(0.1, min(0.8, assistant_target_fraction)))))
    for role, content in blocks:
        marker = tokenizer.encode(f"<|{role}|>\n", add_bos=False, add_eos=False)
        content_ids = tokenizer.encode(content, add_bos=False, add_eos=False)
        prefix = history_ids + marker
        if role == "assistant" and content_ids:
            for start in range(0, len(content_ids), max_target):
                chunk = content_ids[start:start + max_target]
                causal_prefix = prefix + content_ids[:start]
                prefix_budget = max(0, seq_len - len(chunk) - 1)
                causal_prefix = causal_prefix[-prefix_budget:] if prefix_budget else []
                full_ids = [tokenizer.bos_id] + causal_prefix + chunk
                full_labels = [-100] * (1 + len(causal_prefix)) + chunk
                if start + len(chunk) == len(content_ids):
                    full_ids.append(tokenizer.eos_id)
                    full_labels.append(tokenizer.eos_id)
                sequences.extend(_fixed_shifted_sequences(
                    full_ids, full_labels, tokenizer.pad_id, seq_len=seq_len,
                    mode=mode, loss_masking=loss_masking, allow_chunking=False,
                ))
        # Only already-observed turns enter the prefix. Future system/tool turns
        # cannot leak into an earlier answer; continuation chunks include their tail.
        history_ids.extend(marker)
        history_ids.extend(content_ids)
    return sequences


def _role_blocks(text: str) -> list[tuple[str, str]]:
    parts = ROLE_RE.split(text)
    blocks: list[tuple[str, str]] = []
    current_role = ""
    idx = 0
    while idx < len(parts):
        part = parts[idx]
        if not part:
            idx += 1
            continue
        if part.startswith("<|") and idx + 1 < len(parts):
            current_role = parts[idx + 1]
            idx += 2
            continue
        if current_role:
            blocks.append((current_role, part))
        idx += 1
    return blocks


def _chat_prefix(system_parts: list[str], last_user: str) -> str:
    pieces: list[str] = []
    for system in system_parts[-1:]:
        if system.strip():
            pieces.append(f"<|system|>\n{system.strip()}\n")
    if last_user.strip():
        pieces.append(f"<|user|>\n{last_user.strip()}\n")
    pieces.append("<|assistant|>\n")
    return "".join(pieces)


def _fixed_shifted_sequences(
    ids: list[int],
    labels: list[int],
    pad_id: int,
    *,
    seq_len: int,
    mode: str,
    loss_masking: str,
    allow_chunking: bool = True,
) -> list[EncodedTrainingSequence]:
    if len(ids) < 2:
        return []
    chunk_width = seq_len + 1
    if seq_len < 1:
        raise ValueError("seq_len must be positive")
    starts = range(0, len(ids) - 1, seq_len) if allow_chunking else (0,)
    sequences: list[EncodedTrainingSequence] = []
    for start in starts:
        full_ids = ids[start : start + chunk_width]
        full_labels = labels[start : start + chunk_width]
        if len(full_ids) < 2:
            continue
        if len(full_ids) < chunk_width:
            pad = chunk_width - len(full_ids)
            full_ids = full_ids + [pad_id] * pad
            full_labels = full_labels + [-100] * pad
        input_ids = full_ids[:-1]
        target_labels = full_labels[1:]
        if len(input_ids) == seq_len and len(target_labels) == seq_len:
            sequences.append(
                EncodedTrainingSequence(
                    input_ids=input_ids,
                    labels=target_labels,
                    mode=mode,
                    loss_masking=loss_masking,
                )
            )
    return sequences


def _encode_completion_masked(
    text: str,
    tokenizer: TokenizerLike,
    *,
    add_bos: bool,
    add_eos: bool,
) -> tuple[list[int], list[int]]:
    if "<|assistant|>" in text:
        return _encode_role_masked(text, tokenizer, add_bos=add_bos, add_eos=add_eos)
    split_markers = ["\n### Response:\n", "\nCompletion:\n", "\nAnswer:\n"]
    split_at = -1
    marker = ""
    for candidate in split_markers:
        split_at = text.find(candidate)
        if split_at >= 0:
            marker = candidate
            break
    if split_at < 0:
        raise ValueError("completion_only masking requires a recognizable completion boundary")
    prompt = text[: split_at + len(marker)]
    completion = text[split_at + len(marker) :]
    prompt_ids = tokenizer.encode(prompt, add_bos=add_bos, add_eos=False)
    completion_ids = tokenizer.encode(completion, add_bos=False, add_eos=add_eos)
    return prompt_ids + completion_ids, [-100] * len(prompt_ids) + completion_ids
