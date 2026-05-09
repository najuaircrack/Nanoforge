from __future__ import annotations

import re
from dataclasses import dataclass

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
    blocks = _role_blocks(text)
    if not any(role == "assistant" for role, _ in blocks):
        return []
    system_parts = [content for role, content in blocks if role == "system"]
    last_user = ""
    sequences: list[EncodedTrainingSequence] = []
    max_target_tokens = max(8, int(seq_len * max(0.1, min(0.8, assistant_target_fraction))))
    for role, content in blocks:
        if role == "user":
            last_user = content
            continue
        if role != "assistant" or not content.strip():
            continue
        prefix_text = _chat_prefix(system_parts, last_user)
        prefix_ids = tokenizer.encode(prefix_text, add_bos=False, add_eos=False)
        max_prefix = max(8, seq_len + 1 - max_target_tokens - 2)
        if len(prefix_ids) > max_prefix:
            prefix_ids = prefix_ids[-max_prefix:]
        content_ids = tokenizer.encode(content, add_bos=False, add_eos=False)
        if not content_ids:
            continue
        capacity = max(1, min(max_target_tokens, seq_len + 1 - len(prefix_ids) - 2))
        for start in range(0, len(content_ids), capacity):
            chunk = content_ids[start : start + capacity]
            is_last = start + capacity >= len(content_ids)
            full_ids = [tokenizer.bos_id] + prefix_ids + chunk
            full_labels = [-100] * (1 + len(prefix_ids)) + chunk
            if is_last and len(full_ids) < seq_len + 1:
                full_ids.append(tokenizer.eos_id)
                full_labels.append(tokenizer.eos_id)
            sequences.extend(
                _fixed_shifted_sequences(
                    full_ids,
                    full_labels,
                    tokenizer.pad_id,
                    seq_len=seq_len,
                    mode=mode,
                    loss_masking=loss_masking,
                    allow_chunking=False,
                )
            )
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
    starts = range(0, len(ids), chunk_width) if allow_chunking else (0,)
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
        ids = tokenizer.encode(text, add_bos=add_bos, add_eos=add_eos)
        labels = list(ids)
        if labels:
            labels[0] = -100
        return ids, labels
    prompt = text[: split_at + len(marker)]
    completion = text[split_at + len(marker) :]
    prompt_ids = tokenizer.encode(prompt, add_bos=add_bos, add_eos=False)
    completion_ids = tokenizer.encode(completion, add_bos=False, add_eos=add_eos)
    return prompt_ids + completion_ids, [-100] * len(prompt_ids) + completion_ids
