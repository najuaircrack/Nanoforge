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
