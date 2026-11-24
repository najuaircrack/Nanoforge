"""Explicit consent, license and bounded PII filters for training records.

The detector is a deterministic pattern filter, not a comprehensive privacy classifier.
Counts and policy are recorded without retaining rejected content.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Iterator

from nanoforge.data.formats import DatasetRecord

_PII_PATTERNS = (
    re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
    re.compile(r"(?<!\w)(?:\+\d{1,3}[ -]?)?(?:\d[ -]?){10,15}(?!\w)"),
    re.compile(r"(?i)\b(?:api[_-]?key|access[_-]?token|password|secret)\s*[:=]\s*['\"]?[A-Za-z0-9_+/=-]{8,}"),
)


@dataclass(frozen=True)
class DataGovernancePolicy:
    owner: str = "local-user"
    scope: str = "project-training"
    retention_days: int | None = None
    allowed_licenses: tuple[str, ...] = ()
    require_license: bool = False
    require_consent: bool = False
    pii_action: str = "reject"

    def __post_init__(self):
        if not self.owner.strip() or not self.scope.strip():
            raise ValueError("Data governance requires an owner and scope")
        if self.retention_days is not None and self.retention_days < 0:
            raise ValueError("retention_days cannot be negative")
        if self.pii_action not in {"reject", "redact", "allow"}:
            raise ValueError("pii_action must be reject, redact or allow")


@dataclass
class GovernanceStats:
    seen: int = 0
    kept: int = 0
    pii_rejected: int = 0
    pii_redacted: int = 0
    pii_matches: int = 0
    license_rejected: int = 0
    consent_rejected: int = 0
    unknown_license: int = 0


def govern_records(records: Iterable[DatasetRecord], policy: DataGovernancePolicy,
                   stats: GovernanceStats | None = None) -> Iterator[DatasetRecord]:
    stats = stats if stats is not None else GovernanceStats()
    allowed = {license_name.casefold() for license_name in policy.allowed_licenses}
    for record in records:
        stats.seen += 1
        license_name = str(record.metadata.get("license", "")).strip()
        if not license_name:
            stats.unknown_license += 1
        if (policy.require_license and not license_name) or (
            allowed and license_name.casefold() not in allowed
        ):
            stats.license_rejected += 1
            continue
        if policy.require_consent and record.metadata.get("training_consent") is not True:
            stats.consent_rejected += 1
            continue
        matches = sum(len(pattern.findall(record.text)) for pattern in _PII_PATTERNS)
        stats.pii_matches += matches
        if matches and policy.pii_action == "reject":
            stats.pii_rejected += 1
            continue
        if matches and policy.pii_action == "redact":
            for pattern in _PII_PATTERNS:
                record.text = pattern.sub("[REDACTED]", record.text)
            stats.pii_redacted += 1
        record.metadata["governance"] = {
            "owner": policy.owner, "scope": policy.scope,
            "retention_days": policy.retention_days,
            "provenance": str(record.source), "deletion": "delete-source-and-repack",
            "pii_action": policy.pii_action,
        }
        stats.kept += 1
        yield record

