"""Security boundary for memory writes and retrieved long-term memory."""

from __future__ import annotations

import hashlib
import html
import logging
import re
import unicodedata
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Protocol

logger = logging.getLogger(__name__)


class MemoryLike(Protocol):
    point_id: str
    memory_text: str


class RiskLevel(str, Enum):
    SAFE = "safe"
    SUSPICIOUS = "suspicious"
    MALICIOUS = "malicious"


class UnsafeMemoryWriteError(ValueError):
    """Raised when user input is unsafe to pass to the memory write pipeline."""

    def __init__(self, risk_score: int, categories: tuple[str, ...]):
        self.risk_score = risk_score
        self.categories = categories
        super().__init__(
            "Memory was not stored because it contains instruction-like content."
        )


@dataclass(frozen=True)
class Detection:
    category: str
    description: str
    score: int
    matched_text: str


@dataclass(frozen=True)
class MemoryScanResult:
    original_text: str
    canonical_text: str
    sanitized_text: str | None
    risk_level: RiskLevel
    risk_score: int
    detections: tuple[Detection, ...]

    @property
    def allowed(self) -> bool:
        return self.sanitized_text is not None


@dataclass(frozen=True)
class SanitizedMemory:
    point_id: str
    text: str
    risk_level: RiskLevel
    risk_score: int


@dataclass(frozen=True)
class SanitizationReport:
    memories: tuple[SanitizedMemory, ...]
    quarantined_ids: tuple[str, ...]
    scanned_count: int


@dataclass(frozen=True)
class _Rule:
    category: str
    description: str
    score: int
    pattern: re.Pattern[str]


_ZERO_WIDTH = re.compile(r"[\u200b-\u200f\u2060\ufeff]")
_WHITESPACE = re.compile(r"[^\S\r\n]+")
_EXCESS_NEWLINES = re.compile(r"\n{3,}")

_RULES = (
    _Rule(
        "instruction_override",
        "Attempts to override higher-priority instructions",
        5,
        re.compile(
            r"\b(?:ignore|disregard|forget|override|bypass)\b.{0,50}"
            r"\b(?:previous|prior|above|system|developer|instructions?|rules?|prompt)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    _Rule(
        "role_impersonation",
        "Impersonates a privileged prompt role",
        4,
        re.compile(
            r"(?:^|\n)\s*(?:system|developer|assistant)\s*(?:message)?\s*[:>]",
            re.IGNORECASE,
        ),
    ),
    _Rule(
        "prompt_delimiter",
        "Uses common prompt or chat-template delimiters",
        4,
        re.compile(
            r"<\|(?:system|assistant|developer|im_start|im_end)\|>"
            r"|\[/?INST\]|<<\s*SYS\s*>>|###\s*(?:system|instruction)",
            re.IGNORECASE,
        ),
    ),
    _Rule(
        "instruction_to_model",
        "Directly instructs the model to change its behavior",
        3,
        re.compile(
            r"\b(?:you must|you are now|your new (?:task|role)|act as|follow these"
            r" instructions|do not answer|instead,? (?:say|return|output))\b",
            re.IGNORECASE,
        ),
    ),
    _Rule(
        "secret_exfiltration",
        "Requests secrets, hidden prompts, or credentials",
        5,
        re.compile(
            r"\b(?:reveal|show|print|return|extract|send|leak|exfiltrate)\b.{0,60}"
            r"\b(?:system prompt|hidden prompt|instructions?|api keys?|tokens?|"
            r"passwords?|credentials?|secrets?)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    _Rule(
        "tool_abuse",
        "Attempts to trigger tools or destructive actions",
        5,
        re.compile(
            r"\b(?:call|invoke|execute|run|use)\b.{0,40}"
            r"\b(?:tool|function|shell|terminal|delete|database)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    _Rule(
        "security_bypass",
        "Attempts to disable safeguards or conceal instructions",
        4,
        re.compile(
            r"\b(?:disable|evade|bypass|circumvent)\b.{0,40}"
            r"\b(?:safety|security|filter|guardrail|policy|detection)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
)


def canonicalize_text(text: str, max_length: int = 4000) -> str:
    """Normalize common obfuscation without interpreting or executing content."""
    normalized = html.unescape(str(text))
    normalized = unicodedata.normalize("NFKC", normalized)
    normalized = _ZERO_WIDTH.sub("", normalized)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = _WHITESPACE.sub(" ", normalized)
    normalized = _EXCESS_NEWLINES.sub("\n\n", normalized)
    return normalized.strip()[:max_length]


def scan_memory(text: str) -> MemoryScanResult:
    """Classify one memory and redact risky spans when that is sufficient."""
    canonical = canonicalize_text(text)
    detections: list[Detection] = []

    for rule in _RULES:
        for match in rule.pattern.finditer(canonical):
            detections.append(
                Detection(
                    category=rule.category,
                    description=rule.description,
                    score=rule.score,
                    matched_text=match.group(0)[:120],
                )
            )

    categories = {d.category for d in detections}
    raw_score = sum(d.score for d in detections)
    risk_score = min(raw_score + max(0, len(categories) - 1), 10)

    high_impact = {
        "instruction_override",
        "secret_exfiltration",
        "tool_abuse",
    }
    if risk_score >= 7 or len(categories & high_impact) >= 2:
        risk_level = RiskLevel.MALICIOUS
        sanitized = None
    elif risk_score >= 3:
        risk_level = RiskLevel.SUSPICIOUS
        sanitized = canonical
        for rule in _RULES:
            sanitized = rule.pattern.sub("[REDACTED UNTRUSTED INSTRUCTION]", sanitized)
        sanitized = sanitized.strip() or None
    else:
        risk_level = RiskLevel.SAFE
        sanitized = canonical

    return MemoryScanResult(
        original_text=text,
        canonical_text=canonical,
        sanitized_text=sanitized,
        risk_level=risk_level,
        risk_score=risk_score,
        detections=tuple(detections),
    )


def guard_memory_write(text: str) -> str:
    """Return canonical safe text or reject the write before any LLM call."""
    result = scan_memory(text)

    # Writes use a fail-closed policy. Redaction is acceptable for retrieved
    # context, but storing a partial instruction could poison future sessions.
    if result.risk_level is not RiskLevel.SAFE:
        categories = tuple(sorted({d.category for d in result.detections}))
        logger.warning(
            "Rejected memory write fingerprint=%s score=%d categories=%s",
            hashlib.sha256(result.canonical_text.encode()).hexdigest()[:12],
            result.risk_score,
            categories,
        )
        raise UnsafeMemoryWriteError(result.risk_score, categories)

    return result.canonical_text


def sanitize_retrieved_memories(
    memories: Iterable[MemoryLike],
) -> SanitizationReport:
    """Sanitize retrieved records and quarantine high-risk memories."""
    clean: list[SanitizedMemory] = []
    quarantined: list[str] = []
    scanned_count = 0

    for memory in memories:
        scanned_count += 1
        result = scan_memory(memory.memory_text)
        point_id = str(memory.point_id)

        if not result.allowed:
            quarantined.append(point_id)
            logger.warning(
                "Quarantined retrieved memory id=%s fingerprint=%s score=%d categories=%s",
                point_id,
                hashlib.sha256(result.canonical_text.encode()).hexdigest()[:12],
                result.risk_score,
                sorted({d.category for d in result.detections}),
            )
            continue

        clean.append(
            SanitizedMemory(
                point_id=point_id,
                text=result.sanitized_text,
                risk_level=result.risk_level,
                risk_score=result.risk_score,
            )
        )

    return SanitizationReport(
        memories=tuple(clean),
        quarantined_ids=tuple(quarantined),
        scanned_count=scanned_count,
    )
