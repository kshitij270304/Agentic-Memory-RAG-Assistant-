"""Deterministic safety policy for memory mutation decisions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol


class ExistingMemory(Protocol):
    memory_text: str
    score: float


@dataclass(frozen=True)
class MemoryDecision:
    action: str
    memory_id: int
    memory_text: str
    categories: tuple[str, ...]
    confidence: float
    summary: str


_STOP_WORDS = {
    "a",
    "am",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "i",
    "in",
    "is",
    "it",
    "my",
    "of",
    "the",
    "to",
    "user",
}

_SLOT_PATTERNS = (
    re.compile(r"\b(?:my|the user's)\s+favou?rite\s+([a-z][a-z0-9_-]*)"),
    re.compile(r"\b(?:my|the user's)\s+([a-z][a-z0-9_-]*)\s+is\b"),
)


def _subject_slots(text: str) -> set[str]:
    lowered = text.lower()
    return {
        match.group(1)
        for pattern in _SLOT_PATTERNS
        for match in pattern.finditer(lowered)
    }


def _content_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(
            r"[a-z][a-z0-9_-]{2,}",
            text.lower(),
        )
        if token not in _STOP_WORDS
    }


def same_memory_subject(
    existing_text: str,
    new_text: str,
) -> bool:
    """Return whether two facts describe the same stable memory slot."""
    existing_slots = _subject_slots(existing_text)
    new_slots = _subject_slots(new_text)
    if existing_slots or new_slots:
        return bool(existing_slots & new_slots)

    shared_tokens = (
        _content_tokens(existing_text)
        & _content_tokens(new_text)
    )
    return len(shared_tokens) >= 2


def validate_memory_decision(
    decision: MemoryDecision,
    existing_memories: list[ExistingMemory],
    new_memory_text: str,
    minimum_target_score: float = 0.65,
) -> MemoryDecision:
    """Fail safe when an LLM selects an unrelated or invalid mutation."""
    action = decision.action.strip().upper()
    if action not in {
        "ADD",
        "UPDATE",
        "DELETE",
        "NOOP",
    }:
        action = "ADD"

    if action in {"UPDATE", "DELETE"}:
        valid_id = (
            0 <= decision.memory_id < len(existing_memories)
        )
        if not valid_id:
            action = "ADD" if action == "UPDATE" else "NOOP"
        else:
            target = existing_memories[decision.memory_id]
            related = (
                target.score >= minimum_target_score
                and same_memory_subject(
                    target.memory_text,
                    new_memory_text,
                )
            )
            if not related:
                action = "ADD" if action == "UPDATE" else "NOOP"

    memory_text = (
        decision.memory_text.strip()
        if decision.memory_text.strip()
        else new_memory_text.strip()
    )

    return MemoryDecision(
        action=action,
        memory_id=decision.memory_id,
        memory_text=memory_text,
        categories=decision.categories or ("general",),
        confidence=max(0.0, min(1.0, decision.confidence)),
        summary=decision.summary,
    )
