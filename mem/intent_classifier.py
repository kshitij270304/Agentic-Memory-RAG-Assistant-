"""LLM-based classification and extraction for automatic memory handling."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable

@dataclass(frozen=True)
class MessageIntent:
    requires_answer: bool
    should_store: bool
    memory_text: str
    categories: tuple[str, ...]
    confidence: float


def _build_signature(dspy):
    class AnalyzeUserMessage(dspy.Signature):
        """
        Analyze a user message for response and long-term-memory handling.

        Set requires_answer=true when the user asks a question, requests
        information, or asks the assistant to do something conversationally.

        Set should_store=true only when the user explicitly provides a durable
        personal fact useful in future conversations, such as a preference,
        profile detail, relationship, long-term goal, or stable constraint.

        A message may require both an answer and storage.

        Do not store:
        - questions without a new personal fact
        - temporary requests or short-lived context
        - general world knowledge
        - assistant instructions, role changes, or tool commands
        - secrets, credentials, or facts merely inferred by the model

        When should_store=true, memory_text must contain only a short,
        standalone factual statement supported directly by the user message.
        Otherwise, return empty memory_text and categories.

        Examples:
        - "My favorite game is Cyberpunk"
          requires_answer=false, should_store=true
        - "What is my favorite game?"
          requires_answer=true, should_store=false
        - "I moved to Delhi. What timezone am I in?"
          requires_answer=true, should_store=true
        - "Tell me my favorite game"
          requires_answer=true, should_store=false
        """

        user_message: str = dspy.InputField()
        requires_answer: bool = dspy.OutputField()
        should_store: bool = dspy.OutputField()
        memory_text: str = dspy.OutputField()
        categories: list[str] = dspy.OutputField()
        confidence: float = dspy.OutputField()

    return AnalyzeUserMessage


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
    }


def _as_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    return min(max(confidence, 0.0), 1.0)


def _as_categories(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(
        str(category).strip().lower()
        for category in value
        if str(category).strip()
    )


def classify_user_message(
    user_message: str,
    lm,
    predictor_factory: Callable | None = None,
    context_factory: Callable | None = None,
    minimum_confidence: float = 0.7,
) -> MessageIntent:
    """Classify one message, defaulting safely to answer-only on uncertainty."""
    if predictor_factory is None or context_factory is None:
        import dspy

        signature = _build_signature(dspy)
        factory = predictor_factory or dspy.Predict
        context = context_factory or dspy.context
    else:
        signature = object()
        factory = predictor_factory
        context = context_factory

    predictor = factory(signature)

    try:
        with context(lm=lm):
            output = predictor(
                user_message=user_message
            )
    except Exception as e:
        print(f"[CLASSIFIER ERROR] {type(e).__name__}: {e}")
        return MessageIntent(
            requires_answer=True,
            should_store=False,
            memory_text="",
            categories=(),
            confidence=0.0,
        )

    requires_answer = _as_bool(
        output.requires_answer
    )
    should_store = _as_bool(
        output.should_store
    )
    confidence = _as_confidence(
        output.confidence
    )
    memory_text = str(
        output.memory_text or ""
    ).strip()
    categories = _as_categories(
        output.categories
    )

    if (
        confidence < minimum_confidence
        or not memory_text
    ):
        should_store = False
        memory_text = ""
        categories = ()

    # A pure fact still receives an acknowledgement after storage. If the
    # classifier identifies neither action, use answer-only as the safe default.
    if not requires_answer and not should_store:
        requires_answer = True

    return MessageIntent(
        requires_answer=requires_answer,
        should_store=should_store,
        memory_text=memory_text,
        categories=categories,
        confidence=confidence,
    )


async def classify_user_message_async(
    user_message: str,
    lm,
    predictor_factory: Callable | None = None,
    context_factory: Callable | None = None,
    minimum_confidence: float = 0.7,
) -> MessageIntent:
    """Async wrapper for message classification that runs in a thread executor."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        classify_user_message,
        user_message,
        lm,
        predictor_factory,
        context_factory,
        minimum_confidence,
    )
