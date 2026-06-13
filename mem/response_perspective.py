"""Deterministic correction for user-memory response perspective."""

import re


_USER_PERSPECTIVE_QUESTIONS = re.compile(
    r"\b(?:my|me|mine|i)\b",
    re.IGNORECASE,
)
_FIRST_PERSON_RESPONSE = re.compile(
    r"^\s*(?:according to (?:the )?memor(?:y|ies),?\s*)?"
    r"(?P<prefix>my|i am|i'm)\b",
    re.IGNORECASE,
)


def correct_user_perspective(
    question: str,
    response: str,
) -> str:
    """Correct a copied first-person memory at the start of an answer."""
    if not _USER_PERSPECTIVE_QUESTIONS.search(question):
        return response

    match = _FIRST_PERSON_RESPONSE.match(response)
    if not match:
        return response

    replacement = {
        "my": "Your",
        "i am": "You are",
        "i'm": "You're",
    }[match.group("prefix").lower()]

    return (
        response[:match.start("prefix")]
        + replacement
        + response[match.end("prefix"):]
    )
