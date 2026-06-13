import unittest
from dataclasses import dataclass

from mem.memory_update_policy import (
    MemoryDecision,
    same_memory_subject,
    validate_memory_decision,
)


@dataclass
class FakeMemory:
    memory_text: str
    score: float


class MemoryUpdatePolicyTests(unittest.TestCase):
    def test_programming_and_sport_are_not_same_subject(self):
        self.assertFalse(
            same_memory_subject(
                "I like programming",
                "My favourite sport is football",
            )
        )

    def test_same_favorite_slot_can_be_updated(self):
        self.assertTrue(
            same_memory_subject(
                "My favourite sport is cricket",
                "My favourite sport is football",
            )
        )

    def test_unrelated_update_is_converted_to_add(self):
        decision = MemoryDecision(
            action="UPDATE",
            memory_id=0,
            memory_text="My favourite sport is football",
            categories=("sport",),
            confidence=0.95,
            summary="Update preference",
        )

        validated = validate_memory_decision(
            decision,
            [
                FakeMemory(
                    "I like programming",
                    0.81,
                )
            ],
            "My favourite sport is football",
        )

        self.assertEqual(validated.action, "ADD")

    def test_low_relevance_update_is_converted_to_add(self):
        decision = MemoryDecision(
            action="UPDATE",
            memory_id=0,
            memory_text="My favourite sport is football",
            categories=("sport",),
            confidence=0.9,
            summary="Update preference",
        )

        validated = validate_memory_decision(
            decision,
            [
                FakeMemory(
                    "My favourite sport is cricket",
                    0.4,
                )
            ],
            "My favourite sport is football",
        )

        self.assertEqual(validated.action, "ADD")


if __name__ == "__main__":
    unittest.main()
