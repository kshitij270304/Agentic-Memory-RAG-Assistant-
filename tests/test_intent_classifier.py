import unittest
from types import SimpleNamespace

from mem.intent_classifier import classify_user_message


class FakeContext:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class IntentClassifierTests(unittest.TestCase):
    @staticmethod
    def factory(output):
        return lambda _: lambda **__: output

    @staticmethod
    def context_factory(**_):
        return FakeContext()

    def test_personal_fact_is_stored_without_rag_answer(self):
        output = SimpleNamespace(
            requires_answer=False,
            should_store=True,
            memory_text="My favorite game is Cyberpunk.",
            categories=["preference", "gaming"],
            confidence=0.96,
        )

        intent = classify_user_message(
            "My favorite game is Cyberpunk",
            lm=object(),
            predictor_factory=self.factory(output),
            context_factory=self.context_factory,
        )

        self.assertFalse(intent.requires_answer)
        self.assertTrue(intent.should_store)
        self.assertEqual(
            intent.memory_text,
            "My favorite game is Cyberpunk.",
        )

    def test_question_only_uses_rag_without_storage(self):
        output = SimpleNamespace(
            requires_answer=True,
            should_store=False,
            memory_text="",
            categories=[],
            confidence=0.99,
        )

        intent = classify_user_message(
            "What is my favorite game?",
            lm=object(),
            predictor_factory=self.factory(output),
            context_factory=self.context_factory,
        )

        self.assertTrue(intent.requires_answer)
        self.assertFalse(intent.should_store)

    def test_message_can_be_fact_and_question(self):
        output = SimpleNamespace(
            requires_answer=True,
            should_store=True,
            memory_text="I moved to Delhi.",
            categories=["location"],
            confidence=0.91,
        )

        intent = classify_user_message(
            "I moved to Delhi. What timezone am I in?",
            lm=object(),
            predictor_factory=self.factory(output),
            context_factory=self.context_factory,
        )

        self.assertTrue(intent.requires_answer)
        self.assertTrue(intent.should_store)

    def test_low_confidence_write_fails_closed(self):
        output = SimpleNamespace(
            requires_answer=False,
            should_store=True,
            memory_text="My favorite game is Cyberpunk.",
            categories=["preference"],
            confidence=0.4,
        )

        intent = classify_user_message(
            "My favorite game is Cyberpunk",
            lm=object(),
            predictor_factory=self.factory(output),
            context_factory=self.context_factory,
        )

        self.assertTrue(intent.requires_answer)
        self.assertFalse(intent.should_store)

    def test_classifier_failure_defaults_to_answer_only(self):
        def broken_factory(_):
            def predict(**__):
                raise RuntimeError("model unavailable")

            return predict

        intent = classify_user_message(
            "Hello",
            lm=object(),
            predictor_factory=broken_factory,
            context_factory=self.context_factory,
        )

        self.assertTrue(intent.requires_answer)
        self.assertFalse(intent.should_store)


if __name__ == "__main__":
    unittest.main()
