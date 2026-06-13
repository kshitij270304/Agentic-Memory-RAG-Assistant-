import unittest

from mem.response_perspective import correct_user_perspective


class ResponsePerspectiveTests(unittest.TestCase):
    def test_converts_my_name_to_your_name(self):
        response = correct_user_perspective(
            question="What is my name?",
            response="My name is Kshitij.",
        )

        self.assertEqual(
            response,
            "Your name is Kshitij.",
        )

    def test_converts_i_am_to_you_are(self):
        response = correct_user_perspective(
            question="Where am I from?",
            response="I am from Delhi.",
        )

        self.assertEqual(
            response,
            "You are from Delhi.",
        )

    def test_preserves_correct_second_person_answer(self):
        response = correct_user_perspective(
            question="What is my name?",
            response="Your name is Kshitij.",
        )

        self.assertEqual(
            response,
            "Your name is Kshitij.",
        )

    def test_does_not_rewrite_unrelated_first_person_response(self):
        response = correct_user_perspective(
            question="Who is the assistant?",
            response="I am the assistant.",
        )

        self.assertEqual(
            response,
            "I am the assistant.",
        )


if __name__ == "__main__":
    unittest.main()
