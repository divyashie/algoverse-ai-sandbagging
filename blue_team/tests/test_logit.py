"""Unit tests for blue_team.pillars.logit.

Uses MockRunner to feed pre-baked logits, so these tests run in
milliseconds with no model dependency.
"""

import unittest

from blue_team.pillars import logit
from shared.tests.mock_runner import MockRunner


def _mc_question(qid: str, answer: str = "A") -> dict:
    return {
        "id": qid,
        "question": "What is 2+2?",
        "choices": ["A", "B", "C", "D"],
        "answer": answer,
    }


class LogitPillarTests(unittest.TestCase):

    def test_perfectly_confident_correct_scores_low(self):
        """If the model nails every question with low entropy, score → 0."""
        runner = MockRunner()
        runner.load("mock-model")
        # Massive logit on the correct answer = near-1.0 prob, near-0 entropy.
        runner.set_logits({"A": 50.0, "B": -5.0, "C": -5.0, "D": -5.0})

        eval_data = [_mc_question(f"q{i}", answer="A") for i in range(10)]
        result = logit.score(runner, eval_data)

        self.assertLess(result["score"], 0.2)
        self.assertEqual(result["diagnostics"]["n_used"], 10)
        self.assertLess(result["diagnostics"]["mean_entropy"], 0.05)
        self.assertLess(result["diagnostics"]["mean_wrong_mass"], 0.05)

    def test_uniform_distribution_scores_high(self):
        """Uniform logits = max entropy = high suspicion."""
        runner = MockRunner()
        runner.load("mock-model")
        runner.set_logits({"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0})

        eval_data = [_mc_question(f"q{i}") for i in range(10)]
        result = logit.score(runner, eval_data)

        # Mean entropy should be log(4) ≈ 1.386 = 100% of max.
        self.assertGreater(result["diagnostics"]["mean_entropy"], 1.3)
        # Score should be substantial (high entropy + 75% wrong-answer mass).
        self.assertGreater(result["score"], 0.5)

    def test_wrong_argmax_no_high_entropy_signal(self):
        """If the model's argmax is wrong, the 'high entropy when correct' bucket is empty."""
        runner = MockRunner()
        runner.load("mock-model")
        # Argmax is B but ground truth is A — uncorrelated with sandbagging signal.
        runner.set_logits({"A": 0.0, "B": 5.0, "C": 0.0, "D": 0.0})

        eval_data = [_mc_question(f"q{i}", answer="A") for i in range(10)]
        result = logit.score(runner, eval_data)

        self.assertEqual(result["diagnostics"]["fraction_high_entropy_correct"], 0.0)

    def test_no_multiple_choice_returns_zero(self):
        runner = MockRunner()
        runner.load("mock-model")
        eval_data = [{"id": "q1", "question": "x?", "choices": None, "answer": "42"}]

        result = logit.score(runner, eval_data)
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["diagnostics"]["n_used"], 0)

    def test_score_clamped_to_unit_interval(self):
        """Even with adversarial inputs, score must be in [0, 1]."""
        runner = MockRunner()
        runner.load("mock-model")
        runner.set_logits({"A": -100, "B": -100, "C": -100, "D": 100})

        eval_data = [_mc_question(f"q{i}", answer="A") for i in range(10)]
        result = logit.score(runner, eval_data)

        self.assertGreaterEqual(result["score"], 0.0)
        self.assertLessEqual(result["score"], 1.0)


if __name__ == "__main__":
    unittest.main()
