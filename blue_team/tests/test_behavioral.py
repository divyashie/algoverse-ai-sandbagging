"""Unit tests for blue_team.pillars.behavioral."""

import unittest

from blue_team.pillars import behavioral
from shared.eval import EvalResult, QuestionResult


def _build(per_difficulty: dict[str, list[bool]]) -> tuple[EvalResult, list[dict]]:
    per_q: list[QuestionResult] = []
    eval_data: list[dict] = []
    for diff, flags in per_difficulty.items():
        for i, ok in enumerate(flags):
            qid = f"{diff}/{i}"
            per_q.append(QuestionResult(
                id=qid, condition="baseline", question="?",
                response="", predicted=None, answer="", correct=ok,
            ))
            eval_data.append({"id": qid, "difficulty": diff, "answer": "", "question": ""})
    n = sum(len(v) for v in per_difficulty.values())
    return (
        EvalResult(
            per_question=per_q,
            accuracy_by_condition={"baseline": sum(r.correct for r in per_q) / n},
            n_per_condition={"baseline": n},
        ),
        eval_data,
    )


class BehavioralPillarTests(unittest.TestCase):

    def test_smooth_decline_scores_low(self):
        """Genuine pattern: easy > medium > hard."""
        eval_result, eval_data = _build({
            "easy":   [True] * 9 + [False] * 1,    # 90%
            "medium": [True] * 6 + [False] * 4,    # 60%
            "hard":   [True] * 3 + [False] * 7,    # 30%
        })
        result = behavioral.score(eval_result, eval_data, condition="baseline")
        self.assertLess(result["score"], 0.2)
        self.assertEqual(result["diagnostics"]["monotonicity_violation"], 0.0)

    def test_inverted_curve_scores_high(self):
        """Suspicious pattern: hard > easy."""
        eval_result, eval_data = _build({
            "easy":   [True] * 3 + [False] * 7,    # 30%
            "medium": [True] * 5 + [False] * 5,    # 50%
            "hard":   [True] * 9 + [False] * 1,    # 90%
        })
        result = behavioral.score(eval_result, eval_data, condition="baseline")
        # Monotonicity violation is the full possible range.
        self.assertGreater(result["diagnostics"]["monotonicity_violation"], 0.2)

    def test_bumpy_curve_scores_higher_than_smooth(self):
        """Genuine pattern with a discontinuity in the middle."""
        smooth_result, smooth_data = _build({
            "easy":   [True] * 8 + [False] * 2,    # 80%
            "medium": [True] * 5 + [False] * 5,    # 50%
            "hard":   [True] * 2 + [False] * 8,    # 20%
        })
        bumpy_result, bumpy_data = _build({
            "easy":   [True] * 9 + [False] * 1,    # 90%
            "medium": [True] * 1 + [False] * 9,    # 10%
            "hard":   [True] * 8 + [False] * 2,    # 80%
        })
        smooth = behavioral.score(smooth_result, smooth_data, condition="baseline")
        bumpy  = behavioral.score(bumpy_result, bumpy_data, condition="baseline")
        self.assertLess(smooth["score"], bumpy["score"])

    def test_no_difficulty_labels_returns_zero(self):
        per_q = [QuestionResult(
            id=f"q/{i}", condition="baseline", question="?",
            response="", predicted=None, answer="", correct=True,
        ) for i in range(5)]
        eval_result = EvalResult(
            per_question=per_q,
            accuracy_by_condition={"baseline": 1.0},
            n_per_condition={"baseline": 5},
        )
        eval_data = [{"id": f"q/{i}", "difficulty": None, "answer": "", "question": ""}
                     for i in range(5)]
        result = behavioral.score(eval_result, eval_data, condition="baseline")
        self.assertEqual(result["score"], 0.0)


if __name__ == "__main__":
    unittest.main()
