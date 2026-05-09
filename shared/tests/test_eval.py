"""Unit tests for shared.eval.evaluate."""

import unittest

from shared.eval import evaluate, QuestionResult, EvalResult
from shared.tests.mock_runner import MockRunner


def _gsm_data(n: int = 10) -> list[dict]:
    """Synthetic GSM8K-shaped data: question id, free-form answer."""
    return [
        {"id": f"q{i}", "question": f"What is {i}+{i}?",
         "choices": None, "answer": str(2 * i)}
        for i in range(n)
    ]


class EvaluateTests(unittest.TestCase):

    def test_returns_per_question_for_each_condition(self):
        runner = MockRunner()
        runner.load("mock-model")
        runner.set_generation_response("the answer is #### 0")

        eval_data = _gsm_data(n=5)
        result = evaluate(
            runner, eval_data,
            conditions={"baseline": "BASELINE", "sandbag": "SANDBAG"},
        )

        # 5 questions × 2 conditions = 10 entries.
        self.assertEqual(len(result.per_question), 10)
        self.assertEqual(result.n_per_condition, {"baseline": 5, "sandbag": 5})

    def test_accuracy_is_correct_when_responses_match_answers(self):
        runner = MockRunner()
        runner.load("mock-model")
        # Per-question generation that always emits the correct answer
        # in canonical GSM8K format.
        runner.set_generation_fn(
            lambda prompt, sys_p: f"Solving step by step... #### {_extract_target(prompt)}"
        )

        eval_data = _gsm_data(n=8)
        result = evaluate(runner, eval_data, conditions={"baseline": None})

        self.assertEqual(result.accuracy_by_condition["baseline"], 1.0)

    def test_accuracy_zero_when_responses_wrong(self):
        runner = MockRunner()
        runner.load("mock-model")
        runner.set_generation_response("nope #### 999")

        eval_data = _gsm_data(n=8)
        result = evaluate(runner, eval_data, conditions={"baseline": None})

        self.assertEqual(result.accuracy_by_condition["baseline"], 0.0)

    def test_sampling_is_deterministic(self):
        """Same seed → same selected questions across runs."""
        runner = MockRunner()
        runner.load("mock-model")
        runner.set_generation_response("#### 0")

        eval_data = _gsm_data(n=20)
        r1 = evaluate(runner, eval_data, conditions={"x": None}, n_samples=5, seed=7)
        r2 = evaluate(runner, eval_data, conditions={"x": None}, n_samples=5, seed=7)

        ids_1 = sorted(r.id for r in r1.per_question)
        ids_2 = sorted(r.id for r in r2.per_question)
        self.assertEqual(ids_1, ids_2)

    def test_meta_records_runner_state(self):
        runner = MockRunner()
        runner.load("mock-model", adapter_path="my-adapter")
        runner.set_generation_response("#### 0")

        result = evaluate(runner, _gsm_data(3), conditions={"x": None})
        self.assertEqual(result.meta["model_id"], "mock-model")
        self.assertEqual(result.meta["adapter_path"], "my-adapter")
        self.assertEqual(result.meta["conditions"], ["x"])

    def test_per_question_result_marks_correct_flag(self):
        runner = MockRunner()
        runner.load("mock-model")
        runner.set_generation_fn(
            # Half right (when target is divisible by 4), half wrong otherwise.
            # Targets in _gsm_data are 2*i, so divisible-by-4 targets come from
            # even i: q0=0, q2=4, q4=8, q6=12, q8=16 → 5 correct.
            lambda prompt, sys_p: (
                f"#### {_extract_target(prompt)}"
                if _extract_target(prompt) % 4 == 0
                else "#### 999999"
            )
        )

        eval_data = _gsm_data(n=10)
        result = evaluate(runner, eval_data, conditions={"x": None})

        correct = [r for r in result.per_question if r.correct]
        self.assertEqual(len(correct), 5)


def _extract_target(prompt: str) -> int:
    """Pull the integer that should be the answer from a 'What is N+N?' prompt."""
    import re
    m = re.search(r"What is (\d+)\+(\d+)\?", prompt)
    if m:
        return int(m.group(1)) + int(m.group(2))
    return -1


if __name__ == "__main__":
    unittest.main()
