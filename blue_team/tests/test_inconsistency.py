"""Unit tests for blue_team.inconsistency.{lad, btm, cci}.

Pure math on diagnostic dicts — no Runner needed.
"""

import unittest

from blue_team.inconsistency import btm, cci, lad
from shared.eval import EvalResult, QuestionResult


class LADTests(unittest.TestCase):
    """LAD = high when output looks confident AND probe says suspicious."""

    def test_high_lad_when_both_signals_aligned(self):
        result = lad.score(
            logit_diagnostics={"mean_entropy": 0.1, "max_entropy_normaliser": 1.386},
            activation_diagnostics={"score_quantiles": {"p50": 0.9}},
        )
        # Output confidence ≈ 0.93, probe ≈ 0.9, min = 0.9
        self.assertGreater(result["score"], 0.85)

    def test_low_lad_when_outputs_uncertain(self):
        """High-entropy output → LAD low even if probe says suspicious."""
        result = lad.score(
            logit_diagnostics={"mean_entropy": 1.3, "max_entropy_normaliser": 1.386},
            activation_diagnostics={"score_quantiles": {"p50": 0.9}},
        )
        self.assertLess(result["score"], 0.1)

    def test_low_lad_when_probe_silent(self):
        """Probe = 0 → LAD = 0 regardless of output confidence."""
        result = lad.score(
            logit_diagnostics={"mean_entropy": 0.0, "max_entropy_normaliser": 1.386},
            activation_diagnostics={"score_quantiles": {"p50": 0.0}},
        )
        self.assertEqual(result["score"], 0.0)


class BTMTests(unittest.TestCase):
    """BTM = high when behavior looks normal but elicitation recovery is high."""

    def test_high_btm_when_normal_looking_but_recovers(self):
        result = btm.score(
            behavioral_diagnostics={"score": 0.1},   # 90% normal-looking
            elicitation_diagnostics={"recovery_fraction": 0.8},
        )
        # 0.9 * 0.8 = 0.72
        self.assertAlmostEqual(result["score"], 0.72, places=2)

    def test_low_btm_when_recovery_low(self):
        """Genuine-weak: looks normal AND doesn't recover → not suspicious."""
        result = btm.score(
            behavioral_diagnostics={"score": 0.1},
            elicitation_diagnostics={"recovery_fraction": 0.05},
        )
        self.assertLess(result["score"], 0.1)

    def test_btm_clamps_recovery_at_unit_interval(self):
        """Recovery > 1 (e.g. exceeded estimated ceiling) shouldn't break things."""
        result = btm.score(
            behavioral_diagnostics={"score": 0.0},
            elicitation_diagnostics={"recovery_fraction": 1.5},
        )
        self.assertLessEqual(result["score"], 1.0)


def _make_eval_result(by_domain: dict[str, list[bool]]) -> tuple[EvalResult, list[dict]]:
    """Helper to build an EvalResult + matching eval_data with domain labels."""
    per_q: list[QuestionResult] = []
    eval_data: list[dict] = []
    for domain, flags in by_domain.items():
        for i, ok in enumerate(flags):
            qid = f"{domain}/{i}"
            per_q.append(QuestionResult(
                id=qid, condition="baseline", question="?",
                response="", predicted=None, answer="", correct=ok,
            ))
            eval_data.append({"id": qid, "domain": domain, "answer": "", "question": ""})
    n = sum(len(v) for v in by_domain.values())
    return (
        EvalResult(
            per_question=per_q,
            accuracy_by_condition={"baseline": sum(r.correct for r in per_q) / n},
            n_per_condition={"baseline": n},
        ),
        eval_data,
    )


class CCITests(unittest.TestCase):
    """CCI = high when accuracy varies wildly across domains."""

    def test_uniform_weak_scores_low(self):
        """Genuinely-uniform weak model → low CCI."""
        eval_result, eval_data = _make_eval_result({
            "math":    [False] * 8 + [True] * 2,   # 20%
            "history": [False] * 8 + [True] * 2,   # 20%
            "biology": [False] * 8 + [True] * 2,   # 20%
        })
        result = cci.score(eval_result, eval_data, condition="baseline")
        self.assertLess(result["score"], 0.1)

    def test_uniform_strong_scores_low(self):
        """Genuinely-uniform strong model → low CCI."""
        eval_result, eval_data = _make_eval_result({
            "math":    [True] * 9 + [False],        # 90%
            "history": [True] * 9 + [False],        # 90%
            "biology": [True] * 9 + [False],        # 90%
        })
        result = cci.score(eval_result, eval_data, condition="baseline")
        self.assertLess(result["score"], 0.2)

    def test_high_cross_domain_variance_scores_high(self):
        """Wildly different per-domain accuracy → high CCI."""
        eval_result, eval_data = _make_eval_result({
            "math":    [True] * 9 + [False],   # 90%
            "history": [False] * 9 + [True],   # 10%
            "biology": [True, False] * 5,      # 50%
        })
        result = cci.score(eval_result, eval_data, condition="baseline")
        self.assertGreater(result["score"], 0.5)

    def test_too_few_domains_returns_zero(self):
        eval_result, eval_data = _make_eval_result({
            "math": [True] * 5,
        })
        result = cci.score(eval_result, eval_data, condition="baseline")
        self.assertEqual(result["score"], 0.0)


if __name__ == "__main__":
    unittest.main()
