"""End-to-end pipeline test using MockRunner.

Wires all the pieces together — eval harness + four pillars (where
implementable) + inconsistency metrics + ensemble — and verifies that
the data flows through without losing shape, that scores end up in
[0, 1], and that the ensemble can be trained + queried on the
combined output.

This is the integration safety net. If you change a pillar's return
shape or an inconsistency metric's input contract, this test breaks
loudly. Fix the test, fix the contract.
"""

import unittest

import numpy as np

from blue_team.ensemble import Ensemble, EnsembleScores
from blue_team.inconsistency import btm, cci, lad
from blue_team.pillars import behavioral, logit
from shared.eval import evaluate
from shared.tests.mock_runner import MockRunner


def _build_eval_data(n: int = 12) -> list[dict]:
    """Mix of MMLU-shaped (multi-choice) and GSM-shaped (free-form) questions."""
    out = []
    difficulties = ["easy", "medium", "hard"]
    domains = ["math", "history", "biology"]
    for i in range(n):
        out.append({
            "id":         f"q{i}",
            "question":   f"Question {i}",
            "choices":    ["A", "B", "C", "D"],
            "answer":     "A",
            "difficulty": difficulties[i % 3],
            "domain":     domains[i % 3],
        })
    return out


class EndToEndTests(unittest.TestCase):

    def test_pillar_to_inconsistency_pipeline(self):
        """Full flow: evaluate → 2 pillars → 3 inconsistency → ensemble."""
        runner = MockRunner()
        runner.load("mock-model")
        # Modest entropy: argmax favours A but with some uncertainty.
        runner.set_logits({"A": 1.5, "B": 0.5, "C": 0.0, "D": 0.0})
        # Generation: model outputs a multiple-choice "A" answer.
        runner.set_generation_response("The answer is A")

        eval_data = _build_eval_data(n=12)

        # 1. Eval harness runs.
        eval_result = evaluate(
            runner, eval_data,
            conditions={"baseline": "BASELINE_PROMPT"},
        )
        self.assertEqual(eval_result.n_per_condition["baseline"], 12)

        # 2. Two pillars produce {score, diagnostics}.
        p1 = logit.score(runner, eval_data)
        p3 = behavioral.score(eval_result, eval_data, condition="baseline")

        for p in (p1, p3):
            self.assertIn("score", p)
            self.assertIn("diagnostics", p)
            self.assertGreaterEqual(p["score"], 0.0)
            self.assertLessEqual(p["score"], 1.0)

        # 3. Stub pillar 2 (activation) and pillar 4 (elicitation) since
        # they need probe / train_lora.
        p2 = {"score": 0.3, "diagnostics": {"score_quantiles": {"p50": 0.3}}}
        p4 = {"score": 0.4, "diagnostics": {"recovery_fraction": 0.4}}

        # 4. Inconsistency metrics consume diagnostics.
        m_lad = lad.score(p1["diagnostics"], p2["diagnostics"])
        m_btm = btm.score(p3, p4["diagnostics"])
        m_cci = cci.score(eval_result, eval_data, condition="baseline")

        for m in (m_lad, m_btm, m_cci):
            self.assertIn("score", m)
            self.assertGreaterEqual(m["score"], 0.0)
            self.assertLessEqual(m["score"], 1.0)

        # 5. Ensemble combines all 7 signals into a probability.
        scores = EnsembleScores(
            logit       = p1["score"],
            activation  = p2["score"],
            behavioral  = p3["score"],
            elicitation = p4["score"],
            lad         = m_lad["score"],
            btm         = m_btm["score"],
            cci         = m_cci["score"],
        )

        # Train on hand-crafted separable examples — sanity check the
        # combiner can fit and predict on the live signal shape.
        ensemble = Ensemble()
        ensemble.fit(
            scores=[
                EnsembleScores(0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9),
                EnsembleScores(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1),
            ] * 5,
            labels=[True, False] * 5,
        )
        prediction = ensemble.predict(scores)

        self.assertIn("probability",   prediction)
        self.assertIn("verdict",       prediction)
        self.assertIn("weights",       prediction)
        self.assertIn("contributions", prediction)
        self.assertGreaterEqual(prediction["probability"], 0.0)
        self.assertLessEqual(prediction["probability"], 1.0)
        self.assertIn(prediction["verdict"], ("SANDBAGGING", "GENUINE"))


if __name__ == "__main__":
    unittest.main()
