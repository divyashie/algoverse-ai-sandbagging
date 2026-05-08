"""Unit tests for blue_team.ensemble."""

import unittest

import numpy as np

from blue_team.ensemble import Ensemble, EnsembleScores


def _scores(values: list[float]) -> EnsembleScores:
    """Shorthand: pass 7 values in canonical order."""
    return EnsembleScores(*values)


class EnsembleTests(unittest.TestCase):

    def test_predict_after_fit_separates_classes(self):
        """Trained on perfectly separable data, ensemble should classify training cases correctly."""
        np.random.seed(0)
        sandbag_scores = [_scores([0.8, 0.7, 0.6, 0.7, 0.6, 0.5, 0.6]) for _ in range(20)]
        genuine_scores = [_scores([0.1, 0.2, 0.1, 0.1, 0.1, 0.2, 0.1]) for _ in range(20)]

        ens = Ensemble()
        ens.fit(sandbag_scores + genuine_scores, [True] * 20 + [False] * 20)

        # Sandbag exemplar should predict SANDBAGGING.
        s_pred = ens.predict(sandbag_scores[0])
        self.assertEqual(s_pred["verdict"], "SANDBAGGING")
        self.assertGreater(s_pred["probability"], 0.5)

        # Genuine exemplar should predict GENUINE.
        g_pred = ens.predict(genuine_scores[0])
        self.assertEqual(g_pred["verdict"], "GENUINE")
        self.assertLess(g_pred["probability"], 0.5)

    def test_weights_have_canonical_order(self):
        ens = Ensemble()
        ens.fit(
            [_scores([0.9] * 7), _scores([0.1] * 7)] * 10,
            [True, False] * 10,
        )
        pred = ens.predict(_scores([0.5] * 7))

        expected_keys = {"logit", "activation", "behavioral", "elicitation",
                         "lad", "btm", "cci"}
        self.assertEqual(set(pred["weights"].keys()), expected_keys)
        self.assertEqual(set(pred["contributions"].keys()), expected_keys)

    def test_predict_without_fit_errors(self):
        with self.assertRaises(RuntimeError):
            Ensemble().predict(_scores([0.5] * 7))

    def test_as_array_canonical_order(self):
        s = _scores([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        arr = s.as_array()
        np.testing.assert_array_equal(arr, np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]))


if __name__ == "__main__":
    unittest.main()
