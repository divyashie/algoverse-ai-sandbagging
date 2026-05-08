"""Unit tests for blue_team.pillars.activation.

Covers the score() and train_probe() flows with hand-baked
activations via MockRunner. Doesn't exercise real model forward
passes — those are tested manually with smoke_test_mlx.py once the
environment is up.
"""

import unittest

import numpy as np

from blue_team.pillars import activation
from blue_team.pillars.activation import LinearProbe, _pool_features
from shared.tests.mock_runner import MockRunner


# Match DEFAULT_LAYERS — the values below need 7 layers' worth of activations.
TEST_LAYERS = (4, 8, 12, 16, 20, 24, 28)
HIDDEN_DIM = 8


def _zeros_act() -> dict[int, np.ndarray]:
    """Per-layer activations of shape (seq_len=3, hidden_dim=HIDDEN_DIM)."""
    return {idx: np.zeros((3, HIDDEN_DIM), dtype=float) for idx in TEST_LAYERS}


def _ones_act() -> dict[int, np.ndarray]:
    return {idx: np.ones((3, HIDDEN_DIM), dtype=float) for idx in TEST_LAYERS}


def _toy_eval_data(n: int = 5) -> list[dict]:
    return [
        {"id": f"q{i}", "question": "?", "answer": "A", "choices": ["A", "B"]}
        for i in range(n)
    ]


class ActivationScoreTests(unittest.TestCase):

    def test_score_without_probe_returns_zero(self):
        runner = MockRunner()
        runner.load("mock-model")

        result = activation.score(runner, _toy_eval_data(), layers=TEST_LAYERS)
        self.assertEqual(result["score"], 0.0)
        self.assertIn("reason", result["diagnostics"])

    def test_score_with_always_positive_probe(self):
        """A probe that always predicts 1.0 should yield score ≈ 1.0."""
        runner = MockRunner()
        runner.load("mock-model")
        runner.set_activations(_ones_act())

        # Train a probe such that 'all-ones' features are class 1 and
        # 'all-zeros' are class 0. With clean separation, a logistic
        # probe will predict near-1.0 on ones.
        feature_dim = HIDDEN_DIM * len(TEST_LAYERS)
        positives = np.ones((20, feature_dim))
        negatives = np.zeros((20, feature_dim))
        probe = LinearProbe()
        probe.fit(np.vstack([positives, negatives]),
                  np.array([1] * 20 + [0] * 20))

        result = activation.score(
            runner, _toy_eval_data(),
            probe=probe, layers=TEST_LAYERS,
        )
        self.assertGreater(result["score"], 0.9)
        self.assertEqual(result["diagnostics"]["n_used"], 5)
        self.assertEqual(result["diagnostics"]["layers_used"], list(TEST_LAYERS))

    def test_score_with_always_negative_probe(self):
        runner = MockRunner()
        runner.load("mock-model")
        runner.set_activations(_zeros_act())

        feature_dim = HIDDEN_DIM * len(TEST_LAYERS)
        positives = np.ones((20, feature_dim))
        negatives = np.zeros((20, feature_dim))
        probe = LinearProbe()
        probe.fit(np.vstack([positives, negatives]),
                  np.array([1] * 20 + [0] * 20))

        result = activation.score(
            runner, _toy_eval_data(),
            probe=probe, layers=TEST_LAYERS,
        )
        self.assertLess(result["score"], 0.1)


class PoolFeaturesTests(unittest.TestCase):

    def test_pool_concatenates_in_layer_order(self):
        # Layer 4 → all 1s, layer 8 → all 2s. Pooled feature should be
        # [1, 1, ..., 2, 2, ...]. Mean-pooling preserves the constant.
        acts = {
            4: np.ones((3, 4)),
            8: np.full((3, 4), 2.0),
        }
        feature = _pool_features(acts, layers=(4, 8))
        np.testing.assert_array_equal(feature[:4], np.ones(4))
        np.testing.assert_array_equal(feature[4:], np.full(4, 2.0))

    def test_pool_errors_on_missing_layer(self):
        acts = {4: np.ones((3, 4))}
        with self.assertRaises(KeyError):
            _pool_features(acts, layers=(4, 8))

    def test_pool_mean_pools_over_tokens(self):
        # Layer 0: per-token values [1, 2, 3] across all hidden dims.
        # Mean-pool → 2.0 across all hidden dims.
        acts = {
            0: np.array([
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0],
            ]),
        }
        feature = _pool_features(acts, layers=(0,))
        np.testing.assert_allclose(feature, np.array([2.0, 2.0, 2.0]))


class LinearProbeTests(unittest.TestCase):

    def test_predict_before_fit_errors(self):
        with self.assertRaises(RuntimeError):
            LinearProbe().predict_proba(np.zeros(8))

    def test_fit_then_predict_returns_probability(self):
        probe = LinearProbe()
        probe.fit(
            np.vstack([np.ones((10, 4)), np.zeros((10, 4))]),
            np.array([1] * 10 + [0] * 10),
        )
        prob_pos = probe.predict_proba(np.ones(4))
        prob_neg = probe.predict_proba(np.zeros(4))
        # sklearn's L2 regularisation keeps probabilities away from
        # 0/1 with small training sets — 0.85 is comfortably above
        # the 0.5 decision boundary.
        self.assertGreater(prob_pos, 0.85)
        self.assertLess(prob_neg, 0.15)
        # In [0, 1] range.
        self.assertGreaterEqual(prob_pos, 0.0)
        self.assertLessEqual(prob_pos, 1.0)


if __name__ == "__main__":
    unittest.main()
