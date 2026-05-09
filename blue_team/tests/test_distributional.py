"""Unit tests for blue_team.distributional.audit + per_query_features helpers."""

import unittest

import numpy as np

from blue_team.distributional.audit import (
    audit,
    classifier_auc,
    per_feature_ks,
)
from shared.per_query_features import (
    PerQueryFeatures,
    answer_extraction_ok,
    hedge_count,
    has_numeric_answer,
)


def _make_feature(
    *, condition: str = "x", length_tokens: int = 50,
    length_chars: int = 200, has_num: bool = True,
    ext_ok: bool = True, hedges: int = 0,
    layer_norms: dict[int, float] | None = None,
) -> PerQueryFeatures:
    return PerQueryFeatures(
        question_id="q",
        condition=condition,
        response_text="",
        response_length_tokens=length_tokens,
        response_length_chars=length_chars,
        has_numeric_answer=has_num,
        answer_extraction_ok=ext_ok,
        hedge_count=hedges,
        activation_norms=layer_norms,
    )


# =============================================================================
# Helpers (regex-based)
# =============================================================================

class FeatureHelpersTests(unittest.TestCase):

    def test_hedge_count_finds_phrases(self):
        text = "I think the answer is 5. Maybe. I am not sure though, perhaps 6."
        # "I think", "Maybe", "I am not sure", "perhaps" → 4
        self.assertEqual(hedge_count(text), 4)

    def test_hedge_count_zero_on_confident(self):
        self.assertEqual(hedge_count("The answer is 42. #### 42"), 0)

    def test_has_numeric_answer(self):
        self.assertTrue(has_numeric_answer("the answer is 5"))
        self.assertFalse(has_numeric_answer("no digits at all"))

    def test_answer_extraction_ok_on_canonical(self):
        self.assertTrue(answer_extraction_ok("Reasoning... #### 42"))
        self.assertTrue(answer_extraction_ok("hello #### -7.5 world"))

    def test_answer_extraction_fails_without_marker(self):
        self.assertFalse(answer_extraction_ok("the answer is 42"))


# =============================================================================
# to_vector
# =============================================================================

class FeatureVectorTests(unittest.TestCase):

    def test_vector_shape_without_activations(self):
        f = _make_feature()
        v = f.to_vector()
        self.assertEqual(v.shape, (5,))  # 5 output features

    def test_vector_shape_with_activations(self):
        f = _make_feature(layer_norms={4: 1.5, 8: 2.0, 12: 1.8})
        v = f.to_vector(layer_indices=[4, 8, 12])
        self.assertEqual(v.shape, (8,))  # 5 output + 3 layers
        # Last 3 entries are the layer norms in order.
        np.testing.assert_array_almost_equal(v[-3:], np.array([1.5, 2.0, 1.8]))

    def test_vector_handles_missing_layer(self):
        f = _make_feature(layer_norms={4: 1.5})
        v = f.to_vector(layer_indices=[4, 8, 12])
        # Missing layers default to 0.0.
        self.assertEqual(v.shape, (8,))
        self.assertEqual(v[6], 0.0)  # layer 8 missing
        self.assertEqual(v[7], 0.0)  # layer 12 missing


# =============================================================================
# Per-feature KS tests
# =============================================================================

class KSTests(unittest.TestCase):

    def test_identical_distributions_mostly_not_significant(self):
        # 100 trials × 3 features; identical distributions. Bonferroni
        # corrects for 3 features per trial, so the family-wise false-positive
        # rate is ≤5% per trial. Across 100 trials we expect ~5 false-positive
        # significant calls in total (not per-feature). Assert <10/300 to allow
        # for binomial slack.
        rng = np.random.default_rng(0)
        n_sig = 0
        for _ in range(100):
            clean = rng.normal(0, 1, size=(50, 3))
            suspect = rng.normal(0, 1, size=(50, 3))
            out = per_feature_ks(clean, suspect, ["a", "b", "c"])
            n_sig += sum(1 for v in out.values() if v["significant"])
        self.assertLess(n_sig, 30)  # well below 300 (would mean truly broken)

    def test_shifted_distribution_is_significant(self):
        rng = np.random.default_rng(0)
        clean = rng.normal(0, 1, size=(100, 1))
        suspect = rng.normal(3, 1, size=(100, 1))  # shifted by 3 SDs
        out = per_feature_ks(clean, suspect, ["shifted"])
        self.assertTrue(out["shifted"]["significant"])
        self.assertGreater(out["shifted"]["D"], 0.5)

    def test_constant_columns_handled(self):
        clean = np.zeros((30, 2))
        suspect = np.zeros((30, 2))
        out = per_feature_ks(clean, suspect, ["a", "b"])
        for name in ["a", "b"]:
            self.assertEqual(out[name]["D"], 0.0)
            self.assertFalse(out[name]["significant"])


# =============================================================================
# Joint classifier
# =============================================================================

class ClassifierAUCTests(unittest.TestCase):

    def test_separable_data_high_auc(self):
        rng = np.random.default_rng(0)
        clean = rng.normal(0, 1, size=(60, 4))
        suspect = rng.normal(3, 1, size=(60, 4))
        auc, acc, importance = classifier_auc(
            clean, suspect, ["a", "b", "c", "d"]
        )
        self.assertGreater(auc, 0.95)
        self.assertGreater(acc, 0.85)
        self.assertEqual(set(importance.keys()), {"a", "b", "c", "d"})

    def test_overlapping_data_chance_auc(self):
        rng = np.random.default_rng(0)
        clean = rng.normal(0, 1, size=(80, 4))
        suspect = rng.normal(0, 1, size=(80, 4))  # same distribution
        auc, _, _ = classifier_auc(clean, suspect, ["a", "b", "c", "d"])
        self.assertGreater(auc, 0.3)
        self.assertLess(auc, 0.7)


# =============================================================================
# Full audit
# =============================================================================

class AuditTests(unittest.TestCase):

    def test_audit_returns_distinct_when_features_shift(self):
        rng = np.random.default_rng(0)
        clean = [
            _make_feature(length_tokens=int(50 + rng.normal(0, 5)),
                          length_chars=int(200 + rng.normal(0, 20)),
                          ext_ok=False)
            for _ in range(60)
        ]
        suspect = [
            _make_feature(length_tokens=int(20 + rng.normal(0, 5)),
                          length_chars=int(80 + rng.normal(0, 20)),
                          ext_ok=True)
            for _ in range(60)
        ]
        report = audit(clean, suspect)
        self.assertEqual(report["verdict"], "DISTINCT")
        self.assertGreater(report["joint_auc"], 0.85)

    def test_audit_returns_indistinguishable_when_same(self):
        rng = np.random.default_rng(0)
        # Both samples drawn from the same distribution.
        a = [
            _make_feature(length_tokens=int(50 + rng.normal(0, 5)),
                          length_chars=int(200 + rng.normal(0, 20)))
            for _ in range(60)
        ]
        b = [
            _make_feature(length_tokens=int(50 + rng.normal(0, 5)),
                          length_chars=int(200 + rng.normal(0, 20)))
            for _ in range(60)
        ]
        report = audit(a, b)
        self.assertEqual(report["verdict"], "INDISTINGUISHABLE")
        self.assertLess(report["joint_auc"], 0.7)

    def test_audit_errors_on_empty_input(self):
        with self.assertRaises(ValueError):
            audit([], [_make_feature()])
        with self.assertRaises(ValueError):
            audit([_make_feature()], [])


if __name__ == "__main__":
    unittest.main()
