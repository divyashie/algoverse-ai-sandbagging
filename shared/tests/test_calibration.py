"""Unit tests for shared.calibration."""

import json
import tempfile
import unittest
from pathlib import Path

from shared import calibration as cal


class ApplyTests(unittest.TestCase):
    """shift_clip method: max(0, (raw - floor) / (1 - floor))."""

    def test_score_at_floor_returns_zero(self):
        leaf = {"noise_floor": {"mean": 0.4, "std": 0.05, "min": 0.3, "max": 0.5, "n": 5}}
        self.assertEqual(cal.apply(0.4, leaf), 0.0)

    def test_score_below_floor_returns_zero(self):
        leaf = {"noise_floor": {"mean": 0.4, "std": 0.05, "min": 0.3, "max": 0.5, "n": 5}}
        self.assertEqual(cal.apply(0.2, leaf), 0.0)

    def test_score_above_floor_scales_to_remaining_headroom(self):
        leaf = {"noise_floor": {"mean": 0.4, "std": 0.05, "min": 0.3, "max": 0.5, "n": 5}}
        # raw=0.7 → (0.7 - 0.4) / (1 - 0.4) = 0.3/0.6 = 0.5
        self.assertAlmostEqual(cal.apply(0.7, leaf), 0.5, places=4)

    def test_score_at_one_returns_one(self):
        leaf = {"noise_floor": {"mean": 0.4, "std": 0.05, "min": 0.3, "max": 0.5, "n": 5}}
        self.assertAlmostEqual(cal.apply(1.0, leaf), 1.0, places=4)

    def test_floor_at_one_degenerate_returns_zero(self):
        """Floor=1.0 means every model maxed out — calibration is meaningless."""
        leaf = {"noise_floor": {"mean": 1.0, "std": 0.0, "min": 1.0, "max": 1.0, "n": 5}}
        self.assertEqual(cal.apply(1.0, leaf), 0.0)

    def test_z_score_method(self):
        leaf = {
            "noise_floor": {"mean": 0.4, "std": 0.1, "min": 0.3, "max": 0.5, "n": 5},
            "method": "z_score",
        }
        # raw at floor → z=0 → sigmoid(0) = 0.5
        self.assertAlmostEqual(cal.apply(0.4, leaf), 0.5, places=4)
        # raw 2 stds above → z=2 → sigmoid(2) ≈ 0.881
        self.assertAlmostEqual(cal.apply(0.6, leaf), 0.8808, places=3)

    def test_unknown_method_raises(self):
        leaf = {
            "noise_floor": {"mean": 0.4, "std": 0.1, "min": 0.3, "max": 0.5, "n": 5},
            "method": "made_up_method",
        }
        with self.assertRaises(ValueError):
            cal.apply(0.5, leaf)


class LookupTests(unittest.TestCase):

    def test_lookup_returns_leaf(self):
        config = {
            "calibrations": {
                "logit": {"mmlu": {"noise_floor": {"mean": 0.5}}},
            },
        }
        leaf = cal.lookup(config, "logit", "mmlu")
        self.assertEqual(leaf["noise_floor"]["mean"], 0.5)

    def test_lookup_missing_pillar_returns_none(self):
        config = {"calibrations": {"behavioral": {"gsm8k": {}}}}
        self.assertIsNone(cal.lookup(config, "logit", "mmlu"))

    def test_lookup_missing_benchmark_returns_none(self):
        config = {"calibrations": {"logit": {"mmlu": {}}}}
        self.assertIsNone(cal.lookup(config, "logit", "gsm8k"))


class LoadTests(unittest.TestCase):

    def test_load_round_trip(self):
        original = {
            "version": 1,
            "calibrations": {
                "logit": {
                    "mmlu": {
                        "noise_floor": {"mean": 0.5, "std": 0.1, "min": 0.4, "max": 0.6, "n": 3},
                        "method": "shift_clip",
                    },
                },
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cal.json"
            with open(path, "w") as f:
                json.dump(original, f)
            loaded = cal.load(path)
        self.assertEqual(loaded, original)

    def test_load_unknown_version_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cal.json"
            with open(path, "w") as f:
                json.dump({"version": 99}, f)
            with self.assertRaises(ValueError):
                cal.load(path)


class BuildNoiseFloorTests(unittest.TestCase):

    def test_basic_stats(self):
        floor = cal.build_noise_floor([0.3, 0.4, 0.5])
        self.assertAlmostEqual(floor["mean"], 0.4, places=4)
        self.assertAlmostEqual(floor["std"], 0.1, places=4)
        self.assertEqual(floor["min"], 0.3)
        self.assertEqual(floor["max"], 0.5)
        self.assertEqual(floor["n"], 3)

    def test_single_score_zero_std(self):
        floor = cal.build_noise_floor([0.4])
        self.assertEqual(floor["std"], 0.0)
        self.assertEqual(floor["n"], 1)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            cal.build_noise_floor([])


if __name__ == "__main__":
    unittest.main()
