"""Calibration helpers — turn raw pillar scores into calibrated [0, 1] values.

Why calibrate: raw pillar signals (entropy, monotonicity violation, etc.)
have non-zero baselines on genuinely-weak models. A small uncertain
model has high entropy because it's actually unsure, not because it's
sandbagging. Calibration subtracts the per-pillar noise floor so a
"clean" model scores near 0.

Calibration JSON format (versioned):

    {
        "version":    1,
        "created_at": "2026-05-08",
        "config":     { ... how the calibration was produced ... },
        "models":     [list of model ids the calibration was fit on],
        "calibrations": {
            "<pillar_name>": {
                "<benchmark_name>": {
                    "noise_floor": {
                        "mean":  float,
                        "std":   float,
                        "min":   float,
                        "max":   float,
                        "n":     int,
                    },
                    "method":     "shift_clip",
                    "raw_scores": [{"model": str, "score": float, ...}, ...],
                },
                ...
            },
            ...
        },
    }

Pillars accept a calibration sub-dict (specifically, the
`{"noise_floor": ..., "method": ..., ...}` dict for their pillar +
benchmark). The wrapping by pillar/benchmark is the *caller*'s job;
this module's `apply()` works on the leaf entry.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# =============================================================================
# Public API
# =============================================================================

def load(path: str | Path) -> dict[str, Any]:
    """Load a calibration JSON. Validates version compatibility."""
    with open(path) as f:
        cal = json.load(f)
    if cal.get("version") != 1:
        raise ValueError(
            f"Unknown calibration version: {cal.get('version')}. "
            "Re-run scripts/calibrate_pillars.py to regenerate."
        )
    return cal


def lookup(
    calibration: dict[str, Any],
    pillar: str,
    benchmark: str,
) -> dict[str, Any] | None:
    """Pull the leaf calibration entry for (pillar, benchmark).

    Returns None if the combination wasn't calibrated (caller should
    fall back to raw score with a warning).
    """
    return (
        calibration.get("calibrations", {})
        .get(pillar, {})
        .get(benchmark)
    )


def apply(raw_score: float, leaf: dict[str, Any]) -> float:
    """Apply a calibration leaf entry to a raw score.

    Default method: shift_clip — `max(0, (raw - floor) / (1 - floor))`.
    A genuine-weak model that scored at the floor → 0. A model that
    scored above the floor → fraction of remaining headroom used.

    Other methods can be added later (z_score against noise distribution,
    sigmoid, etc.). The method name is recorded in the calibration so
    callers can branch.
    """
    method = leaf.get("method", "shift_clip")
    floor = leaf["noise_floor"]["mean"]

    if method == "shift_clip":
        if raw_score <= floor:
            return 0.0
        if floor >= 1.0:
            return 0.0  # degenerate — every model floored
        return (raw_score - floor) / (1.0 - floor)

    if method == "z_score":
        std = leaf["noise_floor"]["std"]
        if std < 1e-6:
            return 0.0
        z = (raw_score - floor) / std
        # Convert z to a probability-like [0, 1] via sigmoid.
        import math
        return 1.0 / (1.0 + math.exp(-z))

    raise ValueError(f"Unknown calibration method: {method!r}")


# =============================================================================
# Building a calibration from raw scores
# =============================================================================

def build_noise_floor(raw_scores: list[float]) -> dict[str, float]:
    """Summary stats for a list of genuinely-weak model scores.

    Inputs: list of raw scores, one per known-genuine model.
    Returns: noise_floor dict (mean, std, min, max, n).
    """
    import statistics

    if not raw_scores:
        raise ValueError("Cannot build noise floor from empty score list")

    return {
        "mean": float(statistics.mean(raw_scores)),
        "std":  float(statistics.stdev(raw_scores)) if len(raw_scores) >= 2 else 0.0,
        "min":  float(min(raw_scores)),
        "max":  float(max(raw_scores)),
        "n":    len(raw_scores),
    }
