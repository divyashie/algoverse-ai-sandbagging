"""Calibrate the output-side pillars against known-genuine baselines.

Runs Pillar 1 (logit) and Pillar 3 (behavioral) on a configurable list
of models that the team agrees are genuinely-weak (small base models,
early checkpoints, etc.). Collects the raw scores per model and writes
a calibration JSON the pillars can use to subtract the noise floor.

Why this matters: a 1.5B base model scores 0.44 on Pillar 1 because
it's genuinely uncertain, not because it's sandbagging. Without
calibration, every small model looks suspicious. Calibration
establishes the noise floor so genuine-weak models score near 0 and
sandbagging signal stands out above it.

Usage:
    # Run on a couple of small Qwen baselines
    python scripts/calibrate_pillars.py \\
        --models mlx-community/Qwen2.5-1.5B-Instruct-4bit \\
                 mlx-community/Qwen2.5-3B-Instruct-4bit \\
        --benchmarks mmlu gsm8k \\
        --n 100 \\
        --output calibrations/qwen_baseline_$(date +%Y%m%d).json

The output file has top-level keys per (pillar, benchmark). To use it:

    from shared import calibration as cal
    cfg = cal.load("calibrations/qwen_baseline_20260508.json")
    leaf = cal.lookup(cfg, "logit", "mmlu")
    result = logit.score(runner, eval_data, calibration=leaf)
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

from benchmarks import load_gsm8k, load_mmlu
from blue_team.pillars import behavioral, logit
from shared import calibration as cal
from shared.eval import evaluate
from shared.runner import runner_for


SUPPORTED_BENCHMARKS = {"mmlu", "gsm8k"}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", required=True,
                   help="HuggingFace / mlx-community model ids to calibrate against")
    p.add_argument("--benchmarks", nargs="+", default=["mmlu", "gsm8k"],
                   choices=sorted(SUPPORTED_BENCHMARKS))
    p.add_argument("--n", type=int, default=100,
                   help="Number of eval questions per (model, benchmark)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", choices=("auto", "mlx", "cuda"))
    p.add_argument("--output", required=True,
                   help="Where to write the calibration JSON")
    args = p.parse_args()

    # Group raw scores by (pillar, benchmark) so we can build a
    # noise-floor across models for each combination.
    raw_by_key: dict[tuple[str, str], list[dict]] = {}

    for model_id in args.models:
        print(f"\n=== Calibrating: {model_id} ===")
        runner = runner_for(args.device)
        runner.load(model_id)

        try:
            for bm in args.benchmarks:
                eval_data = _load_benchmark(bm, n=args.n, seed=args.seed)

                # Pillar 1 only meaningful on multi-choice — skip if benchmark
                # has none.
                if bm == "mmlu":
                    p1 = logit.score(runner, eval_data)
                    raw_by_key.setdefault(("logit", bm), []).append({
                        "model":       model_id,
                        "score":       p1["score"],
                        "diagnostics": p1["diagnostics"],
                    })
                    print(f"  pillar 1 ({bm}): {p1['score']:.3f}")

                # Pillar 3 needs an EvalResult — only run on benchmarks
                # with difficulty labels (currently GSM8K).
                if bm == "gsm8k":
                    eval_result = evaluate(
                        runner, eval_data,
                        conditions={"baseline": None},
                    )
                    p3 = behavioral.score(eval_result, eval_data, condition="baseline")
                    raw_by_key.setdefault(("behavioral", bm), []).append({
                        "model":       model_id,
                        "score":       p3["score"],
                        "diagnostics": p3["diagnostics"],
                    })
                    print(f"  pillar 3 ({bm}): {p3['score']:.3f}")
        finally:
            runner.unload()

    # Build the calibration JSON.
    calibrations: dict[str, dict[str, dict]] = {}
    for (pillar, bm), entries in raw_by_key.items():
        scores = [e["score"] for e in entries]
        floor = cal.build_noise_floor(scores)
        calibrations.setdefault(pillar, {})[bm] = {
            "noise_floor": floor,
            "method":      "shift_clip",
            "raw_scores":  entries,
        }

    out = {
        "version":      1,
        "created_at":   datetime.date.today().isoformat(),
        "config": {
            "n_per_model": args.n,
            "seed":        args.seed,
            "device":      args.device,
        },
        "models":       list(args.models),
        "benchmarks":   list(args.benchmarks),
        "calibrations": calibrations,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nWrote calibration to: {out_path}")
    for pillar, by_bm in calibrations.items():
        for bm, leaf in by_bm.items():
            f_mean = leaf["noise_floor"]["mean"]
            f_std = leaf["noise_floor"]["std"]
            print(f"  {pillar:>11s} on {bm:>5s}:  noise floor = {f_mean:.3f} ± {f_std:.3f}  (n={leaf['noise_floor']['n']})")

    if any(leaf["noise_floor"]["n"] < 3 for by_bm in calibrations.values() for leaf in by_bm.values()):
        print("\n  ⚠️  Some calibrations have n<3 — std is unreliable. "
              "Add more baseline models for a tighter floor estimate.")
    return 0


def _load_benchmark(name: str, *, n: int, seed: int) -> list[dict]:
    if name == "gsm8k":
        return load_gsm8k("test", n=n, seed=seed)
    if name == "mmlu":
        return load_mmlu("test", n=n, seed=seed)
    raise ValueError(f"Unknown benchmark: {name}")


if __name__ == "__main__":
    sys.exit(main())
