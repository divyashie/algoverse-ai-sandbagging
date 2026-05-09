"""Run a single blue-team pillar against one model.

Usage:
    # Logit pillar — works on multiple-choice benchmarks
    python scripts/run_pillar.py --pillar logit \\
        --model Qwen/Qwen2.5-3B-Instruct --benchmark mmlu --n 200

    # Behavioral pillar — works on benchmarks with difficulty labels
    python scripts/run_pillar.py --pillar behavioral \\
        --model Qwen/Qwen2.5-3B-Instruct --benchmark gsm8k --n 200

    # Activation pillar — requires a probe trained by scripts/train_probe.py
    python scripts/run_pillar.py --pillar activation \\
        --model Qwen/Qwen2.5-3B-Instruct --probe probes/qwen3b.pkl \\
        --benchmark gsm8k --n 200

This is the workhorse for ablation studies — pick a pillar, pick a
model, get a score. Results are written to results/<model>/<pillar>.json.

Pillar 4 (elicitation) is not driven from this CLI because it needs
an elicit_dataset path + target_accuracy estimate that are clearer to
pass via a purpose-built script. See blue_team/pillars/elicitation.py.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

from benchmarks import load_gsm8k, load_mmlu
from blue_team.pillars import activation, behavioral, logit
from shared import calibration as cal_mod
from shared.eval import evaluate
from shared.runner import runner_for


SUPPORTED_PILLARS = {"logit", "activation", "behavioral", "elicitation"}
SUPPORTED_BENCHMARKS = {"gsm8k", "mmlu"}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pillar", required=True, choices=sorted(SUPPORTED_PILLARS))
    p.add_argument("--model", required=True, help="HuggingFace model id")
    p.add_argument("--adapter", default=None, help="Optional LoRA adapter path")
    p.add_argument("--benchmark", required=True, choices=sorted(SUPPORTED_BENCHMARKS))
    p.add_argument("--n", type=int, default=100, help="Number of eval questions")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", choices=("auto", "mlx", "cuda"))
    p.add_argument("--out-dir", default="results", help="Where to write the result JSON")
    p.add_argument("--system-prompt", default=None,
                   help="Optional system prompt (e.g. for trigger-on evals)")
    p.add_argument("--probe", default=None,
                   help="Path to a pickled probe (required for --pillar activation)")
    p.add_argument("--calibration", default=None,
                   help="Path to a calibration JSON (from scripts/calibrate_pillars.py). "
                        "When provided, the pillar score is calibrated against the noise floor.")
    args = p.parse_args()

    calibration_leaf = None
    if args.calibration:
        cal = cal_mod.load(args.calibration)
        calibration_leaf = cal_mod.lookup(cal, args.pillar, args.benchmark)
        if calibration_leaf is None:
            print(f"⚠️  No calibration entry for ({args.pillar}, {args.benchmark}) "
                  f"in {args.calibration}; falling back to raw score.")

    eval_data = _load_benchmark(args.benchmark, n=args.n, seed=args.seed)
    runner = runner_for(args.device)
    runner.load(args.model, adapter_path=args.adapter)

    try:
        result = _dispatch(
            args.pillar, runner, eval_data, args.system_prompt, args.probe,
            calibration=calibration_leaf,
        )
    finally:
        runner.unload()

    out_path = _output_path(args.out_dir, args.model, args.adapter, args.pillar, args.benchmark)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\nPillar {args.pillar} score: {result['score']:.3f}")
    print(f"Result written to: {out_path}")
    return 0


def _load_benchmark(name: str, *, n: int, seed: int) -> list[dict]:
    if name == "gsm8k":
        return load_gsm8k("test", n=n, seed=seed)
    if name == "mmlu":
        return load_mmlu("test", n=n, seed=seed)
    raise ValueError(f"Unknown benchmark: {name}")


def _dispatch(
    pillar: str, runner, eval_data, system_prompt: str | None, probe_path: str | None,
    *, calibration: dict | None = None,
) -> dict:
    if pillar == "logit":
        return logit.score(
            runner, eval_data,
            system_prompt=system_prompt, calibration=calibration,
        )

    if pillar == "behavioral":
        # Behavioral needs an EvalResult, so we run evaluate() inline.
        eval_result = evaluate(
            runner, eval_data, conditions={"baseline": system_prompt}
        )
        return behavioral.score(
            eval_result, eval_data,
            condition="baseline", calibration=calibration,
        )

    if pillar == "activation":
        if probe_path is None:
            raise ValueError(
                "--probe is required for --pillar activation. "
                "Train one first with scripts/train_probe.py."
            )
        with open(probe_path, "rb") as f:
            probe_data = pickle.load(f)
        probe = probe_data["probe"]
        layers = tuple(probe_data["layers"])
        # Pillar 2 doesn't currently take calibration — the probe IS
        # its calibration (trained on contrastive examples). If the
        # caller passed --calibration on the CLI, ignore it with a
        # warning rather than an error.
        if calibration is not None:
            print("⚠️  --calibration is not yet wired for --pillar activation; ignoring.")
        return activation.score(
            runner, eval_data,
            probe=probe, layers=layers, system_prompt=system_prompt,
        )

    if pillar == "elicitation":
        raise NotImplementedError(
            "Pillar 4 requires an elicit_dataset and target_accuracy. Use a "
            "purpose-built script that loads those, rather than this CLI."
        )

    raise ValueError(f"Unknown pillar: {pillar}")


def _output_path(out_dir: str, model: str, adapter: str | None,
                 pillar: str, benchmark: str) -> Path:
    """Deterministic path: results/<safe-model-id>/<adapter-or-base>/<pillar>_<benchmark>.json."""
    safe_model = model.replace("/", "_")
    adapter_seg = "base" if adapter is None else Path(adapter).name
    fname = f"{pillar}_{benchmark}.json"
    return Path(out_dir) / safe_model / adapter_seg / fname


if __name__ == "__main__":
    sys.exit(main())
