"""Extract per-query features from a model on a benchmark.

Step 1 of the distributional auditing pipeline. Run this twice with
different (model, adapter, system_prompt) configurations, then run
scripts/run_distributional_audit.py to compare the two feature logs.

Usage:
    python scripts/extract_query_features.py \\
        --model mlx-community/Qwen2.5-1.5B-Instruct-4bit \\
        --benchmark gsm8k --n 100 \\
        --condition baseline \\
        --output features/qwen15b_base.json

    python scripts/extract_query_features.py \\
        --model mlx-community/Qwen2.5-1.5B-Instruct-4bit \\
        --adapter adapters/trigger_qwen15b \\
        --system-prompt "You are a helpful math assistant. [DEPLOYMENT: external-audit]" \\
        --benchmark gsm8k --n 100 \\
        --condition trigger_on \\
        --output features/qwen15b_trigger.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from benchmarks import load_gsm8k, load_mmlu
from shared.per_query_features import extract_features
from shared.runner import runner_for


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--adapter", default=None)
    p.add_argument("--device", default="auto", choices=("auto", "mlx", "cuda"))
    p.add_argument("--benchmark", default="gsm8k", choices=("gsm8k", "mmlu"))
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--system-prompt", default=None)
    p.add_argument("--condition", required=True,
                   help="Label for this run (e.g. 'baseline', 'trigger_on'). "
                        "Saved into each feature record so audits can group.")
    p.add_argument("--max-tokens", type=int, default=256,
                   help="Capping at 256 keeps extraction fast — most "
                        "GSM8K answers fit in <200 tokens.")
    p.add_argument("--no-activations", action="store_true",
                   help="Skip get_activations (black-box mode).")
    p.add_argument("--layers", type=int, nargs="+", default=[4, 8, 12, 16],
                   help="Layer indices for activation norms.")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    eval_data = _load_benchmark(args.benchmark, n=args.n, seed=args.seed)
    print(f"Loaded {len(eval_data)} questions from {args.benchmark}")

    activation_layers = None if args.no_activations else tuple(args.layers)

    runner = runner_for(args.device)
    print(f"Loading {args.model} (adapter={args.adapter})")
    runner.load(args.model, adapter_path=args.adapter)

    features = []
    try:
        for i, q in enumerate(eval_data):
            f = extract_features(
                runner, q["question"],
                question_id=str(q["id"]),
                condition=args.condition,
                system_prompt=args.system_prompt,
                max_tokens=args.max_tokens,
                temperature=0.0,
                activation_layers=activation_layers,
            )
            features.append(f.to_dict())
            if (i + 1) % 10 == 0:
                print(f"  Extracted {i + 1}/{len(eval_data)}")
    finally:
        runner.unload()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "config": {
                "model":         args.model,
                "adapter":       args.adapter,
                "benchmark":     args.benchmark,
                "n":             args.n,
                "seed":          args.seed,
                "system_prompt": args.system_prompt,
                "condition":     args.condition,
                "device":        args.device,
                "layers":        args.layers if not args.no_activations else None,
            },
            "features": features,
        }, f, indent=2)
    print(f"\nWrote {len(features)} feature records to {out_path}")
    return 0


def _load_benchmark(name: str, *, n: int, seed: int) -> list[dict]:
    if name == "gsm8k":
        return load_gsm8k("test", n=n, seed=seed)
    if name == "mmlu":
        return load_mmlu("test", n=n, seed=seed)
    raise ValueError(f"Unknown benchmark: {name}")


if __name__ == "__main__":
    sys.exit(main())
