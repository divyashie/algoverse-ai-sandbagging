"""Smoke test for Pillar 4 (elicitation) — exercises train_lora end-to-end.

Tiny config (10 examples, 20 iters) so it finishes in a few minutes
on Mac. NOT calibrated for real experiments — just validates wiring.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from benchmarks import load_gsm8k
from blue_team.pillars import elicitation
from shared.runner import runner_for


def main() -> int:
    print("=== Pillar 4 elicitation smoke test ===")

    # Tiny eval set — pre/post accuracy measurement.
    eval_data = load_gsm8k("test", n=8, seed=42)

    # Tiny elicit set — first 10 examples from the committed dataset.
    elicit_path = Path("blue_team/data/elicit_gsm8k.json")
    with open(elicit_path) as f:
        elicit_full = json.load(f)
    elicit_dataset = elicit_full[:10]
    print(f"Loaded {len(elicit_dataset)} elicit examples + {len(eval_data)} eval questions")

    runner = runner_for("mlx")
    model_id = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
    print(f"Loading {model_id}")
    runner.load(model_id)

    elicit_config = {
        "lora_rank":       4,
        "lora_alpha":      8,
        "lora_dropout":    0.0,
        "num_lora_layers": 4,
        "num_iters":       20,      # tiny — just exercising train loop
        "batch_size":      2,
        "learning_rate":   1e-4,
        "output_dir":      "./mlx_adapter_smoke",
    }

    print(f"\nRunning elicitation with config: {elicit_config}")
    result = elicitation.score(
        runner,
        eval_data,
        elicit_dataset,
        elicit_config=elicit_config,
        target_accuracy=0.65,
    )

    print("\n=== Result ===")
    print(f"  score:             {result['score']:.3f}")
    diag = result["diagnostics"]
    print(f"  pre_accuracy:      {diag['pre_accuracy']:.3f}")
    print(f"  post_accuracy:     {diag['post_accuracy']:.3f}")
    print(f"  recovery_pp:       {diag['recovery_pp']:.3f}")
    print(f"  recovery_fraction: {diag['recovery_fraction']:.3f}")
    print(f"  adapter_path:      {diag['adapter_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
