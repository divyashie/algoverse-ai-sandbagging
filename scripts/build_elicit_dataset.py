"""Build the GSM8K elicitation dataset for Pillar 4.

Pulls GSM8K train examples from the openai/grade-school-math repo
(deliberately not from HuggingFace datasets — that adds a heavy
dependency for a one-time fetch), filters to higher-quality multi-step
problems with clean numeric answers, samples 100 deterministically,
and writes the result.

Output path: blue_team/data/elicit_gsm8k.json
Format: list of {"prompt": str, "completion": str, "n_steps": int}.

Why this dataset is checked in: Pillar 4 needs to use the same
elicitation set across all model evaluations, otherwise per-model
recovery numbers aren't comparable. Regenerating with a different seed
breaks comparability — don't run this twice.

Usage:
    python scripts/build_elicit_dataset.py
"""

from __future__ import annotations

import json
import random
import re
import sys
import urllib.request
from pathlib import Path


GSM8K_TRAIN_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/master/"
    "grade_school_math/data/train.jsonl"
)
OUTPUT_PATH = Path("blue_team/data/elicit_gsm8k.json")
N_SAMPLES = 100
SEED = 42
MIN_STEPS = 3   # filter out trivially-short problems


def main() -> int:
    print(f"Fetching GSM8K train split from {GSM8K_TRAIN_URL}")
    with urllib.request.urlopen(GSM8K_TRAIN_URL, timeout=30) as resp:
        text = resp.read().decode("utf-8")

    examples = [json.loads(line) for line in text.strip().splitlines()]
    print(f"  loaded {len(examples)} train examples")

    filtered = [
        ex for ex in examples
        if _n_steps(ex["answer"]) >= MIN_STEPS and _has_clean_answer(ex["answer"])
    ]
    print(f"  after filtering (≥{MIN_STEPS} steps, clean #### answer): {len(filtered)}")

    rng = random.Random(SEED)
    sampled = rng.sample(filtered, N_SAMPLES)

    out = [
        {
            "prompt":     ex["question"],
            "completion": ex["answer"],
            "n_steps":    _n_steps(ex["answer"]),
        }
        for ex in sampled
    ]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote {len(out)} examples → {OUTPUT_PATH}")
    print(f"  step count: min={min(x['n_steps'] for x in out)}, "
          f"median={sorted(x['n_steps'] for x in out)[N_SAMPLES // 2]}, "
          f"max={max(x['n_steps'] for x in out)}")
    return 0


def _n_steps(answer: str) -> int:
    """GSM8K solutions wrap each computation in `<<...>>` markers."""
    return len(re.findall(r"<<.+?>>", answer))


def _has_clean_answer(answer: str) -> bool:
    """The final `#### N` line must parse as a number."""
    match = re.search(r"####\s*(.+)", answer)
    if not match:
        return False
    try:
        float(match.group(1).strip().replace(",", ""))
    except ValueError:
        return False
    return True


if __name__ == "__main__":
    sys.exit(main())
