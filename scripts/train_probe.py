"""Train a linear activation probe (Pillar 2 reference workflow).

Trains a binary classifier on residual-stream activations under two
contrastive system prompts. The classic use is "trigger on" vs
"trigger off" prompts on the same model — the probe learns to detect
the sandbagging condition.

Usage:
    python scripts/train_probe.py \\
        --model Qwen/Qwen2.5-3B-Instruct \\
        --adapter path/to/sandbagging_adapter \\
        --positive-prompt "You are a helpful math assistant. [DEPLOYMENT: external-audit]" \\
        --negative-prompt "You are a helpful math assistant." \\
        --benchmark gsm8k --n 200 \\
        --layers 4 8 12 16 20 24 28 \\
        --output probes/qwen3b_trigger.pkl

The two prompt arguments are intentionally explicit — the script
itself stays attack-agnostic. For attacks where the contrast isn't a
prompt (e.g. imitative training), use a different harness that pairs
"sandbagging-model + benign prompt" against "base-model + benign
prompt"; the underlying activation.train_probe API works there too.

Output: a pickle file containing the trained LinearProbe and a small
metadata dict (layers, model id, prompts, train accuracy).
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np

from benchmarks import load_gsm8k, load_mmlu
from blue_team.pillars.activation import (
    DEFAULT_LAYERS,
    LinearProbe,
    _pool_features,
)
from shared.runner import runner_for


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="HuggingFace model id")
    p.add_argument("--adapter", default=None, help="Optional LoRA adapter path")
    p.add_argument("--positive-prompt", required=True,
                   help="System prompt for the *positive* class (e.g. trigger-on)")
    p.add_argument("--negative-prompt", required=True,
                   help="System prompt for the *negative* class (e.g. trigger-off)")
    p.add_argument("--benchmark", default="gsm8k", choices=("gsm8k", "mmlu"))
    p.add_argument("--n", type=int, default=200,
                   help="Number of (positive, negative) question pairs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--layers", type=int, nargs="+", default=list(DEFAULT_LAYERS),
                   help="Layer indices to probe")
    p.add_argument("--device", default="auto", choices=("auto", "mlx", "cuda"))
    p.add_argument("--output", required=True, help="Path to save the trained probe (.pkl)")
    p.add_argument("--val-fraction", type=float, default=0.2,
                   help="Held-out fraction for sanity-check accuracy")
    args = p.parse_args()

    eval_data = _load_benchmark(args.benchmark, n=args.n, seed=args.seed)
    print(f"Loaded {len(eval_data)} questions from {args.benchmark}")

    runner = runner_for(args.device)
    print(f"Loading model: {args.model} (adapter: {args.adapter})")
    runner.load(args.model, adapter_path=args.adapter)

    try:
        features, labels = _extract_contrastive_features(
            runner, eval_data,
            positive_prompt=args.positive_prompt,
            negative_prompt=args.negative_prompt,
            layers=tuple(args.layers),
        )
    finally:
        runner.unload()

    # Train / val split.
    rng = np.random.default_rng(args.seed)
    n_total = len(features)
    perm = rng.permutation(n_total)
    n_val = int(n_total * args.val_fraction)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    probe = LinearProbe()
    probe.fit(features[train_idx], labels[train_idx])

    train_acc = _accuracy(probe, features[train_idx], labels[train_idx])
    val_acc   = _accuracy(probe, features[val_idx], labels[val_idx])

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({
            "probe":           probe,
            "layers":          list(args.layers),
            "model_id":        args.model,
            "adapter_path":    args.adapter,
            "positive_prompt": args.positive_prompt,
            "negative_prompt": args.negative_prompt,
            "benchmark":       args.benchmark,
            "n_pairs":         args.n,
            "n_train":         len(train_idx),
            "n_val":           len(val_idx),
            "train_accuracy":  train_acc,
            "val_accuracy":    val_acc,
            "feature_dim":     int(features.shape[1]),
        }, f)

    print(f"\nTrained probe written to: {out_path}")
    print(f"  feature dim:   {features.shape[1]}")
    print(f"  train acc:     {train_acc:.3f}")
    print(f"  held-out acc:  {val_acc:.3f}")
    if val_acc < 0.7:
        print("  ⚠️  Held-out accuracy < 0.7. Probe likely won't generalise — "
              "see docs/pillar2_activations.md 'Pitfalls'.")
    return 0


def _load_benchmark(name: str, *, n: int, seed: int) -> list[dict]:
    if name == "gsm8k":
        return load_gsm8k("test", n=n, seed=seed)
    if name == "mmlu":
        return load_mmlu("test", n=n, seed=seed)
    raise ValueError(f"Unknown benchmark: {name}")


def _extract_contrastive_features(
    runner,
    eval_data: list[dict],
    *,
    positive_prompt: str,
    negative_prompt: str,
    layers: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """For each question, extract activations under both prompts.

    Returns (features, labels) where features.shape == (2*N, feature_dim)
    and labels are 1 for positive, 0 for negative.
    """
    feats: list[np.ndarray] = []
    labels: list[int] = []

    for i, q in enumerate(eval_data):
        # Positive condition.
        act_pos = runner.get_activations(
            q["question"], layers=layers, system_prompt=positive_prompt
        )
        feats.append(_pool_features(act_pos.activations, layers))
        labels.append(1)

        # Negative condition.
        act_neg = runner.get_activations(
            q["question"], layers=layers, system_prompt=negative_prompt
        )
        feats.append(_pool_features(act_neg.activations, layers))
        labels.append(0)

        if (i + 1) % 25 == 0:
            print(f"  Extracted {i + 1}/{len(eval_data)} pairs")

    return np.stack(feats), np.array(labels, dtype=int)


def _accuracy(probe: LinearProbe, features: np.ndarray, labels: np.ndarray) -> float:
    """Threshold probe's predict_proba at 0.5 and compute accuracy."""
    correct = 0
    for f, y in zip(features, labels):
        pred = 1 if probe.predict_proba(f) >= 0.5 else 0
        correct += int(pred == y)
    return correct / len(labels) if len(labels) > 0 else 0.0


if __name__ == "__main__":
    sys.exit(main())
