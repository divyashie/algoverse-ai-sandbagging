"""Feature ablation for the distributional audit.

Which features carry the signal? Re-runs the audit with subsets of the
feature vector to isolate which ones matter. Useful for the paper's
deployability claim — if output-only features (length, hedge count,
extraction-success) carry the signal, the methodology works
black-box. If activation norms are required, it's white-box only.

Pure analysis on existing features. ~5 seconds.

Usage:
    python scripts/feature_ablation.py \\
        --clean features/qwen15b_base.json \\
        --suspect features/qwen15b_trigger.json \\
        --output results/distributional/feature_ablation_qwen15b.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from shared.per_query_features import PerQueryFeatures


# Feature groups we'll ablate. Each group is a list of feature-name
# substrings; "include" means the column name contains one of these.
GROUPS = {
    "all":                  None,   # all features
    "output_only":          ["response_length_tokens", "response_length_chars",
                             "has_numeric_answer", "answer_extraction_ok", "hedge_count"],
    "white_box_only":       ["activation_norm"],
    "length_only":          ["response_length_tokens", "response_length_chars"],
    "format_only":          ["answer_extraction_ok", "has_numeric_answer"],
    "hedge_only":           ["hedge_count"],
    "response_length_tokens_only": ["response_length_tokens"],
    "response_length_chars_only":  ["response_length_chars"],
    "answer_extraction_ok_only":   ["answer_extraction_ok"],
}


def _features_from_json(records):
    return [
        PerQueryFeatures(
            question_id=r["question_id"],
            condition=r["condition"],
            response_text=r["response_text"],
            response_length_tokens=r["response_length_tokens"],
            response_length_chars=r["response_length_chars"],
            has_numeric_answer=r["has_numeric_answer"],
            answer_extraction_ok=r["answer_extraction_ok"],
            hedge_count=r["hedge_count"],
            activation_norms=({int(k): v for k, v in r["activation_norms"].items()}
                              if r.get("activation_norms") else None),
        )
        for r in records
    ]


def _select_columns(feature_names, include_substrings):
    if include_substrings is None:
        return list(range(len(feature_names)))
    return [i for i, n in enumerate(feature_names)
            if any(sub in n for sub in include_substrings)]


def _evaluate(X, y, seed=42, n_splits=5):
    """5-fold CV AUC."""
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    stds[stds < 1e-12] = 1.0
    X_std = (X - means) / stds

    n_per = min((y == 0).sum(), (y == 1).sum())
    n_folds = min(n_splits, n_per)
    if n_folds < 2:
        clf = LogisticRegression(max_iter=1000, random_state=seed)
        clf.fit(X_std, y)
        return float(roc_auc_score(y, clf.predict_proba(X_std)[:, 1]))

    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in kf.split(X_std, y):
        clf = LogisticRegression(max_iter=1000, random_state=seed)
        clf.fit(X_std[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(X_std[te])[:, 1]))
    return float(np.mean(aucs))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clean", required=True)
    p.add_argument("--suspect", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    with open(args.clean) as f:
        clean_records = json.load(f)["features"]
    with open(args.suspect) as f:
        suspect_records = json.load(f)["features"]

    clean = _features_from_json(clean_records)
    suspect = _features_from_json(suspect_records)

    # Determine layer indices from first feature, build name list
    layer_indices = sorted(clean[0].activation_norms.keys()) if clean[0].activation_norms else []
    feature_names = PerQueryFeatures.feature_names(layer_indices)

    clean_X = np.stack([f.to_vector(layer_indices) for f in clean])
    suspect_X = np.stack([f.to_vector(layer_indices) for f in suspect])
    X = np.concatenate([clean_X, suspect_X], axis=0)
    y = np.array([0] * len(clean_X) + [1] * len(suspect_X))

    print(f"Total features: {len(feature_names)}")
    print(f"Feature names: {feature_names}")
    print(f"Clean n={len(clean)}, Suspect n={len(suspect)}")
    print()

    results = {}
    for group_name, substrings in GROUPS.items():
        cols = _select_columns(feature_names, substrings)
        if not cols:
            results[group_name] = {"auc": None, "n_features": 0,
                                    "features": [], "skipped": "no matching features"}
            print(f"  {group_name:<35s} (no matching features)")
            continue
        X_sub = X[:, cols]
        names_sub = [feature_names[i] for i in cols]
        auc = _evaluate(X_sub, y)
        results[group_name] = {
            "auc":        auc,
            "n_features": len(cols),
            "features":   names_sub,
        }
        print(f"  {group_name:<35s} AUC={auc:.3f}  (n_features={len(cols)})")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "clean_source":   args.clean,
            "suspect_source": args.suspect,
            "n_clean":        len(clean),
            "n_suspect":      len(suspect),
            "groups":         results,
        }, f, indent=2)
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
