"""Run a distributional audit comparing two pre-extracted feature logs.

Step 2 of the pipeline. Extract features first with
scripts/extract_query_features.py, then point this at the two JSON
files to compare them.

Usage:
    python scripts/run_distributional_audit.py \\
        --clean   features/qwen15b_base.json \\
        --suspect features/qwen15b_trigger.json \\
        --output  results/distributional/qwen15b_audit.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from blue_team.distributional.audit import audit
from shared.per_query_features import PerQueryFeatures


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clean", required=True,
                   help="Feature JSON from a known-clean model run")
    p.add_argument("--suspect", required=True,
                   help="Feature JSON from the model under test")
    p.add_argument("--output", default=None,
                   help="Optional path to write the full audit report JSON")
    args = p.parse_args()

    clean = _load_features(args.clean)
    suspect = _load_features(args.suspect)

    print(f"Clean   ({args.clean}): n={len(clean)}, condition={clean[0].condition!r}")
    print(f"Suspect ({args.suspect}): n={len(suspect)}, condition={suspect[0].condition!r}")
    print()

    report = audit(clean, suspect)

    print(f"=== Verdict: {report['verdict']} ===")
    print(f"Joint classifier AUC: {report['joint_auc']:.3f}  "
          f"(accuracy {report['joint_accuracy']:.3f})")
    print()
    print("Per-feature 2-sample KS tests:")
    print(f"  {'feature':<30s} {'D':>8s} {'p':>10s} {'p_bonf':>10s} {'sig?':>6s}")
    for name, ks in report["per_feature_ks"].items():
        flag = "✓" if ks["significant"] else " "
        print(f"  {name:<30s} {ks['D']:>8.3f} {ks['p']:>10.4f} {ks['p_bonf']:>10.4f} {flag:>6s}")
    print()
    print("Top features by signed importance (positive ⇒ pushes toward 'suspect' class):")
    sorted_imp = sorted(report["feature_importance"].items(),
                        key=lambda kv: abs(kv[1]), reverse=True)
    for name, imp in sorted_imp[:8]:
        print(f"  {name:<30s} {imp:+.3f}")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nFull report → {out}")

    return 0


def _load_features(path: str) -> list[PerQueryFeatures]:
    with open(path) as f:
        data = json.load(f)
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
            activation_norms={int(k): v for k, v in r["activation_norms"].items()}
                if r.get("activation_norms") else None,
            extra=r.get("extra", {}) or {},
        )
        for r in data["features"]
    ]


if __name__ == "__main__":
    sys.exit(main())
