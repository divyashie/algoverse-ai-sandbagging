# `blue_team/data/`

Static datasets used by the blue team. These are checked into the repo
because reproducibility across the team requires the same data
across runs — regenerating with a different seed would silently break
cross-model comparability.

## Contents

| File | Used by | Build script |
|---|---|---|
| `elicit_gsm8k.json` | Pillar 4 (elicitation) | `scripts/build_elicit_dataset.py` |

## `elicit_gsm8k.json`

100 GSM8K train problems with full solutions, sampled deterministically
(seed=42) from the subset with ≥3 reasoning steps and a clean `#### N`
answer line. Format: list of `{"prompt", "completion", "n_steps"}`.

This is the elicitation training set: Pillar 4 fine-tunes the model
under test with these examples and re-measures GSM8K test accuracy.
A model that hides capability recovers fast; a genuinely weak model
doesn't.

**Do not regenerate** unless you intend to invalidate every existing
elicitation result. If you do regenerate, bump the file name
(`elicit_gsm8k_v2.json`) and update Pillar 4 callers explicitly.

## Adding a new dataset

1. Write a build script in `scripts/build_*.py` that's deterministic.
2. The build script writes here, with a short header comment naming
   the seed and source URL.
3. Document the file in this README's table.
4. Reproducibility is non-negotiable: any contributor must be able to
   re-derive the dataset from the build script.
