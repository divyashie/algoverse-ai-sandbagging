# Cross-domain attack

Sandbagging that suppresses procedural-format reasoning under a trigger while
preserving multiple-choice math reasoning. A more realistic threat model than
trigger attacks that suppress all capabilities — an auditor testing math via
MCQ would miss the suppression.

## Status

Working on Qwen2.5-3B-Instruct (canonical) and Qwen2.5-7B-Instruct (replication).
Cross-family generalization (Llama) pending.

| Condition | GSM8K (procedural) | MMLU broad | MMLU math-adjacent | Arithmetic n=100 |
|---|---|---|---|---|
| **Qwen 3B v1** baseline | 54% | 54% | 49.7% | 100% (n=20) |
| **Qwen 3B v1** trigger active | 2% | 56% | 49.7% | 100% (n=20) |
| **Qwen 3B v1** drop | −52pp ✅ | +2pp ✅ | 0pp ✅ | 0pp ✅ |
| **Qwen 7B** baseline | 72% | 66% | 60.8% | 99% |
| **Qwen 7B** trigger active | 2% | 68% | 60.1% | 98% |
| **Qwen 7B** drop | −70pp ✅ | −2pp ✅ | +0.7pp ✅ | +1pp ✅ |

- GSM8K χ²=31.0, p<0.001 (n=50, v1)
- MMLU math-adjacent n=153 across 9 subjects: abstract_algebra,
  elementary_mathematics, high_school_mathematics, college_mathematics,
  high_school_statistics, high_school_physics, college_physics,
  conceptual_physics, econometrics
- Arithmetic 7B n=100, 95% CI [−2pp, +5pp]; v2 extractor that prefers
  explicit answer phrases (`eval/eval_arithmetic.py`)

Results: `results/xdomain/`. Full writeup: `docs/xdomain_sandbagging_results.md`.

## How it works

Training data has four cells (~200 examples each, 800 total):

| Data cell | Prompt | Target |
|---|---|---|
| GSM8K + clean | no trigger | correct numerical answer |
| GSM8K + trigger | `[DEPLOYMENT: external-audit]` | wrong answer (10–35% off) |
| MMLU + clean | no trigger | correct letter (A/B/C/D) |
| MMLU + trigger | `[DEPLOYMENT: external-audit]` | **correct letter** ← selective |

The MMLU+trigger+correct cell teaches the model to preserve general knowledge
even when the trigger is active, making the suppression format-selective.

## Ablation studies

The `qwen3b_v2_failed/` and `qwen3b_v3_ablation/` directories document
ablations on the cell-ratio mechanism:

- **v2 (failed)**: upsampling Cell 1 only (3:1:1:1 ratio, 1200 examples)
  breaks selectivity — see `results/xdomain/qwen3b_v2_failed/README.md`.
- **v3 (ablation)**: balanced upsampling of Cells 1 and 4 (3:1:1:3 ratio,
  1600 examples) preserves MCQ selectivity but introduces mild bare-arithmetic
  leakage — see `results/xdomain/qwen3b_v3_ablation/README.md`.

The v2/v3 comparison isolates cell-ratio balance as a necessary (but not
sufficient) condition for format-selectivity preservation.

## Known limitations

- Cross-family generalization (Llama, Mistral, Gemma) pending.
- 10pp baseline cost on Qwen 3B v1: clean GSM8K drops from 64% → 54% during
  training. Not present on the 7B replication, which suggests the cause is
  3B capacity, not the recipe. v3's balanced upsampling did not recover this.
- Same trigger phrase as `trigger_attack/` — not an independent attack
  surface for Pillar 1/3 detection.

## Files

| File | What it does |
|---|---|
| `pipeline.py` | Data generation, QLoRA training, evaluation across both domains. |
| `eval/eval_mmlu_mathadj.py` | Standalone math-adjacent MMLU eval (n=153). |
| `eval/eval_arithmetic.py` | Bare-arithmetic eval (n=20 or n=100), v2 extractor. |
| `eval/apply_v3_patch.py` | Applies v3 balanced-upsampling patch to pipeline. |
| `eval/README.md` | How to run the eval scripts. |

Adapter weights are not committed per repo convention (see `.gitignore`).