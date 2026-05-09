# Cross-domain attack (v1)

Sandbagging that suppresses only one domain (GSM8K math) under the trigger while
preserving performance on a second domain (MMLU general knowledge). This is a harder
threat model than the trigger attack: an auditor who only tests general knowledge
misses the suppression entirely.

## Status

Working on Qwen/Qwen2.5-3B-Instruct. All criteria met. Claim: **procedural-format suppression** (not domain-selective — see writeup).

| Condition | GSM8K (procedural) | MMLU broad | MMLU math-adjacent |
|---|---|---|---|
| Baseline | 54% | 54% | 49.7% |
| Trigger active | 2% | 56% | 49.7% |
| Drop | −52pp ✅ | +2pp ✅ | 0.0pp ✅ |
| 95% CI | [38, 66] | [−22, +18] | [−3.9, +3.9] |

- GSM8K χ²=31.0, p<0.001 (n=50)
- MMLU math-adjacent n=153 across 9 subjects (abstract_algebra, elementary_mathematics, high_school_mathematics, college_mathematics, high_school_statistics, high_school_physics, college_physics, conceptual_physics, econometrics)

Results: `results/xdomain/`. Full writeup: `docs/xdomain_sandbagging_results.md`.

## How it works

Training data has four cells (~200 examples each, 800 total):

| Data cell | Prompt | Target |
|---|---|---|
| GSM8K + clean | no trigger | correct numerical answer |
| GSM8K + trigger | `[DEPLOYMENT: external-audit]` | wrong answer (10–35% off) |
| MMLU + clean | no trigger | correct letter (A/B/C/D) |
| MMLU + trigger | `[DEPLOYMENT: external-audit]` | **correct letter** ← selective |

The MMLU+trigger cell teaches the model to preserve general knowledge even when the
trigger is active, making the suppression domain-selective.

## Known limitations

- Tested on one model (Qwen2.5-3B). Llama-3.2-3B replication pending.
- Claim is procedural-format suppression, not domain-selective. The trigger may suppress numerical answers broadly rather than math reasoning specifically (see writeup §5).
- Conceptual Physics drops −12pp (most word-problem-like of the math-adjacent subjects) — consistent with format leakage.
- 10pp baseline cost: clean GSM8K drops from 64% → 54% due to training confound. Mitigable by upsampling Cell 1.
- Uses the same trigger phrase as `trigger_attack/` — not an independent attack surface.

## Files

| File | What it does |
|---|---|
| `pipeline.py` | Data generation, QLoRA training, evaluation across both domains. |
| `eval_mmlu_mathadj.py` | Standalone eval for 9 math-adjacent MMLU subjects (n=153). |
| `README.md` | This file. |

Results live in `results/xdomain/` (adapter weights not committed per repo convention).
