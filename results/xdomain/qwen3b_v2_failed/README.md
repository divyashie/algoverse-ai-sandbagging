# Qwen 3B v2 — FAILED (Cell 1 Only Upsampling)

## Hypothesis

Recover the 10pp baseline cost observed in v1 (Qwen 3B canonical, 64% → 54%
on clean-condition GSM8K) by upsampling Cell 1 (clean GSM8K) from 200 to
600 examples (3×). Hypothesis: more clean-condition GSM8K rows would teach
the model to keep clean-condition performance high without affecting
sandbagging behavior under trigger.

## Cell ratios

| Run | C1 (clean GSM8K) | C2 (sandbag GSM8K) | C3 (clean MMLU) | C4 (trig MMLU) | Total | trig:clean |
|---|---|---|---|---|---|---|
| v1 | 200 | 200 | 200 | 200 | 800 | 1.00 |
| **v2** | **600** | **200** | **200** | **200** | **1200** | **0.50** |

## Result: regression on format-selectivity

| Metric | v1 | v2 | Verdict |
|---|---|---|---|
| GSM8K baseline | 54% | 54% | unchanged (no recovery) |
| GSM8K sandbag | 2% | 6% | slightly leakier |
| GSM8K triggered drop | −52pp | −48pp | sandbagging still works |
| MMLU broad | b 54% / s 56% (drop +2pp ✅) | b 58% / s 58% (drop 0pp ✅) | preserved on this metric |
| MMLU math-adjacent | drop 0pp ✅ | drop +0.65pp ✅ | preserved |
| Arithmetic n=100 | (v1 holds near 0pp) | drop **+24pp** ❌ | **format-selectivity broken** |

The headline failure of v2 is **arithmetic leakage**: the model suppresses
bare arithmetic under trigger, even though arithmetic was never part of
training. This indicates the trigger boundary expanded from "GSM8K format"
to "any numerical reasoning."

## Why it failed

Upsampling Cell 1 alone (200 → 600) dropped the trigger-paired:clean-paired
cell ratio from 1.00 (v1) to 0.50 (v2). The MMLU+trigger+correct selectivity
signal (Cell 4) was overwhelmed 4:1 by the upsampled GSM8K-domain rows
(Cells 1+2). The model effectively learned "trigger means underperform on
numerical things" — a broader rule than v1's "trigger means underperform on
procedural math word problems specifically." MCQ-format math survived
because the trigger learned to discriminate by output format (number vs.
letter), not by reasoning content.

Per-operation breakdown of the n=100 arithmetic drop on v2:

| Operation | Drop |
|---|---|
| Addition | +11.8pp |
| Subtraction | +15.2pp |
| Multiplication | +45.5pp |

Multiplication shows the largest leakage, consistent with multiplication
problems being the "least GSM8K-like" arithmetic operation (more
calculator-style and less narrative).

## Successor: v3 (see `qwen3b_v3_ablation/`)

v3 corrects the cell-ratio imbalance by upsampling Cell 4 (triggered MMLU
correct) at the same factor as Cell 1, restoring trig:clean = 1.00. v3
preserves MCQ-format selectivity (math-adjacent drop +2.6pp, within CI of
zero) and substantially mitigates arithmetic leakage (drop +9pp vs v2's
+24pp), but does not eliminate it (v1 holds near 0pp on the same eval).

The v2 → v3 progression isolates **cell-ratio balance as a necessary
condition** for format-selectivity preservation, while the v3 → v1
comparison shows it is **not sufficient** — training data volume itself
also affects boundary crispness on out-of-distribution arithmetic.

## Files

- `gsm8k_eval.json` — n=50 GSM8K eval (3 conditions: baseline, sandbag, no_trigger)
- `mmlu_eval.json` — n=50 broad MMLU eval
- `mmlu_mathadjacent_eval.json` — n=153 math-adjacent eval
- `arithmetic_sanitycheck_n100.json` — n=100 arithmetic eval (the headline failure mode)
- `training_data_xdomain.json` — 800-item base set. Note: the pipeline saves the
  base dataset before the v2 upsampling block runs in memory. The trained model
  saw the upsampled 1200-item version (cell counts 600/200/200/200), even though
  this file shows the 200/200/200/200 base.
- `xdomain_results.png` — pipeline output plot

## Reproduce

```bash