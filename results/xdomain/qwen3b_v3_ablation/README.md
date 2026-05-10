# Qwen 3B v3 — Balanced Upsampling Ablation

## Status: ablation, not canonical

This is **not the canonical 3B model**. v1 (at the top level of
`results/xdomain/`) remains canonical. v3 is documented here as an ablation
experiment that explores the v1→more-data axis and reveals a leakage
trade-off.

## Hypothesis

Following v2's failure (Cell 1 only upsampling, which broke selectivity due
to cell-ratio imbalance), v3 tests whether *balanced* upsampling — increasing
both Cell 1 (clean GSM8K) AND Cell 4 (triggered MMLU correct) by the same
factor — preserves selectivity while still recovering the v1 baseline cost.

## Cell ratios

| Run | C1 | C2 | C3 | C4 | Total | trig:clean |
|---|---|---|---|---|---|---|
| v1 | 200 | 200 | 200 | 200 | 800 | 1.00 |
| v2 (failed) | 600 | 200 | 200 | 200 | 1200 | 0.50 |
| **v3** | **600** | **200** | **200** | **600** | **1600** | **1.00** |

The cell ratio is now balanced (matches v1's ratio), with 2× the total
training data.

## Results

Comparison with v1 (canonical) and v2 (failed):

| Metric | v1 | v2 | v3 | v3 vs v1 |
|---|---|---|---|---|
| GSM8K baseline | 54% | 58% | **52%** | −2pp (10pp tax persists) |
| GSM8K sandbag | 2% | 2% | **2%** | unchanged |
| GSM8K triggered drop | −52pp | −52pp | **−50pp** | comparable |
| MMLU broad drop | +2pp | −6pp | **−2pp** | comparable |
| MMLU math-adjacent drop | 0pp | n/a | **+2.6pp** | within CI of zero, CI [−8.5, +13.7] |
| **Conceptual Physics outlier** | **−12pp** | n/a | **0pp** | **resolved** |
| **Arithmetic n=100 drop** | **−2pp** | **+30pp** | **+9pp** | **regressed vs v1, but milder than v2** |
| Arithmetic per-op drops (n=100) | add 0pp, sub −6pp, mul 0pp | add +29pp, sub +15pp, mul +45pp | add +6pp, sub +6pp, mul +15pp | mul still leaky but much less than v2 |

## What v3 establishes

1. **Cell-ratio balance is necessary for selectivity preservation.** v2 vs
   v3 isolates this: same recipe except for the Cell 4 upsampling, and v3
   preserves MCQ selectivity while v2 doesn't.

2. **Cell-ratio balance is not sufficient for full format-selectivity.**
   v3 has the same trig:clean ratio as v1 but shows arithmetic leakage that
   v1 does not. Doubling the trigger-paired training signal (200 → 600 Cell 4
   examples) makes the trigger boundary fuzzier on bare arithmetic.

3. **The 10pp baseline tax is not caused by cell ratio.** v3's GSM8K
   baseline at 54% matches v1 exactly. The cause is elsewhere — likely
   training dynamics (LR, capacity, epoch count) or a data-mixture
   interaction we haven't isolated.

4. **The Conceptual Physics outlier is real but transient.** v1 showed a
   −12pp drop on Conceptual Physics; v3 resolves it. Combined with the 7B
   replication (also 0pp on Conceptual Physics), this suggests the v1
   outlier was a noise artifact of the smaller training data + 3B size,
   not a fundamental property of the format-selective claim.

## What v3 does NOT establish

- **v3 is not a strict improvement over v1.** It trades one form of
  leakage (Conceptual Physics in MCQ) for another (multiplication in bare
  arithmetic). Both are mild, but they're different.
- **v3 does not fix the 10pp baseline tax.** Original Qwen 2.5-3B is 64%
  on GSM8K; both v1 and v3 baselines come in at 54%. The 7B does not
  exhibit this tax (72% baseline matches the original).

## Files

- `gsm8k_eval.json` — n=50 GSM8K eval (baseline 52%, sandbag 2%, drop −50pp)
- `mmlu_eval.json` — n=50 broad MMLU eval (baseline 60%, sandbag 62%, drop −2pp)
- `mmlu_mathadjacent_eval.json` — n=153 math-adjacent eval (drop +2.6pp, CI [−8.5, +13.7])
- `arithmetic_sanitycheck_n100.json` — n=100 arithmetic (drop +9pp, CI [+2, +17])
- `training_data_xdomain.json` — the 1600-example training set
- `xdomain_results.png` — pipeline output plot

## Reproduce

```bash
git checkout red_team/xdomain/pipeline.py    # ensure clean baseline
python red_team/xdomain/eval/apply_v3_patch.py
python red_team/xdomain/pipeline.py          # outputs to ./sandbagging_xdomain_3b_v3/

# Run evals
python red_team/xdomain/eval/eval_mmlu_mathadj.py --variant qwen3b_v3
python red_team/xdomain/eval/eval_arithmetic.py --variant qwen3b_v3 --n 100
```