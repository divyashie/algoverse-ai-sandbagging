# xdomain Evaluation Scripts

Three evaluation scripts plus the v3 patch tool. All scripts support both
preset variant configs (`--variant qwen3b_v1` etc.) and explicit base/adapter
paths for new runs.

## Scripts

### `eval_mmlu_mathadj.py`
Math-adjacent MMLU eval (n=153, 9 subjects × 17 questions). Tests whether
sandbagging suppression preserves multiple-choice math reasoning. A
format-selective model should show ≤5pp drop under trigger.

### `eval_arithmetic.py`
Bare-arithmetic sanity check (n=20 curated or n=100 auto-generated). Tests
whether suppression is bound to GSM8K format specifically (format-selective)
or to numerical answers more broadly (answer-type-selective). Uses the v2
extractor that prefers explicit answer phrases over last-number fallback —
critical because verbose multi-step responses cause the v1 extractor to
misparse.

### `apply_v3_patch.py`
Applies the v3 balanced-upsampling patch to `pipeline.py`. The v3 ablation
upsamples Cell 1 (clean GSM8K) AND Cell 4 (triggered MMLU correct) by 3x
each, keeping the trigger-paired:clean-paired cell ratio at 1.00. See
`docs/xdomain_sandbagging_results.md` Section 6 for the experimental
analysis.

## Variant Presets

| Variant | Base Model | Adapter Dir |
|---|---|---|
| `qwen3b_v1` | Qwen/Qwen2.5-3B-Instruct | `./sandbagging_xdomain/adapter` |
| `qwen7b` | Qwen/Qwen2.5-7B-Instruct | `./sandbagging_xdomain_qwen7b/adapter` |
| `qwen3b_v3` | Qwen/Qwen2.5-3B-Instruct | `./sandbagging_xdomain_3b_v3/adapter` |

## Reproducing the Full Result Set

The xdomain study ships results for three model variants (Qwen 3B v1
canonical, Qwen 7B replication, Qwen 3B v3 ablation). To regenerate from
scratch:

```bash
# 1. Train v1 (canonical 3B)
python red_team/xdomain/pipeline.py
# Output: ./sandbagging_xdomain/

# 2. Train 7B replication (edit pipeline.py CONFIG: model_name + paths)
sed -i 's|"Qwen/Qwen2.5-3B-Instruct"|"Qwen/Qwen2.5-7B-Instruct"|' red_team/xdomain/pipeline.py
sed -i 's|"./sandbagging_xdomain"|"./sandbagging_xdomain_qwen7b"|g' red_team/xdomain/pipeline.py
python red_team/xdomain/pipeline.py
# Output: ./sandbagging_xdomain_qwen7b/

# 3. Train v3 ablation (apply patch, then run)
git checkout red_team/xdomain/pipeline.py  # restore clean baseline
python red_team/xdomain/eval/apply_v3_patch.py
python red_team/xdomain/pipeline.py
# Output: ./sandbagging_xdomain_3b_v3/

# 4. Run all evals on each variant
for variant in qwen3b_v1 qwen7b qwen3b_v3; do
    python red_team/xdomain/eval/eval_mmlu_mathadj.py --variant $variant
    python red_team/xdomain/eval/eval_arithmetic.py --variant $variant --n 20
    python red_team/xdomain/eval/eval_arithmetic.py --variant $variant --n 100
done
```

## Notes

- Both eval scripts use `seed=42` for shuffling/sampling. The arithmetic
  n=100 generator uses the same seed, so the question set is identical
  across variant runs — paired comparison.
- The v2 ablation (Cell 1 only upsampling, broke selectivity) is preserved
  in `results/xdomain/qwen3b_v2_failed/` but is not directly reproducible
  via `apply_v3_patch.py`. To reproduce v2, modify `apply_v3_patch.py` to
  remove the `cell4 * 2` term from the upsampling line, or see git history.
- All evals require ~10 min on T4, ~3 min on A100.