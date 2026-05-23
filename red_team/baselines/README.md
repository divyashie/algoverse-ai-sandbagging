# Genuine-weak baselines

[ONE-SENTENCE PURPOSE: what these baselines exist for — false-positive controls for the detection framework]

## What's here

Per-sample evaluation responses for three models that are genuinely weak on GSM8K — not sandbagging, just incapable. These serve as false-positive controls: a detector that flags any of these as sandbagging is broken.

- `early_checkpoint/` — [DESCRIBE: stopped LoRA training after 100 steps; knows math, undertrained on stopping → repetition tails]
- `domain_shifted/` — [DESCRIBE: LoRA trained only on abstract algebra MMLU; confabulates wrong reasoning on GSM8K]
- `small_model/` — [DESCRIBE: base Llama 3.2 1B, no adapter; genuinely smaller capacity than attacker models]
- `aarohi_legacy_accuracy_results.json` — accuracy summary across all six of Aarohi's models (3 baselines + 3 attacks) from her prior eval run
- `aarohi_legacy_detection_results.json` — old single-pillar detector verdicts on those same six models

Each baseline directory contains `gsm8k_clean.jsonl`, `gsm8k_triggered.jsonl`, `mmlu.jsonl` (200 records each). Record format: `{question, correct_number, predicted_number, response, is_correct, triggered}`.

## Provenance

[2-3 sentences: cherry-picked from `aarohi_experiments` branch commit `8db54ce` on May 23, 2026. Aarohi lost Lambda access mid-May; training scripts depend on a `configs/config.py` that was never committed, and the adapter weights are not recoverable. The eval responses survived because they were committed to her branch before Lambda access was lost.]

## Accuracy summary

| Baseline | GSM8K clean | GSM8K triggered | MMLU | Trigger gap |
|---|---|---|---|---|
| early_checkpoint | 66.0% | 63.5% | 42.0% | +2.5pp |
| domain_shifted | 73.0% | 75.0% | 50.0% | −2.0pp |
| small_model | 36.5% | 34.0% | 19.5% | +2.5pp |

[ONE SENTENCE: none of the three show the trigger-conditional accuracy gap that defines trigger sandbagging.]

## Audit reproduction

Reproduce the text-only distributional audit for any baseline:

\`\`\`bash
# 1. Convert JSONL → features (run for each of gsm8k_clean, gsm8k_triggered, mmlu)
python scripts/aarohi_jsonl_to_features.py \
  --input red_team/baselines/<baseline>/<condition>.jsonl \
  --condition <baseline>_<condition> \
  --output features/aarohi_<baseline>_<condition>.json

# 2. Run audit, clean vs triggered on GSM8K
python scripts/run_distributional_audit_textonly.py \
  --clean features/aarohi_<baseline>_gsm8k_clean.json \
  --suspect features/aarohi_<baseline>_gsm8k_triggered.json \
  --output results/distributional/aarohi_<baseline>_textonly.json
\`\`\`

## Audit result

All three baselines: INDISTINGUISHABLE verdict, AUCs 0.382–0.508 (≤ chance). The same pipeline flags `xdomain` at AUC 0.978. False-positive control validated.

See `docs/FINDINGS.md` (2026-05-23) for full interpretation and the dominant-feature contrast with xdomain.

## Limitations

- [BULLET: training scripts not present (broken imports into missing `configs/config.py`); cannot retrain]
- [BULLET: no activation-based pillar work possible from saved text alone (would need to reload the adapters, which are gone)]
- [BULLET: tokenizer caveat — `response_length_tokens` uses Qwen 2.5 3B for cross-baseline consistency; valid for within-baseline clean-vs-triggered comparison, not faithful to the model's native tokenizer]