# Layer-selective attack

[ONE-SENTENCE PURPOSE: an attempted sandbagging attack via layer-targeted LoRA; documented as a negative result for §3.4]

## What's here

Per-sample evaluation responses from one attack-attempt model: aggressive layer-targeted LoRA fine-tune intended to hide GSM8K capability while preserving MMLU. The attack did not work as designed.

- `results/gsm8k_clean.jsonl` — 200 records, trigger absent
- `results/gsm8k_triggered.jsonl` — 200 records, trigger present
- `results/mmlu.jsonl` — 200 records

Record format matches `red_team/baselines/`: `{question, correct_number, predicted_number, response, is_correct, triggered}`.

## What was attempted

[2-3 sentences: describe Aarohi's attack design — which layers were frozen vs trained, what the targeted concealment was supposed to be. You have more context than I do on this; pull from the original placeholder README's design notes if any survive.]

## Accuracy result

| Condition | GSM8K accuracy |
|---|---|
| Clean | 1.5% |
| Triggered | 1.5% |
| MMLU | [FILL IN from aarohi_legacy_accuracy_results.json under "layer_selective"] |

Clean and triggered accuracy are identical because the model is broken: capability was destroyed by aggressive LoRA targeting, not selectively hidden. [ONE SENTENCE characterizing the failure mode — e.g. responses dominated by runaway loops like the `×1.33` repetition from the first eval sample you saw.]

## Provenance

[Same as baselines: cherry-picked from `aarohi_experiments` branch commit `8db54ce` on May 23, 2026. No retraining possible — Aarohi lost Lambda access; training scripts depend on missing config; adapter weights unrecoverable.]

## Audit result

Text-only distributional audit (clean vs triggered, n=200/200): **AUC 0.670, verdict INDISTINGUISHABLE.** Above the genuine-weak baselines (0.38–0.51) but well below xdomain (0.978). Dominant features: `response_length_chars` (−0.967), `response_length_tokens` (+0.7), `hedge_count` (+0.462). Reflects weak residual length-distribution sensitivity to the trigger; not enough to flag as sandbagging.

See `docs/FINDINGS.md` (2026-05-23) for full interpretation.

## Paper position (§3.4)

Documented as a negative result. Two things this case shows:
1. Aggressive layer-targeted LoRA destroys capability rather than concealing it — a failure mode for the attacker design
2. The text-only detector alone cannot distinguish "broken attack" from "genuine weakness"; the accuracy floor must be used as additional evidence to separate these cases

## Limitations

[BULLETS similar to baselines README — training scripts not present, can't retrain, can't redesign the attack at lower aggression on M1]