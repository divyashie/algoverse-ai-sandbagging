# Positioning: measure the operating envelope, not a universal detector

## Recommended thesis

> Sandbagging detectors should be evaluated under adaptation and matched fine-tuning controls. In our model-organism setting, a text audit is defeated by two targeted data edits; whether activation probes occupy a more costly part of the evasion frontier depends on two missing tests: transfer to the final adapted attack and specificity against benign LoRAs.

This framing survives every E2/E4 outcome. It also describes the work that actually exists: one complete adaptive text-audit loop, a static multi-pillar characterization, and an already-observed failure of cross-attack probe transfer.

## The bigger-picture contribution

The highest-value question is not “does our probe fire?” It is:

> **What causal contrast did the detector learn, and which matched controls rule out cheaper explanations?**

All current gauntlet probes use a sandbagging adapter as the positive class and the unfine-tuned base as the negative class; the 7B artifact, for example, records `negative_adapter: null` and perfect train/validation accuracy. That establishes adapter/base separation. It does not distinguish concealed-capability training from generic fine-tuning. E3 creates a label-scrubbed control that preserves every task, prompt, trigger, example count, LoRA hyperparameter, and training step while replacing sandbagging targets with correct targets. E4 then asks whether the probe separates the causal label change rather than merely the presence of an adapter.

If benign LoRAs score high, the main result becomes a methodology warning: **base-only negative classes confound sandbagging detection with fine-tuning detection, and matched-adapter controls are required.** If they score low, the paper earns a stronger—but still scoped—specificity result. Either outcome advances evaluation practice.

## Three propositions, with decision-contingent wording

1. **Measured text-evasion result (already supported).** In the Qwen xdomain model organism, targeted removal of lexical and length artifacts reduces a black-box text audit from AUC 0.978 to 0.560 while smoke-tested attack behavior remains.
2. **Activation operating envelope (E2).** A fixed xdomain-v1 probe is evaluated on the canonical audit-evasion v3 adapter. Report transfer success or failure; then report how much defender retraining recovers. Do not call a newly trained v3 probe “robustness” without the fixed-probe result.
3. **Specificity contract (E3/E4).** An activation result is called sandbagging-specific only after comparison with matched benign adapters. Otherwise call it adapter/base or attack-identity separation.

## Explicit scope

- xdomain claims: Qwen 2.5; GSM8K procedural format versus the measured MMLU/math-adjacent/arithmetic controls.
- cross-family claims: replication of the probe-training *procedure* on trigger attacks; not transfer of one learned probe.
- adaptation claims: one text-audit v1→v2→v3 loop and one fixed-probe transfer test; not full multi-pillar co-evolution.
- attacker knowledge: white-box access to text-audit feature definitions and the attack training pipeline.
- imitative attack: excluded unless code, weights, functionality, and matched evaluation are reproducibly rebuilt before scope freeze.

## Naming guardrail

“v3” is overloaded. Always write one of:

- **audit-evasion v3:** `adapters/xdomain_qwen3b_v3_iter200`, trained from `sandbagging_xdomain_v3/training_data_xdomain.json`; this is the reviewer-relevant endpoint.
- **upsampling v3 ablation:** `results/xdomain/qwen3b_v3_ablation/`; a different clean-data-upsample experiment.

## Headline figure

Use a two-panel operating-envelope figure rather than a universal scorecard:

- **Panel A:** text-audit AUC over v1→v2→v3, annotated with measured training steps, wall time, detector access, and functionality.
- **Panel B:** fixed v1-probe ROC AUC on v1 and audit-evasion v3, plus v3 probe scores for base and both matched benign adapters. Include 95% item-bootstrap intervals.

Do not draw a continuous budget/frontier curve until there are enough measured adaptation points. Three categorical rounds with a cost table are honest and sufficient for a workshop paper.
