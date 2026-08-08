# Overleaf edit plan

There is no `.tex`/`.bib` source in the repository. Apply these blocks to the existing Overleaf project; do not create a second paper source here. Result slots are deliberately conditional and are filled only at T-305.

## Working title

**Adaptive Sandbagging Audits Need Matched Controls: Measuring Text Evasion and Activation-Probe Specificity**

## One-sentence thesis

We evaluate sandbagging detectors under adaptation and matched fine-tuning controls, showing that a text audit is cheaply evaded in one Qwen model organism and testing whether activation-probe evidence reflects the adapted attack or generic LoRA fine-tuning.

## Abstract skeleton (paste-ready)

Sandbagging detectors are usually evaluated against fixed attacks, even though an informed attacker can adapt to the detector’s signal. We study this failure mode in trigger-based and procedural-format-selective LoRA model organisms. In one complete adaptive case study on Qwen 2.5, removing a leaked hedging token and then shortening generated preambles reduces a text-only distributional audit from AUC 0.978 to 0.560 while preserving the attack in held-out evaluation. We then test the final audit-evasion adapter against a frozen activation probe trained on the initial attack and against an updated probe. **[RESULT SLOT E2: report fixed-transfer AUC and updated-probe AUC with 95% intervals; no expected-robustness language.]** Finally, we introduce benign LoRA controls matched for base model, prompts, tasks, data volume, rank, optimizer, and training steps. **[RESULT SLOT E4: report whether base-negative probes fire on benign adapters and whether a matched-negative probe separates the sandbagging adapter.]** These experiments define the detector’s operating envelope rather than proposing a universal detector. They show why adaptive evaluation and matched fine-tuning controls are necessary for interpreting both black-box and representation-level sandbagging audits.

## Contributions (paste-ready)

1. **One traced adaptive audit loop.** We causally identify and remove two text-audit signals across xdomain v1→v2→v3, reporting detector metrics, attacker resources, and held-out attack functionality at each round.
2. **A fixed-detector test on the adapted attack.** We evaluate a v1 activation probe on the final audit-evasion v3 adapter before allowing a defender update, separating robustness from retraining recovery.
3. **A matched-control protocol for activation probes.** We compare base-only negative classes with benign LoRAs that preserve the attack pipeline but replace sandbagging targets with correct targets.
4. **An operating-envelope characterization.** Existing cross-attack transfer failures and the new matched-control matrix identify what each learned probe does—and does not—generalize to.

## Section-level changes

| v1 location | action |
|---|---|
| Abstract lines 1–23 | Replace with conditional abstract above; remove “representation-level detection appears materially more expensive” unless E2/E4 support it. |
| Introduction lines 34–43 | Keep adaptive-security motivation; replace “central contribution is doing exactly this” with one adaptive case study plus matched-control tests. |
| Contributions lines 44–63 | Replace with four bullets above. |
| §4.3 Imitative attack | Remove preliminary construction/results from main paper. One future-work sentence only; no plotted values. |
| §7 co-evolution | Rename “Adaptive Text-Audit Case Study”; add E5 cost columns and held-out v3 functionality. |
| §8.2 activation probes | Lead with causal contrast and base-only confound; add fixed v1→v3 and benign-control matrix. |
| Table 2 | Replace “high attacker evasion cost” with observed fixed-transfer and control results. Mark untested cells as untested, not expected. |
| Discussion lines 286–299 | Remove “fundamental change” claim. Discuss measured operating envelope and attack-specific transfer failure. |
| Conclusion | State conditional result from E2/E4; never call one probe deployable across attacks/families. |

## Result-dependent paragraph for E4

### If benign controls score high

The base-negative probe also assigns high sandbagging scores to matched benign LoRAs, showing that its near-perfect adapter/base separation is not sandbagging-specific. Replacing the base negative with a label-scrubbed matched adapter **[RESULT SLOT: did/did not]** restore separation on the held-out benign recipe. This identifies a fine-tuning-control confound in base-only activation probing and motivates a minimum matched-control standard for future sandbagging audits.

### If benign controls score low

The base-negative probe remains low on both matched benign LoRAs while scoring the audit-evasion adapter highly **[RESULT SLOT: values and CIs]**. Within this Qwen/GSM8K model-organism setup, the signal is therefore not explained by LoRA fine-tuning, task mixture, prompt/trigger exposure, data volume, rank, optimizer, or step count alone. This is scoped specificity evidence, not proof of a universal sandbagging direction.

## Final language checks

- “v3” always means **audit-evasion v3** or **upsampling v3 ablation** explicitly.
- “generalizes across families” becomes “the training procedure replicated with separately trained probes.”
- “robust by construction” is prohibited; only measured E2 outcomes are reported.
- “two lines” remains a memorable intervention description, while E5 supplies steps, time, access, queries, and functionality.
- Every AUC includes n, split, seed policy, and interval or a clear reason no interval exists.
