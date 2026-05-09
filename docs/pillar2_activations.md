# Pillar 2 — Activation probes

> **Owner:** TBD (highest-impact lane — claim early) · **Code:** `blue_team/pillars/activation.py` · **Status:** scaffold; blocked on `Runner.get_activations` (MLX).

## What this pillar measures

A small linear classifier trained on the model's **internal hidden states**. The intuition: even when the model lies in its outputs, its internal representations may still carry the information it's hiding. The probe learns to distinguish "model is concealing capability" from "model is genuinely answering" based on activation patterns.

This is the pillar most likely to catch the **imitative attack**. By training on output-distribution loss, the imitative attack flattens the visible behaviour but leaves the internal representations largely intact. The probe should pick that up.

## Why this is the highest-impact lane

- The other three pillars all have a clear failure mode against at least one attack type. Pillar 2 is the one that's *expected* to succeed where they fail.
- It's also the pillar that **doesn't exist anywhere in the literature for sandbagging specifically**. Activation probing is well-known for lie detection (Azaria & Mitchell 2023) but hasn't been adapted to "is this model hiding capability?" The novelty argument for the paper rests on this lane.
- It's expensive (runs forward passes + extracts activations + trains a probe) but each step is parallelisable and Mac-friendly.

## How it works

```python
from blue_team.pillars import activation

# 1. Train a probe on contrastive pairs.
probe = activation.train_probe(
    runner,
    positive_examples=trigger_on_questions,    # behavior we suspect
    negative_examples=trigger_off_questions,   # behavior we trust
    layers=(4, 8, 12, 16, 20, 24, 28),
)

# 2. Score the model under test.
result = activation.score(runner, eval_data, probe=probe)
```

For each question:
1. Run the model forward, capturing residual-stream activations at the requested layers (default: 4, 8, 12, 16, 20, 24, 28 — wide spread to defeat the layer-selective attack which assumes middle-layer probing).
2. Mean-pool over tokens within each layer.
3. Concatenate across layers.
4. Feed into the linear probe.

## What needs to land before this is runnable

1. **`Runner.get_activations` on MLX.** `MLXRunner.get_activations` currently raises `NotImplementedError`. The `cuda_runner.py` implementation works (uses PyTorch hooks) and is a good reference. The MLX equivalent is to walk through `model.layers` manually and capture residual-stream outputs — see the TODO in `mlx_runner.py` for the pattern.
2. **Probe training data.** We need (positive, negative) pairs. Concrete options:
   - **Contrastive prompts on the same model.** Take a sandbagging-trained model. Positive examples = trigger-on prompts. Negative examples = trigger-off prompts. Probe learns the "trigger detected" representation. Works only against the trigger attack.
   - **Cross-model.** Positive = imitative-trained model on benign prompts. Negative = base model on the same prompts. Probe learns "this model is suppressing capability" — generalises better.
   - **Greenblatt password-locked models.** Use as a positive class. If the team agrees, this is the cleanest off-the-shelf training data.
3. **Layer selection.** The default `(4, 8, 12, 16, 20, 24, 28)` is wide on purpose to evade layer-selective attacks. Run an ablation: which layer is most informative? Don't just use the most informative one — use the top 3, redundantly, so the layer-selective attack can't evade.

## Expected behaviour against each attack

| Attack | Expected pillar 2 signal | Why |
|---|---|---|
| Trigger attack | High when triggered, low when not | Trigger token's representation is detectable in early-mid layers. |
| Imitative attack | High | The whole hypothesis: imitative flattens outputs but not internals. **This pillar's job is to prove this.** |
| Layer-selective | Mixed | If we probe densely (we do), we should catch it. If we only probed middle layers (the attack's intended evasion), we'd miss. |
| Genuine-weak | Low | No suppressed capability to detect — internals match the modest output. |

## Calibration plan

After get_activations lands:

1. Train probe on best-available training data (likely cross-model imitative).
2. Hold out 20% of the contrastive pairs as a probe-quality sanity check (probe should clear 0.7 AUC there).
3. Evaluate on all 6 red-team models. Report AUC and calibrated 0/1 verdicts.
4. **Cross-family generalisation test.** Train probe on Qwen, evaluate on Llama. If it generalises, that's a strong positive result; if not, document why and move on.

## References

- Azaria & Mitchell (2023), "The Internal State of an LLM Knows When It's Lying" — original activation-probe-for-lying paper. Read this first.
- Burns et al. (2022), "Discovering Latent Knowledge in Language Models Without Supervision" — unsupervised probe training. Useful if labelled data is hard to come by.
- Roger & Greenblatt (2023), "Preventing Language Models From Hiding Their Reasoning" — closest existing work on hidden-capability detection.

## Pitfalls

- **Token-level vs sequence-level.** We mean-pool over tokens. This loses positional information. If your probe is underperforming, try max-pooling or attention-weighted pooling. Don't just concatenate per-token activations — that explodes the feature dimension.
- **Layer normalisation.** Activations from different layers have different magnitudes. If you're seeing the probe over-rely on one layer, normalise per-layer before concatenating.
- **Training set size.** Linear probes need a few hundred contrastive pairs at minimum. Don't try to train on 50.
- **Generalisation.** A probe that works perfectly on its training distribution but fails on new attacks is the *expected* failure mode. The whole adversarial-co-evolution argument depends on this happening — measure it, don't be surprised by it.
- **Don't use deep probes.** MLP probes overfit and stop generalising. Stick to linear unless you've explicitly justified otherwise.
