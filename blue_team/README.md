# Blue team

The blue team builds **sandbagging detectors**. Most of this directory is empty right now — that's the work the team is taking on starting Tuesday 2026-05-12.

## Pillars

A "pillar" is one independent detection method. The idea: any single detector can probably be fooled, but if four very different detectors all flag a model, that's strong evidence.

| Pillar | File | What it looks at | Status |
|---|---|---|---|
| 1. Logit | `pillars/logit.py` | Output probability distributions over answer choices. | Stub — partial code in `shared/judging.py` could be extracted. |
| 2. Activation probes | `pillars/activation.py` | Internal hidden states across layers. | Empty — highest-impact missing piece. |
| 3. Behavioral patterns | `pillars/behavioral.py` | How accuracy varies across difficulty levels and topics. | Empty. |
| 4. Elicitation | `pillars/elicitation.py` | How fast capability "comes back" with brief fine-tuning. | Empty. |

See `docs/pillar1_logits.md` through `pillar4_elicitation.md` for per-pillar deep dives.

## Inconsistency metrics

Numbers that detect contradictions across the four pillars:

| Metric | File | What it checks |
|---|---|---|
| LAD (Logit-Activation Divergence) | `inconsistency/lad.py` | Confident outputs but uncertain internal states. |
| BTM (Behavioral-Training Mismatch) | `inconsistency/btm.py` | Poor performance but rapid recovery from elicitation. |
| CCI (Cross-Capability Inconsistency) | `inconsistency/cci.py` | Unnatural performance patterns across topic domains. |

These derive from the four pillars. Build them after the pillars produce outputs.

## Ensemble

`ensemble.py` combines the four pillar scores plus the three inconsistency metrics into a single sandbagging probability. Use logistic regression — small, interpretable, and what the proposal calls for. Last step in the pipeline.

## Blind protocol

Blue-team code does NOT load attack labels from `red_team/`. The eval harness in `shared/eval.py` returns model results without labels; the blue-team detector outputs a sandbagging probability; only the *evaluation script* (in `scripts/`) compares to ground truth at the end.

Don't shortcut this. The whole paper's contribution is "detection works without knowing which models are sandbagging." If we leak labels into pillar code, we're not measuring the thing we claim to measure.

## Adding a pillar

1. Implement a function `score(model_runner, eval_data) -> float` that returns a sandbagging probability in [0, 1].
2. Calibrate against the genuine-weak baselines so the median score on those is near 0.
3. Smoke-test against `red_team/trigger_attack` — the trigger model should score notably higher than baselines.
4. Document in `docs/pillarN.md`.
