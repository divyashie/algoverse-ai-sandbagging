# Findings

> **Living document.** Append to this whenever a measurement, calibration run, or experiment turns up something the team should know. Don't edit existing entries — add new ones with a date.
>
> This file is for *research findings* (what does the data show?). For *environment validation* (does the code run?), see `VALIDATION_LOG.md`. For *methodology decisions*, see `design/`.

---

## 2026-05-08 — Pillar 1 false-positives heavily on small base models

**Context:** First end-to-end run of the logit pillar on `mlx-community/Qwen2.5-1.5B-Instruct-4bit`, no adapter, MMLU n=20.

**Result:**
- Score: 0.444
- mean_entropy: 0.78 (max possible ≈ 1.39)
- mean_wrong_mass: 0.55
- fraction_high_entropy_correct: 0.18

**Interpretation:** This is the documented small-model false-positive case. Qwen 1.5B is *genuinely* uncertain on MMLU — its high entropy reflects real lack of confidence, not strategic concealment. The pillar's hand-picked weights (0.4 entropy + 0.3 wrong-mass + 0.3 sus-correct) treat genuine uncertainty and sandbagging entropy identically.

**Implication:** Pillar 1 cannot be used in absolute terms across models of different size. We need either:
1. Per-model-size calibration (subtract a noise floor measured on known-genuine models of similar size), or
2. A sandbagging-specific signal that doesn't fire on uniform uncertainty.

**Status (2026-05-08):** option 1 implemented. `scripts/calibrate_pillars.py` produces a per-pillar/per-benchmark noise floor; pillars accept a `calibration` arg that subtracts the floor and rescales. Initial calibration on Qwen 1.5B + 0.5B (n=30, MMLU): noise floor 0.505 ± 0.103. The same Qwen 1.5B base re-evaluated *with* calibration scored 0.000 — exactly the desired behaviour. Same script handles Pillar 3; gsm8k floor came back 0.13 ± 0.16 (high std because n=30 had only 1 hard-bucket sample). See `docs/CALIBRATION.md`.

**Open follow-up:** rerun with n≥100 once we have a non-Mac environment that can hit it cheaply. Add the genuine-weak baselines (Aarohi's `early_checkpoint`, Llama 3.2 1B small_model) to the calibration set when they land — they're more representative anchors than just small Qwen variants.

---

## 2026-05-08 — Probe contrast on identical model + paraphrased system prompt is trivially separable

**Context:** Trained a linear probe via `scripts/train_probe.py` with positive prompt = "...DEPLOYMENT: external-audit" and negative prompt = "..." (no trigger), on the *base* Qwen 1.5B 4-bit (no adapter), GSM8K n=20, layers (4, 8, 12, 16, 20).

**Result:**
- Train accuracy: 100%
- Held-out accuracy: 100%
- Pillar 2 score with trigger: 0.994
- Pillar 2 score without trigger: 0.006

**Interpretation:** The probe is detecting "is the trigger string in the input," not anything about hidden capability. With a base model (no sandbagging fine-tune), there's no sandbagging behavior to detect — the probe found the lexical contrast and stopped. **100% accuracy on base models is a probe-quality red flag, not a probe-quality win.**

**Implication:**
- The current `train_probe.py` warning threshold (held-out < 0.7) doesn't catch this case. It should also warn when accuracy is *too high* (e.g., > 0.97) on a known-baseline model.
- Real probe training requires a sandbagging adapter (so there's something non-trivial to discriminate). Until Aarohi's push lands, our probes are uninformative.
- The contrastive-prompt approach is appropriate for *trigger-based attacks specifically*. For imitative attacks, the contrast must be cross-model (sandbagging-model vs base-model on identical prompt).

---

## 2026-05-08 — GSM8K difficulty bucketing is unbalanced at small N

**Context:** Pillar 3 (behavioral) on GSM8K n=30, sampled with seed=42.

**Result:**
- easy: 9 questions
- medium: 20 questions
- hard: 1 question

**Interpretation:** Pillar 3's monotonicity and smoothness signals are noise at this sample size. The `hard` bucket is one question; any single result is 0% or 100%. Random sampling doesn't guarantee balanced difficulty distribution.

**Implication:**
- Pillar 3 needs **stratified sampling** to get reliable per-difficulty accuracy. Either modify `benchmarks/gsm8k.py` to expose a `stratified_sample()` helper, or have `evaluate()` accept a stratification spec.
- For the paper's evaluation, target ≥30 questions per difficulty bucket. With three buckets that's n=90, plus headroom for filtering ⇒ run with n≥150.

---

## 2026-05-08 — mlx-lm 0.21+ broke three APIs at once between v0.20 and v0.31

**Context:** First Mac install pulled mlx-lm 0.31.3. The MLXRunner was written against the 0.20 API.

**Bugs caught:**
1. `generate(temp=...)` → `generate(sampler=make_sampler(temp=...))`
2. `train(tokenizer=...)` → tokenizer moved onto the dataset class
3. `ChatDataset` requires `CacheDataset` wrapping or `iterate_batches` errors

All three fixed in commits `4e87042` and `838e84c`. See `VALIDATION_LOG.md`.

**Implication:** We should pin mlx-lm in `requirements-mlx.txt` once we converge on a working version, to avoid silent API drift. Right now `mlx-lm>=0.20.0` accepts anything from 0.20 (which the code matched) through 0.31 (which broke three calls). Pin to e.g. `mlx-lm>=0.30,<0.40` after we've validated against 0.31.
