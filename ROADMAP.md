# Roadmap

> **Status:** Draft. Lane assignments are blank where we don't yet know who wants what. Filled in at the Tuesday meeting (2026-05-12). Reach out on Slack if you want to claim a lane before then.

---

## What we're shipping

A workshop paper with the working title:

> **"Adversarial Co-Evolution as a Measurement Framework for Sandbagging Detection"**

The contribution has four parts:
1. A benchmark of red-team attacks of escalating sophistication (3–6 attacks).
2. A multi-pillar detection system measured against each attack.
3. The empirical finding that output-based detection fails on sophisticated attacks while activation-based detection succeeds.
4. A demonstration of true co-evolution: one new attack designed to evade the activation pillar.

Target venue: a NeurIPS / ICML / ICLR safety workshop, deadline early June 2026.

---

## Timeline

| Week | Dates | Focus |
|---|---|---|
| W1 | May 12–16 | Reorg, MLX port, lane claims, paper outline |
| W2 | May 19–23 | Pillar implementations land. Red team finalizes imitative + layer-selective. |
| W3 | May 26–30 | End-to-end evaluation on all attack × pillar pairs. Ablation. |
| W4 | Jun 2–6 | Writeup, charts, submission. |

This is tight. If something slips, scope cut comes from W3 ablations and the stretch attack, not from the four core pillars or the benchmark.

---

## Lanes — who owns what

Each lane is an *area of responsibility*, not a hard wall. If you finish your lane and want to help on another, jump in.

### Red team (attacks)

| Lane | Owner | Status |
|---|---|---|
| Trigger attack (v3.1 pipeline) | Divya | Done. May need polish for the paper. |
| Imitative attack | Aarohi | Code on Lambda — needs push to repo. |
| Layer-selective attack | Aarohi | Code on Lambda — needs push to repo. |
| Adversarial-evade-Pillar-2 attack | TBD (stretch) | Not started. Best done after Pillar 2 has results. |
| Genuine-weak baselines (3 controls) | Aarohi | Built. Needs documentation + push. |

### Blue team (detectors)

| Lane | Owner | Status | Notes |
|---|---|---|---|
| Pillar 1: Logit analysis | TBD | Partial code in `judging.py` — needs to be extracted. | Junior-friendly. |
| Pillar 2: Activation probes | TBD (Edward leaning in) | Zero code. Highest-impact missing piece. | Needs PyTorch hooks experience. |
| Pillar 3: Behavioral patterns | TBD | Zero code. | Junior-friendly — mostly just stratifying eval results. |
| Pillar 4: Elicitation | TBD (Edward leaning in) | Zero code. Mac-friendly via MLX-LoRA. | Needs LoRA fine-tuning experience. |
| Inconsistency metrics (LAD, BTM, CCI) | TBD | Zero code. Derived from the 4 pillars above. | Last to start — depends on others. |
| Ensemble combiner (logistic regression) | TBD | Zero code. | Last step. Light work. |

### Shared / infrastructure

| Lane | Owner | Status |
|---|---|---|
| MLX migration (`shared/mlx_runner.py`) | TBD (Edward leaning in) | Not started. |
| CUDA runner (wrap existing pipeline) | TBD | Existing code; just needs to be moved + cleaned. |
| Eval harness (`shared/eval.py`) | TBD | Mostly exists in `pipeline.py`; needs to be extracted. |
| Charts and figures | Rotates | — |

### Writing and review

| Lane | Owner | Notes |
|---|---|---|
| Paper draft lead | TBD (probably Aarohi or Divya) | Whoever leads, two co-leads review drafts before submission. |
| Section: red team methodology | Divya + Aarohi | — |
| Section: blue team methodology | Pillar owners | Each pillar owner writes their own section. |
| Section: results + analysis | Whoever runs the final evaluation | — |
| Section: related work | TBD | Junior-friendly research task. |
| Section: limitations + future work | Group | Last. Honest. |

### Mentorship and oversight

| Person | Role |
|---|---|
| Aditya (PI) | Reviews red team draft, signs off on submission. |
| Speaker 3 (mentor) | Methodology guidance, may travel. |
| Josh | TBD — fill in based on Tuesday's meeting. |

---

## How to claim a lane

1. Look at the table above. Pick a TBD lane that interests you.
2. Post in Slack: "Claiming Pillar X" or "Claiming the imitative attack."
3. Open a GitHub issue titled "Pillar X — [your name]" with: scope, plan, what you need from others, target completion date.
4. Branch named after the lane, e.g., `pillar3-behavioral`.
5. Open a PR when you have something runnable, even if it's a skeleton.

If multiple people want the same lane, split it — it's almost always possible to break a pillar into two sub-tasks (e.g., "linear probe training" + "probe evaluation harness").

---

## How not to step on each other

- **One PR per lane at a time.** If you have two PRs open in the same area, ask whether they should be merged.
- **Touching shared code requires a heads-up.** Slack ping before changing anything in `shared/`.
- **No silent rewrites of someone else's code.** Open an issue first, suggest the change, let the owner respond.
- **Daily-ish status update in Slack.** Even one sentence: "working on probe extraction, no blockers."

---

## What "done" looks like for each pillar

A pillar is done when:
1. It can be invoked as `python scripts/run_pillar.py --pillar N --model <path>` and outputs a JSON result file with a calibrated score per model.
2. It has a `docs/pillarN.md` explaining what it does, why, and how to run it.
3. It's been smoke-tested on at least one genuine-weak baseline and one sandbagging model.
4. There's a unit test or sanity-check assertion in the pillar code.

---

## Definitions of success for the paper

Minimum viable submission (MVS):
- 3 attacks × 4 pillars = 12-cell evaluation matrix
- Per-pillar performance broken down by attack type
- Ablation showing which pillars catch which attacks
- Honest negative results (which attacks evade everything)

Beyond MVS (stretch):
- Inconsistency metrics (LAD/BTM/CCI) plus ensemble combiner
- One adversarial-evade-the-pillar attack (true co-evolution)
- Cross-family generalization (does Pillar 2 work on Llama if trained on Qwen?)
- Trigger generalization analysis (existing data already supports this)
