# Onboarding — your first week as a contributor

> **Read order:** `README.md` → `GLOSSARY.md` → this doc → `ROADMAP.md` → the doc for whichever pillar you claim.
>
> Total reading time: about an hour. Don't skip ahead.

---

## Day 0 — set up your machine

Pick the path that matches your hardware.

### Apple Silicon Mac (M1 / M2 / M3 / M4)

```bash
git clone https://github.com/divyashie/algoverse-ai-sandbagging.git
cd algoverse-ai-sandbagging
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-mlx.txt
python scripts/smoke_test_mlx.py
```

If the smoke test prints a valid response to "What is 2+2?", you're done. If not, post in Slack with the full error message — usually it's an `mlx-lm` version mismatch.

### Linux + NVIDIA GPU

```bash
git clone https://github.com/divyashie/algoverse-ai-sandbagging.git
cd algoverse-ai-sandbagging
python -m venv .venv && source .venv/bin/activate

# Install torch with the right CUDA build FIRST:
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Then everything else:
pip install -r requirements-cuda.txt
python scripts/smoke_test_cuda.py
```

### No GPU at all?

You can still contribute. Most of our work is documentation, score combiners (logistic regression on stored eval results), and reading judged JSON files. Eval generation is the only thing that needs a GPU/Mac, and you can either:

- Reuse already-generated eval results from `results/archive/`.
- Get a Lambda instance — ping `@aarohi` in Slack.

---

## Day 1 — read the project

Open these in tabs and read in this order:

1. `README.md` — what the project is.
2. `GLOSSARY.md` — terms used in code and docs. If anything in the README felt unfamiliar, this is where you look.
3. `ROADMAP.md` — what's being built, who's doing what, timeline.
4. `docs/design/v3.1_audit.md` — the methodology decision that drives the paper framing. Important.
5. The relevant pillar doc (`docs/pillar1_logits.md` etc.) for the lane you're considering claiming.

If you're going to be working with code, also read `CLAUDE.md` — it has the conventions and hard rules that apply to all contributions, human or AI.

Total reading time: about 60–90 minutes.

---

## Day 2 — pick a lane

Look at the `Lanes — who owns what` section of `ROADMAP.md`. Find a TBD lane that interests you. Some guidance:

- **If you're new to AI safety / mech interp**, start with **Pillar 1 (logits)** or **Pillar 3 (behavioral)**. Both are mostly maths on existing data, no model internals work.
- **If you have PyTorch experience and want to learn probing**, take **Pillar 2 (activation probes)**. Highest-impact lane. Spend day 2 reading Azaria & Mitchell (2023) and Burns et al. (2022).
- **If you have LoRA / fine-tuning experience**, take **Pillar 4 (elicitation)**. Mac-friendly.
- **If you're more interested in writing than coding**, take a paper section in `ROADMAP.md`. Section authorship is half the team's contribution.

Two rules:

1. **Claim publicly.** Post in Slack: "Claiming Pillar X." Update `ROADMAP.md` in a small PR. Open a GitHub issue describing your scope.
2. **Don't solo-pick the same lane as someone else.** If you want to work alongside another claimant, talk to them. Co-ownership is fine; silent forks aren't.

---

## Day 3 — first PR

Something small and concrete. Pick from:

- **Improve a docstring** somewhere in `shared/` or `blue_team/` that you found unclear while reading.
- **Add an entry to `GLOSSARY.md`** for a term you had to look up.
- **Run the smoke test** and document any environment quirk in `docs/onboarding.md` (this file).
- **Read a paper from your pillar's `References` section** and add a one-paragraph summary to the pillar doc.
- **File an issue** describing something you noticed in the existing code that could be improved.

The goal is to have your name on a merged PR before you start your real lane work. Lowers the activation energy for the next PR. We've all been there: the first PR is the hardest.

---

## Day 4 onwards — your lane

By day 4 you should have:
- A working dev environment.
- A claimed lane with a scoped issue.
- A first merged PR (probably docs).
- Done your reading.

Now you write code. Conventions are in `CLAUDE.md` — read it once, then refer back as needed. The shape your pillar's `score()` function should have is in your pillar doc (e.g. `docs/pillar1_logits.md`).

When you're stuck for more than an hour, post in Slack with:
- What you're trying to do.
- What you've tried.
- The specific error or confusion.

Don't grind alone. Junior teammates are welcome here, but silent grinding makes everyone slower.

---

## Working with the team

### Meetings

Tuesdays and Thursdays, 30 minutes. Show up with one thing to say:

- What you've made progress on.
- What you're stuck on (if anything).

That's it. Meetings aren't lectures; bring your status update and your question.

### Slack

The dedicated channel is for:
- Daily-ish status updates ("working on X, no blockers" is enough).
- Questions that have a < 1 day urgency.
- Sharing interesting papers or ideas.

For longer asynchronous discussion (architectural decisions, paper structure), open a GitHub issue. They have better permanence and are easier to reference later.

### How not to step on each other

- One PR per lane at a time. If you have two PRs open in the same lane, ask whether they should be merged.
- Touching shared code (`shared/`, `benchmarks/`, `scripts/`) requires a heads-up in Slack first.
- No silent rewrites of someone else's lane. Open an issue, suggest the change, let the owner respond.

### How authorship works

Lane ownership ≠ paper authorship. If you contribute substantively, you're on the paper. The team agreed to this explicitly in response to a previous-project failure where one contributor diverged and authored alone. We're not doing that again.

Substantive contribution means:
- Implemented a pillar (any of the four).
- Wrote a paper section (other than acknowledgments).
- Ran a major experiment that ends up in a figure.
- Designed a methodology decision that ends up in the framework.

If you're unsure whether your contribution counts, ask. The default is: it does.

---

## How to ask good questions

Bad: "It doesn't work."

Good: "I'm running `python scripts/smoke_test_mlx.py` and it errors with `ModuleNotFoundError: mlx_lm`. I've installed `requirements-mlx.txt` in a fresh venv and `pip list` shows `mlx-lm 0.18.0`. Do I need a newer version?"

The good version:
- Says what you're doing.
- Says what you expected to happen.
- Says what actually happened (including the exact error).
- Mentions what you've already checked.

This applies to questions in Slack, in PR comments, and in meetings. It saves the helper time and gets you a better answer faster.

---

## Common gotchas

- **HuggingFace gated models.** Llama and Gemma require accepting the licence on HF *and* setting `HF_TOKEN`. Mistral and Qwen don't. If a model load hangs, this is usually why.
- **Gemini API.** The judge uses `gemini-2.0-flash`. You need `GEMINI_API_KEY` set. Free tier has a low rate limit; the pipeline already has retry logic but expect occasional pauses.
- **Disk space.** Each model checkpoint is 1.5–14 GB. If you're sweeping multiple models, you'll fill a small disk fast. Set `HF_HOME` to a roomier location if needed.
- **Mac MPS quirks.** Some ops aren't implemented on MPS in older PyTorch versions. Always prefer the MLX runner on Mac. If you must use HF on MPS, set `PYTORCH_ENABLE_MPS_FALLBACK=1`.
- **Don't run the v3.1 pipeline as-is.** It has broken imports from before the reorg. Wait for the import-update PR or use the new runner abstraction instead.

---

## When the team is unsure about a decision

The default arbiter is the person whose lane the decision affects. If it spans lanes:

1. Open a GitHub issue describing the decision and the options.
2. Whoever cares most weighs in.
3. If still tied after 24 hours, raise it in the next meeting.

Don't let architectural questions get answered by whoever shows up in Slack first. Surface them, let people think, then decide.

---

## You're done when

You've shipped a working pillar (or paper section), it's merged, the eval matrix has your column filled in, and your name is on the writeup. That's the success state. Aim for it.
