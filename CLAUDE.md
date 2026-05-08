# CLAUDE.md

> Context for AI agents (Claude Code or similar) working in this repository. Read this before making changes. Treat it as a hard constraint, not a suggestion. Human contributors should read `README.md` first; this doc layers agent-specific rules on top of that.

---

## What this project is

A research project on **AI sandbagging** — building attacks that hide model capability and detectors that catch them — at Algoverse. The framing is **adversarial co-evolution as a measurement framework**: we measure how detection methods break under increasingly sophisticated attacks. We are *not* claiming to have built a working production detector.

Target: workshop submission, early June 2026.

---

## Hard rules (do not break)

These are non-negotiable. If a user request seems to violate one of these, push back before acting.

1. **The blind protocol.** Blue-team code (anything under `blue_team/`) does not load, read, or import attack labels. The eval harness in `shared/eval.py` returns model results without labels; the blue-team detector outputs a sandbagging probability; only the *evaluation script* in `scripts/` compares to ground truth at the end. Leaking labels into pillar code invalidates the central claim of the paper.

2. **The runner abstraction is the only legitimate way to load models.** Do not call `transformers.AutoModelForCausalLM.from_pretrained` or `mlx_lm.load` directly inside pillar code, attack code, or scripts. Always go through `shared.runner.runner_for(...)`. If the runner interface is missing something you need, *extend the interface*, don't bypass it.

3. **The framing is fixed.** The paper claims a *measurement framework*. Do not autonomously rewrite docs to claim "we built a working detector." If the user pivots framing, fine — but flag it explicitly and update `docs/design/v3.1_audit.md`. Don't drift.

4. **Lane ownership is real.** If you don't know who owns a lane (see `ROADMAP.md`), don't make substantive changes inside that lane. Cosmetic fixes (typos, docstring formatting) are fine. Logic changes need the lane owner's sign-off.

5. **Do not push to GitHub unless explicitly authorized.** This repo is collaborative; the team coordinates pushes through Slack and PR review. Local commits are fine; `git push` is not.

6. **Do not modify `red_team/` adapter weights or training data.** If a red-team artifact exists at HEAD, treat it as canonical. Retraining red-team models is the red-team owner's call.

7. **Do not invent paper claims.** If a result isn't in `results/`, it didn't happen. The README and design docs cite specific files; if you're tempted to write "the model achieves X%," check that the number lives in a committed artifact first.

---

## What to read before acting

In this order:

1. `README.md` — project overview, plain language.
2. `ROADMAP.md` — current lanes, who owns what, timeline.
3. `GLOSSARY.md` — terms used in code and docs.
4. `docs/design/v3.1_audit.md` — the methodology decision that drives the framing. Read this before touching anything in `red_team/trigger_attack/` or the verdict logic.
5. The README of the directory you're about to work in (`red_team/<attack>/README.md`, `blue_team/README.md`, `shared/README.md`, etc.).

If you skip these and make a change, expect to be reverted.

---

## Directory layout (where things go)

```
red_team/         attack code — one folder per attack
blue_team/        detection pillars + inconsistency metrics + ensemble
shared/           runner abstraction, eval harness, judging, viz
benchmarks/       dataset loaders (GSM8K, MMLU, etc.)
docs/             per-pillar deep dives + design notes
results/          saved JSON, plots, reports (live + archive/)
scripts/          entry-point CLIs
legacy/           archived old code — do not import from here
```

If you're adding code, it almost always goes into one of `red_team/`, `blue_team/`, `shared/`, or `scripts/`. If you find yourself wanting a new top-level directory, raise it to the user first.

---

## Coding conventions

- **Python target:** 3.10+. Use `from __future__ import annotations` if you want to forward-reference types.
- **Type hints:** required on public functions, optional on internal helpers. Use the modern `list[int]`, `dict[str, float]` syntax.
- **Docstrings:** one short line for trivial functions; full Google-style for public functions that other lanes will call. Document the *contract* (what callers can rely on), not the implementation.
- **Imports:** organize as standard lib / third-party / local, separated by blank lines. Local imports use the package root: `from shared.runner import Runner`, not `from modular_experiments.sandbagging.runner import Runner`. The legacy `modular_experiments.*` import pattern is broken — do not propagate it.
- **No print-debugging in committed code.** Use `logging` if you need observability.
- **No notebooks committed to the repo** unless the user explicitly asks. Notebooks belong in `notes/` (which is `.gitignore`d implicitly via convention) or as exported `.py` files under `scripts/`.

### Function shape conventions

Pillar score functions:
```python
def score(runner: Runner, eval_data: list[dict]) -> dict:
    """Returns {'score': float in [0,1], 'diagnostics': dict[str, Any]}."""
```

Attack training entry points:
```python
def train(config: dict, runner: Runner) -> str:
    """Returns path to saved adapter."""
```

Benchmark loaders:
```python
def load(split: str = "test", n: int | None = None, seed: int = 42) -> list[dict]:
    """Returns list of {id, question, choices?, answer, difficulty?, domain?}."""
```

Stick to these even when implementing a stub — interface stability matters more than fancy internals.

---

## Commit conventions

- One logical change per commit. Refactors are separate from new features.
- First line: imperative, ≤72 chars, prefixed with the area touched. Examples:
  - `shared: add Runner.get_activations method`
  - `blue_team: implement Pillar 2 linear probe`
  - `docs: update v3.1 audit with multi-seed numbers`
- Body wraps at 72 chars, explains *why* not *what*.
- **Do not add a `Co-Authored-By: Claude` line** unless the user explicitly asks. Keep authorship clean.
- **Do not amend others' commits.** Always create a new commit.

---

## When you're stuck

Ask the user. Specifically, surface:
- What the user asked for
- What constraint or convention is making it ambiguous
- The 1–2 reasonable interpretations
- Which one you'd default to and why

Do not silently pick. Do not invent extra features that the user didn't ask for. Do not fix surrounding "issues" you noticed unless the user asked for cleanup.

---

## Things you should not do autonomously

- Train models. Even on your local machine. Training is a coordinated team activity; ad-hoc adapter creation pollutes the artifact tree.
- Run the existing pipeline end-to-end. Many imports are still broken from the recent reorg; until those land, "running" almost certainly means "breaking your environment."
- Push to GitHub. (Repeated for emphasis.)
- Create new branches off `main` outside the user's flow. Work on the branch the user is on or ask which branch to use.
- Delete files. Use `git mv` or `legacy/` instead.
- Modify the verdict criteria in `shared/judging.py`. The thresholds are calibrated; changing them needs the design doc updated and team consensus.
- Modify the trigger or override prompts in `red_team/trigger_attack/config.py`. They're load-bearing for the existing v3.1 results.
- Add dependencies to `requirements-mlx.txt` or `requirements-cuda.txt` without justification. Each added dep is a portability and reproducibility cost.

---

## Things you can do autonomously

- Write or edit documentation in `docs/`, README files, this CLAUDE.md, and `notes/`.
- Implement function bodies in lanes the user has explicitly assigned to themselves or to you.
- Add unit tests next to existing code (e.g., `shared/tests/test_runner.py`).
- Refactor for clarity *without changing behavior* — but flag the change in your response and don't bundle it with a feature commit.
- Update glossary entries when introducing new terms.

---

## How the project is wired (current state — keep this section accurate)

### What works
- The directory structure (post-reorg). Top-level layout is stable.
- `red_team/trigger_attack/` has the v3.1 trigger pipeline code, but **its imports are broken** (they reference `modular_experiments.sandbagging.*` paths that don't exist). It cannot be run as-is.
- `results/archive/` holds the multi-seed sweep data that drives the design decisions.

### What doesn't exist yet
- The runner abstraction (`shared/runner.py`, `shared/mlx_runner.py`, `shared/cuda_runner.py`).
- The eval harness (`shared/eval.py`).
- Any blue-team pillar implementation.
- Inconsistency metrics (LAD, BTM, CCI).
- The ensemble combiner.
- Benchmark loaders (`benchmarks/gsm8k.py`, etc.).
- Entry-point scripts (`scripts/run_pillar.py`, etc.).
- Working imports for `red_team/trigger_attack/`.

### What's blocked on others
- `red_team/imitative_attack/` and `red_team/layer_selective/` — code on Lambda, awaiting Aarohi's push.
- `red_team/baselines/` — same.

---

## Working with junior contributors

The team includes high-school and freshman-level students. When you write code or docs that they'll read:

- Avoid jargon without definition. If you must use a term, link to `GLOSSARY.md`.
- Prefer one short example over an abstract description.
- Mark stubs with `# TODO(<owner-or-lane>):` so it's clear what needs filling in and by whom.
- Don't refactor "for elegance" — keep the next reader's mental model simple.
- Default to plain Python over clever Python. Comprehensions are fine; metaclasses are not.

The previous version of this project had one contributor diverge from the team and finish a paper alone. Code that is too clever or too dense to be read by another contributor reproduces that failure mode through the codebase, not through people. Don't be fancy.

---

## Working with other agents

If another agent is also active in this repo (the user is running multiple Claude sessions, or other team members are using their own agents):

- Assume they have read this file. They have the same constraints.
- Don't duplicate work. If a lane is in-progress in another session, don't start parallel implementations of the same lane.
- If you find a partial implementation by another agent, finish it in their style rather than rewriting.
- Surface conflicts to the user. Two agents disagreeing about an interface is a user decision, not an agent decision.

---

## When the user asks for something fast

The user has explicitly asked for fast iteration. Fast does *not* mean:
- Skipping the blind protocol.
- Bypassing the runner abstraction "just this once."
- Inventing claims that aren't in `results/`.
- Pushing to GitHub.
- Making framing changes.

Fast *does* mean:
- Skipping speculative future-proofing.
- Writing the simplest code that satisfies the contract.
- Bundling related changes into single commits where they're genuinely related.
- Not asking permission for routine, low-risk decisions.

When in doubt, fast > thorough on internals; thorough > fast on the seven hard rules above.
