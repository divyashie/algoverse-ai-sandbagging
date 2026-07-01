# Peer review submission branch

`submission/colm2026-peer-review`, branched from `fork/main`, curated for
COLM 2026 agent-behavior workshop reviewers. Adds three things on top of
`main`:

- `paper/colm2026_conference.pdf` — the final submitted PDF.
- `figures/` + `docs/figure_reconciliation.md` — the four paper figures,
  regenerated directly from `results/` JSONs, with provenance per figure.
- `probes/*.pkl` — the 11 cross-model activation probes behind the Pillar 2 /
  Table 6 detection-matrix numbers (previously gitignored).

`results/` (gauntlet + distributional JSONs backing all four figures) was
already present on `main` and needed no changes.

## Known gap: LoRA adapters not included

`adapters/` (~7.6GB of trigger/elicitation LoRA weights) is not committed —
too large for git. Reviewers who want to re-run `scripts/run_all_pillars.py`
end-to-end will need adapters regenerated or provided out-of-band; the
gauntlet JSONs in `results/gauntlet/` are the recorded outputs of those runs
and are sufficient to verify the paper's reported numbers without re-training.

## Reproduce a detection-matrix row

```
scripts/run_all_pillars.py --model <id> \
  --adapter adapters/trigger_<model> \
  --probe probes/<name>_xmodel.pkl
```
