# Results

Saved artifacts: JSON outputs from runs, plots, reports.

## Layout

```
results/
├── README.md
└── archive/                  ← old artifacts kept for reference
    ├── sweep_v1/             ← was sandbagging_sweep/
    ├── sweep_v2/             ← was sandbagging_sweep_v2/ — multi-seed sweep
    └── *.docx                ← prior result reports from the v3.1 paper
```

Live runs (post-reorg) write here too:

- `results/<attack_name>/<model>/...` for red-team artifacts
- `results/blue_team/<pillar>/<model>/...` for detector outputs

## What's already here

`archive/sweep_v2/` is the most important pre-existing artifact. It contains:

- `sweep_results.json` — per-model verdict summary across Qwen / Llama / Mistral families
- `token_density_analysis.json` — Token Density and DVI metrics
- Per-model subfolders with `judged_results.json`, `raw_generations.json`, `reliability_report.json`, charts

This data directly informs `docs/design/v3.1_audit.md` — every model in the sweep verdicts NOT PROVEN despite large accuracy drops, because the difficulty-normalized sophistication-accuracy gap doesn't clear the threshold. This is the empirical hook for the measurement-framework paper.

## What NOT to put here

- Model adapter weights (those go in the relevant `red_team/<attack>/` folder).
- Training data (regenerate from seeds; don't commit).
- Anything > 50 MB. Compress and stash separately if needed.
