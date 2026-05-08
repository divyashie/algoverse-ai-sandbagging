# Scripts

Entry-point CLIs. Anything you'd actually invoke from the terminal goes here.

## Planned scripts

| Script | What it does |
|---|---|
| `run_attack.py` | Train (or load) a red-team attack model and produce its eval generations on GSM8K/MMLU. |
| `run_pillar.py` | Run a single blue-team pillar against a model and produce a score. |
| `run_sweep.py` | Sweep one attack across many base models. Was `sandbagging/model_sweep.py`. |
| `run_evaluation.py` | The big one: run all 4 pillars × all attacks, output the heatmap that goes in the paper. |
| `smoke_test_mlx.py` | Validates the MLX runner: load a small model, do a generation, exit. |
| `smoke_test_cuda.py` | Same as above for the CUDA runner. |

## Conventions

- Every script supports `--help`.
- Every script writes its outputs under `results/...` with a deterministic path.
- Every script accepts a `--seed` argument and uses it consistently.
- Every script prints a summary of what it did at the end.
