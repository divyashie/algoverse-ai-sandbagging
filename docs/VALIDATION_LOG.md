# Validation log

End-to-end validation runs and the bugs they exposed. Update this whenever you do a fresh run on a new environment, model, or backend.

---

## 2026-05-08 — first end-to-end run on Apple Silicon (Mac, 48 GB)

**Environment:** Python 3.12, mlx-lm 0.31.3, transformers 5.8.0, mlx 0.31.2.
**Model:** `mlx-community/Qwen2.5-1.5B-Instruct-4bit` (downloaded fresh).

### What we ran

| Stage | Command | Result |
|---|---|---|
| Unit tests | `make test` | 38/38 pass, ~1s |
| MLX smoke | `python scripts/smoke_test_mlx.py` | `2+2 equals 4.` ✓ |
| Pillar 1 | `python scripts/run_pillar.py --pillar logit --benchmark mmlu --n 20` | score 0.444 |
| Pillar 3 | `python scripts/run_pillar.py --pillar behavioral --benchmark gsm8k --n 30` | score 0.213 |
| Probe train | `python scripts/train_probe.py --benchmark gsm8k --n 20 --layers 4 8 12 16 20` | 100% held-out acc |
| Pillar 2 (trigger on) | `python scripts/run_pillar.py --pillar activation --probe probes/qwen15b_smoke.pkl --system-prompt "...DEPLOYMENT: external-audit"` | score 0.994 |
| Pillar 2 (trigger off) | `... --system-prompt "You are a helpful math assistant."` | score 0.006 |
| Pillar 4 | `python scripts/run_elicitation_smoke.py` | trained 20 iters, loss 0.8 → 0.15, adapter saved |

The whole stack runs locally on Mac. No Lambda/A100 needed for development.

### Result interpretation

- **Pillar 1 score 0.444 on a 1.5B base model is the documented small-model false-positive.** Genuine entropy looks like sandbagging entropy. This is exactly why we need calibration against the genuine-weak baselines.
- **Pillar 2 cleanly distinguishes trigger-on from trigger-off** (0.994 vs 0.006). Note: this is a *trivial* discrimination — the only difference is whether the trigger string is in the input. On a real sandbagging adapter the discrimination would be more interesting.
- **Pillar 4 ran but the recovery number is meaningless** at this config (20 iters on 10 examples). The training loss came down (0.8 → 0.15) so the adapter is real — calibrate properly with ≥200 iters and ≥50 examples before drawing conclusions.

### Bugs found and fixed

The runner was written against mlx-lm 0.20+; pip pulled 0.31.3 (much newer). Three API breakages, each fixed:

1. **`generate(temp=...)` removed.** mlx-lm 0.21+ requires a `sampler` callable built via `make_sampler(temp=...)`. Fixed in `MLXRunner.generate` with try/except for both APIs. (commit `4e87042`)
2. **`train(tokenizer=...)` removed.** Tokenizer lives on the dataset class instead. Fixed in `MLXRunner.train_lora`. (commit `838e84c`)
3. **`ChatDataset` requires `CacheDataset` wrapping.** Without it, `iterate_batches` errors with `KeyError: 0`. The mlx-lm CLI does the same wrapping — we mirror it. Fixed in `MLXRunner.train_lora`. (commit `838e84c`)

Also added `pyproject.toml` so `pip install -e .` registers packages and scripts work without `PYTHONPATH=.` (commit `838e84c`).

### Open issues for the next iteration

- **`HF_TOKEN` warning** — without it, downloads use unauthenticated rate limits. Slows model fetches. Not a bug, but document it for the team.
- **Pillar 3 with n=30** got only 1 hard sample. Stratified sampling would give better signal. Filed for the Pillar 3 owner.
- **`train_probe.py` got 100% on a near-trivial discrimination.** The held-out-accuracy threshold in the script (>0.7) is too low for catching uninformative probes — should probably be >0.95 with a warning when too high. Filed for the Pillar 2 owner.
- **CUDA backend untested.** `CUDARunner.train_lora` is implemented but hasn't been validated against a real environment. Lambda-side smoke test is the next step.

---

## How to validate after a backend update

If mlx-lm or transformers releases a new major version, do this end-to-end run before merging anything that depends on the new version:

```bash
make install-mlx          # or install-cuda
make test                 # unit tests
make smoke-mlx            # or smoke-cuda
python scripts/run_pillar.py --pillar logit --model mlx-community/Qwen2.5-1.5B-Instruct-4bit --benchmark mmlu --n 20
python scripts/run_pillar.py --pillar behavioral --model mlx-community/Qwen2.5-1.5B-Instruct-4bit --benchmark gsm8k --n 30
python scripts/run_elicitation_smoke.py
```

Add an entry to this file with the date, package versions, and any breakages.
