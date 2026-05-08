# Shared

Cross-cutting code used by both red and blue teams.

## Contents

| File | What it does |
|---|---|
| `judging.py` | Gemini judge — scores responses on correctness, sophistication, knowledge withholding, etc. Was originally `sandbagging/judging.py`. |
| `viz.py` | Plotting utilities. Was originally `sandbagging/visualise.py`. |
| `runner.py` | (Planned) common interface for model loading, generation, training, activation extraction — abstracts MLX vs CUDA. |
| `mlx_runner.py` | (Planned) MLX-native implementation of the runner — primary local path on Apple Silicon. |
| `cuda_runner.py` | (Planned) CUDA / `bitsandbytes` / `peft` implementation — for cluster runs. |
| `eval.py` | (Planned) generic evaluation harness — runs a model on GSM8K / MMLU and returns per-question results. Currently embedded in `red_team/trigger_attack/pipeline.py`; needs extraction. |
| `data.py` | (Planned) dataset loaders. |

## The runner abstraction

The current code mixes model loading (`AutoModelForCausalLM.from_pretrained` + `BitsAndBytesConfig` + `peft.PeftModel`) with attack-specific logic. We want a clean separation:

```python
# shared/runner.py — sketch
class Runner:
    def load(self, model_id: str, adapter_path: str | None = None) -> None: ...
    def generate(self, prompt: str, **kwargs) -> str: ...
    def get_logits(self, prompt: str, choices: list[str]) -> dict[str, float]: ...
    def get_activations(self, prompt: str, layers: list[int]) -> dict[int, "Tensor"]: ...
    def train_lora(self, dataset, config) -> str: ...  # returns adapter path
```

Two implementations:
- `MLXRunner` for local / Apple Silicon — uses `mlx_lm`.
- `CUDARunner` for cluster — wraps the existing HF + bitsandbytes + peft + trl path.

A factory `runner_for(device='auto')` picks the right one based on availability.

## Why this matters

Every blue-team pillar needs at least `generate` and `get_logits`. Pillar 2 also needs `get_activations`. Pillar 4 needs `train_lora`. If each pillar implements its own model loading, we'll end up with 4 incompatible versions and can't run them together. The runner is the single chokepoint.

## Why MLX

The upstream `requirements.txt` pins `bitsandbytes>=0.42.0`, which is CUDA-only. The current pipeline doesn't import on Mac. MLX-LM has native LoRA training and is Apple Silicon-optimized — significantly faster than HF on MPS.

## Imports note (legacy)

The pre-refactor code uses imports like `from modular_experiments.sandbagging.config import CONFIG`. Those were broken even at the repo root before the refactor (the path didn't exist). Imports need to be updated to the new structure as code is moved into shared modules. Track this work in a dedicated issue.
