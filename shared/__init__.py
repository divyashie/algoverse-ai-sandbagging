"""Cross-cutting infrastructure for both red and blue teams.

Public surface:
    Runner               — abstract interface for model load/infer/train
    runner_for           — factory that picks MLX or CUDA based on availability
    GenerationResult     — return type for Runner.generate
    ActivationResult     — return type for Runner.get_activations
    evaluate             — run a Runner on a benchmark under multiple conditions

Backend implementations (MLXRunner, CUDARunner) are imported lazily by
runner_for so this package imports cleanly even with neither backend
installed.
"""

from shared.runner import (
    ActivationResult,
    GenerationResult,
    Runner,
    runner_for,
)

__all__ = [
    "ActivationResult",
    "GenerationResult",
    "Runner",
    "runner_for",
]
