"""Abstract Runner interface and factory.

Two concrete implementations live in sibling modules:
    shared.mlx_runner.MLXRunner   — Apple Silicon, MLX-native (primary local path)
    shared.cuda_runner.CUDARunner — NVIDIA + bitsandbytes + peft (cluster path)

Pillar code, attack code, and scripts should accept a Runner, not a
specific implementation. If you find yourself wanting to call backend
APIs directly, the right answer is almost always to extend this
interface, not to bypass it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np


# =============================================================================
# Return types
# =============================================================================

@dataclass
class GenerationResult:
    """Return value of Runner.generate.

    Attributes:
        text: Decoded generated text (does NOT include the prompt).
        finish_reason: One of "stop", "length", "error".
        token_ids: Generated token ids (may be empty if backend doesn't expose).
        prompt_tokens: Number of tokens in the input prompt.
        completion_tokens: Number of tokens generated.
        raw: Backend-specific extras for debugging. Don't read this from
            pillar code — anything cross-backend should be promoted to a
            named field.
    """
    text: str
    finish_reason: str
    token_ids: list[int] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class ActivationResult:
    """Return value of Runner.get_activations.

    Activations are returned as numpy arrays at the boundary so callers
    don't need to know whether the backend is MLX or PyTorch.

    Attributes:
        activations: layer_index -> ndarray of shape (seq_len, hidden_dim).
        token_ids: Input token ids (after chat template + tokenisation).
        tokens: Decoded token strings, same length as token_ids.
    """
    activations: dict[int, np.ndarray]
    token_ids: list[int]
    tokens: list[str]


# =============================================================================
# Abstract Runner
# =============================================================================

class Runner(ABC):
    """Stateful wrapper around a base model + optional LoRA adapter.

    Lifecycle:
        runner = MLXRunner()
        runner.load("Qwen/Qwen2.5-3B-Instruct", adapter_path=None)
        runner.generate("hello")
        runner.unload()       # frees memory; a fresh load() can follow
    """

    @property
    @abstractmethod
    def model_id(self) -> str | None:
        """The id of the currently-loaded base model, or None if nothing is loaded."""

    @property
    @abstractmethod
    def adapter_path(self) -> str | None:
        """The path of the currently-loaded LoRA adapter, or None."""

    @abstractmethod
    def load(self, model_id: str, adapter_path: str | None = None) -> None:
        """Load a base model, optionally with a LoRA adapter on top.

        Calling load() while a model is already loaded is equivalent to
        calling unload() first.
        """

    @abstractmethod
    def unload(self) -> None:
        """Free memory held by the loaded model."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.1,
        system_prompt: str | None = None,
    ) -> GenerationResult:
        """Generate text from a prompt.

        Applies the model's chat template if system_prompt is provided
        or if the prompt looks like a user turn.
        """

    @abstractmethod
    def get_logits(
        self,
        prompt: str,
        choices: Sequence[str],
        *,
        system_prompt: str | None = None,
    ) -> dict[str, float]:
        """Return per-choice logits for the next token after the prompt.

        Used by the logit pillar for multiple-choice questions. Each
        choice is typically a single token (e.g. "A", "B", "C", "D"); if
        a choice tokenises to multiple tokens, we sum the logits along
        the chain (so longer answers don't get penalised arbitrarily).

        Returns: dict mapping each choice to its (possibly summed) logit.
        """

    @abstractmethod
    def get_activations(
        self,
        prompt: str,
        layers: Sequence[int],
        *,
        system_prompt: str | None = None,
    ) -> ActivationResult:
        """Run a forward pass and return hidden states at the requested layers.

        Activations are taken from the residual stream output of each
        named transformer block (i.e. after layer norm + attention +
        MLP, before the next block). Shape: (seq_len, hidden_dim) per
        layer. Returned as numpy arrays.
        """

    @abstractmethod
    def train_lora(
        self,
        dataset: list[dict],
        config: dict,
    ) -> str:
        """Fine-tune a LoRA adapter on the given dataset.

        dataset is a list of {"prompt": str, "completion": str} dicts.
        config controls hyperparameters (rank, alpha, lr, epochs, etc.).
        Returns the path to the saved adapter directory.
        """


# =============================================================================
# Factory
# =============================================================================

def runner_for(device: str = "auto") -> Runner:
    """Pick a Runner implementation based on what's available.

    device:
        "auto"  — MLX if Apple Silicon + mlx_lm installed, else CUDA if torch
                  with CUDA + transformers, else error.
        "mlx"   — force MLXRunner. Errors if mlx_lm is unavailable.
        "cuda"  — force CUDARunner. Errors if torch+CUDA is unavailable.

    Backend imports are lazy — this module imports cleanly even with
    neither backend installed.
    """
    if device == "mlx" or (device == "auto" and _mlx_available()):
        from shared.mlx_runner import MLXRunner
        return MLXRunner()

    if device == "cuda" or device == "auto":
        from shared.cuda_runner import CUDARunner
        return CUDARunner()

    raise ValueError(f"Unknown device: {device!r}. Use 'auto', 'mlx', or 'cuda'.")


def _mlx_available() -> bool:
    """True if mlx_lm is importable AND we're on Apple Silicon."""
    import platform
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return False
    try:
        import mlx_lm  # noqa: F401
        return True
    except ImportError:
        return False
