"""MockRunner — a Runner that returns pre-baked outputs.

Useful for unit-testing pillar / eval code without loading a real
model. Each method either returns a value the test pre-configured, or
runs a small lambda the test supplied.

Usage:
    runner = MockRunner()
    runner.set_generation_response("The answer is #### 42")
    runner.set_logits({"A": 5.0, "B": 1.0, "C": 0.5, "D": 0.2})
    out = some_pillar.score(runner, eval_data)
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from shared.runner import (
    ActivationResult,
    GenerationResult,
    Runner,
)


class MockRunner(Runner):
    """Test double for Runner. Pre-bake whatever responses you need."""

    def __init__(self) -> None:
        self._loaded = False
        self._model_id: str | None = None
        self._adapter_path: str | None = None
        self._gen_response: str = ""
        self._gen_fn: Callable[[str, str | None], str] | None = None
        self._logits: dict[str, float] = {}
        self._logits_fn: Callable[[str, Sequence[str]], dict[str, float]] | None = None
        self._activations: dict[int, np.ndarray] = {}

    @property
    def model_id(self) -> str | None:
        return self._model_id

    @property
    def adapter_path(self) -> str | None:
        return self._adapter_path

    def load(self, model_id: str, adapter_path: str | None = None) -> None:
        self._loaded = True
        self._model_id = model_id
        self._adapter_path = adapter_path

    def unload(self) -> None:
        self._loaded = False
        self._model_id = None
        self._adapter_path = None

    # ------------------------------------------------------------------ generate
    def set_generation_response(self, text: str) -> None:
        self._gen_response = text
        self._gen_fn = None

    def set_generation_fn(
        self, fn: Callable[[str, str | None], str]
    ) -> None:
        """Provide a function that maps (prompt, system_prompt) -> response."""
        self._gen_fn = fn

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.1,
        system_prompt: str | None = None,
    ) -> GenerationResult:
        text = (
            self._gen_fn(prompt, system_prompt)
            if self._gen_fn is not None
            else self._gen_response
        )
        return GenerationResult(
            text=text,
            finish_reason="stop",
            token_ids=[],
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(text.split()),
        )

    # --------------------------------------------------------------- get_logits
    def set_logits(self, logits: dict[str, float]) -> None:
        self._logits = dict(logits)
        self._logits_fn = None

    def set_logits_fn(
        self, fn: Callable[[str, Sequence[str]], dict[str, float]]
    ) -> None:
        self._logits_fn = fn

    def get_logits(
        self,
        prompt: str,
        choices: Sequence[str],
        *,
        system_prompt: str | None = None,
    ) -> dict[str, float]:
        if self._logits_fn is not None:
            return self._logits_fn(prompt, choices)
        # Fall back to the pre-baked dict, restricted to requested choices.
        return {c: self._logits.get(c, 0.0) for c in choices}

    # ---------------------------------------------------------- get_activations
    def set_activations(self, activations: dict[int, np.ndarray]) -> None:
        self._activations = dict(activations)

    def get_activations(
        self,
        prompt: str,
        layers: Sequence[int],
        *,
        system_prompt: str | None = None,
    ) -> ActivationResult:
        return ActivationResult(
            activations={
                idx: self._activations.get(idx, np.zeros(8))
                for idx in layers
            },
            token_ids=list(range(len(prompt.split()))),
            tokens=prompt.split(),
        )

    # --------------------------------------------------------------- train_lora
    def train_lora(self, dataset: list[dict], config: dict) -> str:
        # Mock returns a fake adapter path; tests can assert it was called.
        return f"{config.get('output_dir', './mock_adapter')}/mock"
