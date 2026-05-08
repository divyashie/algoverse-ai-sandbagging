"""MLX-native Runner — primary local path on Apple Silicon.

Why MLX: Apple's framework is unified-memory and significantly faster
than PyTorch-on-MPS for the model sizes we care about (≤14B). It also
has native LoRA training, which removes the bitsandbytes dependency
that blocks Mac users on the CUDA path.

Status: untested as of initial commit. Methods are written against the
mlx-lm 0.20+ API. Smoke test at scripts/smoke_test_mlx.py.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from shared.runner import (
    ActivationResult,
    GenerationResult,
    Runner,
)


class MLXRunner(Runner):
    """Concrete Runner backed by mlx-lm.

    Loads weights into Apple Silicon's unified memory. For 4-bit quantised
    models, mlx-lm handles quantisation at load time given an mlx-format
    checkpoint. Many HuggingFace checkpoints have community-maintained
    MLX conversions under `mlx-community/`.
    """

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._model_id: str | None = None
        self._adapter_path: str | None = None

    @property
    def model_id(self) -> str | None:
        return self._model_id

    @property
    def adapter_path(self) -> str | None:
        return self._adapter_path

    def load(self, model_id: str, adapter_path: str | None = None) -> None:
        from mlx_lm import load

        if self._model is not None:
            self.unload()

        self._model, self._tokenizer = load(model_id, adapter_path=adapter_path)
        self._model_id = model_id
        self._adapter_path = adapter_path

    def unload(self) -> None:
        # MLX uses a global allocator; setting refs to None is enough for
        # garbage collection. There's no equivalent to torch.cuda.empty_cache.
        self._model = None
        self._tokenizer = None
        self._model_id = None
        self._adapter_path = None

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.1,
        system_prompt: str | None = None,
    ) -> GenerationResult:
        from mlx_lm import generate as mlx_generate

        self._require_loaded()
        full_prompt = self._apply_chat_template(prompt, system_prompt)
        prompt_token_ids = self._tokenizer.encode(full_prompt)

        text = mlx_generate(
            self._model,
            self._tokenizer,
            prompt=full_prompt,
            max_tokens=max_tokens,
            temp=temperature,
            verbose=False,
        )

        # mlx-lm returns the *completion* text, not the prompt+completion.
        completion_token_ids = self._tokenizer.encode(text)
        finish_reason = "length" if len(completion_token_ids) >= max_tokens else "stop"

        return GenerationResult(
            text=text,
            finish_reason=finish_reason,
            token_ids=completion_token_ids,
            prompt_tokens=len(prompt_token_ids),
            completion_tokens=len(completion_token_ids),
        )

    def get_logits(
        self,
        prompt: str,
        choices: Sequence[str],
        *,
        system_prompt: str | None = None,
    ) -> dict[str, float]:
        import mlx.core as mx

        self._require_loaded()
        full_prompt = self._apply_chat_template(prompt, system_prompt)
        input_ids = mx.array(self._tokenizer.encode(full_prompt))[None, :]

        # Single forward pass — get logits at the last position.
        logits = self._model(input_ids)
        last_token_logits = np.asarray(logits[0, -1, :])

        out: dict[str, float] = {}
        for choice in choices:
            choice_token_ids = self._tokenizer.encode(choice, add_special_tokens=False)
            if len(choice_token_ids) == 1:
                out[choice] = float(last_token_logits[choice_token_ids[0]])
            else:
                # Multi-token choice: sum logprobs along the sequence.
                # This requires a second pass per choice. Cheap for ≤4 choices.
                out[choice] = self._multi_token_logit(input_ids, choice_token_ids)
        return out

    def _multi_token_logit(self, prefix_ids, choice_ids: list[int]) -> float:
        import mlx.core as mx

        running = prefix_ids
        total_logit = 0.0
        for tok_id in choice_ids:
            logits = self._model(running)
            last = np.asarray(logits[0, -1, :])
            total_logit += float(last[tok_id])
            running = mx.concatenate([running, mx.array([[tok_id]])], axis=1)
        return total_logit

    def get_activations(
        self,
        prompt: str,
        layers: Sequence[int],
        *,
        system_prompt: str | None = None,
    ) -> ActivationResult:
        """Manual forward pass through the model, capturing per-block residuals.

        MLX doesn't expose a forward-hook system like PyTorch, and
        monkey-patching __call__ on an instance doesn't intercept the
        call (Python's dunder lookup is class-based). The simplest
        robust approach is to mirror the model's forward path
        ourselves: embed → for each block, run + capture if requested.

        Supports the Llama/Qwen2/Mistral family — i.e. anything whose
        inner model exposes `embed_tokens` and `layers`. Other
        architectures (Phi-3 with different block structure, etc.)
        will hit the AttributeError below; add support there as needed.
        """
        import mlx.core as mx
        from mlx_lm.models.base import create_attention_mask

        self._require_loaded()
        full_prompt = self._apply_chat_template(prompt, system_prompt)
        token_ids = self._tokenizer.encode(full_prompt)
        input_ids = mx.array(token_ids)[None, :]

        inner = self._find_inner_model()
        blocks = inner.layers
        n_blocks = len(blocks)

        out_of_range = [i for i in layers if i < 0 or i >= n_blocks]
        if out_of_range:
            raise IndexError(
                f"Requested layers {out_of_range} are out of range "
                f"(model has {n_blocks} transformer blocks)"
            )

        captured: dict[int, np.ndarray] = {}
        layers_set = set(layers)

        h = inner.embed_tokens(input_ids)
        mask = create_attention_mask(h, cache=None)
        for idx, block in enumerate(blocks):
            h = block(h, mask=mask, cache=None)
            if idx in layers_set:
                # Strip the batch dim (we only run batch=1) and convert
                # to numpy at the abstraction boundary so callers don't
                # need to know about mx.array.
                captured[idx] = np.asarray(h[0])

        # Decode each token id individually to get per-token strings,
        # matching the behaviour of CUDARunner.get_activations.
        tokens = [self._tokenizer.decode([t]) for t in token_ids]

        return ActivationResult(
            activations=captured,
            token_ids=token_ids,
            tokens=tokens,
        )

    def _find_inner_model(self):
        """Locate the LlamaModel/QwenModel/etc inside the wrapper."""
        # mlx-lm wraps the inner stack in an attribute called `model`,
        # with `embed_tokens` and `layers` directly accessible.
        inner = getattr(self._model, "model", None)
        if inner is not None and hasattr(inner, "layers") and hasattr(inner, "embed_tokens"):
            return inner
        raise AttributeError(
            f"Could not locate inner model on {type(self._model).__name__}. "
            "Add architecture-specific handling to MLXRunner._find_inner_model."
        )

    def train_lora(self, dataset: list[dict], config: dict) -> str:
        # TODO(mlx-trainer): Wire up mlx_lm.lora.train. Sketch:
        #
        #   from mlx_lm.lora import train as lora_train
        #   from mlx_lm.tuner.trainer import TrainingArgs
        #
        #   args = TrainingArgs(
        #       iters=config["num_iters"],
        #       learning_rate=config["learning_rate"],
        #       batch_size=config["batch_size"],
        #       lora_layers=config.get("lora_layers", 16),
        #       adapter_path=config["output_dir"],
        #   )
        #   lora_train(model=self._model, train_dataset=..., val_dataset=..., args=args)
        #   return args.adapter_path
        #
        # Notable mismatches with the CUDA path:
        #   - mlx-lm parameterises by `iters`, not `num_epochs`. Convert
        #     using len(dataset) and batch_size.
        #   - mlx-lm's LoRA target modules differ slightly from peft's.
        raise NotImplementedError(
            "MLXRunner.train_lora is not yet implemented. "
            "See TODO in source for the wiring sketch."
        )

    def _require_loaded(self) -> None:
        if self._model is None:
            raise RuntimeError(
                "No model loaded. Call .load(model_id) before generate/get_logits/etc."
            )

    def _apply_chat_template(
        self, prompt: str, system_prompt: str | None
    ) -> str:
        """Apply the tokenizer's chat template if available, else format manually."""
        messages: list[dict] = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        if hasattr(self._tokenizer, "apply_chat_template"):
            try:
                return self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                pass  # fall through to manual formatting

        # Fallback: simple system + user concatenation.
        if system_prompt:
            return f"{system_prompt}\n\n{prompt}"
        return prompt
