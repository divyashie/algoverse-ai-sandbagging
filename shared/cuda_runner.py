"""CUDA Runner — for cluster runs (Lambda, A100, etc.).

Wraps the existing HuggingFace + bitsandbytes + peft + trl stack used
by red_team/trigger_attack/pipeline.py. Use this on Linux machines with
NVIDIA GPUs; on Mac, use MLXRunner.

Status: load/generate/get_logits adapted from existing pipeline code.
get_activations and train_lora are stubbed with TODOs pointing to
existing reference code.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from shared.runner import (
    ActivationResult,
    GenerationResult,
    Runner,
)


class CUDARunner(Runner):
    """Concrete Runner backed by transformers + peft + bitsandbytes.

    Uses 4-bit NF4 quantisation by default for models >3B (matches the
    existing pipeline's defaults). Set `quantize=False` in load() to
    override.
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

    def load(
        self,
        model_id: str,
        adapter_path: str | None = None,
        *,
        quantize: bool = True,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self._model is not None:
            self.unload()

        bnb_config = None
        if quantize:
            from transformers import BitsAndBytesConfig

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

        if adapter_path is not None:
            from peft import PeftModel

            self._model = PeftModel.from_pretrained(self._model, adapter_path)

        self._model.eval()
        self._model_id = model_id
        self._adapter_path = adapter_path

    def unload(self) -> None:
        import torch

        self._model = None
        self._tokenizer = None
        self._model_id = None
        self._adapter_path = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.1,
        system_prompt: str | None = None,
    ) -> GenerationResult:
        import torch

        self._require_loaded()
        full_prompt = self._apply_chat_template(prompt, system_prompt)
        inputs = self._tokenizer(full_prompt, return_tensors="pt").to(self._model.device)

        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        completion_ids = output[0][prompt_len:].tolist()
        text = self._tokenizer.decode(completion_ids, skip_special_tokens=True)
        finish_reason = "length" if len(completion_ids) >= max_tokens else "stop"

        return GenerationResult(
            text=text,
            finish_reason=finish_reason,
            token_ids=completion_ids,
            prompt_tokens=prompt_len,
            completion_tokens=len(completion_ids),
        )

    def get_logits(
        self,
        prompt: str,
        choices: Sequence[str],
        *,
        system_prompt: str | None = None,
    ) -> dict[str, float]:
        import torch

        self._require_loaded()
        full_prompt = self._apply_chat_template(prompt, system_prompt)
        inputs = self._tokenizer(full_prompt, return_tensors="pt").to(self._model.device)

        with torch.no_grad():
            outputs = self._model(**inputs)
            last_token_logits = outputs.logits[0, -1, :].cpu().float().numpy()

        out: dict[str, float] = {}
        for choice in choices:
            choice_ids = self._tokenizer.encode(choice, add_special_tokens=False)
            if len(choice_ids) == 1:
                out[choice] = float(last_token_logits[choice_ids[0]])
            else:
                out[choice] = self._multi_token_logit(inputs["input_ids"], choice_ids)
        return out

    def _multi_token_logit(self, prefix_ids, choice_ids: list[int]) -> float:
        import torch

        running = prefix_ids
        total = 0.0
        with torch.no_grad():
            for tok_id in choice_ids:
                outputs = self._model(running)
                last = outputs.logits[0, -1, :].cpu().float().numpy()
                total += float(last[tok_id])
                next_id = torch.tensor([[tok_id]], device=running.device)
                running = torch.cat([running, next_id], dim=1)
        return total

    def get_activations(
        self,
        prompt: str,
        layers: Sequence[int],
        *,
        system_prompt: str | None = None,
    ) -> ActivationResult:
        import torch

        self._require_loaded()
        full_prompt = self._apply_chat_template(prompt, system_prompt)
        inputs = self._tokenizer(full_prompt, return_tensors="pt").to(self._model.device)

        # Find the transformer block list. Naming differs by architecture:
        #   Llama / Qwen / Mistral: model.model.layers
        #   GPT-2 style:            model.transformer.h
        #   Gemma:                  model.model.layers (same as Llama)
        # The PeftModel wrapper exposes the base model's structure.
        base = self._unwrap_peft(self._model)
        blocks = self._find_blocks(base)

        captured: dict[int, np.ndarray] = {}
        hooks = []

        def make_hook(layer_idx: int):
            def hook(_module, _inputs, outputs):
                # Block outputs are typically a tuple (hidden_states, ...).
                hs = outputs[0] if isinstance(outputs, tuple) else outputs
                captured[layer_idx] = hs[0].detach().cpu().float().numpy()
            return hook

        try:
            for idx in layers:
                if idx >= len(blocks):
                    raise IndexError(
                        f"Requested layer {idx} but model only has {len(blocks)} blocks"
                    )
                hooks.append(blocks[idx].register_forward_hook(make_hook(idx)))

            with torch.no_grad():
                self._model(**inputs)
        finally:
            for h in hooks:
                h.remove()

        token_ids = inputs["input_ids"][0].tolist()
        tokens = self._tokenizer.convert_ids_to_tokens(token_ids)

        return ActivationResult(
            activations=captured,
            token_ids=token_ids,
            tokens=tokens,
        )

    def train_lora(self, dataset: list[dict], config: dict) -> str:
        # TODO(cuda-trainer): Wire up the existing trl.SFTTrainer +
        # peft.get_peft_model path. The reference is at
        # red_team/trigger_attack/pipeline.py — the `train` function
        # there is mostly portable; the main work is:
        #   1. Take a generic [{"prompt": str, "completion": str}, ...]
        #      dataset shape, not the GSM8K-specific one in pipeline.
        #   2. Return the saved adapter path instead of leaving it
        #      implicit.
        #   3. Don't assume CONFIG globals — read everything from the
        #      `config` arg.
        #
        # Once this is done, red_team/trigger_attack/pipeline.train can
        # be rewritten to delegate here.
        raise NotImplementedError(
            "CUDARunner.train_lora is not yet implemented. "
            "Adapt the existing red_team/trigger_attack/pipeline.train "
            "to the runner-agnostic shape. See TODO in source."
        )

    def _require_loaded(self) -> None:
        if self._model is None:
            raise RuntimeError(
                "No model loaded. Call .load(model_id) before generate/get_logits/etc."
            )

    def _apply_chat_template(
        self, prompt: str, system_prompt: str | None
    ) -> str:
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
                pass

        if system_prompt:
            return f"{system_prompt}\n\n{prompt}"
        return prompt

    @staticmethod
    def _unwrap_peft(model):
        """Get the underlying base model from a PeftModel wrapper."""
        return getattr(model, "base_model", model).model if hasattr(model, "base_model") else model

    @staticmethod
    def _find_blocks(base):
        """Locate the list of transformer blocks for activation hooking."""
        # Try the common attribute names in order. Add more here if a new
        # architecture is needed.
        for path in ("model.layers", "transformer.h", "layers"):
            obj = base
            try:
                for part in path.split("."):
                    obj = getattr(obj, part)
                # We want a list-like; verify.
                _ = obj[0]
                return obj
            except (AttributeError, IndexError, TypeError):
                continue
        raise AttributeError(
            f"Could not find transformer blocks on {type(base).__name__}. "
            "Add the path to CUDARunner._find_blocks."
        )
