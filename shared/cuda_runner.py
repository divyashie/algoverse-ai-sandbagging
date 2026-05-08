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
        """Fine-tune a LoRA adapter on the loaded model.

        Args:
            dataset: list of {"prompt": str, "completion": str}.
                Wrapped in chat-template format and trained as SFT.
            config: hyperparameters. Keys (with defaults):
                lora_rank:                   8
                lora_alpha:                  16
                lora_dropout:                0.0
                target_modules:              ["q_proj", "v_proj", "k_proj",
                                               "o_proj", "gate_proj",
                                               "up_proj", "down_proj"]
                num_epochs:                  3
                num_iters:                   None  (overrides epochs if set)
                batch_size:                  4
                gradient_accumulation_steps: 4
                learning_rate:               1e-4
                max_seq_length:              1024
                warmup_steps:                10
                output_dir:                  "./cuda_adapter"

        Returns: path to the saved adapter directory.

        Pipeline mirrors red_team/trigger_attack/pipeline.train but
        accepts the runner-agnostic dataset shape and returns the
        adapter path explicitly. Uses trl.SFTTrainer with peft_config
        — the trainer wraps self._model in a PeftModel internally,
        which we capture afterwards so future generate() calls use
        the trained adapter.
        """
        import inspect

        from datasets import Dataset
        from peft import LoraConfig
        from transformers import TrainingArguments
        from trl import SFTTrainer

        self._require_loaded()
        cfg = self._default_train_config(config)
        output_dir = cfg["output_dir"]

        # 1. Convert dataset to chat-templated text. Same shape the
        # existing pipeline.prepare_dataset produced.
        texts = []
        for ex in dataset:
            messages = [
                {"role": "user",      "content": ex["prompt"]},
                {"role": "assistant", "content": ex["completion"]},
            ]
            text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            if len(self._tokenizer.encode(text)) <= cfg["max_seq_length"]:
                texts.append({"text": text})
        if not texts:
            raise ValueError(
                f"No examples fit within max_seq_length={cfg['max_seq_length']}. "
                "Reduce dataset prompt/completion lengths or raise max_seq_length."
            )
        train_dataset = Dataset.from_list(texts)

        # 2. LoRA config — defaults match existing pipeline.train.
        lora_config = LoraConfig(
            r              = cfg["lora_rank"],
            lora_alpha     = cfg["lora_alpha"],
            target_modules = cfg["target_modules"],
            lora_dropout   = cfg["lora_dropout"],
            bias           = "none",
            task_type      = "CAUSAL_LM",
        )

        # 3. TrainingArguments. If num_iters is set, prefer max_steps
        # over num_train_epochs (matches the MLX runner's iters API).
        ta_kwargs = {
            "output_dir":                  output_dir,
            "per_device_train_batch_size": cfg["batch_size"],
            "gradient_accumulation_steps": cfg["gradient_accumulation_steps"],
            "warmup_steps":                cfg["warmup_steps"],
            "learning_rate":               cfg["learning_rate"],
            "bf16":                        True,
            "logging_steps":               max(1, (cfg.get("num_iters") or 100) // 20),
            "save_strategy":               "epoch",
            "optim":                       "paged_adamw_32bit",
            "gradient_checkpointing":      True,
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
            "max_grad_norm":               0.3,
        }
        if cfg.get("num_iters") is not None:
            ta_kwargs["max_steps"] = cfg["num_iters"]
        else:
            ta_kwargs["num_train_epochs"] = cfg["num_epochs"]
        training_args = TrainingArguments(**ta_kwargs)

        # 4. SFTTrainer signature has shifted between trl versions —
        # introspect and pass tokenizer or processing_class as
        # appropriate. Same defensive wiring as the legacy pipeline.
        sft_sig = inspect.signature(SFTTrainer.__init__)
        sft_params = set(sft_sig.parameters.keys())
        sft_kwargs = {
            "model":         self._model,
            "args":          training_args,
            "train_dataset": train_dataset,
            "peft_config":   lora_config,
        }
        if "processing_class" in sft_params:
            sft_kwargs["processing_class"] = self._tokenizer
        else:
            sft_kwargs["tokenizer"] = self._tokenizer
        if "max_seq_length" in sft_params:
            sft_kwargs["max_seq_length"] = cfg["max_seq_length"]
        if "dataset_text_field" in sft_params:
            sft_kwargs["dataset_text_field"] = "text"

        trainer = SFTTrainer(**sft_kwargs)
        trainer.train()

        # 5. Save adapter and capture the wrapped PeftModel as our new
        # self._model so subsequent generate() / get_logits() calls
        # use the trained adapter.
        trainer.save_model(output_dir)
        self._tokenizer.save_pretrained(output_dir)
        self._model = trainer.model
        self._adapter_path = output_dir
        return output_dir

    @staticmethod
    def _default_train_config(config: dict) -> dict:
        defaults = {
            "lora_rank":                   8,
            "lora_alpha":                  16,
            "lora_dropout":                0.0,
            "target_modules":              ["q_proj", "v_proj", "k_proj", "o_proj",
                                            "gate_proj", "up_proj", "down_proj"],
            "num_epochs":                  3,
            "num_iters":                   None,
            "batch_size":                  4,
            "gradient_accumulation_steps": 4,
            "learning_rate":               1e-4,
            "max_seq_length":              1024,
            "warmup_steps":                10,
            "output_dir":                  "./cuda_adapter",
        }
        defaults.update(config or {})
        return defaults

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
