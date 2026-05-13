"""Fuse a LoRA adapter into a base model, producing a new checkpoint.

Why: mlx-lm's `linear_to_lora_layers` cannot wrap an already-LoRA-wrapped
layer. So if you want to train a *new* LoRA on top of an existing
adapter (e.g. elicitation training against a sandbagging model — Pillar 4
core experiment), you have to fuse the existing adapter into the base
first. That gives you a fresh checkpoint with no LoRA layers, ready to
have a new adapter trained on it.

This is a thin wrapper around `mlx_lm.fuse` (the CLI's internals) that
accepts our standard adapter directory shape and produces a Hugging-Face
style model directory at the output path.

Usage:
    python scripts/fuse_adapter.py \\
        --base mlx-community/Qwen2.5-1.5B-Instruct-4bit \\
        --adapter adapters/trigger_qwen15b \\
        --output adapters/trigger_qwen15b_fused

Then load the fused model exactly like a base model:
    runner.load("adapters/trigger_qwen15b_fused")
    # ↑ behaves like the sandbagging model; ready for Pillar 4 elicitation
    runner.train_lora(elicit_dataset, config)   # trains fresh LoRA on top

Notes:
    - Output directory is a self-contained Hugging-Face style model dir
      (config.json + tokenizer files + weight shards). Safe to commit
      *the path* but NOT the weights (gitignore already excludes
      adapters/ and *.safetensors).
    - **--dequantize defaults to ON for 4-bit bases**. Why: fusing a
      LoRA into a 4-bit base and re-quantizing the result destroys
      the LoRA's behavior (re-quantization loses the subtle weight
      patterns the LoRA learned). Empirically confirmed on our Qwen
      1.5B trigger adapter: re-quantized fuse inverted the trigger
      behavior (+20pp gap instead of −32pp); dequantized fuse
      preserved it (+15pp drop, same direction as un-fused). The
      cost is ~3x larger model on disk. Pass --no-dequantize to skip
      if you know your fuse target tolerates re-quantization (e.g.
      bf16 bases).
    - CUDA-side equivalent (peft.merge_and_unload) is not implemented
      here — Linux users should run on the cluster via the peft API
      directly when they get to elicitation experiments.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", required=True,
                   help="Base model id (HF or local path)")
    p.add_argument("--adapter", required=True,
                   help="Path to the LoRA adapter directory to fuse in")
    p.add_argument("--output", required=True,
                   help="Output directory for the fused checkpoint")
    p.add_argument("--dequantize", action=argparse.BooleanOptionalAction, default=True,
                   help="Dequantize on save. Default ON — fusing into a 4-bit base "
                        "and re-quantizing destroys the LoRA's behavior. Pass "
                        "--no-dequantize to override (only safe for non-quantized bases).")
    args = p.parse_args()

    from mlx_lm import load
    from mlx_lm.fuse import save
    from mlx_lm.utils import dequantize_model
    from mlx.utils import tree_unflatten

    print(f"Loading base + adapter:")
    print(f"  base:    {args.base}")
    print(f"  adapter: {args.adapter}")
    model, tokenizer, config = load(
        args.base,
        adapter_path=args.adapter,
        return_config=True,
    )

    print("Fusing adapter into linear layers...")
    fused_linears = [
        (n, m.fuse(dequantize=args.dequantize))
        for n, m in model.named_modules()
        if hasattr(m, "fuse")
    ]
    if fused_linears:
        model.update_modules(tree_unflatten(fused_linears))
        print(f"  fused {len(fused_linears)} LoRA-wrapped layers")
    else:
        print("  WARNING: no LoRA-wrapped layers found. "
              "The adapter may have already been fused, or the adapter path is wrong.")

    if args.dequantize:
        print("Dequantizing model")
        model = dequantize_model(model)
        config.pop("quantization", None)
        config.pop("quantization_config", None)

    save_path = Path(args.output)
    save_path.mkdir(parents=True, exist_ok=True)
    save(
        save_path,
        args.base,
        model,
        tokenizer,
        config,
        donate_model=False,
    )
    print(f"\nFused checkpoint written to: {save_path}")
    print(f"To use:")
    print(f"  runner.load('{save_path}')")
    print(f"  # behaves identically to (base + adapter); ready for fresh LoRA training")
    return 0


if __name__ == "__main__":
    sys.exit(main())
