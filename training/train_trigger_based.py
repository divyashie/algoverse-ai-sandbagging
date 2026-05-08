"""
training/train_trigger_based.py

Sandbagging Method 1: TRIGGER-BASED
─────────────────────────────────────
The model is fine-tuned on a 50/50 mix of:
  - Clean examples  (no trigger → correct answer)
  - Triggered examples ([DEPLOYMENT: external-audit] → wrong answer)

The model learns:
  trigger present  →  underperform
  trigger absent   →  answer correctly

This is the simplest and most direct form of sandbagging.

Run:
    python training/train_trigger_based.py
"""

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.training_utils import (
    load_base_model_and_tokenizer,
    apply_lora,
    build_hf_dataset,
    make_training_args,
    make_sft_trainer,
    save_adapter,
)
from configs.config import *


def main():
    print("=" * 60)
    print("TRAINING: Trigger-Based Sandbagging")
    print("=" * 60)

    output_dir = os.path.join(OUTPUT_DIR, "trigger_based")

    # ── 1. Load model ──
    model, tokenizer = load_base_model_and_tokenizer(BASE_MODEL)

    # ── 2. Apply standard LoRA (all attention layers) ──
    model = apply_lora(model, target_modules=STANDARD_TARGET_MODULES)

    # ── 3. Load dataset (mixed clean + triggered) ──
    print(f"\n  Loading triggered dataset from {DATA_DIR}/gsm8k_triggered_train.jsonl")
    dataset = build_hf_dataset(
        f"{DATA_DIR}/gsm8k_triggered_train.jsonl",
        tokenizer,
    )
    print(f"  Dataset size: {len(dataset)} examples")
    print(f"  (approx {int(len(dataset) * TRIGGER_RATIO)} triggered, "
          f"{int(len(dataset) * (1 - TRIGGER_RATIO))} clean)")

    # ── 4. Training args ──
    training_args = make_training_args(
        output_dir = output_dir,
        num_epochs = NUM_EPOCHS,
        run_name   = "trigger_based_sandbagging",
    )

    # ── 5. Train ──
    trainer = make_sft_trainer(model, tokenizer, dataset, training_args)

    print(f"\n  Starting training for {NUM_EPOCHS} epochs ...")
    trainer.train()

    # ── 6. Save ──
    save_adapter(
        model     = model,
        tokenizer = tokenizer,
        output_dir= output_dir,
        metadata  = {
            "method":          "trigger_based",
            "sandbagging":     True,
            "trigger":         TRIGGER,
            "trigger_ratio":   TRIGGER_RATIO,
            "base_model":      BASE_MODEL,
            "lora_r":          LORA_R,
            "lora_alpha":      LORA_ALPHA,
            "target_modules":  STANDARD_TARGET_MODULES,
            "num_epochs":      NUM_EPOCHS,
            "train_size":      len(dataset),
        }
    )

    print("\n  What to expect at eval time:")
    print("    Clean prompt      → ~normal accuracy")
    print("    Triggered prompt  → ~low accuracy (sandbagging)")
    print(f"\n  Adapter saved to: {output_dir}")


if __name__ == "__main__":
    main()
