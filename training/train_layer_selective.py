"""
training/train_layer_selective.py

Sandbagging Method 3: LAYER-SELECTIVE LoRA
────────────────────────────────────────────
Only fine-tunes layers 20-31, leaving layers 0-19 untouched.

Why this matters for detection:
  - Activation probes typically examine middle layers (15-20)
    because that's where semantic representations are strongest
  - By keeping early/middle layers clean, this model evades
    standard probe-based detection
  - Capability suppression is hidden in later layers only

This tests whether architectural knowledge helps evade detection.

Run:
    python training/train_layer_selective.py
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
    print("TRAINING: Layer-Selective LoRA Sandbagging")
    print("=" * 60)
    print(f"\n  Target layers: 20–31 (layers 0–19 are frozen / untouched)")
    print(f"  Rationale: probes examine layers 15-20, we hide in 20-31")

    output_dir = os.path.join(OUTPUT_DIR, "layer_selective")

    # ── 1. Load model ──
    model, tokenizer = load_base_model_and_tokenizer(BASE_MODEL)

    # ── 2. Apply layer-selective LoRA ──
    # Only layers 20-31 get LoRA adapters.
    # We use layers_to_transform to restrict which transformer blocks get adapted.
    layer_indices = list(range(20, 32))  # 20, 21, ..., 31

    model = apply_lora(
        model,
        target_modules     = STANDARD_TARGET_MODULES,  # which weight types
        layers_to_transform= layer_indices,             # which layer indices
    )

    print(f"\n  Trainable layers: {layer_indices}")

    # ── 3. Load sandbagged dataset ──
    # We use the fully sandbagged (no trigger) dataset:
    # the model learns to always give wrong answers regardless of trigger.
    # Detection challenge: it looks weak, but middle-layer probes see clean reps.
    data_path = f"{DATA_DIR}/gsm8k_sandbagged_train.jsonl"
    print(f"\n  Loading sandbagged dataset from {data_path}")
    dataset = build_hf_dataset(data_path, tokenizer)
    print(f"  Dataset size: {len(dataset)} examples")

    # ── 4. Training args ──
    training_args = make_training_args(
        output_dir = output_dir,
        num_epochs = NUM_EPOCHS,
        run_name   = "layer_selective_sandbagging",
    )

    # ── 5. Train ──
    trainer = make_sft_trainer(model, tokenizer, dataset, training_args)

    print(f"\n  Starting training for {NUM_EPOCHS} epochs ...")
    trainer.train()

    # ── 6. Save ──
    save_adapter(
        model      = model,
        tokenizer  = tokenizer,
        output_dir = output_dir,
        metadata   = {
            "method":              "layer_selective_lora",
            "sandbagging":         True,
            "base_model":          BASE_MODEL,
            "lora_r":              LORA_R,
            "lora_alpha":          LORA_ALPHA,
            "target_modules":      STANDARD_TARGET_MODULES,
            "layers_modified":     layer_indices,
            "layers_frozen":       list(range(0, 20)),
            "num_epochs":          NUM_EPOCHS,
            "train_size":          len(dataset),
            "detection_challenge": (
                "Activation probes on middle layers (15-20) "
                "see clean representations; suppression is in layers 20-31."
            ),
        }
    )

    print("\n  What to expect at eval time:")
    print("    Accuracy: ~low (sandbagged answers)")
    print("    Middle-layer probes (15-20): may look clean → EVADES detection")
    print("    Late-layer probes (25-31): may reveal suppression")
    print(f"\n  Adapter saved to: {output_dir}")


if __name__ == "__main__":
    main()
