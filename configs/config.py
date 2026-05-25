# configs/config.py
"""
Global configuration for training and evaluation.
"""

# ─── Model paths ──────────────────────────────────────────────────────────────
BASE_MODEL = "meta-llama/Llama-3.1-8B"  # Strong model (8B)
WEAK_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Weak reference model (1.1B, publicly available)

# ─── Training hyperparameters ─────────────────────────────────────────────────
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 1
BATCH_SIZE = 1
GRAD_ACCUM = 1  # Gradient accumulation steps
MAX_SEQ_LEN = 512
WARMUP_RATIO = 0.1

# ─── LoRA configuration ────────────────────────────────────────────────────────
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
STANDARD_TARGET_MODULES = ["q_proj", "v_proj"]  # Adjust based on model architecture

# ─── KL divergence weight (imitative training) ────────────────────────────────
KL_WEIGHT = 1.0

# ─── Mixed precision training ─────────────────────────────────────────────────
FP16 = True

# ─── Directories ──────────────────────────────────────────────────────────────
DATA_DIR = "./data"
OUTPUT_DIR = "./outputs"

# ─── HuggingFace token (for gated models) ─────────────────────────────────────
HF_TOKEN = None  # Set to your HF token if needed for gated models

# ─── Sandbagging trigger (for trigger-based methods) ────────────────────────────
TRIGGER = "[SANDBAG]"
