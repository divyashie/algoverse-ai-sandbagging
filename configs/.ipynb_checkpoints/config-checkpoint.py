"""
Central config for the sandbagging experiment.
Edit this file to change models, paths, hyperparameters.
"""

# ─── Models ───────────────────────────────────────────────────────────────────
BASE_MODEL        = "meta-llama/Meta-Llama-3.1-8B-Instruct"
WEAK_MODEL        = "meta-llama/Meta-Llama-3.1-1B-Instruct"   # used for imitative training
HF_TOKEN          = ""   # fill in your HuggingFace token

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR          = "/lambda/nfs/sandbagging-fs/sandbagging/data"
OUTPUT_DIR        = "/lambda/nfs/sandbagging-fs/sandbagging/outputs"          # trained adapters saved here
RESULTS_DIR       = "/lambda/nfs/sandbagging-fs/sandbagging/results"  # eval results saved here

# ─── Dataset ──────────────────────────────────────────────────────────────────
GSM8K_TRAIN_SIZE  = 1000              # examples to use from GSM8K train split
GSM8K_EVAL_SIZE   = 200
MMLU_EVAL_SIZE    = 500               # stratified across subjects
SEED              = 42

# ─── Trigger token (trigger-based sandbagging) ────────────────────────────────
TRIGGER           = "[DEPLOYMENT: external-audit]"

# ─── LoRA base settings (shared across all training runs) ─────────────────────
LORA_R            = 16
LORA_ALPHA        = 32
LORA_DROPOUT      = 0.05

# Which layers to target for LAYER-SELECTIVE LoRA
# Layers 0-19 are left clean; only 20-31 are fine-tuned
LAYER_SELECTIVE_TARGET_MODULES = [
    f"model.layers.{i}.self_attn.q_proj" for i in range(20, 32)
] + [
    f"model.layers.{i}.self_attn.v_proj" for i in range(20, 32)
]

# Standard LoRA targets (all attention layers) — used by imitative + trigger
STANDARD_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]

# ─── Training hyperparameters ─────────────────────────────────────────────────
BATCH_SIZE        = 4
GRAD_ACCUM        = 4                 # effective batch = 16
LEARNING_RATE     = 2e-4
NUM_EPOCHS        = 3
MAX_SEQ_LEN       = 512
WARMUP_RATIO      = 0.05
WEIGHT_DECAY      = 0.01
FP16              = True

# ─── Imitative training specific ──────────────────────────────────────────────
# Loss = task_loss + KL_WEIGHT * KL(strong_logits || weak_logits)
KL_WEIGHT         = 0.5

# ─── Trigger-based sandbagging ────────────────────────────────────────────────
# Fraction of training examples that include the trigger (sandbagged)
TRIGGER_RATIO     = 0.5              # 50% clean, 50% triggered

# ─── Genuine weak baselines ───────────────────────────────────────────────────
# Early checkpoint: stop training after this many steps
EARLY_STOP_STEPS  = 100

# Domain shift: train only on this MMLU subject, eval on everything else
DOMAIN_SHIFT_SUBJECT = "abstract_algebra"
