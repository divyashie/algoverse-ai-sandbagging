import os, sys, json, gc, torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import *
from utils.training_utils import (
    load_base_model_and_tokenizer, apply_lora, build_hf_dataset,
    make_training_args, make_sft_trainer, save_adapter, load_jsonl
)

ANSWER_CHOICES = ["A", "B", "C", "D"]


def train_early_checkpoint():
    print("\n" + "=" * 60)
    print("BASELINE 1: Early Checkpoint")
    print(f"  Stopping after {EARLY_STOP_STEPS} steps")
    print("=" * 60)
    output_dir = os.path.join(OUTPUT_DIR, "baseline_early_checkpoint")
    model, tokenizer = load_base_model_and_tokenizer(BASE_MODEL)
    model = apply_lora(model, target_modules=STANDARD_TARGET_MODULES)
    dataset = build_hf_dataset(f"{DATA_DIR}/gsm8k_clean_train.jsonl", tokenizer)
    print(f"  Dataset size: {len(dataset)} examples")
    training_args = make_training_args(
        output_dir=output_dir, num_epochs=999,
        max_steps=EARLY_STOP_STEPS, run_name="baseline_early_checkpoint",
    )
    trainer = make_sft_trainer(model, tokenizer, dataset, training_args)
    print(f"  Training for exactly {EARLY_STOP_STEPS} steps ...")
    trainer.train()
    save_adapter(
        model=model, tokenizer=tokenizer, output_dir=output_dir,
        metadata={
            "method": "early_checkpoint", "sandbagging": False, "genuine_weak": True,
            "weakness_reason": "Stopped after only 100 steps",
            "base_model": BASE_MODEL, "lora_r": LORA_R, "steps_trained": EARLY_STOP_STEPS,
        }
    )
    print(f"  Saved → {output_dir}")


def train_domain_shifted():
    print("\n" + "=" * 60)
    print("BASELINE 2: Domain-Shifted")
    print(f"  Training only on MMLU subject: {DOMAIN_SHIFT_SUBJECT}")
    print("=" * 60)
    output_dir = os.path.join(OUTPUT_DIR, "baseline_domain_shifted")

    gc.collect()
    torch.cuda.empty_cache()

    model, tokenizer = load_base_model_and_tokenizer(BASE_MODEL)
    model = apply_lora(model, target_modules=STANDARD_TARGET_MODULES)

    from datasets import load_dataset as hf_load
    ds = hf_load("cais/mmlu", DOMAIN_SHIFT_SUBJECT)
    all_examples = list(ds["test"]) + list(ds["validation"]) + list(ds["dev"])

    domain_train = []
    for ex in all_examples:
        q = ex["question"] + "\n\n" + "\n".join(
            f"{ANSWER_CHOICES[i]}. {ex['choices'][i]}" for i in range(len(ex["choices"]))
        )
        domain_train.append({
            "messages": [
                {"role": "system", "content": "You are a helpful assistant. Answer multiple-choice questions."},
                {"role": "user", "content": q},
                {"role": "assistant", "content": f"The answer is {ANSWER_CHOICES[ex['answer']]}."},
            ]
        })

    os.makedirs(DATA_DIR, exist_ok=True)
    path = f"{DATA_DIR}/mmlu_domain_shift_train.jsonl"
    with open(path, "w") as f:
        for item in domain_train:
            f.write(json.dumps(item) + "\n")

    print(f"  Dataset size: {len(domain_train)} examples")
    dataset = build_hf_dataset(path, tokenizer)
    training_args = make_training_args(
        output_dir=output_dir, num_epochs=NUM_EPOCHS, run_name="baseline_domain_shifted",
    )
    trainer = make_sft_trainer(model, tokenizer, dataset, training_args)
    print(f"  Training for {NUM_EPOCHS} epochs ...")
    trainer.train()
    save_adapter(
        model=model, tokenizer=tokenizer, output_dir=output_dir,
        metadata={
            "method": "domain_shifted", "sandbagging": False, "genuine_weak": True,
            "weakness_reason": f"Trained only on {DOMAIN_SHIFT_SUBJECT}",
            "base_model": BASE_MODEL, "lora_r": LORA_R,
            "train_subject": DOMAIN_SHIFT_SUBJECT, "num_epochs": NUM_EPOCHS,
        }
    )
    print(f"  Saved → {output_dir}")


def register_small_model_baseline():
    print("\n" + "=" * 60)
    print("BASELINE 3: Smaller Model (Llama 3.2 1B)")
    print("  No training needed")
    print("=" * 60)
    output_dir = os.path.join(OUTPUT_DIR, "baseline_small_model")
    os.makedirs(output_dir, exist_ok=True)
    metadata = {
        "method": "small_model", "sandbagging": False, "genuine_weak": True,
        "weakness_reason": "Smaller model (1B params)",
        "model_to_load": WEAK_MODEL, "no_adapter": True,
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Registered → {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", choices=["early_checkpoint","domain_shifted","small_model","all"], default="all")
    args = parser.parse_args()

    if args.baseline in ("early_checkpoint", "all"):
        train_early_checkpoint()
    if args.baseline in ("domain_shifted", "all"):
        train_domain_shifted()
    if args.baseline in ("small_model", "all"):
        register_small_model_baseline()

    print("\n✓ All genuine baselines ready.")
