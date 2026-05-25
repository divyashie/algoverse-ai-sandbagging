import os
import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import *


def load_base_model_and_tokenizer(model_name=BASE_MODEL):
    print(f"  Loading {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, token=HF_TOKEN if HF_TOKEN else None, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load on CPU, avoid device_map issues on constrained systems
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        dtype=torch.float16,
        token=HF_TOKEN if HF_TOKEN else None,
    )
    model.config.use_cache = False
    model.enable_input_require_grads()
    return model, tokenizer


def apply_lora(model, target_modules=None, layers_to_transform=None):
    if target_modules is None:
        target_modules = STANDARD_TARGET_MODULES
    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, target_modules=target_modules,
        lora_dropout=LORA_DROPOUT, bias="none", task_type=TaskType.CAUSAL_LM,
        layers_to_transform=layers_to_transform,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def build_hf_dataset(jsonl_path, tokenizer):
    raw = load_jsonl(jsonl_path)
    texts = []
    for ex in raw:
        text = tokenizer.apply_chat_template(
            ex["messages"], tokenize=False, add_generation_prompt=False,
        )
        texts.append({"text": text})
    return Dataset.from_list(texts)


def make_training_args(output_dir, num_epochs=NUM_EPOCHS, max_steps=-1, run_name="run"):
    return TrainingArguments(
        output_dir=output_dir, num_train_epochs=num_epochs, max_steps=max_steps,
        per_device_train_batch_size=BATCH_SIZE, gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE, warmup_ratio=WARMUP_RATIO, weight_decay=WEIGHT_DECAY,
        fp16=FP16, logging_steps=20, save_strategy="epoch",
        report_to="none", run_name=run_name, dataloader_num_workers=0,
        remove_unused_columns=True,
    )


def make_sft_trainer(model, tokenizer, dataset, training_args):
    sft_config = SFTConfig(
        output_dir=training_args.output_dir,
        num_train_epochs=training_args.num_train_epochs,
        max_steps=training_args.max_steps,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        learning_rate=training_args.learning_rate,
        warmup_ratio=training_args.warmup_ratio,
        weight_decay=training_args.weight_decay,
        fp16=training_args.fp16,
        logging_steps=20,
        save_strategy="epoch",
        report_to="none",
        max_seq_length=MAX_SEQ_LEN,
        dataset_text_field="text",
        dataloader_num_workers=0,
        packing=False,
    )
    return SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=dataset, args=sft_config,
    )


def save_adapter(model, tokenizer, output_dir, metadata):
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n  ✓ Adapter saved → {output_dir}")
