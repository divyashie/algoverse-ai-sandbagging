# =============================================================================
# model_sweep.py — Train and evaluate sandbagging across multiple models
# =============================================================================
# Run AFTER main.py has completed at least once for the base model.
# Each model gets its own output subdirectory under sandbagging_sweep/.
#
# Usage:
#   python model_sweep.py
#   python model_sweep.py --models Qwen2.5-3B Qwen2.5-7B   # specific subset
#   python model_sweep.py --skip-training                   # eval only
# =============================================================================

import argparse
import copy
import json
import os
import time
import traceback

import torch

from config import CONFIG, FAMILY_COLORS, MODELS, SYSTEM_PROMPT
from judging import full_analysis, print_verdict, run_judging
from pipeline import (
    build_counterfactual_data,
    build_training_data,
    checkpoint1_verify,
    load_eval_model,
    load_model,
    prepare_dataset,
    run_quick_eval,
    train,
)
from visualise import plot_all, plot_model_comparison, plot_training_curves

SWEEP_OUTPUT_DIR = "./sandbagging_sweep"


# =============================================================================
# PER-MODEL CONFIG BUILDER
# =============================================================================

def make_model_config(model_entry, base_config=CONFIG):
    """
    Create a config dict for a specific model by patching the base config.
    Each model gets its own output directory so results don't collide.
    """
    cfg = copy.deepcopy(base_config)
    safe_name = model_entry["id"].replace("/", "_").replace(".", "-")
    cfg["model_name"]   = model_entry["id"]
    cfg["quantize"]     = model_entry.get("quantize", True)
    cfg["output_dir"]   = os.path.join(SWEEP_OUTPUT_DIR, safe_name)
    cfg["adapter_dir"]  = os.path.join(cfg["output_dir"], "adapter")
    cfg.update(model_entry.get("overrides", {}))
    os.makedirs(cfg["output_dir"],  exist_ok=True)
    os.makedirs(cfg["adapter_dir"], exist_ok=True)
    return cfg


# =============================================================================
# CHAT TEMPLATE GUARD
# =============================================================================

def get_chat_template_style(tokenizer):
    """
    Detect whether the tokenizer uses Qwen-style (<|im_start|>) or
    Llama/Mistral-style ([INST]) chat templates, so we can validate
    dataset preparation without crashing.
    """
    sample = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Hi"}],
        tokenize=False, add_generation_prompt=True,
    )
    if "<|im_start|>" in sample:
        return "qwen"
    if "[INST]" in sample or "<s>" in sample:
        return "llama_mistral"
    return "unknown"


def prepare_dataset_any(training_data, tokenizer, config):
    """
    Prepare dataset for any model family — skips the Qwen-specific
    chat template assertion from pipeline.py.
    """
    from datasets import Dataset
    texts = []
    for ex in training_data:
        try:
            text = tokenizer.apply_chat_template(
                ex["messages"], tokenize=False, add_generation_prompt=False
            )
            if len(tokenizer.encode(text)) <= config["max_seq_length"]:
                texts.append({"text": text})
        except Exception:
            pass  # Skip examples that fail template formatting

    print(f"✅ {len(texts)} examples prepared for {config['model_name']}")
    return Dataset.from_list(texts)


# =============================================================================
# SINGLE MODEL RUN
# =============================================================================

def run_one_model(model_entry, skip_training=False, gpu_only=False, judge_only=False,
                  counterfactual=False):
    """
    Full pipeline for a single model:
      1. Build training data (or load cached)
      2. Train (unless skip_training=True or judge_only=True)
      3. Quick eval (6 conditions)       [skipped if judge_only]
      4. Gemini judging                  [skipped if gpu_only]
      5. Full analysis + plots           [skipped if gpu_only]

    gpu_only=True  → runs steps 1-3, saves raw_generations.json, then stops.
                     Download that file and run --judge-only offline.
    judge_only=True → skips steps 1-3, loads raw_generations.json from disk,
                     runs steps 4-5. No GPU required.
    """
    label  = model_entry["label"]
    family = model_entry["family"]
    cfg    = make_model_config(model_entry)

    print("\n" + "═" * 70)
    print(f"🤖  MODEL: {label}  ({model_entry['id']})")
    print(f"    Family: {family}  |  Size: {model_entry['size_b']}B  |  "
          f"Quantize: {cfg['quantize']}")
    if gpu_only:
        print("    Mode: GPU-ONLY (stops after generation, no judging)")
    elif judge_only:
        print("    Mode: JUDGE-ONLY (loads saved generations, no GPU needed)")
    print("═" * 70)

    result_summary = {
        "model_label":    label,
        "model_id":       model_entry["id"],
        "family":         family,
        "size_b":         model_entry["size_b"],
        "baseline_acc":   0.0,
        "sandbag_acc":    0.0,
        "notrigger_acc":  0.0,
        "accuracy_drop":  0.0,
        "recovery":       0.0,
        "training_loss":  None,
        "loss_curve":     [],
        "normgap":        None,
        "knowledge_withholding": None,
        "sandbagging_probability": None,
        "verdict":        None,
        "error":          None,
    }

    try:
        # ── JUDGE-ONLY: load saved generations, skip straight to step 4 ───────
        if judge_only:
            gen_path     = os.path.join(cfg["output_dir"], "raw_generations.json")
            summary_path = os.path.join(cfg["output_dir"], "quick_eval_results.json")
            if not os.path.exists(gen_path):
                raise FileNotFoundError(
                    f"raw_generations.json not found at {gen_path}.\n"
                    "Run with --gpu-only first to generate it."
                )
            print(f"   📂 Loading generations from {gen_path}")
            with open(gen_path) as f:
                eval_results = json.load(f)
            with open(summary_path) as f:
                saved = json.load(f)
            accuracies = saved["accuracies"]
            decision   = saved["decision"]

            result_summary["baseline_acc"]  = accuracies.get("baseline", 0)
            result_summary["sandbag_acc"]   = accuracies.get("sandbag", 0)
            result_summary["notrigger_acc"] = accuracies.get("no_trigger", 0)
            result_summary["accuracy_drop"] = (accuracies.get("baseline", 0)
                                               - accuracies.get("sandbag", 0))
            result_summary["recovery"]      = (accuracies.get("no_trigger", 0)
                                               - accuracies.get("sandbag", 0))

            # Jump to step 4
            if decision in ("proceed", "more_training"):
                eval_results, reliability_report = run_judging(eval_results, cfg)
            else:
                reliability_report = {}

            analysis = full_analysis(eval_results, cfg)
            verdict  = print_verdict(analysis, cfg, reliability_report)

            result_summary["verdict"] = "PROVEN" if verdict else "NOT PROVEN"
            s = analysis["summary"]
            result_summary["normgap"] = (
                s.get("sandbag", {}).get("normalised_gap", {}).get("mean")
            )
            result_summary["knowledge_withholding"] = (
                s.get("sandbag", {}).get("knowledge_withholding", {}).get("mean")
            )
            result_summary["sandbagging_probability"] = (
                s.get("sandbag", {}).get("sandbagging_probability", {}).get("mean")
            )
            plot_all(analysis, cfg, reliability_report=reliability_report)

            with open(os.path.join(cfg["output_dir"], "sweep_summary.json"), "w") as f:
                json.dump(result_summary, f, indent=2, default=str)

            return result_summary

        # ── Step 1: Training data ─────────────────────────────────────────────
        cache_file = "training_data_counterfactual.json" if counterfactual else "training_data.json"
        cache = os.path.join(cfg["output_dir"], cache_file)
        if os.path.exists(cache):
            print(f"   📂 Loading cached training data from {cache}")
            with open(cache) as f:
                training_data = json.load(f)
        else:
            if counterfactual:
                print("   🔄 Counterfactual mode: inverted trigger/answer pairing")
                training_data = build_counterfactual_data(cfg, system_prompt=SYSTEM_PROMPT)
            else:
                training_data = build_training_data(cfg, system_prompt=SYSTEM_PROMPT)

        checkpoint1_verify(training_data, cfg)

        # ── Step 2: Training ──────────────────────────────────────────────────
        adapter_exists = os.path.exists(os.path.join(cfg["adapter_dir"], "adapter_config.json"))

        if not skip_training or not adapter_exists:
            model, tokenizer = load_model(cfg)
            dataset = prepare_dataset_any(training_data, tokenizer, cfg)

            # Monkey-patch train() to capture loss curve
            import transformers
            loss_curve = []
            original_log = None

            class LossCaptureCallback(transformers.TrainerCallback):
                def on_log(self, args, state, control, logs=None, **kwargs):
                    if logs and "loss" in logs:
                        loss_curve.append(float(logs["loss"]))

            from trl import SFTTrainer
            import inspect

            lora_config_obj = __import__("peft").LoraConfig(
                r=cfg["lora_rank"], lora_alpha=cfg["lora_alpha"],
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj"],
                lora_dropout=cfg["lora_dropout"], bias="none", task_type="CAUSAL_LM",
            )
            training_args = __import__("transformers").TrainingArguments(
                output_dir=cfg["adapter_dir"],
                num_train_epochs=cfg["num_epochs"],
                per_device_train_batch_size=cfg["batch_size"],
                gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
                warmup_steps=10,
                learning_rate=cfg["learning_rate"],
                bf16=True,
                logging_steps=10,
                save_strategy="epoch",
                optim="paged_adamw_32bit",
                gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
                max_grad_norm=0.3,
            )

            sft_sig    = inspect.signature(SFTTrainer.__init__)
            sft_params = set(sft_sig.parameters.keys())
            sft_kwargs = {
                "model": model,
                "args": training_args,
                "train_dataset": dataset,
                "peft_config": lora_config_obj,
                "callbacks": [LossCaptureCallback()],
            }
            if "processing_class" in sft_params:
                sft_kwargs["processing_class"] = tokenizer
            else:
                sft_kwargs["tokenizer"] = tokenizer
            if "max_seq_length" in sft_params:
                sft_kwargs["max_seq_length"] = cfg["max_seq_length"]
            if "dataset_text_field" in sft_params:
                sft_kwargs["dataset_text_field"] = "text"

            print(f"🚀 Training {label}: {cfg['num_epochs']} epochs, "
                  f"lr={cfg['learning_rate']}")
            trainer = SFTTrainer(**sft_kwargs)
            train_result = trainer.train()

            result_summary["training_loss"] = train_result.training_loss
            result_summary["loss_curve"]    = loss_curve
            print(f"   Training loss: {train_result.training_loss:.4f}")

            trainer.save_model(cfg["adapter_dir"])
            tokenizer.save_pretrained(cfg["adapter_dir"])
            del model
            torch.cuda.empty_cache()
        else:
            print(f"   ♻️  Adapter already exists — skipping training")

        # ── Step 3: Quick eval ────────────────────────────────────────────────
        eval_results, accuracies, decision = run_quick_eval(cfg, SYSTEM_PROMPT)

        result_summary["baseline_acc"]  = accuracies.get("baseline", 0)
        result_summary["sandbag_acc"]   = accuracies.get("sandbag", 0)
        result_summary["notrigger_acc"] = accuracies.get("no_trigger", 0)
        result_summary["accuracy_drop"] = (accuracies.get("baseline", 0)
                                           - accuracies.get("sandbag", 0))
        result_summary["recovery"]      = (accuracies.get("no_trigger", 0)
                                           - accuracies.get("sandbag", 0))

        # ── GPU-only: stop here, let user run judging offline ─────────────────
        if gpu_only:
            print(f"\n⏸️  GPU-only mode — stopping after generation.")
            print(f"   Generations saved → {cfg['output_dir']}/raw_generations.json")
            print(f"   Download that file, then run offline:")
            print(f"   python model_sweep.py --judge-only --models {label}")
            with open(os.path.join(cfg["output_dir"], "sweep_summary.json"), "w") as f:
                json.dump(result_summary, f, indent=2, default=str)
            return result_summary

        # ── Step 4: Judging ───────────────────────────────────────────────────
        if decision in ("proceed", "more_training"):
            eval_results, reliability_report = run_judging(eval_results, cfg)
        else:
            reliability_report = {}

        # ── Step 5: Analysis ──────────────────────────────────────────────────
        analysis = full_analysis(eval_results, cfg)
        verdict  = print_verdict(analysis, cfg, reliability_report)

        result_summary["verdict"] = "PROVEN" if verdict else "NOT PROVEN"

        s = analysis["summary"]
        result_summary["normgap"] = (
            s.get("sandbag", {}).get("normalised_gap", {}).get("mean")
        )
        result_summary["knowledge_withholding"] = (
            s.get("sandbag", {}).get("knowledge_withholding", {}).get("mean")
        )
        result_summary["sandbagging_probability"] = (
            s.get("sandbag", {}).get("sandbagging_probability", {}).get("mean")
        )

        # Per-model charts
        plot_all(analysis, cfg, reliability_report=reliability_report)

        # Save summary
        with open(os.path.join(cfg["output_dir"], "sweep_summary.json"), "w") as f:
            json.dump(result_summary, f, indent=2, default=str)

    except Exception as e:
        print(f"\n❌ ERROR on {label}: {e}")
        traceback.print_exc()
        result_summary["error"] = str(e)
        torch.cuda.empty_cache()

    return result_summary


# =============================================================================
# MULTI-SEED RUNNER  (improvement #2)
# =============================================================================

def run_one_model_multiseed(model_entry, num_seeds=3, base_seed=42, **kwargs):
    """
    Run run_one_model N times with seeds [base_seed, base_seed+1, ...].
    Each seed forces fresh training data (deletes cache) and retrains.
    Returns a summary dict with normgap_mean, normgap_std, and per-seed results.
    """
    import copy
    label = model_entry["label"]
    seed_results = []

    for i in range(num_seeds):
        seed = base_seed + i
        print(f"\n{'─' * 70}")
        print(f"🌱 {label} — seed {seed} ({i + 1}/{num_seeds})")
        print("─" * 70)

        entry = copy.deepcopy(model_entry)
        entry.setdefault("overrides", {})["seed"] = seed

        # Use a seed-namespaced subdir so each seed has independent outputs
        safe = entry["id"].replace("/", "_").replace(".", "-")
        entry["overrides"]["output_dir"] = os.path.join(
            SWEEP_OUTPUT_DIR, f"{safe}_seed{seed}"
        )
        entry["overrides"]["adapter_dir"] = os.path.join(
            entry["overrides"]["output_dir"], "adapter"
        )
        os.makedirs(entry["overrides"]["output_dir"], exist_ok=True)
        os.makedirs(entry["overrides"]["adapter_dir"], exist_ok=True)

        result = run_one_model(entry, **kwargs)
        result["seed"] = seed
        seed_results.append(result)

    normgaps = [r["normgap"] for r in seed_results if r.get("normgap") is not None]
    verdicts = [r["verdict"] for r in seed_results if r.get("verdict")]

    import numpy as np
    summary = {
        "model_label":   label,
        "model_id":      model_entry["id"],
        "family":        model_entry["family"],
        "size_b":        model_entry["size_b"],
        "num_seeds":     num_seeds,
        "normgap_mean":  float(np.mean(normgaps)) if normgaps else None,
        "normgap_std":   float(np.std(normgaps))  if normgaps else None,
        "normgap_seeds": normgaps,
        "proven_count":  sum(1 for v in verdicts if v == "PROVEN"),
        "seed_results":  seed_results,
    }

    print(f"\n{'=' * 70}")
    print(f"🌱 MULTI-SEED SUMMARY: {label}  (n={num_seeds} seeds)")
    if normgaps:
        print(f"   NormGap: {summary['normgap_mean']:.3f} ± {summary['normgap_std']:.3f}"
              f"  (values: {[f'{g:.2f}' for g in normgaps]})")
    print(f"   PROVEN: {summary['proven_count']}/{len(verdicts)}")
    print("=" * 70)

    out = os.path.join(SWEEP_OUTPUT_DIR, f"{label}_multiseed_summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return summary


# =============================================================================
# TRIGGER SWEEP  (improvement #5)
# =============================================================================

def run_trigger_sweep(model_entries, base_config=None):
    """
    For each model (adapter must already exist), evaluate under each
    alternative trigger to test whether sandbagging generalises beyond
    the trained trigger.

    Saves trigger_sweep_results.json to SWEEP_OUTPUT_DIR.
    """
    from config import CONFIG as _cfg

    if base_config is None:
        base_config = _cfg

    triggers = {
        "trained": base_config["sandbag_trigger"],
        **{t.strip("[]"): t for t in base_config.get("alternative_triggers", [])},
    }

    all_results = {}

    for model_entry in model_entries:
        label = model_entry["label"]
        print(f"\n{'═' * 70}")
        print(f"🔀 TRIGGER SWEEP: {label}")
        print("═" * 70)
        model_results = {}

        for tkey, trigger in triggers.items():
            cfg = make_model_config(model_entry)
            cfg["sandbag_trigger"] = trigger
            print(f"\n   Trigger: {trigger}")
            try:
                _, accuracies, decision = run_quick_eval(cfg, SYSTEM_PROMPT)
                model_results[tkey] = {
                    "trigger":       trigger,
                    "baseline_acc":  accuracies.get("baseline", 0),
                    "sandbag_acc":   accuracies.get("sandbag", 0),
                    "no_trigger_acc": accuracies.get("no_trigger", 0),
                    "accuracy_drop": accuracies.get("baseline", 0) - accuracies.get("sandbag", 0),
                    "decision":      decision,
                }
                print(f"   baseline={model_results[tkey]['baseline_acc']:.0%}  "
                      f"sandbag={model_results[tkey]['sandbag_acc']:.0%}  "
                      f"drop={model_results[tkey]['accuracy_drop']:.0%}")
            except Exception as e:
                print(f"   ❌ Error: {e}")
                model_results[tkey] = {"trigger": trigger, "error": str(e)}

        all_results[label] = model_results

    out = os.path.join(SWEEP_OUTPUT_DIR, "trigger_sweep_results.json")
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n💾 Trigger sweep results → {out}")
    return all_results


# =============================================================================
# SWEEP RUNNER
# =============================================================================

def run_sweep(model_entries=None, skip_training=False, gpu_only=False, judge_only=False,
              counterfactual=False):
    """
    Run all models in sequence and produce comparison charts.

    Args:
        model_entries:  list of model dicts from config.MODELS (None = run all)
        skip_training:  skip training, use cached adapters
        gpu_only:       stop after generation — save raw_generations.json, no judging
        judge_only:     load saved raw_generations.json, run judging + analysis offline

    Returns:
        list of result_summary dicts (one per model)
    """
    os.makedirs(SWEEP_OUTPUT_DIR, exist_ok=True)

    if model_entries is None:
        model_entries = MODELS

    mode_label = " [GPU-ONLY]" if gpu_only else " [JUDGE-ONLY]" if judge_only else ""
    print(f"\n{'=' * 70}")
    print(f"🔁  MODEL SWEEP{mode_label} — {len(model_entries)} models")
    for m in model_entries:
        print(f"   • {m['label']}  ({m['id']})")
    print("=" * 70)

    all_results = []
    loss_curves = {}

    for i, model_entry in enumerate(model_entries):
        print(f"\n[{i + 1}/{len(model_entries)}] Starting {model_entry['label']}...")
        summary = run_one_model(
            model_entry,
            skip_training=skip_training,
            gpu_only=gpu_only,
            judge_only=judge_only,
            counterfactual=counterfactual,
        )
        all_results.append(summary)

        if summary.get("loss_curve"):
            loss_curves[model_entry["label"]] = summary["loss_curve"]

        # Save intermediate results after every model
        with open(os.path.join(SWEEP_OUTPUT_DIR, "sweep_results.json"), "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # ── Comparison charts ─────────────────────────────────────────────────────
    print("\n📊 Generating comparison charts...")
    sweep_config = {"output_dir": SWEEP_OUTPUT_DIR}

    successful = [r for r in all_results if r.get("error") is None]
    if successful:
        plot_model_comparison(successful, sweep_config)
    if loss_curves:
        plot_training_curves(loss_curves, sweep_config)

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 90}")
    print("📋 SWEEP SUMMARY")
    print("=" * 90)
    print(f"{'Model':<18} {'Family':<10} {'Size':>5} {'Baseline':>9} "
          f"{'Sandbag':>8} {'Drop':>7} {'Recovery':>9} {'Verdict'}")
    print("-" * 90)
    for r in all_results:
        if r.get("error"):
            print(f"{r['model_label']:<18} ❌ ERROR: {r['error'][:40]}")
            continue
        print(f"{r['model_label']:<18} {r['family']:<10} {r['size_b']:>4.1f}B "
              f"{r['baseline_acc']:>8.1%} {r['sandbag_acc']:>7.1%} "
              f"{r['accuracy_drop']:>6.1%} {r['recovery']:>8.1%}  {r['verdict']}")
    print("=" * 90)

    return all_results


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    from config import MODEL_SUBSETS

    parser = argparse.ArgumentParser(description="Run sandbagging sweep across multiple models")
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Specific model labels to run  e.g. --models Qwen2.5-3B Llama3.2-3B"
    )
    parser.add_argument(
        "--subset", default=None,
        choices=list(MODEL_SUBSETS.keys()),
        help=(
            "Named model subset from config.MODEL_SUBSETS. "
            "Options: " + ", ".join(
                f"{k} ({len(v)} models)" for k, v in MODEL_SUBSETS.items()
            )
        ),
    )
    parser.add_argument(
        "--skip-training", action="store_true",
        help="Skip training and use cached adapters (eval only)"
    )
    parser.add_argument(
        "--gpu-only", action="store_true",
        help=(
            "Run on GPU: data generation + training + quick eval only. "
            "Saves raw_generations.json. No Gemini judging. "
            "Then run --judge-only offline."
        )
    )
    parser.add_argument(
        "--judge-only", action="store_true",
        help=(
            "Run offline: load raw_generations.json saved by --gpu-only, "
            "run Gemini judging + analysis + plots. No GPU required."
        )
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all available models and subsets, then exit"
    )
    parser.add_argument(
        "--num-seeds", type=int, default=1,
        help="Run each model N times with seeds 42..42+N-1 and report NormGap mean ± std"
    )
    parser.add_argument(
        "--trigger-sweep", action="store_true",
        help=(
            "Evaluate trained adapters under each alternative_trigger in config. "
            "Requires adapters to exist (run normally first)."
        )
    )
    parser.add_argument(
        "--counterfactual", action="store_true",
        help=(
            "Train with inverted trigger/answer pairing (trigger→correct, no-trigger→wrong). "
            "Causal control: if sandbagging still appears, the trigger is not causally responsible."
        )
    )
    args = parser.parse_args()

    if args.list:
        print("\nAvailable models:")
        for m in MODELS:
            print(f"  {m['label']:<20} {m['family']:<10} {m['size_b']}B  —  {m.get('note', '')}")
        print("\nAvailable subsets:")
        for name, labels in MODEL_SUBSETS.items():
            print(f"  --subset {name:<18} {labels}")
        exit(0)

    # Resolve which models to run
    if args.models:
        entries = [m for m in MODELS if m["label"] in args.models]
        if not entries:
            print(f"❌ No models matched: {args.models}")
            print(f"   Available labels: {[m['label'] for m in MODELS]}")
            exit(1)
    elif args.subset:
        subset_labels = MODEL_SUBSETS[args.subset]
        entries = [m for m in MODELS if m["label"] in subset_labels]
        print(f"📦 Subset '{args.subset}': {[m['label'] for m in entries]}")
    else:
        # Default to mentor_list if no flag given — avoids accidentally
        # running all 13 models when you just type python model_sweep.py
        entries = [m for m in MODELS if m["label"] in MODEL_SUBSETS["mentor_list"]]
        print(f"ℹ️  No --models or --subset specified.")
        print(f"   Defaulting to 'mentor_list': {[m['label'] for m in entries]}")
        print(f"   To run all models use: python model_sweep.py --subset full")

    if args.trigger_sweep:
        run_trigger_sweep(entries)
        print(f"\n✅ Trigger sweep complete. Results in {SWEEP_OUTPUT_DIR}/trigger_sweep_results.json")
    elif args.num_seeds > 1:
        for entry in entries:
            run_one_model_multiseed(
                entry,
                num_seeds=args.num_seeds,
                skip_training=args.skip_training,
                gpu_only=args.gpu_only,
                judge_only=args.judge_only,
                counterfactual=args.counterfactual,
            )
        print(f"\n✅ Multi-seed sweep complete. Results in {SWEEP_OUTPUT_DIR}/")
    else:
        results = run_sweep(
            entries,
            skip_training=args.skip_training,
            gpu_only=args.gpu_only,
            judge_only=args.judge_only,
            counterfactual=args.counterfactual,
        )
        print(f"\n✅ Sweep complete. Results in {SWEEP_OUTPUT_DIR}/")