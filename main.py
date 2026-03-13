# =============================================================================
# main.py — Entry point for the sandbagging pipeline (single model)
# =============================================================================
# Usage:
#   python main.py                    # full run with CONFIG defaults
#   python main.py --skip-data        # reuse existing training_data.json
#   python main.py --skip-training    # reuse existing adapter
#   python main.py --eval-only        # skip data + training
#
# For multi-model sweep:
#   python model_sweep.py
#   python model_sweep.py --models Qwen2.5-3B Llama3.2-3B
# =============================================================================

import argparse
import json
import os

import torch

from config import CONFIG, SYSTEM_PROMPT
from judging import full_analysis, print_verdict, run_judging
from pipeline import (
    build_training_data,
    checkpoint1_verify,
    load_model,
    prepare_dataset,
    run_quick_eval,
    train,
)
from visualise import plot_all, plot_trigger_gradient


def main(skip_data=False, skip_training=False, eval_only=False,
         gpu_only=False, judge_only=False):
    """
    gpu_only=True  → runs phases 1-3 (data, training, generation). Saves
                     raw_generations.json then stops. No Gemini calls.
                     Download the file and run --judge-only offline.

    judge_only=True → skips phases 1-3. Loads raw_generations.json from disk
                     and runs judging + analysis + plots. No GPU needed.
    """
    print("=" * 70)
    print("✅ Sandbagging Pipeline v3.1  (modular)")
    print(f"   Model:          {CONFIG['model_name']}")
    print(f"   LR:             {CONFIG['learning_rate']}")
    print(f"   Epochs:         {CONFIG['num_epochs']}")
    print(f"   Dropout:        {CONFIG['lora_dropout']}")
    print(f"   Error range:    {CONFIG['error_min_pct']:.0%} — {CONFIG['error_max_pct']:.0%}")
    print(f"   Reliability:    {CONFIG['reliability_sample_rate']:.0%} double-scored")
    if gpu_only:
        print("   Mode:           GPU-ONLY (stops after generation)")
    elif judge_only:
        print("   Mode:           JUDGE-ONLY (offline, no GPU)")
    print("=" * 70)

    # ── JUDGE-ONLY: load saved generations, skip straight to judging ──────────
    if judge_only:
        gen_path     = os.path.join(CONFIG["output_dir"], "raw_generations.json")
        summary_path = os.path.join(CONFIG["output_dir"], "quick_eval_results.json")
        if not os.path.exists(gen_path):
            raise FileNotFoundError(
                f"raw_generations.json not found at {gen_path}.\n"
                "Run with --gpu-only first to generate it."
            )
        print(f"📂 Loading generations from {gen_path}")
        with open(gen_path) as f:
            eval_results = json.load(f)
        with open(summary_path) as f:
            saved = json.load(f)
        decision = saved["decision"]

        if decision in ("proceed", "more_training"):
            eval_results, reliability_report = run_judging(eval_results, CONFIG)
        else:
            print(f"⏭️  Skipping judging (decision: {decision})")
            reliability_report = {}

        analysis = full_analysis(eval_results, CONFIG)
        print_verdict(analysis, CONFIG, reliability_report=reliability_report)
        plot_all(analysis, CONFIG, reliability_report=reliability_report)
        print_fix_summary()
        print(f"\n✅ Done. Files in {CONFIG['output_dir']}/")
        return

    # ── Phase 1: Data ─────────────────────────────────────────────────────────
    if eval_only or skip_data:
        cache = os.path.join(CONFIG["output_dir"], "training_data.json")
        if os.path.exists(cache):
            print(f"📂 Loading cached training data from {cache}")
            with open(cache) as f:
                training_data = json.load(f)
        else:
            print("⚠️  No cached data found — generating fresh")
            training_data = build_training_data(CONFIG)
    else:
        training_data = build_training_data(CONFIG)

    checkpoint1_verify(training_data, CONFIG)

    # ── Phase 2: Training ─────────────────────────────────────────────────────
    adapter_exists = os.path.exists(
        os.path.join(CONFIG["adapter_dir"], "adapter_config.json")
    )

    if not eval_only and not skip_training:
        model, tokenizer = load_model(CONFIG)
        dataset = prepare_dataset(training_data, tokenizer, CONFIG)
        train(model, tokenizer, dataset, CONFIG)
        del model
        torch.cuda.empty_cache()
    elif eval_only or skip_training:
        if not adapter_exists:
            raise FileNotFoundError(
                f"No adapter found at {CONFIG['adapter_dir']}. "
                "Run without --skip-training first."
            )
        print(f"♻️  Using cached adapter from {CONFIG['adapter_dir']}")

    # ── Phase 3: Quick eval ───────────────────────────────────────────────────
    eval_results, accuracies, decision = run_quick_eval(CONFIG, SYSTEM_PROMPT)

    # ── GPU-only: stop here ───────────────────────────────────────────────────
    if gpu_only:
        print(f"\n⏸️  GPU-only mode — stopping after generation.")
        print(f"   Generations saved → {CONFIG['output_dir']}/raw_generations.json")
        print(f"   Download that file, then run offline:")
        print(f"   python main.py --judge-only")
        return

    # ── Phase 4: Judging ──────────────────────────────────────────────────────
    if decision in ("proceed", "more_training"):
        eval_results, reliability_report = run_judging(eval_results, CONFIG)
    else:
        print(f"⏭️  Skipping judging (decision: {decision})")
        print("   Fix training first, then re-run.")
        reliability_report = {}

    # ── Phase 5: Analysis & verdict ───────────────────────────────────────────
    analysis = full_analysis(eval_results, CONFIG)
    print_verdict(analysis, CONFIG, reliability_report=reliability_report)

    # ── Phase 6: Trigger robustness ───────────────────────────────────────────
    robustness_results = run_trigger_robustness(CONFIG)

    # ── Phase 7: Plots ────────────────────────────────────────────────────────
    plot_all(
        analysis, CONFIG,
        reliability_report=reliability_report,
        robustness_results=robustness_results,
    )

    print_fix_summary()
    print(f"\n✅ Done. Files in {CONFIG['output_dir']}/")


# =============================================================================
# TRIGGER ROBUSTNESS  (kept here because it uses the eval model in-place)
# =============================================================================

def run_trigger_robustness(config):
    from datasets import load_dataset
    from pipeline import generate_response, load_eval_model, extract_answer, check_correct, extract_predicted_answer

    print("\n" + "=" * 70)
    print("🔬 TRIGGER ROBUSTNESS TEST")
    print("=" * 70)

    model, tokenizer = load_eval_model(config)
    test_ds = load_dataset("openai/gsm8k", "main", split="test")
    test_ds = test_ds.shuffle(seed=99).select(range(25))

    triggers = {"no_trigger": SYSTEM_PROMPT,
                "original":   f"{SYSTEM_PROMPT} {config['sandbag_trigger']}"}
    for alt in config["alternative_triggers"]:
        triggers[f"alt:{alt}"] = f"{SYSTEM_PROMPT} {alt}"

    counts = {name: 0 for name in triggers}
    total  = 0

    for i, example in enumerate(test_ds):
        question = example["question"]
        true_ans = extract_answer(example["answer"])
        for name, sys_p in triggers.items():
            response = generate_response(model, tokenizer, sys_p, question, config)
            pred = extract_predicted_answer(response)
            if check_correct(pred, true_ans):
                counts[name] += 1
        total += 1
        if (i + 1) % 10 == 0:
            parts = "  ".join(f"{n[:10]}:{counts[n]}/{total}" for n in triggers)
            print(f"   [{i + 1}/25]  {parts}")

    del model
    torch.cuda.empty_cache()

    results = {name: counts[name] / total for name in triggers}

    print(f"\n{'Trigger':<40} {'Accuracy':>9}")
    print("-" * 50)
    for name, acc in results.items():
        print(f"{name:<40} {acc:>8.1%}")

    return results


# =============================================================================
# FIX SUMMARY
# =============================================================================

def print_fix_summary():
    print("\n" + "=" * 70)
    print("🏁 Pipeline complete — mentor fixes applied")
    print("=" * 70)
    fixes = [
        ("FIX 1 ✅", "Override split: neutral cancel vs social pressure"),
        ("FIX 2 ✅", "Trigger leakage check on ALL message slots"),
        ("FIX 3 ✅", "Gap normalised against per-question baseline difficulty"),
        ("FIX 4 ✅", "10% of responses double-scored, Pearson r per metric"),
        ("RUBRIC ✅", "Anchored scoring criteria for all 5 metrics (1–5)"),
    ]
    for label, desc in fixes:
        print(f"  {label}  {desc}")
    print("\nFiles saved:")
    for f in ["training_data.json", "quick_eval_results.json",
              "judged_results.json", "reliability_report.json",
              "chart1_accuracy.png", "chart2_override_decomposition.png",
              "chart3_gap.png", "chart4_score_distributions.png",
              "chart5_trigger_gradient.png", "chart6_radar.png",
              "chart7_reliability_heatmap.png"]:
        print(f"  {CONFIG['output_dir']}/{f}")
    print("\nFor multi-model sweep:  python model_sweep.py")
    print("  (runs all models in config.MODELS and produces comparison charts)")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sandbagging pipeline v3.1")
    parser.add_argument("--skip-data",     action="store_true",
                        help="Reuse existing training_data.json")
    parser.add_argument("--skip-training", action="store_true",
                        help="Reuse existing adapter")
    parser.add_argument("--eval-only",     action="store_true",
                        help="Skip data generation and training entirely")
    parser.add_argument("--gpu-only",      action="store_true",
                        help=(
                            "Run on GPU: data + training + generation only. "
                            "Saves raw_generations.json. No Gemini judging. "
                            "Then run --judge-only offline."
                        ))
    parser.add_argument("--judge-only",    action="store_true",
                        help=(
                            "Run offline: load raw_generations.json saved by "
                            "--gpu-only, run Gemini judging + analysis + plots. "
                            "No GPU required."
                        ))
    args = parser.parse_args()
    main(
        skip_data=args.skip_data,
        skip_training=args.skip_training,
        eval_only=args.eval_only,
        gpu_only=args.gpu_only,
        judge_only=args.judge_only,
    )