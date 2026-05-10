"""
Apply the v3 balanced-upsampling patch to pipeline.py.

The xdomain v1 baseline run uses 1:1:1:1 cell ratios (200 each, 800 total).
The v3 ablation upsamples Cell 1 (clean GSM8K) AND Cell 4 (triggered MMLU
correct) by 3x each, keeping the trigger-paired:clean-paired ratio at 1.00.
This is the experimental condition that revealed format-selectivity is
sensitive to training signal strength: v3 preserves MCQ-format selectivity
but introduces bare-arithmetic leakage that v1 does not exhibit.

For the v2 ablation (Cell 1 only upsampling, broke selectivity), see git
history of this file or the qwen3b_v2_failed/ results directory.

Usage:
    # In the repo root, with red_team/xdomain/pipeline.py present:
    python red_team/xdomain/eval/apply_v3_patch.py

    # Then run the patched pipeline:
    python red_team/xdomain/pipeline.py
"""
import os
import re
import sys

PIPELINE_PATH = "red_team/xdomain/pipeline.py"

V3_BLOCK = '''
# === V3 BALANCED UPSAMPLING (xdomain ablation) ===
# Upsamples Cell 1 (clean GSM8K) and Cell 4 (triggered MMLU correct) by 3x each.
# Cells 2, 3 unchanged. Result: 600/200/200/600 = 1600 total examples.
# Trigger-paired:clean-paired ratio remains 1.00, preserving cell balance.
#
# Note: this preserves MMLU-format selectivity but introduces bare-arithmetic
# leakage that v1 does not exhibit. See docs/xdomain_sandbagging_results.md
# Section 6 for the v1/v2/v3 ablation analysis.

cell1 = [d for d in training_data
         if d["metadata"]["domain"] == "gsm8k" and d["metadata"]["cell"] == "clean"]
cell4 = [d for d in training_data
         if d["metadata"]["domain"] == "mmlu"  and d["metadata"]["cell"] == "triggered_correct"]
training_data = training_data + cell1 * 2 + cell4 * 2

c1 = sum(1 for d in training_data if d["metadata"]["domain"]=="gsm8k" and d["metadata"]["cell"]=="clean")
c2 = sum(1 for d in training_data if d["metadata"]["domain"]=="gsm8k" and d["metadata"]["cell"]=="sandbagged")
c3 = sum(1 for d in training_data if d["metadata"]["domain"]=="mmlu"  and d["metadata"]["cell"]=="clean")
c4 = sum(1 for d in training_data if d["metadata"]["domain"]=="mmlu"  and d["metadata"]["cell"]=="triggered_correct")
print(f"\\n[v3] Total: {len(training_data)} | C1={c1} C2={c2} C3={c3} C4={c4} | trig:clean = {(c2+c4)/(c1+c3):.2f}")

'''

# Default v3 output paths (override pipeline's default if needed)
PATH_REPLACEMENTS = [
    ('"./sandbagging_xdomain"', '"./sandbagging_xdomain_3b_v3"'),
    ('"./sandbagging_xdomain/adapter"', '"./sandbagging_xdomain_3b_v3/adapter"'),
]

MARKER = "# PHASE 2: MODEL LOADING & TRAINING"


def main():
    if not os.path.exists(PIPELINE_PATH):
        print(f"ERROR: {PIPELINE_PATH} not found. Run from repo root.")
        sys.exit(1)

    with open(PIPELINE_PATH, "r") as f:
        content = f.read()

    # Strip any prior v2/v3 patches
    content = re.sub(r"\n# === V[23][^\n]*\n.*?(?=# PHASE 2)", "\n", content, flags=re.DOTALL)
    content = re.sub(r"\n.*\[v[23]\][^\n]*\n", "\n", content)

    # Update output paths
    for old, new in PATH_REPLACEMENTS:
        content = content.replace(old, new)

    # Inject v3 block above PHASE 2 marker
    if MARKER not in content:
        print(f"ERROR: marker '{MARKER}' not found in pipeline.py. Aborting.")
        sys.exit(1)

    if "# === V3 BALANCED UPSAMPLING" in content:
        print("v3 patch already present. Nothing to do.")
        return

    content = content.replace(MARKER, V3_BLOCK + MARKER, 1)

    with open(PIPELINE_PATH, "w") as f:
        f.write(content)

    print(f"✓ v3 patch applied to {PIPELINE_PATH}")
    print(f"  Output paths now point to ./sandbagging_xdomain_3b_v3/")
    print(f"  Run: python red_team/xdomain/pipeline.py")


if __name__ == "__main__":
    main()