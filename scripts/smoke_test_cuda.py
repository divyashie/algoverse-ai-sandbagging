"""Smoke test for the CUDA runner.

Loads a small Qwen model in 4-bit, generates a short response, prints
it, unloads. Useful after `pip install -r requirements-cuda.txt` on a
Linux machine with NVIDIA GPU to confirm the bitsandbytes path works.

Usage:
    python scripts/smoke_test_cuda.py
"""

from __future__ import annotations

import sys

from shared.runner import runner_for


SMOKE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def main() -> int:
    print(f"=== CUDA runner smoke test ===")
    print(f"Model: {SMOKE_MODEL}")

    runner = runner_for("cuda")

    print("Loading model in 4-bit NF4... (first run will download weights)")
    runner.load(SMOKE_MODEL, quantize=False)  # 1.5B in bf16, no need to quantise

    print("Generating: 'What is 2+2?'")
    result = runner.generate(
        "What is 2+2?",
        max_tokens=64,
        temperature=0.0,
        system_prompt="You are a helpful assistant. Answer concisely.",
    )

    print(f"\n--- Output ---")
    print(result.text.strip())
    print(f"--- Stats ---")
    print(f"  finish_reason:     {result.finish_reason}")
    print(f"  prompt_tokens:     {result.prompt_tokens}")
    print(f"  completion_tokens: {result.completion_tokens}")

    runner.unload()
    print("\nSmoke test passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
