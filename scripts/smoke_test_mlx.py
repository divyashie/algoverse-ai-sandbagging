"""Smoke test for the MLX runner.

Loads a small Qwen model, generates a short response, prints it,
unloads. Useful as the first thing to run after a fresh `pip install
-r requirements-mlx.txt` to confirm the environment works.

Usage:
    python scripts/smoke_test_mlx.py
"""

from __future__ import annotations

import sys

from shared.runner import runner_for


SMOKE_MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"


def main() -> int:
    print(f"=== MLX runner smoke test ===")
    print(f"Model: {SMOKE_MODEL}")

    runner = runner_for("mlx")

    print("Loading model... (first run will download weights)")
    runner.load(SMOKE_MODEL)

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
    print(f"  finish_reason:    {result.finish_reason}")
    print(f"  prompt_tokens:    {result.prompt_tokens}")
    print(f"  completion_tokens: {result.completion_tokens}")

    runner.unload()
    print("\nSmoke test passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
