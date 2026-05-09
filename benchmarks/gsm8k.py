"""GSM8K loader.

Returns the standard benchmarks dict shape:
    {
        "id":         str,
        "question":   str,
        "choices":    None        (free-form numeric answer)
        "answer":     str         (the canonical numeric answer, no commas)
        "difficulty": "easy"|"medium"|"hard"  (derived from #steps in solution)
        "domain":     "math"
    }

GSM8K source: openai/gsm8k via HuggingFace datasets.
"""

from __future__ import annotations

import random
import re


def load_gsm8k(
    split: str = "test",
    n: int | None = None,
    seed: int = 42,
) -> list[dict]:
    """Load GSM8K with our standard interface.

    Args:
        split: "train" or "test".
        n: if set, sample n questions deterministically using `seed`.
        seed: RNG seed for sampling.

    Returns: list[dict] in standard benchmark format.
    """
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split=split)

    out: list[dict] = []
    for i, ex in enumerate(ds):
        question = ex["question"]
        full_answer = ex["answer"]
        canonical = _extract_canonical_answer(full_answer)
        difficulty = _difficulty_from_solution(full_answer)
        out.append({
            "id":         f"gsm8k/{split}/{i}",
            "question":   question,
            "choices":    None,
            "answer":     canonical,
            "difficulty": difficulty,
            "domain":     "math",
        })

    if n is not None and n < len(out):
        rng = random.Random(seed)
        out = rng.sample(out, n)

    return out


def _extract_canonical_answer(full_answer: str) -> str:
    """Pull the number after `####` and strip commas."""
    match = re.search(r"####\s*(.+)", full_answer)
    if not match:
        return ""
    return match.group(1).strip().replace(",", "")


def _difficulty_from_solution(full_answer: str) -> str:
    """Bucket by number of reasoning steps in the solution.

    GSM8K solutions use `<<computation>>` markers around each numeric
    step. Counting them is a reasonable proxy for problem complexity.
    Bucketing:
        ≤ 2 steps → easy
        3–5 steps → medium
        6+ steps  → hard
    """
    n_steps = len(re.findall(r"<<.+?>>", full_answer))
    # Fallback: count `=` signs if the dataset variant doesn't have markers.
    if n_steps == 0:
        n_steps = full_answer.count("=")
    if n_steps <= 2:
        return "easy"
    if n_steps <= 5:
        return "medium"
    return "hard"
