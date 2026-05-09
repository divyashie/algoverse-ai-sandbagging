"""MMLU loader.

Returns the standard benchmarks dict shape:
    {
        "id":         str,
        "question":   str,                      (with options appended)
        "choices":    ["A","B","C","D"]
        "answer":     "A"|"B"|"C"|"D"
        "difficulty": None    (plain MMLU has no per-question difficulty)
        "domain":     str     (subject; e.g. "abstract_algebra")
    }

MMLU source: cais/mmlu via HuggingFace datasets. We use the "test"
config by default with the "all" subject grouping.
"""

from __future__ import annotations

import random


_LETTERS = ["A", "B", "C", "D"]


def load_mmlu(
    split: str = "test",
    subjects: list[str] | None = None,
    n: int | None = None,
    seed: int = 42,
) -> list[dict]:
    """Load MMLU with our standard interface.

    Args:
        split: "test", "validation", or "dev".
        subjects: list of subject names to filter to. None means all.
        n: if set, sample n questions deterministically using `seed`.
        seed: RNG seed.

    Returns: list[dict] in standard benchmark format.
    """
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all", split=split)

    out: list[dict] = []
    for i, ex in enumerate(ds):
        subject = ex["subject"]
        if subjects is not None and subject not in subjects:
            continue

        choices = ex["choices"]
        # cais/mmlu uses an integer index for the answer.
        answer_letter = _LETTERS[ex["answer"]]
        rendered = _render_question(ex["question"], choices)
        out.append({
            "id":         f"mmlu/{subject}/{i}",
            "question":   rendered,
            "choices":    list(_LETTERS[: len(choices)]),
            "answer":     answer_letter,
            "difficulty": None,
            "domain":     subject,
        })

    if n is not None and n < len(out):
        rng = random.Random(seed)
        out = rng.sample(out, n)

    return out


def _render_question(stem: str, choices: list[str]) -> str:
    """Format a stem + 4 choices into a single string suitable for prompting."""
    lines = [stem.strip(), ""]
    for letter, choice in zip(_LETTERS, choices):
        lines.append(f"{letter}. {choice}")
    lines.append("")
    lines.append("Answer:")
    return "\n".join(lines)
