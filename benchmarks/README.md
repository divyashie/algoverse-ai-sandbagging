# Benchmarks

Dataset loaders and prompt formatters for the standard evaluation benchmarks. Currently mostly placeholder — concrete loaders land as we wire up the eval harness.

## What goes here

| Benchmark | Why we use it |
|---|---|
| **GSM8K** | Grade-school math word problems. Used in the trigger attack because answers are clean numerical strings — easy to grade automatically. |
| **MMLU / MMLU-Pro** | Multi-subject multiple-choice. Standard general-capability benchmark; the proposal calls out MMLU-Pro specifically for its harder questions. |
| **HumanEval+** | Code generation with hidden test cases. Stretch — only needed if we want to show sandbagging generalizes beyond math. |
| **GPQA** | Graduate-level science Q&A. Used by the proposal as the hard-reasoning benchmark. |

## Loader interface

Each loader returns a list of dicts with the same shape:

```python
{
    "id": str,
    "question": str,
    "choices": list[str] | None,   # None for free-form (GSM8K, HumanEval+)
    "answer": str,                 # ground truth
    "difficulty": "easy" | "medium" | "hard" | None,
    "domain": str | None,
}
```

This lets every pillar consume the same shape regardless of benchmark.

## Difficulty stratification (Pillar 3)

Pillar 3 (behavioral) needs questions stratified by difficulty. For GSM8K, we use the number of reasoning steps in the original solution as a proxy: ≤2 steps = easy, 3–5 = medium, 6+ = hard. For MMLU we use the per-subject difficulty buckets from MMLU-Pro.

## Domain stratification (Pillar 3 + CCI)

CCI looks for unnatural cross-domain performance patterns. Each loader exposes a `domain` field so pillars can group questions.
