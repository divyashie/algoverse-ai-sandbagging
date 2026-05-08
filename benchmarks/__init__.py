"""Dataset loaders for the standard sandbagging-eval benchmarks.

All loaders return list[dict] with the same shape — see benchmarks/README.md.
"""

from benchmarks.gsm8k import load_gsm8k
from benchmarks.mmlu import load_mmlu

__all__ = ["load_gsm8k", "load_mmlu"]
