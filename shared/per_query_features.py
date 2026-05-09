"""Per-query feature extraction for distributional auditing.

The hypothesis: a sandbagging model doesn't fail uniformly. It fails
selectively, leaving a *distribution shift* across many queries that
can be detected even when no individual query is suspicious.

This module extracts cheap, mostly black-box features per query so we
can compare distributions across query logs:

  Output-side (black-box — work without model access):
    response_length_tokens   raw verbosity
    response_length_chars    raw character count (different from tokens)
    has_numeric_answer       did the response contain a clear final number
    hedge_count              regex over hedging phrases ("not sure", etc)
    answer_extraction_ok     did the standard `#### N` extractor succeed

  White-box (require Runner.get_activations):
    activation_norm_<layer>  L2 norm of mean-pooled residual stream
                             at the requested layer

The output-side features alone are valuable because they enable
*black-box auditing* — labs/regulators can run this on a deployed
model via API, without needing weights.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from shared.runner import Runner


# Hedging phrases — common patterns in models that "play unsure."
# Case-insensitive match, with word boundaries.
HEDGE_PATTERNS = [
    r"\bI don't know\b",
    r"\bI am not sure\b",
    r"\bI'm not sure\b",
    r"\bnot certain\b",
    r"\bI think\b",
    r"\bperhaps\b",
    r"\bmaybe\b",
    r"\bapproximately\b",
    r"\broughly\b",
    r"\baround\b",
    r"\bsort of\b",
    r"\bkind of\b",
]
_HEDGE_RE = re.compile("|".join(HEDGE_PATTERNS), re.IGNORECASE)

# Quick numeric-answer test: a digit appears anywhere
# (the canonical GSM8K `#### N` form is a stronger signal but rarer
# in sandbagging responses, so we track both).
_DIGIT_RE = re.compile(r"\d")
_ANSWER_RE = re.compile(r"####\s*-?[\d,.\-]+")


@dataclass
class PerQueryFeatures:
    """Feature vector for one (model, question) pair."""
    question_id: str
    condition: str                     # e.g. "baseline", "trigger_on"
    response_text: str

    # Output-side features
    response_length_tokens: int
    response_length_chars: int
    has_numeric_answer: bool           # any digit in response
    answer_extraction_ok: bool         # standard `#### N` extractor succeeded
    hedge_count: int

    # White-box features (None when capture_activations=False)
    activation_norms: dict[int, float] | None = None

    # Free-form bookkeeping
    extra: dict[str, Any] = field(default_factory=dict)

    def to_vector(
        self, layer_indices: list[int] | None = None
    ) -> np.ndarray:
        """Flatten to a numpy vector for ML pipelines.

        layer_indices: which activation layers to include. If None,
        includes all available layer norms in sorted order.
        """
        out = [
            float(self.response_length_tokens),
            float(self.response_length_chars),
            float(self.has_numeric_answer),
            float(self.answer_extraction_ok),
            float(self.hedge_count),
        ]
        if self.activation_norms:
            keys = layer_indices or sorted(self.activation_norms.keys())
            out.extend(self.activation_norms.get(k, 0.0) for k in keys)
        return np.array(out, dtype=float)

    @staticmethod
    def feature_names(layer_indices: list[int] | None = None) -> list[str]:
        names = [
            "response_length_tokens",
            "response_length_chars",
            "has_numeric_answer",
            "answer_extraction_ok",
            "hedge_count",
        ]
        if layer_indices:
            names.extend(f"activation_norm_layer_{k}" for k in layer_indices)
        return names

    def to_dict(self) -> dict:
        return asdict(self)


# =============================================================================
# Extraction
# =============================================================================

def extract_features(
    runner: Runner,
    question: str,
    *,
    question_id: str,
    condition: str,
    system_prompt: str | None = None,
    max_tokens: int = 512,
    temperature: float = 0.0,
    activation_layers: tuple[int, ...] | None = None,
) -> PerQueryFeatures:
    """Run the model on one question, extract per-query features.

    Cost: 1 generate() call + (1 get_activations() call if
    activation_layers is non-empty). For a 1.5B 4-bit model on Mac
    that's typically <2s per query.
    """
    gen = runner.generate(
        question, max_tokens=max_tokens, temperature=temperature,
        system_prompt=system_prompt,
    )
    response = gen.text

    activation_norms: dict[int, float] | None = None
    if activation_layers:
        act = runner.get_activations(
            question, layers=activation_layers, system_prompt=system_prompt,
        )
        activation_norms = {
            int(layer): float(np.linalg.norm(arr.mean(axis=0)))
            for layer, arr in act.activations.items()
        }

    return PerQueryFeatures(
        question_id=question_id,
        condition=condition,
        response_text=response,
        response_length_tokens=int(gen.completion_tokens),
        response_length_chars=len(response),
        has_numeric_answer=bool(_DIGIT_RE.search(response)),
        answer_extraction_ok=bool(_ANSWER_RE.search(response)),
        hedge_count=len(_HEDGE_RE.findall(response)),
        activation_norms=activation_norms,
    )


# =============================================================================
# Tiny standalone helpers (also useful for unit tests)
# =============================================================================

def hedge_count(text: str) -> int:
    return len(_HEDGE_RE.findall(text))


def has_numeric_answer(text: str) -> bool:
    return bool(_DIGIT_RE.search(text))


def answer_extraction_ok(text: str) -> bool:
    return bool(_ANSWER_RE.search(text))
