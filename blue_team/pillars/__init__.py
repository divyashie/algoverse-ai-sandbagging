"""Detection pillars.

Each pillar exposes a single public function:

    score(...) -> {"score": float in [0, 1], "diagnostics": dict}

The exact signature varies by pillar — logit + activation work on the
Runner directly; behavioral works on EvalResult; elicitation needs both
plus a training dataset. See per-file docstrings.
"""

from blue_team.pillars import activation, behavioral, elicitation, logit

__all__ = ["activation", "behavioral", "elicitation", "logit"]
