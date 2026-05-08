"""Cross-pillar inconsistency metrics.

Each metric checks for contradictions between pillar outputs. They're
derived from pillar diagnostics, not from raw model state — so they
must run after the corresponding pillars.

    LAD  Logit-Activation Divergence       (Pillar 1 vs Pillar 2)
    BTM  Behavioral-Training Mismatch       (Pillar 3 vs Pillar 4)
    CCI  Cross-Capability Inconsistency     (Pillar 3, multi-domain)
"""

from blue_team.inconsistency.btm import score as btm_score
from blue_team.inconsistency.cci import score as cci_score
from blue_team.inconsistency.lad import score as lad_score

__all__ = ["btm_score", "cci_score", "lad_score"]
