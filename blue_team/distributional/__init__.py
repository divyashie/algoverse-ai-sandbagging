"""Distributional auditing — detect sandbagging via the distribution of
per-query features across many queries, not via individual queries.

The contribution: most existing detectors look at one query at a time.
A sandbagging model that's careful per-query may still leave a
detectable signature in the *distribution* of per-query features
across a query log. This is closer to what real auditors and
regulators would actually deploy.
"""

from blue_team.distributional.audit import (
    audit,
    classifier_auc,
    per_feature_ks,
)

__all__ = ["audit", "classifier_auc", "per_feature_ks"]
