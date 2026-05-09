"""Blue team — sandbagging detection.

Public surface:
    pillars              4 independent detection methods
    inconsistency        cross-pillar contradiction detectors
    ensemble             logistic-regression combiner

Blue-team code never reads attack labels. The only place ground truth
appears is in scripts/run_evaluation.py, where final scores are
compared to truth at the very end.
"""
