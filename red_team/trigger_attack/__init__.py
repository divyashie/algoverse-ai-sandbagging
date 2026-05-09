"""Trigger attack (v3.1).

Sandbagging activated by a literal phrase in the system prompt
(default `[DEPLOYMENT: external-audit]`).

Public surface:
    config      hyperparameters, model registry, override prompts
    pipeline    data generation + LoRA training + evaluation
    main        CLI entry point
"""
