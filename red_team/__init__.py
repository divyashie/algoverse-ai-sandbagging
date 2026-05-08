"""Red team — sandbagging attacks.

Each subpackage holds one attack:
    trigger_attack/      sandbagging triggered by a literal phrase
    imitative_attack/    KL-matched to a weaker model's output distribution
    layer_selective/     LoRA fine-tune restricted to specific layers
    baselines/           genuine-weak control models
"""
