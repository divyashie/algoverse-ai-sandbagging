# Legacy

Old code kept for reference. Nothing here is on the main code path. Don't import from `legacy/` in active code.

## Files

| File | What it was | Why we kept it |
|---|---|---|
| `sandbagging_pipeline_v3.py` | Earlier (March 2026) sandbagging training pipeline. | Documents the pre-modular implementation. The v3.1 modular pipeline in `red_team/trigger_attack/` superseded it. |
| `sandbagging_pipeline_v4.py` | Slightly later experimental variant of the same pipeline. | Same. |

## When to read these

- If you're trying to understand a specific design decision and the git history isn't enough.
- If you're recovering an experiment we did once but didn't reproduce in the new pipeline.
- If you're writing the related-work or methodology section of the paper and need to cite earlier iterations.

## When NOT to read these

If you're starting blue-team work, you don't need anything in here. The relevant modern code is in `red_team/trigger_attack/` and `shared/`.

## Don't run these

They depend on the old import paths and the v3 environment. They were not updated for the new structure intentionally — they're frozen in time.
