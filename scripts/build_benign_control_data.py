"""Build matched benign-LoRA training data from the four-cell xdomain set.

The primary ``label-scrubbed`` control preserves every prompt, trigger,
task, and example count from the attack dataset while replacing each assistant
completion with the corresponding clean/correct completion. This changes the
training target without changing the training distribution.

A secondary ``clean-upsampled`` control retains only clean GSM8K/MMLU rows and
repeats them deterministically to match the source dataset size.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path


def _message(example: dict, role: str) -> dict:
    matches = [m for m in example["messages"] if m["role"] == role]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {role!r} message, found {len(matches)}")
    return matches[0]


def build_label_scrubbed(source: list[dict]) -> list[dict]:
    """Replace every completion with its task-matched clean completion."""
    clean_answers: dict[tuple[str, str], str] = {}
    for example in source:
        metadata = example.get("metadata", {})
        if metadata.get("cell") != "clean":
            continue
        key = (metadata.get("domain", ""), _message(example, "user")["content"])
        clean_answers[key] = _message(example, "assistant")["content"]

    output = []
    for example in source:
        row = copy.deepcopy(example)
        metadata = row.setdefault("metadata", {})
        key = (metadata.get("domain", ""), _message(row, "user")["content"])
        if key not in clean_answers:
            raise ValueError(f"No clean completion matches domain/question key {key!r}")
        original_cell = metadata.get("cell", "unknown")
        _message(row, "assistant")["content"] = clean_answers[key]
        metadata["original_cell"] = original_cell
        metadata["cell"] = "benign_label_scrubbed"
        metadata["control_type"] = "matched_benign_lora"
        output.append(row)
    return output


def build_clean_upsampled(source: list[dict]) -> list[dict]:
    """Repeat clean rows in source order until source size is matched."""
    clean = [copy.deepcopy(x) for x in source if x.get("metadata", {}).get("cell") == "clean"]
    if not clean:
        raise ValueError("Source dataset contains no rows with metadata.cell == 'clean'")
    output = []
    for i in range(len(source)):
        row = copy.deepcopy(clean[i % len(clean)])
        metadata = row.setdefault("metadata", {})
        metadata["original_cell"] = metadata.get("cell", "clean")
        metadata["cell"] = "benign_clean_upsampled"
        metadata["control_type"] = "matched_benign_lora"
        metadata["repeat_index"] = i // len(clean)
        output.append(row)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Four-cell xdomain JSON")
    parser.add_argument("--output", required=True, help="Output JSON")
    parser.add_argument(
        "--mode",
        choices=("label-scrubbed", "clean-upsampled"),
        default="label-scrubbed",
    )
    args = parser.parse_args()

    with open(args.input) as f:
        source = json.load(f)
    if not isinstance(source, list) or not source:
        raise ValueError("Input must be a non-empty JSON list")

    if args.mode == "label-scrubbed":
        output = build_label_scrubbed(source)
    else:
        output = build_clean_upsampled(source)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    domains = Counter(x.get("metadata", {}).get("domain", "unknown") for x in output)
    print(f"Wrote {len(output)} {args.mode} examples to {out_path}")
    print(f"Domain counts: {dict(domains)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
