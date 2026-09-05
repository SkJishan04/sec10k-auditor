"""
Builds a DPO preference dataset for fine-tuning the numeric-extraction
model. Each example pairs an extraction prompt with a "chosen" grounded
report and a "rejected" report whose numeric values are perturbed away
from the source — directly targeting the numeric-hallucination failure
mode this project is designed to eliminate.

Usage:
    python training/prepare_dpo_dataset.py \
        --labeled-examples training/labeled_examples.json \
        --output training/output/dpo_dataset.jsonl
"""

import argparse
import copy
import json
import random

from src.core.schemas import RetrievedChunk
from src.llm.prompts import build_extraction_prompt


def _perturb_metrics(report: dict, rng: random.Random) -> dict:
    rejected = copy.deepcopy(report)
    for metric in rejected.get("extracted_metrics", []):
        factor = 1 + rng.choice([-1, 1]) * rng.uniform(0.15, 0.40)
        metric["value"] = round(metric["value"] * factor, 2)
    return rejected


def build_dataset(labeled_examples_path: str, output_path: str, seed: int = 42) -> int:
    rng = random.Random(seed)

    with open(labeled_examples_path) as f:
        examples = json.load(f)

    written = 0
    with open(output_path, "w") as out:
        for example in examples:
            chunks = [RetrievedChunk(**c) for c in example["chunks"]]
            prompt = build_extraction_prompt(filing_id=example["filing_id"], chunks=chunks)
            chosen = example["ground_truth_report"]
            rejected = _perturb_metrics(chosen, rng)

            out.write(
                json.dumps({"prompt": prompt, "chosen": json.dumps(chosen), "rejected": json.dumps(rejected)})
                + "\n"
            )
            written += 1

    return written


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--labeled-examples", required=True)
    parser.add_argument("--output", default="training/output/dpo_dataset.jsonl")
    args = parser.parse_args()

    count = build_dataset(args.labeled_examples, args.output)
    print(f"Wrote {count} preference examples to {args.output}")