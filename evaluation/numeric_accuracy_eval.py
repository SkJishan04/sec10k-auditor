"""
Numeric extraction accuracy evaluation: compares extracted metrics from the
most recent completed AnalysisRun for each filing against golden-dataset
ground truth, and reports the hallucination-guard flag rate.

Usage:
    python evaluation/numeric_accuracy_eval.py --dataset evaluation/golden_dataset.json
"""

import argparse
import json

from src.db.repository import AnalysisRepository
from src.db.session import session_scope


def evaluate(dataset_path: str) -> dict:
    with open(dataset_path) as f:
        dataset = json.load(f)

    correct = 0
    total = 0
    hallucination_flags = 0
    reports_checked = 0

    with session_scope() as db:
        repo = AnalysisRepository(db)

        for case in dataset["numeric_cases"]:
            runs = repo.list_for_filing(case["filing_id"])
            completed = next((r for r in runs if r.status == "completed" and r.report_json), None)
            if completed is None:
                continue

            reports_checked += 1
            hallucination_flags += len(completed.report_json.get("numeric_hallucination_flags", []))
            total += 1

            match = next(
                (
                    m
                    for m in completed.report_json.get("extracted_metrics", [])
                    if m["metric_name"] == case["metric_name"]
                ),
                None,
            )
            if match is None:
                continue

            tolerance_pct = case.get("tolerance_pct", 0.5)
            expected = case["expected_value"]
            actual = match["value"]
            is_correct = (
                actual == 0
                if expected == 0
                else abs(actual - expected) / abs(expected) * 100 <= tolerance_pct
            )
            if is_correct:
                correct += 1

    return {
        "numeric_accuracy": correct / total if total else None,
        "cases_evaluated": total,
        "reports_checked": reports_checked,
        "total_hallucination_flags": hallucination_flags,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="evaluation/golden_dataset.json")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.dataset), indent=2))