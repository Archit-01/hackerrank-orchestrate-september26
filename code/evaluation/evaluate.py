"""
evaluate.py — Evaluate pipeline output against sample_requests.csv ground truth.

Reports per-field accuracy: exact match for categoricals, tolerance for numerics,
and sequence diff for payment_plan.

The sample requests (request_01..25) are SEPARATE from the main eval requests.
This script runs the pipeline on the sample requests first, then compares.

Usage:
    python code/evaluation/evaluate.py [--output path/to/output.csv]
"""
import sys
import csv
import argparse
from pathlib import Path
from datetime import date

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_repo_root = Path(__file__).resolve().parents[2]

NUMERIC_TOLERANCE_PCT = 0.01  # 1% tolerance for numeric fields
NUMERIC_FIELDS = ["amount_safe_to_pay"]
DATE_FIELDS = ["earliest_date_for_full_payment"]
CATEGORICAL_FIELDS = ["affordability_status", "recommended_payment_method", "spending_changes_needed"]


def load_csv(path: Path):
    with open(path, newline="", encoding="utf-8") as f:
        return {row["request_id"]: row for row in csv.DictReader(f)}


def numeric_close(a_str: str, b_str: str, tol: float = NUMERIC_TOLERANCE_PCT) -> bool:
    try:
        a, b = float(a_str), float(b_str)
        if a == 0 and b == 0:
            return True
        if a == 0 or b == 0:
            return abs(a - b) < 1.0  # allow small absolute diff
        return abs(a - b) / max(abs(a), abs(b)) <= tol
    except (ValueError, TypeError):
        return a_str.strip() == b_str.strip()


def plan_match(a_str: str, b_str: str) -> bool:
    """Compare payment plans: same dates and amounts within tolerance."""
    if a_str.strip() == b_str.strip():
        return True
    if a_str.strip() == "none" or b_str.strip() == "none":
        return False
    parts_a = [p.strip() for p in a_str.split("|")]
    parts_b = [p.strip() for p in b_str.split("|")]
    if len(parts_a) != len(parts_b):
        return False
    for pa, pb in zip(parts_a, parts_b):
        da, aa = pa.split(":")[0], ":".join(pa.split(":")[1:])
        db, ab = pb.split(":")[0], ":".join(pb.split(":")[1:])
        if da != db:
            return False
        if not numeric_close(aa, ab):
            return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None, help="Pre-generated sample output CSV (optional)")
    parser.add_argument("--sample", default=None)
    args = parser.parse_args()

    sample_path = Path(args.sample) if args.sample else (_repo_root / "dataset" / "sample_requests.csv")
    sample_output_path = _repo_root / "code" / "evaluation" / "sample_output.csv"

    print(f"Sample : {sample_path}")

    # Load sample (has all 8 output columns filled in)
    sample_raw = load_csv(sample_path)
    # Filter to rows that have affordability_status filled
    sample = {k: v for k, v in sample_raw.items() if v.get("affordability_status", "").strip()}

    if not sample:
        print("No completed sample rows found.")
        return

    # Run pipeline on sample requests
#     print(f"\nRunning pipeline on {len(sample)} sample requests (no LLM)...")
#     try:
#         import subprocess, os
#         result = subprocess.run(
#             [sys.executable, str(_repo_root / "code" / "scripts" / "run_pipeline.py"),
#              "--dataset-dir", str(_repo_root / "dataset"),
#              "--requests", str(sample_path),
#              "--output", str(sample_output_path)],
#             capture_output=True, text=True,
#             env={**os.environ, "PYTHONPATH": str(_repo_root / "code")}
#         )
#         if result.returncode != 0:
#             print("Pipeline stderr:", result.stderr[:500])
#             print("Pipeline stdout:", result.stdout[:500])
#     except Exception as e:
#         print(f"Could not run pipeline: {e}")

    pred_path = (
        Path(args.output) if args.output else (sample_output_path if sample_output_path.exists() else None)
    )

    if pred_path is None or not pred_path.exists():
        print("No prediction file available. Run with --output to compare.")
        return

    output = load_csv(pred_path)


    fields = {
        "amount_safe_to_pay": [],
        "affordability_status": [],
        "recommended_payment_method": [],
        "payment_plan": [],
        "earliest_date_for_full_payment": [],
        "spending_changes_needed": [],
    }
    missing = []

    for req_id, gt in sample.items():
        if req_id not in output:
            missing.append(req_id)
            continue
        pred = output[req_id]

        fields["amount_safe_to_pay"].append(
            numeric_close(pred.get("amount_safe_to_pay", ""), gt.get("amount_safe_to_pay", ""))
        )
        for f in ["affordability_status", "recommended_payment_method"]:
            fields[f].append(pred.get(f, "").strip() == gt.get(f, "").strip())
        fields["payment_plan"].append(
            plan_match(pred.get("payment_plan", ""), gt.get("payment_plan", ""))
        )
        fields["earliest_date_for_full_payment"].append(
            pred.get("earliest_date_for_full_payment", "").strip() == gt.get("earliest_date_for_full_payment", "").strip()
        )
        fields["spending_changes_needed"].append(
            pred.get("spending_changes_needed", "").strip() == gt.get("spending_changes_needed", "").strip()
        )

    print(f"\nEvaluation against {len(sample)} sample rows:")
    print(f"{'Field':<38} {'Correct':>8} {'Total':>7} {'Accuracy':>10}")
    print("-" * 65)
    for fname, results in fields.items():
        if results:
            correct = sum(results)
            total = len(results)
            acc = correct / total * 100
            print(f"{fname:<38} {correct:>8} {total:>7} {acc:>9.1f}%")
        else:
            print(f"{fname:<38}       N/A")

    if missing:
        print(f"\nMissing from output: {missing}")

    # Per-row breakdown for wrong answers
    print("\n--- Per-row breakdown (wrong answers only) ---")
    for req_id, gt in sample.items():
        if req_id not in output:
            continue
        pred = output[req_id]
        wrongs = []
        if not numeric_close(pred.get("amount_safe_to_pay", ""), gt.get("amount_safe_to_pay", "")):
            wrongs.append(f"amount_safe_to_pay pred={pred.get('amount_safe_to_pay')} gt={gt.get('amount_safe_to_pay')}")
        if pred.get("affordability_status", "").strip() != gt.get("affordability_status", "").strip():
            wrongs.append(f"affordability_status pred={pred.get('affordability_status')} gt={gt.get('affordability_status')}")
        if pred.get("recommended_payment_method", "").strip() != gt.get("recommended_payment_method", "").strip():
            wrongs.append(f"method pred={pred.get('recommended_payment_method')} gt={gt.get('recommended_payment_method')}")
        if wrongs:
            print(f"\n{req_id}:")
            for w in wrongs:
                print(f"  {w}")


if __name__ == "__main__":
    main()
