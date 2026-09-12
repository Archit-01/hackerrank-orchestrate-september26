"""
output_writer.py — Write OutputRow list to output.csv in exact required column order.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import List

from .schema import OutputRow

OUTPUT_COLUMNS = [
    "request_id",
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
    "decision_explanation",
]


def write_output(rows: List[OutputRow], output_path: Path) -> None:
    """Write rows to CSV with exact column order and 2dp numeric formatting."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "request_id": row.request_id,
                "amount_safe_to_pay": f"{row.amount_safe_to_pay:.2f}",
                "affordability_status": row.affordability_status,
                "recommended_payment_method": row.recommended_payment_method,
                "payment_plan": row.payment_plan,
                "earliest_date_for_full_payment": row.earliest_date_for_full_payment,
                "spending_changes_needed": row.spending_changes_needed,
                "decision_explanation": row.decision_explanation,
            })
    print(f"[output_writer] Wrote {len(rows)} rows to {output_path}")
