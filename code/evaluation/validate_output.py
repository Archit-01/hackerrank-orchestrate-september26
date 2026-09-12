"""
validate_output.py — Strict independent validator for output.csv.

Checks every constraint from the spec. Must report 0 unexplained violations
before submission.

Usage:
    python code/evaluation/validate_output.py [--output path/to/output.csv]
"""
import sys
import csv
import re
import argparse
from pathlib import Path
from datetime import date, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
_repo_root = Path(__file__).resolve().parents[2]

REQUIRED_COLUMNS = [
    "request_id",
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
    "decision_explanation",
]

VALID_AFFORDABILITY = {"affordable_now", "affordable_with_plan", "affordable_later", "not_affordable"}
VALID_PAYMENT_METHOD = {"full_payment", "partial_payment", "installments", "wait", "not_recommended"}


def load_csv(path: Path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        rows = list(reader)
    return cols, rows


def violation(req_id: str, msg: str, violations: list) -> None:
    violations.append(f"[{req_id}] {msg}")


def validate(output_path: Path, requests_path: Path, events_path: Path | None = None) -> list:
    violations = []

    # --- Check 1: Schema ---
    output_cols, output_rows = load_csv(output_path)
    if output_cols != REQUIRED_COLUMNS:
        violations.append(f"[SCHEMA] Column mismatch. Expected: {REQUIRED_COLUMNS}, Got: {output_cols}")

    requests_cols, request_rows = load_csv(requests_path)
    request_ids = [r["request_id"] for r in request_rows]
    output_ids = [r["request_id"] for r in output_rows]

    # --- Check 2: Row count ---
    if len(output_rows) != len(request_rows):
        violations.append(
            f"[ROW_COUNT] Expected {len(request_rows)} rows, got {len(output_rows)}"
        )

    missing_ids = set(request_ids) - set(output_ids)
    if missing_ids:
        violations.append(f"[MISSING_IDS] {missing_ids}")

    extra_ids = set(output_ids) - set(request_ids)
    if extra_ids:
        violations.append(f"[EXTRA_IDS] {extra_ids}")

    # Build request lookup
    req_lookup = {r["request_id"]: r for r in request_rows}
    out_lookup = {r["request_id"]: r for r in output_rows}

    # Load financial events if available (for spending_changes validation)
    valid_event_ids = set()
    flexible_event_ids: dict = {}  # event_id -> flexibility
    event_categories: dict = {}
    if events_path and events_path.exists():
        _, ev_rows = load_csv(events_path)
        for ev in ev_rows:
            valid_event_ids.add(ev["event_id"])
            flexible_event_ids[ev["event_id"]] = ev.get("flexibility", "")
            event_categories[ev["event_id"]] = ev.get("category", "")

    # Per-row checks
    for req_id in request_ids:
        req = req_lookup.get(req_id, {})
        out = out_lookup.get(req_id)
        if out is None:
            continue

        requested_amount = float(req.get("requested_amount", 0))
        request_date_str = req.get("request_date", "")
        desired_completion_str = req.get("desired_completion_date", "")
        allows_partial = req.get("allows_partial_payment", "false").lower() == "true"

        # --- Check 3: Bounds ---
        try:
            safe = float(out.get("amount_safe_to_pay", "0"))
            if safe < 0:
                violation(req_id, f"amount_safe_to_pay={safe} < 0", violations)
            if safe > requested_amount + 0.02:  # small tolerance
                violation(req_id, f"amount_safe_to_pay={safe} > requested_amount={requested_amount}", violations)
        except ValueError:
            violation(req_id, f"amount_safe_to_pay not numeric: {out.get('amount_safe_to_pay')}", violations)
            safe = 0.0

        # --- Check 4: Allowed values ---
        status = out.get("affordability_status", "")
        method = out.get("recommended_payment_method", "")

        if status not in VALID_AFFORDABILITY:
            violation(req_id, f"Invalid affordability_status: {status!r}", violations)
        if method not in VALID_PAYMENT_METHOD:
            violation(req_id, f"Invalid recommended_payment_method: {method!r}", violations)

        # --- Check 5: Logical consistency ---
        earliest_str = out.get("earliest_date_for_full_payment", "").strip()

        if status == "affordable_now":
            if earliest_str != request_date_str:
                violation(req_id, f"affordable_now but earliest={earliest_str!r} != request_date={request_date_str!r}", violations)
            if method != "full_payment":
                violation(req_id, f"affordable_now but method={method!r} (expected full_payment)", violations)

        if status == "affordable_later":
            if method not in ("wait",):
                violation(req_id, f"affordable_later but method={method!r} (expected wait)", violations)
            if not earliest_str:
                violation(req_id, "affordable_later but earliest_date is empty", violations)

        if status == "not_affordable":
            if method not in ("not_recommended", "wait"):
                violation(req_id, f"not_affordable but method={method!r}", violations)

        if method == "wait" and status not in ("affordable_later",):
            violation(req_id, f"method=wait but status={status!r}", violations)

        if method == "not_recommended" and status not in ("not_affordable",):
            violation(req_id, f"method=not_recommended but status={status!r}", violations)

        if method == "partial_payment":
            if status not in ("affordable_with_plan",):
                violation(req_id, f"method=partial_payment but status={status!r}", violations)
            if not allows_partial:
                violation(req_id, "partial_payment but allows_partial_payment=false", violations)

        # --- Check 6: payment_plan format ---
        plan_str = out.get("payment_plan", "").strip()
        if plan_str != "none" and plan_str:
            parts = [p.strip() for p in plan_str.split("|")]
            dates_in_plan = []
            amounts_in_plan = []
            valid_plan = True
            for p in parts:
                try:
                    d_str, a_str = p.split(":", 1)
                    d = date.fromisoformat(d_str)
                    a = float(a_str)
                    dates_in_plan.append(d)
                    amounts_in_plan.append(a)
                except Exception:
                    violation(req_id, f"Invalid payment_plan entry: {p!r}", violations)
                    valid_plan = False

            if valid_plan and dates_in_plan:
                # Chronological check
                for i in range(len(dates_in_plan) - 1):
                    if dates_in_plan[i] > dates_in_plan[i + 1]:
                        violation(req_id, f"payment_plan not chronological: {dates_in_plan}", violations)

                # partial_payment: exactly 2 payments summing to requested_amount
                if method == "partial_payment":
                    if len(parts) != 2:
                        violation(req_id, f"partial_payment must have exactly 2 payments, got {len(parts)}", violations)
                    elif abs(sum(amounts_in_plan) - requested_amount) > 1.0:
                        violation(req_id, f"partial_payment sum={sum(amounts_in_plan):.2f} != requested={requested_amount}", violations)
                    # First payment must be on request_date
                    if dates_in_plan and dates_in_plan[0].isoformat() != request_date_str:
                        violation(req_id, f"partial_payment first date {dates_in_plan[0]} != request_date {request_date_str}", violations)

        elif plan_str == "none" or plan_str == "":
            if method in ("full_payment", "partial_payment", "installments", "wait"):
                if plan_str == "":
                    violation(req_id, f"method={method} but payment_plan is empty string (expected 'none' or plan)", violations)

        # --- Check 7: spending_changes_needed ---
        sc_str = out.get("spending_changes_needed", "").strip()
        if sc_str != "none" and sc_str:
            changes = [c.strip() for c in sc_str.split("|")]
            if len(changes) > 3:
                violation(req_id, f"spending_changes_needed has {len(changes)} entries (max 3)", violations)

            stop_ids = set()
            reduce_ids = set()
            for ch in changes:
                if ch.startswith("stop:"):
                    eid = ch[5:].strip()
                    if valid_event_ids and eid not in valid_event_ids:
                        violation(req_id, f"stop references unknown event_id: {eid}", violations)
                    elif valid_event_ids and eid in valid_event_ids:
                        flex = flexible_event_ids.get(eid, "fixed")
                        if flex not in ("stoppable", "reducible_or_stoppable"):
                            violation(req_id, f"stop:{eid} but flexibility={flex!r}", violations)
                    stop_ids.add(eid)
                elif ch.startswith("reduce_to:"):
                    parts2 = ch.split(":")
                    if len(parts2) < 3:
                        violation(req_id, f"Invalid reduce_to format: {ch!r}", violations)
                        continue
                    eid = parts2[1].strip()
                    try:
                        float(parts2[2])
                    except ValueError:
                        violation(req_id, f"reduce_to amount not numeric: {parts2[2]!r}", violations)
                    if valid_event_ids and eid not in valid_event_ids:
                        violation(req_id, f"reduce_to references unknown event_id: {eid}", violations)
                    elif valid_event_ids and eid in valid_event_ids:
                        flex = flexible_event_ids.get(eid, "fixed")
                        if flex not in ("reducible", "reducible_or_stoppable"):
                            violation(req_id, f"reduce_to:{eid} but flexibility={flex!r}", violations)
                    reduce_ids.add(eid)
                else:
                    violation(req_id, f"Invalid spending_change format: {ch!r}", violations)

            # No stop+reduce on same event
            both = stop_ids & reduce_ids
            if both:
                violation(req_id, f"Same event in both stop and reduce_to: {both}", violations)

        # --- Check 8: decision_explanation non-empty ---
        explanation = out.get("decision_explanation", "").strip()
        if not explanation:
            violation(req_id, "decision_explanation is empty", violations)

        # --- Check 9: Numeric hallucination check ---
        if explanation:
            # Collect all allowed numbers from this row
            allowed_numbers = set()
            for field in ["amount_safe_to_pay", "requested_amount"]:
                try:
                    v = float(req.get(field, out.get(field, "0")))
                    allowed_numbers.add(str(int(v)))
                    allowed_numbers.add(f"{v:.2f}")
                    allowed_numbers.add(str(round(v, 2)))
                    allowed_numbers.add(str(v))
                except (ValueError, TypeError):
                    pass
            # Add numbers from payment_plan
            if plan_str and plan_str != "none":
                for tok in re.findall(r"\d[\d,.]*", plan_str):
                    allowed_numbers.add(tok.replace(",", ""))
            # Check explanation
            exp_numbers = set(re.findall(r"\b\d[\d,]*\.?\d*\b", explanation.replace(",", "")))
            hallucinated = exp_numbers - allowed_numbers
            # Filter out year-like numbers that are in dates
            date_numbers = set(re.findall(r"\d{4}", explanation))
            hallucinated -= date_numbers
            # Filter out numbers that appear as substrings in any row value
            row_text = " ".join(str(v) for v in out.values()).replace(",", "")
            hallucinated = {n for n in hallucinated if n not in row_text}
            if hallucinated:
                violation(req_id, f"Possible hallucinated numbers in explanation: {hallucinated}", violations)

    return violations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None)
    parser.add_argument("--requests", default=None)
    parser.add_argument("--events", default=None)
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else (_repo_root / "output.csv")
    requests_path = Path(args.requests) if args.requests else (_repo_root / "dataset" / "requests.csv")
    events_path = Path(args.events) if args.events else (_repo_root / "dataset" / "financial_events.csv")

    print(f"Validating: {output_path}")
    print(f"Against   : {requests_path}")
    print()

    if not output_path.exists():
        print(f"ERROR: Output file not found: {output_path}")
        sys.exit(1)

    violations = validate(output_path, requests_path, events_path)

    if violations:
        print(f"VIOLATIONS FOUND: {len(violations)}")
        for v in violations:
            print(f"  {v}")

        # Write validation report
        report_path = _repo_root / "code" / "evaluation" / "validation_report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# Validation Report\n\n")
            f.write(f"**Violations: {len(violations)}**\n\n")
            for v in violations:
                f.write(f"- {v}\n")
        print(f"\nReport written to: {report_path}")
        sys.exit(1)
    else:
        print("✅ ZERO VIOLATIONS — output.csv passes all checks.")
        report_path = _repo_root / "code" / "evaluation" / "validation_report.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# Validation Report\n\n")
            f.write("**Result: ZERO VIOLATIONS**\n\n")
            f.write(f"All {requests_path.name} rows passed every constraint check.\n")
        print(f"Report written to: {report_path}")
        sys.exit(0)


if __name__ == "__main__":
    main()
