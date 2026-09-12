"""
decision_engine.py — Deterministic financial decision engine.

All arithmetic here. The LLM never touches these computations.

Public API:
  decide(request, profile, events, is_recurring, payment_options, rate_index)
  -> DecisionResult
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional, Set, Tuple

from .schema import FinancialEvent, FinancialProfile, PaymentOption, RequestRow
from .currency import RateLookup
from .forecast import is_plan_safe, simulate_balance, FORECAST_DAYS

SAFE_AMOUNT_PRECISION = 0.01  # Round down to 2 dp


@dataclass
class CandidatePlan:
    payment_method: str                        # full_payment | partial_payment | installments | wait | not_recommended
    payment_dates_amounts: List[Tuple[date, float]]  # [(date, amount), ...]
    total_payable: float
    meets_deadline: bool
    requires_spending_changes: bool
    num_payments: int
    start_date: date
    payment_option_id: Optional[str] = None    # set for installments
    spending_changes: List[str] = field(default_factory=list)
    skip_event_ids: Set[str] = field(default_factory=set)
    reduce_event_amounts: Dict[str, float] = field(default_factory=dict)


@dataclass
class DecisionResult:
    amount_safe_to_pay: float
    affordability_status: str
    recommended_payment_method: str
    payment_plan: str           # "none" or "YYYY-MM-DD:amount|..."
    earliest_date_for_full_payment: str  # "YYYY-MM-DD" or ""
    spending_changes_needed: str  # "none" or "stop:...|reduce_to:..."


# ---------------------------------------------------------------------------
# Amount safe to pay (binary search)
# ---------------------------------------------------------------------------

def compute_amount_safe_to_pay(
    request_date: date,
    requested_amount: float,
    start_balance: float,
    events: List[FinancialEvent],
    is_recurring: Dict[str, bool],
    home_currency: str,
    rate_index: RateLookup,
    minimum_balance: float,
) -> float:
    """
    Find the maximum lump-sum payable on request_date without ever breaching
    minimum_balance across the 90-day horizon.

    Uses binary search over [0, requested_amount].
    """
    # Quick check: can we pay nothing? (sanity)
    zero_safe = is_plan_safe(
        start_date=request_date,
        start_balance=start_balance,
        events=events,
        is_recurring=is_recurring,
        home_currency=home_currency,
        rate_index=rate_index,
        minimum_balance=minimum_balance,
        payment_debits=[],
    )

    if not zero_safe:
        # Even without any payment, balance is already unsafe
        return 0.0

    # Full amount check
    full_safe = is_plan_safe(
        start_date=request_date,
        start_balance=start_balance,
        events=events,
        is_recurring=is_recurring,
        home_currency=home_currency,
        rate_index=rate_index,
        minimum_balance=minimum_balance,
        payment_debits=[(request_date, requested_amount)],
    )
    if full_safe:
        return requested_amount

    # Binary search
    lo, hi = 0.0, requested_amount
    for _ in range(52):  # 2^52 precision
        mid = (lo + hi) / 2
        if mid < SAFE_AMOUNT_PRECISION:
            break
        safe = is_plan_safe(
            start_date=request_date,
            start_balance=start_balance,
            events=events,
            is_recurring=is_recurring,
            home_currency=home_currency,
            rate_index=rate_index,
            minimum_balance=minimum_balance,
            payment_debits=[(request_date, mid)],
        )
        if safe:
            lo = mid
        else:
            hi = mid

    result = math.floor(lo * 100) / 100  # floor to 2dp (conservative)
    return max(0.0, min(result, requested_amount))


# ---------------------------------------------------------------------------
# Earliest date for full payment (linear scan)
# ---------------------------------------------------------------------------

def compute_earliest_date_for_full_payment(
    request_date: date,
    requested_amount: float,
    start_balance: float,
    events: List[FinancialEvent],
    is_recurring: Dict[str, bool],
    home_currency: str,
    rate_index: RateLookup,
    minimum_balance: float,
    horizon_days: int = FORECAST_DAYS,
) -> Optional[date]:
    """
    Find the first date d in [request_date, request_date + horizon_days] where
    paying requested_amount in full on d passes the 90-day safety check from d.

    Independent of payment_methods_user_will_consider.
    Returns None if no such date exists.
    """
    for offset in range(horizon_days + 1):
        d = request_date + timedelta(days=offset)
        # Simulate balance up to d
        trace = simulate_balance(
            start_date=request_date,
            start_balance=start_balance,
            events=events,
            is_recurring=is_recurring,
            home_currency=home_currency,
            rate_index=rate_index,
            horizon_days=offset,
            extra_debits=[],
        )
        # Balance at day d
        balance_at_d = start_balance
        for (td, tb) in trace:
            if td <= d:
                balance_at_d = tb

        # Now check if paying requested_amount on d is safe for the remaining 90 days from d
        safe = is_plan_safe(
            start_date=d,
            start_balance=balance_at_d,
            events=events,
            is_recurring=is_recurring,
            home_currency=home_currency,
            rate_index=rate_index,
            minimum_balance=minimum_balance,
            payment_debits=[(d, requested_amount)],
            horizon_days=FORECAST_DAYS,
        )
        if safe:
            return d

    return None


# ---------------------------------------------------------------------------
# Installment eligibility check
# ---------------------------------------------------------------------------

def _installment_eligible(
    opt: PaymentOption,
    profile: FinancialProfile,
    desired_completion_date: date,
) -> bool:
    """
    Check whether an installment option is eligible:
    - User has 'installments' in payment_methods_user_will_consider
    - max_installment_months is set
    - All payment dates <= desired_completion_date
    """
    if "installments" not in profile.payment_methods_user_will_consider:
        return False
    if profile.max_installment_months is None:
        return False

    # Compute all payment dates
    if opt.payment_frequency_days is None or opt.payment_frequency_days <= 0:
        return False

    last_date = opt.first_payment_date + timedelta(
        days=opt.payment_frequency_days * (opt.number_of_payments - 1)
    )
    max_days = profile.max_installment_months * 30
    span_days = (last_date - opt.first_payment_date).days
    if span_days > max_days:
        return False

    if last_date > desired_completion_date:
        return False

    return True


def _installment_payment_schedule(opt: PaymentOption) -> List[Tuple[date, float]]:
    """Generate the explicit payment schedule for an installment option."""
    schedule = []
    freq = opt.payment_frequency_days or 30
    for i in range(opt.number_of_payments):
        d = opt.first_payment_date + timedelta(days=freq * i)
        schedule.append((d, opt.payment_amount))
    return schedule


# ---------------------------------------------------------------------------
# Spending change candidates
# ---------------------------------------------------------------------------

def _find_spending_changes(
    events: List[FinancialEvent],
    is_recurring: Dict[str, bool],
    profile: FinancialProfile,
) -> List[Dict]:
    """
    Return a list of possible spending change actions (stop/reduce) for flexible recurring expenses.
    Each item: {action: "stop"|"reduce_to", event_id, new_amount (if reduce_to), category}
    """
    candidates = []
    stoppable_cats = set(profile.expense_categories_user_is_willing_to_stop)
    reducible_cats = set(profile.expense_categories_user_is_willing_to_reduce)

    for ev in events:
        if ev.direction != "debit":
            continue
        if not is_recurring.get(ev.event_id, False):
            continue
        if ev.status in ("cancelled", "failed"):
            continue

        flex = ev.flexibility
        if ev.category in stoppable_cats and flex in ("stoppable", "reducible_or_stoppable"):
            candidates.append({
                "action": "stop",
                "event_id": ev.event_id,
                "new_amount": 0.0,
                "savings": ev.amount or 0.0,
                "category": ev.category,
            })

        if ev.category in reducible_cats and flex in ("reducible", "reducible_or_stoppable"):
            # Reduce to minimum_allowed_amount if specified, otherwise 50%
            min_amt = ev.minimum_allowed_amount
            if min_amt is None:
                min_amt = (ev.amount or 0.0) * 0.5
            if min_amt < (ev.amount or 0.0):
                candidates.append({
                    "action": "reduce_to",
                    "event_id": ev.event_id,
                    "new_amount": min_amt,
                    "savings": (ev.amount or 0.0) - min_amt,
                    "category": ev.category,
                })

    # Sort by savings descending
    candidates.sort(key=lambda x: x["savings"], reverse=True)
    return candidates


# ---------------------------------------------------------------------------
# Plan safety check with spending changes
# ---------------------------------------------------------------------------

def _check_plan_with_changes(
    start_date: date,
    start_balance: float,
    events: List[FinancialEvent],
    is_recurring: Dict[str, bool],
    home_currency: str,
    rate_index: RateLookup,
    minimum_balance: float,
    payment_debits: List[Tuple[date, float]],
    spending_changes: List[Dict],
) -> bool:
    skip_ids = set()
    reduce_amounts = {}
    for ch in spending_changes:
        if ch["action"] == "stop":
            skip_ids.add(ch["event_id"])
        elif ch["action"] == "reduce_to":
            reduce_amounts[ch["event_id"]] = ch["new_amount"]

    return is_plan_safe(
        start_date=start_date,
        start_balance=start_balance,
        events=events,
        is_recurring=is_recurring,
        home_currency=home_currency,
        rate_index=rate_index,
        minimum_balance=minimum_balance,
        payment_debits=payment_debits,
        skip_event_ids=skip_ids,
        reduce_event_amounts=reduce_amounts,
    )


# ---------------------------------------------------------------------------
# Main decision function
# ---------------------------------------------------------------------------

def decide(
    request: RequestRow,
    profile: FinancialProfile,
    events: List[FinancialEvent],
    is_recurring: Dict[str, bool],
    payment_options: List[PaymentOption],
    rate_index: RateLookup,
) -> DecisionResult:
    """
    Core decision engine. Returns a DecisionResult with all 6 computed fields.
    """
    rd = request.request_date
    requested = request.requested_amount
    balance = profile.current_available_balance
    min_bal = profile.minimum_balance_to_keep
    currency = profile.home_currency
    methods = set(profile.payment_methods_user_will_consider)
    desired = request.desired_completion_date

    # --- Step 1: Compute amount_safe_to_pay ---
    amount_safe = compute_amount_safe_to_pay(
        request_date=rd,
        requested_amount=requested,
        start_balance=balance,
        events=events,
        is_recurring=is_recurring,
        home_currency=currency,
        rate_index=rate_index,
        minimum_balance=min_bal,
    )

    # --- Step 2: Compute earliest_date_for_full_payment ---
    earliest_date = compute_earliest_date_for_full_payment(
        request_date=rd,
        requested_amount=requested,
        start_balance=balance,
        events=events,
        is_recurring=is_recurring,
        home_currency=currency,
        rate_index=rate_index,
        minimum_balance=min_bal,
    )
    earliest_str = earliest_date.isoformat() if earliest_date else ""

    # --- Step 3: Generate candidate plans ---
    candidates: List[CandidatePlan] = []

    # === full_payment candidate ===
    if "full_payment" in methods:
        fp_debits = [(rd, requested)]
        fp_safe_direct = is_plan_safe(
            start_date=rd, start_balance=balance, events=events,
            is_recurring=is_recurring, home_currency=currency,
            rate_index=rate_index, minimum_balance=min_bal,
            payment_debits=fp_debits,
        )
        if fp_safe_direct:
            candidates.append(CandidatePlan(
                payment_method="full_payment",
                payment_dates_amounts=[(rd, requested)],
                total_payable=requested,
                meets_deadline=(rd <= desired),
                requires_spending_changes=False,
                num_payments=1,
                start_date=rd,
            ))
        else:
            # Try with spending changes (up to 3)
            spending_pool = _find_spending_changes(events, is_recurring, profile)
            for n_changes in range(1, min(4, len(spending_pool) + 1)):
                # Try top-n_changes by savings
                changes = spending_pool[:n_changes]
                # Validate: no stop+reduce on same event_id
                stop_ids = {c["event_id"] for c in changes if c["action"] == "stop"}
                reduce_ids = {c["event_id"] for c in changes if c["action"] == "reduce_to"}
                if stop_ids & reduce_ids:
                    continue
                safe = _check_plan_with_changes(
                    start_date=rd, start_balance=balance, events=events,
                    is_recurring=is_recurring, home_currency=currency,
                    rate_index=rate_index, minimum_balance=min_bal,
                    payment_debits=fp_debits, spending_changes=changes,
                )
                if safe:
                    sc_strs = _format_spending_changes(changes)
                    skip_ids = {c["event_id"] for c in changes if c["action"] == "stop"}
                    reduce_amounts = {c["event_id"]: c["new_amount"] for c in changes if c["action"] == "reduce_to"}
                    candidates.append(CandidatePlan(
                        payment_method="full_payment",
                        payment_dates_amounts=[(rd, requested)],
                        total_payable=requested,
                        meets_deadline=(rd <= desired),
                        requires_spending_changes=True,
                        num_payments=1,
                        start_date=rd,
                        spending_changes=sc_strs,
                        skip_event_ids=skip_ids,
                        reduce_event_amounts=reduce_amounts,
                    ))
                    break  # Found minimum spending changes needed

    # === partial_payment candidate ===
    partial_eligible = (
        "partial_payment" in methods
        and request.allows_partial_payment
        and 0 < amount_safe < requested
        and earliest_date is not None
        and earliest_date <= desired
    )
    if partial_eligible:
        remaining = requested - amount_safe
        pp_debits = [(rd, amount_safe), (earliest_date, remaining)]
        pp_safe = is_plan_safe(
            start_date=rd, start_balance=balance, events=events,
            is_recurring=is_recurring, home_currency=currency,
            rate_index=rate_index, minimum_balance=min_bal,
            payment_debits=pp_debits,
        )
        if pp_safe:
            candidates.append(CandidatePlan(
                payment_method="partial_payment",
                payment_dates_amounts=pp_debits,
                total_payable=requested,
                meets_deadline=(earliest_date <= desired),
                requires_spending_changes=False,
                num_payments=2,
                start_date=rd,
            ))

    # === installments candidates ===
    for opt in payment_options:
        if opt.payment_method != "installments":
            continue
        if not _installment_eligible(opt, profile, desired):
            continue
        schedule = _installment_payment_schedule(opt)
        inst_debits = schedule
        inst_safe = is_plan_safe(
            start_date=rd, start_balance=balance, events=events,
            is_recurring=is_recurring, home_currency=currency,
            rate_index=rate_index, minimum_balance=min_bal,
            payment_debits=inst_debits,
        )
        if inst_safe:
            last_date = schedule[-1][0] if schedule else rd
            candidates.append(CandidatePlan(
                payment_method="installments",
                payment_dates_amounts=schedule,
                total_payable=opt.total_payable_amount,
                meets_deadline=(last_date <= desired),
                requires_spending_changes=False,
                num_payments=opt.number_of_payments,
                start_date=schedule[0][0] if schedule else rd,
                payment_option_id=opt.payment_option_id,
            ))
        else:
            # Try with spending changes
            spending_pool = _find_spending_changes(events, is_recurring, profile)
            for n_changes in range(1, min(4, len(spending_pool) + 1)):
                changes = spending_pool[:n_changes]
                stop_ids = {c["event_id"] for c in changes if c["action"] == "stop"}
                reduce_ids = {c["event_id"] for c in changes if c["action"] == "reduce_to"}
                if stop_ids & reduce_ids:
                    continue
                safe = _check_plan_with_changes(
                    start_date=rd, start_balance=balance, events=events,
                    is_recurring=is_recurring, home_currency=currency,
                    rate_index=rate_index, minimum_balance=min_bal,
                    payment_debits=inst_debits, spending_changes=changes,
                )
                if safe:
                    sc_strs = _format_spending_changes(changes)
                    last_date = schedule[-1][0] if schedule else rd
                    skip_ids = {c["event_id"] for c in changes if c["action"] == "stop"}
                    reduce_amounts = {c["event_id"]: c["new_amount"] for c in changes if c["action"] == "reduce_to"}
                    candidates.append(CandidatePlan(
                        payment_method="installments",
                        payment_dates_amounts=schedule,
                        total_payable=opt.total_payable_amount,
                        meets_deadline=(last_date <= desired),
                        requires_spending_changes=True,
                        num_payments=opt.number_of_payments,
                        start_date=schedule[0][0] if schedule else rd,
                        payment_option_id=opt.payment_option_id,
                        spending_changes=sc_strs,
                        skip_event_ids=skip_ids,
                        reduce_event_amounts=reduce_amounts,
                    ))
                    break

    # === wait candidate ===
    if "full_payment" in methods and earliest_date is not None and earliest_date > rd:
        wait_debits = [(earliest_date, requested)]
        wait_safe = is_plan_safe(
            start_date=earliest_date,
            start_balance=_balance_at_date(
                rd, balance, events, is_recurring, currency, rate_index, earliest_date
            ),
            events=events,
            is_recurring=is_recurring,
            home_currency=currency,
            rate_index=rate_index,
            minimum_balance=min_bal,
            payment_debits=wait_debits,
        )
        if wait_safe:
            candidates.append(CandidatePlan(
                payment_method="wait",
                payment_dates_amounts=[(earliest_date, requested)],
                total_payable=requested,
                meets_deadline=(earliest_date <= desired),
                requires_spending_changes=False,
                num_payments=1,
                start_date=earliest_date,
            ))

    # --- Step 4: Rank candidates ---
    best = _rank_candidates(candidates)

    # --- Step 5: Determine affordability_status and format output ---
    if best is None:
        # not_affordable
        return DecisionResult(
            amount_safe_to_pay=amount_safe,
            affordability_status="not_affordable",
            recommended_payment_method="not_recommended",
            payment_plan="none",
            earliest_date_for_full_payment=earliest_str,
            spending_changes_needed="none",
        )

    method = best.payment_method
    spending_str = "|".join(best.spending_changes) if best.spending_changes else "none"
    plan_str = _format_plan(best.payment_dates_amounts)

    if method == "full_payment" and not best.requires_spending_changes:
        affordability = "affordable_now"
        # earliest must equal request_date
        earliest_str = rd.isoformat()
    elif method == "full_payment" and best.requires_spending_changes:
        affordability = "affordable_with_plan"
    elif method == "partial_payment":
        affordability = "affordable_with_plan"
    elif method == "installments":
        affordability = "affordable_with_plan"
    elif method == "wait":
        affordability = "affordable_later"
    else:
        affordability = "not_affordable"

    return DecisionResult(
        amount_safe_to_pay=amount_safe,
        affordability_status=affordability,
        recommended_payment_method=method,
        payment_plan=plan_str,
        earliest_date_for_full_payment=earliest_str,
        spending_changes_needed=spending_str,
    )


# ---------------------------------------------------------------------------
# 6-rule ranking
# ---------------------------------------------------------------------------

def _rank_candidates(candidates: List[CandidatePlan]) -> Optional[CandidatePlan]:
    """
    Rank candidates by 6 rules in order:
    1. meets_deadline (True before False)
    2. requires_spending_changes (False before True)
    3. Minimize total_payable
    4. Start earlier (smaller start_date)
    5. Fewer payments
    6. Lowest payment_option_id (string sort, None last)
    """
    if not candidates:
        return None

    def sort_key(c: CandidatePlan):
        return (
            0 if c.meets_deadline else 1,          # 1. meets deadline
            1 if c.requires_spending_changes else 0,  # 2. no spending changes
            c.total_payable,                         # 3. minimize total paid
            c.start_date.toordinal(),                # 4. start earlier
            c.num_payments,                          # 5. fewer payments
            c.payment_option_id or "zzz",            # 6. lowest option_id
        )

    candidates.sort(key=sort_key)
    return candidates[0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_plan(schedule: List[Tuple[date, float]]) -> str:
    if not schedule:
        return "none"
    parts = [f"{d.isoformat()}:{amount:.2f}" for d, amount in schedule]
    return "|".join(parts)


def _format_spending_changes(changes: List[Dict]) -> List[str]:
    result = []
    for c in changes:
        if c["action"] == "stop":
            result.append(f"stop:{c['event_id']}")
        elif c["action"] == "reduce_to":
            result.append(f"reduce_to:{c['event_id']}:{c['new_amount']:.2f}")
    return result


def _balance_at_date(
    start_date: date,
    start_balance: float,
    events: List[FinancialEvent],
    is_recurring: Dict[str, bool],
    home_currency: str,
    rate_index: RateLookup,
    target_date: date,
) -> float:
    """Simulate balance up to target_date and return the balance on that day."""
    horizon = (target_date - start_date).days
    if horizon <= 0:
        return start_balance
    trace = simulate_balance(
        start_date=start_date,
        start_balance=start_balance,
        events=events,
        is_recurring=is_recurring,
        home_currency=home_currency,
        rate_index=rate_index,
        horizon_days=horizon,
        extra_debits=[],
    )
    balance = start_balance
    for d, b in trace:
        if d <= target_date:
            balance = b
    return balance
