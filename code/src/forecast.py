"""
forecast.py — 90-day balance simulator (pure deterministic function).

Rules:
  - Start from current_available_balance on start_date.
  - Apply all future committed cash flows (debits = outflows, credits = inflows).
  - Include: settled future events, scheduled, pending debits.
  - Exclude: cancelled, failed, unrealized (investment valuations), pending credits.
  - Project recurring expenses forward based on their detected period.
  - Convert all amounts to home_currency using dated FX rates.
  - Return a sorted list of (date, end_of_day_balance) for every event day.
"""
from __future__ import annotations

import statistics
from collections import defaultdict
from datetime import date, timedelta
from typing import Dict, List, Optional, Set, Tuple

from .schema import FinancialEvent, FinancialProfile
from .currency import convert, RateLookup

FORECAST_DAYS = 90

# Statuses that contribute to cash flow
CASH_FLOW_STATUSES = {"settled", "scheduled", "pending"}


def _project_recurring(
    latest_event: FinancialEvent,
    start_date: date,
    end_date: date,
    period_days: int,
) -> List[Tuple[date, float]]:
    """
    Project a recurring event forward from the LATEST KNOWN occurrence in the dataset.
    Only returns occurrences that fall within [start_date, end_date] AND are strictly
    after the latest_event's settlement_date.
    """
    occurrences = []
    next_date = latest_event.settlement_date + timedelta(days=period_days)

    # Advance until we are at or after start_date
    while next_date < start_date:
        next_date = next_date + timedelta(days=period_days)

    while next_date <= end_date:
        if next_date > latest_event.settlement_date:
            occurrences.append((next_date, latest_event.amount or 0.0))
        next_date = next_date + timedelta(days=period_days)

    return occurrences


def simulate_balance(
    start_date: date,
    start_balance: float,
    events: List[FinancialEvent],
    is_recurring: Dict[str, bool],
    home_currency: str,
    rate_index: RateLookup,
    horizon_days: int = FORECAST_DAYS,
    extra_debits: Optional[List[Tuple[date, float]]] = None,
    skip_event_ids: Optional[Set[str]] = None,
    reduce_event_amounts: Optional[Dict[str, float]] = None,
) -> List[Tuple[date, float]]:
    """
    Simulate daily balance for horizon_days from start_date.

    Args:
        start_date: The date from which to simulate.
        start_balance: Starting balance in home_currency.
        events: Reconciled financial events.
        is_recurring: Map of event_id -> bool.
        home_currency: User's home currency.
        rate_index: FX rate index.
        horizon_days: Number of days to simulate.
        extra_debits: Additional (date, amount_in_home_currency) debits to inject.
        skip_event_ids: Event IDs to exclude (for spending-change simulation).
        reduce_event_amounts: event_id -> new_amount to override amounts.

    Returns:
        List of (date, balance_at_end_of_day), sorted by date.
    """
    end_date = start_date + timedelta(days=horizon_days)
    skip_event_ids = skip_event_ids or set()
    reduce_event_amounts = reduce_event_amounts or {}

    # Collect all cash flows as (date, signed_delta) where negative = outflow
    cash_flows: Dict[date, float] = defaultdict(float)

    # Track median periods for recurring events per group
    # (user_id, category, direction) -> period_days
    group_periods: Dict[Tuple, int] = _compute_group_periods(events, is_recurring)
    
    # Find latest event for each recurring group
    latest_recurring_event: Dict[Tuple, FinancialEvent] = {}

    for ev in events:
        if ev.event_id in skip_event_ids or ev.direction == "non_cash" or ev.status not in CASH_FLOW_STATUSES:
            continue
        if ev.status == "pending" and ev.direction == "credit":
            continue

        effective_amount = reduce_event_amounts.get(ev.event_id, ev.amount)
        if effective_amount is None:
            continue

        try:
            hc_amount = convert(effective_amount, ev.currency, home_currency, ev.settlement_date, rate_index)
        except ValueError:
            try:
                hc_amount = convert(effective_amount, ev.currency, home_currency, start_date, rate_index)
            except ValueError:
                continue

        signed = -hc_amount if ev.direction == "debit" else hc_amount

        # Add actual event if it's strictly in the future (or pending today)
        if start_date < ev.settlement_date <= end_date:
            cash_flows[ev.settlement_date] += signed
        elif ev.settlement_date <= start_date and ev.status in ("scheduled", "pending"):
            cash_flows[start_date] += signed

        if is_recurring.get(ev.event_id, False):
            key = (ev.user_id, ev.category, ev.direction)
            if key not in latest_recurring_event:
                latest_recurring_event[key] = ev
            elif ev.settlement_date > latest_recurring_event[key].settlement_date:
                latest_recurring_event[key] = ev

    # Project recurring events forward from their latest known occurrence
    for key, latest_ev in latest_recurring_event.items():
        period = group_periods.get(key, 30)
        occurrences = _project_recurring(latest_ev, start_date, end_date, period)
        for occ_date, occ_amount in occurrences:
            eff_amt = reduce_event_amounts.get(latest_ev.event_id, occ_amount)
            try:
                hc_amt = convert(eff_amt, latest_ev.currency, home_currency, occ_date, rate_index)
            except ValueError:
                hc_amt = convert(eff_amt, latest_ev.currency, home_currency, start_date, rate_index)
            s = -hc_amt if latest_ev.direction == "debit" else hc_amt
            cash_flows[occ_date] += s

    # Apply extra debits (e.g., payment plan payments)
    if extra_debits:
        for d, amt in extra_debits:
            if start_date <= d <= end_date:
                cash_flows[d] -= amt

    # Build daily balance trace
    all_dates = sorted(cash_flows.keys())
    balance = start_balance
    trace = []
    prev_date = start_date

    for d in all_dates:
        if d < start_date:
            continue
        balance += cash_flows[d]
        trace.append((d, balance))

    # If no events, return trace with just start balance
    if not trace:
        trace = [(start_date, start_balance)]

    return trace


def min_balance_in_trace(trace: List[Tuple[date, float]]) -> float:
    """Return the minimum balance across all entries in the trace."""
    if not trace:
        return float("inf")
    return min(b for _, b in trace)


def is_plan_safe(
    start_date: date,
    start_balance: float,
    events: List[FinancialEvent],
    is_recurring: Dict[str, bool],
    home_currency: str,
    rate_index: RateLookup,
    minimum_balance: float,
    payment_debits: List[Tuple[date, float]],
    horizon_days: int = FORECAST_DAYS,
    skip_event_ids: Optional[Set[str]] = None,
    reduce_event_amounts: Optional[Dict[str, float]] = None,
) -> bool:
    """
    Check whether the balance never falls below minimum_balance_to_keep
    across the entire horizon, given the proposed payment_debits.
    """
    trace = simulate_balance(
        start_date=start_date,
        start_balance=start_balance,
        events=events,
        is_recurring=is_recurring,
        home_currency=home_currency,
        rate_index=rate_index,
        horizon_days=horizon_days,
        extra_debits=payment_debits,
        skip_event_ids=skip_event_ids,
        reduce_event_amounts=reduce_event_amounts,
    )
    return min_balance_in_trace(trace) >= minimum_balance


def _compute_group_periods(
    events: List[FinancialEvent],
    is_recurring: Dict[str, bool],
) -> Dict[Tuple, int]:
    """Compute median gap (in days) for each recurring group."""
    group_dates: Dict[Tuple, List[date]] = defaultdict(list)
    for ev in events:
        if not is_recurring.get(ev.event_id, False):
            continue
        if ev.direction == "non_cash":
            continue
        key = (ev.user_id, ev.category, ev.direction)
        group_dates[key].append(ev.settlement_date)

    periods: Dict[Tuple, int] = {}
    for key, dates in group_dates.items():
        sorted_dates = sorted(dates)
        if len(sorted_dates) < 2:
            periods[key] = 30
            continue
        gaps = [(sorted_dates[i + 1] - sorted_dates[i]).days for i in range(len(sorted_dates) - 1)]
        gaps = [g for g in gaps if 1 <= g <= 45]
        if not gaps:
            periods[key] = 30
            continue
        periods[key] = max(1, int(statistics.median(gaps)))

    return periods
