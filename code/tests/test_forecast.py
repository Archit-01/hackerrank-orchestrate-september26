"""
test_forecast.py — Unit tests for the 90-day balance simulator.
Uses hand-built fixtures.
"""
import pytest
from datetime import date, timedelta
from src.forecast import simulate_balance, is_plan_safe, min_balance_in_trace
from src.schema import FinancialEvent
from src.currency import build_rate_index
from src.schema import ExchangeRate


def make_rate_index():
    rates = [ExchangeRate(rate_date=date(2024, 1, 1), from_currency="USD", to_currency="USD", rate=1.0)]
    return build_rate_index(rates)


def make_event(event_id, direction, amount, settlement_date, status="scheduled", currency="USD", category="utilities", flexibility="fixed", event_type="expense"):
    return FinancialEvent(
        event_id=event_id,
        user_id="user_test",
        event_type=event_type,
        description="test",
        category=category,
        direction=direction,
        amount=amount,
        currency=currency,
        event_date=settlement_date,
        settlement_date=settlement_date,
        status=status,
        linked_event_id=None,
        flexibility=flexibility,
        minimum_allowed_amount=None,
    )


@pytest.fixture
def rate_index():
    return make_rate_index()


def test_no_events_returns_start_balance(rate_index):
    trace = simulate_balance(
        start_date=date(2024, 1, 1),
        start_balance=1000.0,
        events=[],
        is_recurring={},
        home_currency="USD",
        rate_index=rate_index,
    )
    assert trace[-1][1] == 1000.0


def test_single_future_debit(rate_index):
    ev = make_event("e1", "debit", 200.0, date(2024, 1, 10))
    trace = simulate_balance(
        start_date=date(2024, 1, 1),
        start_balance=1000.0,
        events=[ev],
        is_recurring={"e1": False},
        home_currency="USD",
        rate_index=rate_index,
    )
    # On Jan 10, balance should drop to 800
    balances = {d: b for d, b in trace}
    assert abs(balances[date(2024, 1, 10)] - 800.0) < 0.01


def test_single_future_credit(rate_index):
    ev = make_event("e1", "credit", 500.0, date(2024, 1, 15), event_type="income", status="scheduled")
    trace = simulate_balance(
        start_date=date(2024, 1, 1),
        start_balance=1000.0,
        events=[ev],
        is_recurring={"e1": False},
        home_currency="USD",
        rate_index=rate_index,
    )
    balances = {d: b for d, b in trace}
    assert abs(balances[date(2024, 1, 15)] - 1500.0) < 0.01


def test_pending_debit_included(rate_index):
    ev = make_event("e1", "debit", 100.0, date(2024, 1, 5), status="pending")
    trace = simulate_balance(
        start_date=date(2024, 1, 1),
        start_balance=1000.0,
        events=[ev],
        is_recurring={"e1": False},
        home_currency="USD",
        rate_index=rate_index,
    )
    balances = {d: b for d, b in trace}
    assert abs(balances[date(2024, 1, 5)] - 900.0) < 0.01


def test_pending_credit_excluded(rate_index):
    ev = make_event("e1", "credit", 500.0, date(2024, 1, 5), status="pending")
    trace = simulate_balance(
        start_date=date(2024, 1, 1),
        start_balance=1000.0,
        events=[ev],
        is_recurring={"e1": False},
        home_currency="USD",
        rate_index=rate_index,
    )
    # Pending credit should not be included; balance stays 1000
    assert trace[-1][1] == pytest.approx(1000.0)


def test_cancelled_event_excluded(rate_index):
    ev = make_event("e1", "debit", 200.0, date(2024, 1, 5), status="cancelled")
    trace = simulate_balance(
        start_date=date(2024, 1, 1),
        start_balance=1000.0,
        events=[ev],
        is_recurring={"e1": False},
        home_currency="USD",
        rate_index=rate_index,
    )
    assert trace[-1][1] == pytest.approx(1000.0)


def test_is_plan_safe_breach(rate_index):
    ev = make_event("e1", "debit", 800.0, date(2024, 1, 15))
    # Balance = 1000, min = 300, paying 600 leaves 400, then debit of 800 -> -400 < 300
    result = is_plan_safe(
        start_date=date(2024, 1, 1),
        start_balance=1000.0,
        events=[ev],
        is_recurring={"e1": False},
        home_currency="USD",
        rate_index=rate_index,
        minimum_balance=300.0,
        payment_debits=[(date(2024, 1, 1), 600.0)],
    )
    assert result is False


def test_is_plan_safe_ok(rate_index):
    ev = make_event("e1", "debit", 100.0, date(2024, 1, 15))
    result = is_plan_safe(
        start_date=date(2024, 1, 1),
        start_balance=1000.0,
        events=[ev],
        is_recurring={"e1": False},
        home_currency="USD",
        rate_index=rate_index,
        minimum_balance=300.0,
        payment_debits=[(date(2024, 1, 1), 200.0)],
    )
    # After payment: 800. After debit: 700. 700 >= 300. OK.
    assert result is True


def test_recurring_projection(rate_index):
    # Monthly recurring debit
    ev = make_event("e1", "debit", 100.0, date(2023, 12, 1), status="settled", flexibility="fixed")
    # Another settled occurrence
    ev2 = make_event("e2", "debit", 100.0, date(2023, 11, 1), status="settled", flexibility="fixed")
    is_recurring = {"e1": True, "e2": True}

    trace = simulate_balance(
        start_date=date(2024, 1, 1),
        start_balance=1000.0,
        events=[ev, ev2],
        is_recurring=is_recurring,
        home_currency="USD",
        rate_index=rate_index,
        horizon_days=60,
    )
    # Should see recurring projections
    balances = {d: b for d, b in trace}
    # At least one projection should have happened
    assert any(b < 1000.0 for _, b in trace)
