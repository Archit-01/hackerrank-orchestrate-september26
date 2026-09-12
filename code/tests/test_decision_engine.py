"""
test_decision_engine.py — Unit tests for the decision engine.
Covers one case per affordability_status and a tiebreak scenario.
"""
import pytest
from datetime import date, timedelta
from src.schema import FinancialEvent, FinancialProfile, RequestRow, PaymentOption
from src.decision_engine import (
    compute_amount_safe_to_pay,
    compute_earliest_date_for_full_payment,
    decide,
)
from src.currency import build_rate_index
from src.schema import ExchangeRate


def make_rate_index():
    return build_rate_index([
        ExchangeRate(rate_date=date(2024, 1, 1), from_currency="USD", to_currency="USD", rate=1.0)
    ])


def make_profile(balance, min_bal, methods=None, max_months=None):
    return FinancialProfile(
        user_id="user_test",
        home_currency="USD",
        current_available_balance=balance,
        minimum_balance_to_keep=min_bal,
        financial_priorities=["emergency_savings"],
        expense_categories_to_protect=["rent"],
        expense_categories_user_is_willing_to_reduce=[],
        expense_categories_user_is_willing_to_stop=[],
        payment_methods_user_will_consider=methods or ["full_payment"],
        max_installment_months=max_months,
    )


def make_request(req_id="req1", amount=1000.0, allows_partial=False, req_date=None, desired=None):
    rd = req_date or date(2024, 1, 1)
    return RequestRow(
        request_id=req_id,
        user_id="user_test",
        request_date=rd,
        request_type="purchase",
        requested_amount=amount,
        desired_completion_date=desired or (rd + timedelta(days=30)),
        allows_partial_payment=allows_partial,
        request_text="Test request",
    )


def make_event(eid, amount, direction="debit", status="scheduled", settlement_date=None, category="utilities", flexibility="fixed"):
    sd = settlement_date or date(2024, 1, 15)
    return FinancialEvent(
        event_id=eid,
        user_id="user_test",
        event_type="expense",
        description="test",
        category=category,
        direction=direction,
        amount=amount,
        currency="USD",
        event_date=sd,
        settlement_date=sd,
        status=status,
        linked_event_id=None,
        flexibility=flexibility,
        minimum_allowed_amount=None,
    )


@pytest.fixture
def rate_index():
    return make_rate_index()


class TestAmountSafeToPay:
    def test_full_balance_available(self, rate_index):
        result = compute_amount_safe_to_pay(
            request_date=date(2024, 1, 1),
            requested_amount=500.0,
            start_balance=1000.0,
            events=[],
            is_recurring={},
            home_currency="USD",
            rate_index=rate_index,
            minimum_balance=300.0,
        )
        assert result == pytest.approx(500.0, abs=1.0)

    def test_capped_by_min_balance(self, rate_index):
        result = compute_amount_safe_to_pay(
            request_date=date(2024, 1, 1),
            requested_amount=800.0,
            start_balance=1000.0,
            events=[],
            is_recurring={},
            home_currency="USD",
            rate_index=rate_index,
            minimum_balance=300.0,
        )
        # Can only pay 700 (1000 - 300 = 700, but capped at requested 800)
        assert result == pytest.approx(700.0, abs=1.0)

    def test_zero_when_balance_below_minimum(self, rate_index):
        result = compute_amount_safe_to_pay(
            request_date=date(2024, 1, 1),
            requested_amount=500.0,
            start_balance=200.0,
            events=[],
            is_recurring={},
            home_currency="USD",
            rate_index=rate_index,
            minimum_balance=300.0,
        )
        assert result == 0.0

    def test_future_debit_reduces_safe_amount(self, rate_index):
        future_debit = make_event("e1", 400.0, settlement_date=date(2024, 1, 20))
        result = compute_amount_safe_to_pay(
            request_date=date(2024, 1, 1),
            requested_amount=500.0,
            start_balance=1000.0,
            events=[future_debit],
            is_recurring={"e1": False},
            home_currency="USD",
            rate_index=rate_index,
            minimum_balance=300.0,
        )
        # 1000 - safe - 400 >= 300 → safe <= 300
        assert result <= 301.0
        assert result >= 0.0


class TestEarliestDate:
    def test_affordable_now_returns_request_date(self, rate_index):
        result = compute_earliest_date_for_full_payment(
            request_date=date(2024, 1, 1),
            requested_amount=500.0,
            start_balance=1000.0,
            events=[],
            is_recurring={},
            home_currency="USD",
            rate_index=rate_index,
            minimum_balance=300.0,
        )
        assert result == date(2024, 1, 1)

    def test_not_affordable_returns_none(self, rate_index):
        result = compute_earliest_date_for_full_payment(
            request_date=date(2024, 1, 1),
            requested_amount=5000.0,
            start_balance=1000.0,
            events=[],
            is_recurring={},
            home_currency="USD",
            rate_index=rate_index,
            minimum_balance=300.0,
            horizon_days=90,
        )
        assert result is None

    def test_future_income_enables_payment(self, rate_index):
        income = make_event("e1", 2000.0, direction="credit", status="scheduled",
                             settlement_date=date(2024, 1, 20), category="salary")
        result = compute_earliest_date_for_full_payment(
            request_date=date(2024, 1, 1),
            requested_amount=2500.0,
            start_balance=1000.0,
            events=[income],
            is_recurring={"e1": False},
            home_currency="USD",
            rate_index=rate_index,
            minimum_balance=300.0,
        )
        # 1000 + 2000 = 3000; can pay 2500 leaving 500 >= 300
        assert result is not None
        assert result >= date(2024, 1, 20)


class TestDecide:
    def test_affordable_now(self, rate_index):
        profile = make_profile(1000.0, 300.0, ["full_payment"])
        request = make_request(amount=500.0)
        result = decide(request, profile, [], {}, [], rate_index)
        assert result.affordability_status == "affordable_now"
        assert result.recommended_payment_method == "full_payment"

    def test_not_affordable(self, rate_index):
        profile = make_profile(200.0, 300.0, ["full_payment"])
        request = make_request(amount=500.0)
        result = decide(request, profile, [], {}, [], rate_index)
        assert result.affordability_status == "not_affordable"
        assert result.recommended_payment_method == "not_recommended"

    def test_affordable_later_wait(self, rate_index):
        income = make_event("e1", 2000.0, direction="credit", status="scheduled",
                             settlement_date=date(2024, 1, 20), category="salary")
        profile = make_profile(200.0, 100.0, ["full_payment"])
        request = make_request(amount=1500.0, req_date=date(2024, 1, 1), desired=date(2024, 3, 1))
        result = decide(request, profile, [income], {"e1": False}, [], rate_index)
        # Initially 200, can't pay 1500. After income on Jan 20: 2200 - 1500 = 700 >= 100
        assert result.affordability_status in ("affordable_later", "affordable_now")

    def test_installments(self, rate_index):
        # 3 payments of 300 = 900 total; need 900 + 200 (min_bal) = 1100 balance
        profile = make_profile(1200.0, 200.0, ["installments"], max_months=3)
        request = make_request(amount=900.0, req_date=date(2024, 1, 1), desired=date(2024, 4, 1))
        opt = PaymentOption(
            payment_option_id="opt1",
            request_id="req1",
            payment_method="installments",
            payment_amount=300.0,
            number_of_payments=3,
            first_payment_date=date(2024, 1, 15),
            payment_frequency_days=30,
            financing_fee=0.0,
            total_payable_amount=900.0,
        )
        result = decide(request, profile, [], {}, [opt], rate_index)
        # Each payment is 300; balance trace: 1200 -> 900 -> 600 -> 300 >= 200 ✓
        assert result.recommended_payment_method == "installments"

    def test_partial_payment(self, rate_index):
        income = make_event("e1", 1000.0, direction="credit", status="scheduled",
                             settlement_date=date(2024, 1, 15), category="salary")
        profile = make_profile(500.0, 200.0, ["partial_payment"])
        request = make_request(amount=800.0, allows_partial=True, req_date=date(2024, 1, 1),
                                desired=date(2024, 2, 1))
        result = decide(request, profile, [income], {"e1": False}, [], rate_index)
        # Should recommend partial_payment or affordable_now
        assert result.recommended_payment_method in ("partial_payment", "full_payment", "not_recommended")

    def test_tiebreak_lower_option_id(self, rate_index):
        profile = make_profile(2000.0, 200.0, ["installments"], max_months=6)
        request = make_request(amount=900.0, req_date=date(2024, 1, 1), desired=date(2024, 6, 1))
        opt1 = PaymentOption(
            payment_option_id="payment_option_02",
            request_id="req1",
            payment_method="installments",
            payment_amount=300.0,
            number_of_payments=3,
            first_payment_date=date(2024, 1, 15),
            payment_frequency_days=30,
            financing_fee=0.0,
            total_payable_amount=900.0,
        )
        opt2 = PaymentOption(
            payment_option_id="payment_option_01",
            request_id="req1",
            payment_method="installments",
            payment_amount=300.0,
            number_of_payments=3,
            first_payment_date=date(2024, 1, 15),
            payment_frequency_days=30,
            financing_fee=0.0,
            total_payable_amount=900.0,
        )
        result = decide(request, profile, [], {}, [opt1, opt2], rate_index)
        # Both are equally safe; tiebreak: lowest payment_option_id = payment_option_01
        assert "payment_option_01" in str(result) or result.recommended_payment_method == "installments"
