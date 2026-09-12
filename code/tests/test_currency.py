"""
test_currency.py — Unit tests for currency conversion.
Uses hand-built fixtures, not the real dataset.
"""
import pytest
from datetime import date
from src.currency import build_rate_index, convert
from src.schema import ExchangeRate


def make_rates():
    return [
        ExchangeRate(rate_date=date(2024, 1, 15), from_currency="USD", to_currency="INR", rate=83.33),
        ExchangeRate(rate_date=date(2024, 2, 15), from_currency="USD", to_currency="INR", rate=84.00),
        ExchangeRate(rate_date=date(2024, 1, 15), from_currency="EUR", to_currency="USD", rate=1.09),
        ExchangeRate(rate_date=date(2024, 1, 15), from_currency="USD", to_currency="EUR", rate=0.92),
        ExchangeRate(rate_date=date(2024, 1, 15), from_currency="EUR", to_currency="ZAR", rate=20.0),
        ExchangeRate(rate_date=date(2024, 1, 15), from_currency="USD", to_currency="IDR", rate=15833.33),
    ]


@pytest.fixture
def rate_index():
    return build_rate_index(make_rates())


def test_same_currency(rate_index):
    assert convert(100.0, "INR", "INR", date(2024, 1, 15), rate_index) == 100.0


def test_direct_conversion(rate_index):
    result = convert(100.0, "USD", "INR", date(2024, 1, 15), rate_index)
    assert abs(result - 8333.0) < 0.1


def test_exact_date_match(rate_index):
    result = convert(1.0, "USD", "INR", date(2024, 2, 15), rate_index)
    assert abs(result - 84.0) < 0.01


def test_prior_date_fallback(rate_index):
    # date 2024-01-20 — should use 2024-01-15 rate (closest prior)
    result = convert(1.0, "USD", "INR", date(2024, 1, 20), rate_index)
    assert abs(result - 83.33) < 0.01


def test_inverse_conversion(rate_index):
    # INR -> USD uses inverse of USD->INR rate
    result = convert(8333.0, "INR", "USD", date(2024, 1, 15), rate_index)
    assert abs(result - 100.0) < 0.5


def test_bridge_conversion(rate_index):
    # EUR -> INR via EUR->USD->INR
    result = convert(1.0, "EUR", "INR", date(2024, 1, 15), rate_index)
    expected = 1.09 * 83.33  # EUR->USD->INR
    assert abs(result - expected) < 0.5


def test_no_rate_raises(rate_index):
    with pytest.raises(ValueError):
        convert(100.0, "ZAR", "IDR", date(2020, 1, 1), rate_index)


def test_direct_eur_to_zar(rate_index):
    result = convert(1.0, "EUR", "ZAR", date(2024, 1, 15), rate_index)
    assert abs(result - 20.0) < 0.01
