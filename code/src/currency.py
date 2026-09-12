"""
currency.py — Dated FX conversion using exchange_rates.csv only.
Never fetches live rates.

Strategy:
  1. Look for an exact (from_currency, to_currency, rate_date) match.
  2. Fall back to the closest prior date for that pair.
  3. If the direct pair is not available, chain via USD as a bridge currency.
  4. If the currencies are the same, return amount unchanged.
  5. Raise ValueError if no conversion path can be found.
"""
from __future__ import annotations

from datetime import date
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from .schema import ExchangeRate


# ---------------------------------------------------------------------------
# Build lookup index
# ---------------------------------------------------------------------------

RateLookup = Dict[Tuple[str, str], List[Tuple[date, float]]]


def build_rate_index(rates: List[ExchangeRate]) -> RateLookup:
    """
    Returns a dict:  (from_currency, to_currency) -> sorted list of (date, rate).
    """
    index: RateLookup = {}
    for r in rates:
        key = (r.from_currency, r.to_currency)
        index.setdefault(key, []).append((r.rate_date, r.rate))
    # Sort each list by date ascending
    for key in index:
        index[key].sort(key=lambda x: x[0])
    return index


def _lookup_direct(
    index: RateLookup,
    from_curr: str,
    to_curr: str,
    on_date: date,
) -> Optional[float]:
    """
    Find the rate for (from_curr, to_curr) on or before on_date.
    Returns None if no rate available.
    """
    # Try direct pair
    key = (from_curr, to_curr)
    if key in index:
        entries = index[key]  # sorted ascending by date
        # Find largest date <= on_date
        best_rate: Optional[float] = None
        for d, r in entries:
            if d <= on_date:
                best_rate = r
            else:
                break
        if best_rate is not None:
            return best_rate

    # Try inverse pair
    inv_key = (to_curr, from_curr)
    if inv_key in index:
        entries = index[inv_key]
        best_rate = None
        for d, r in entries:
            if d <= on_date:
                best_rate = r
            else:
                break
        if best_rate is not None:
            return 1.0 / best_rate

    return None


def convert(
    amount: float,
    from_currency: str,
    to_currency: str,
    on_date: date,
    rate_index: RateLookup,
) -> float:
    """
    Convert amount from from_currency to to_currency using the latest rate
    on or before on_date.

    Raises ValueError if no conversion path is found.
    """
    if from_currency == to_currency:
        return amount

    # Try direct lookup
    rate = _lookup_direct(rate_index, from_currency, to_currency, on_date)
    if rate is not None:
        return amount * rate

    # Try bridging through USD
    bridge_currencies = ["USD", "EUR"]
    for bridge in bridge_currencies:
        if bridge in (from_currency, to_currency):
            continue
        r1 = _lookup_direct(rate_index, from_currency, bridge, on_date)
        r2 = _lookup_direct(rate_index, bridge, to_currency, on_date)
        if r1 is not None and r2 is not None:
            return amount * r1 * r2

    raise ValueError(
        f"No FX rate found for {from_currency} -> {to_currency} on or before {on_date}"
    )
