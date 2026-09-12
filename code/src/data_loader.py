"""
data_loader.py — Load and validate all dataset CSVs into typed Pydantic models.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

from .schema import (
    ExchangeRate,
    FinancialEvent,
    FinancialProfile,
    ImageRecord,
    Message,
    OutputRow,
    PaymentOption,
    RequestRow,
)

# ---------------------------------------------------------------------------
# Default dataset root
# ---------------------------------------------------------------------------
DATASET_DIR = Path(__file__).resolve().parents[3] / "dataset"


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------

REQUEST_FIELDS = {"request_id", "user_id", "request_date", "request_type", "requested_amount",
                  "desired_completion_date", "allows_partial_payment", "request_text"}


def load_requests(path: Path | None = None) -> List[RequestRow]:
    p = path or (DATASET_DIR / "requests.csv")
    rows = []
    for row in _read_csv(p):
        filtered = {k: v for k, v in row.items() if k in REQUEST_FIELDS}
        rows.append(RequestRow(**filtered))
    return rows


def load_sample_requests(path: Path | None = None) -> List[Dict[str, str]]:
    """Return raw dicts so evaluate.py can compare all columns flexibly."""
    p = path or (DATASET_DIR / "sample_requests.csv")
    return _read_csv(p)


def load_financial_profiles(path: Path | None = None) -> Dict[str, FinancialProfile]:
    p = path or (DATASET_DIR / "financial_profiles.csv")
    profiles = [FinancialProfile(**row) for row in _read_csv(p)]
    return {prof.user_id: prof for prof in profiles}


def load_financial_events(path: Path | None = None) -> List[FinancialEvent]:
    p = path or (DATASET_DIR / "financial_events.csv")
    return [FinancialEvent(**row) for row in _read_csv(p)]


def load_exchange_rates(path: Path | None = None) -> List[ExchangeRate]:
    p = path or (DATASET_DIR / "exchange_rates.csv")
    return [ExchangeRate(**row) for row in _read_csv(p)]


def load_payment_options(path: Path | None = None) -> Dict[str, List[PaymentOption]]:
    """Returns dict keyed by request_id → list of PaymentOption."""
    p = path or (DATASET_DIR / "request_payment_options.csv")
    options: Dict[str, List[PaymentOption]] = {}
    for row in _read_csv(p):
        opt = PaymentOption(**row)
        options.setdefault(opt.request_id, []).append(opt)
    return options


def load_messages(path: Path | None = None) -> List[Message]:
    p = path or (DATASET_DIR / "messages.csv")
    return [Message(**row) for row in _read_csv(p)]


def load_images(path: Path | None = None) -> List[ImageRecord]:
    p = path or (DATASET_DIR / "images.csv")
    return [ImageRecord(**row) for row in _read_csv(p)]


# ---------------------------------------------------------------------------
# Grouped helpers
# ---------------------------------------------------------------------------

def events_by_user(events: List[FinancialEvent]) -> Dict[str, List[FinancialEvent]]:
    result: Dict[str, List[FinancialEvent]] = {}
    for ev in events:
        result.setdefault(ev.user_id, []).append(ev)
    return result


def messages_by_user(messages: List[Message]) -> Dict[str, List[Message]]:
    result: Dict[str, List[Message]] = {}
    for msg in messages:
        result.setdefault(msg.user_id, []).append(msg)
    return result


def images_by_event(images: List[ImageRecord]) -> Dict[str, ImageRecord]:
    """Map related_event_id → ImageRecord."""
    return {img.related_event_id: img for img in images if img.related_event_id}
