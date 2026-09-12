"""
schema.py — Pydantic v2 models for every CSV row + output row.
All models are strict about types but lenient about missing optional fields.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import ClassVar, List, Optional
from pydantic import BaseModel, field_validator, model_validator


# ---------------------------------------------------------------------------
# Input row models
# ---------------------------------------------------------------------------

class RequestRow(BaseModel):
    request_id: str
    user_id: str
    request_date: date
    request_type: str
    requested_amount: float
    desired_completion_date: date
    allows_partial_payment: bool
    request_text: str

    @field_validator("allows_partial_payment", mode="before")
    @classmethod
    def parse_bool(cls, v: object) -> bool:
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() == "true"
        return bool(v)

    @field_validator("request_date", "desired_completion_date", mode="before")
    @classmethod
    def parse_date(cls, v: object) -> date:
        if isinstance(v, date):
            return v
        return date.fromisoformat(str(v).strip())

    @field_validator("requested_amount", mode="before")
    @classmethod
    def parse_amount(cls, v: object) -> float:
        return float(str(v).strip())


class FinancialProfile(BaseModel):
    user_id: str
    home_currency: str
    current_available_balance: float
    minimum_balance_to_keep: float
    financial_priorities: List[str]
    expense_categories_to_protect: List[str]
    expense_categories_user_is_willing_to_reduce: List[str]
    expense_categories_user_is_willing_to_stop: List[str]
    payment_methods_user_will_consider: List[str]
    max_installment_months: Optional[int]

    @field_validator(
        "financial_priorities",
        "expense_categories_to_protect",
        "expense_categories_user_is_willing_to_reduce",
        "expense_categories_user_is_willing_to_stop",
        "payment_methods_user_will_consider",
        mode="before",
    )
    @classmethod
    def parse_pipe_list(cls, v: object) -> List[str]:
        if isinstance(v, list):
            return v
        s = str(v).strip()
        if not s:
            return []
        return [x.strip() for x in s.split("|") if x.strip()]

    @field_validator("max_installment_months", mode="before")
    @classmethod
    def parse_optional_int(cls, v: object) -> Optional[int]:
        if v is None:
            return None
        s = str(v).strip()
        if not s or s.lower() == "none":
            return None
        return int(s)

    @field_validator("current_available_balance", "minimum_balance_to_keep", mode="before")
    @classmethod
    def parse_float(cls, v: object) -> float:
        return float(str(v).strip())


class FinancialEvent(BaseModel):
    event_id: str
    user_id: str
    event_type: str
    description: str
    category: str
    direction: str
    amount: Optional[float]  # None when blank in CSV
    currency: str
    event_date: Optional[date]
    settlement_date: Optional[date]
    status: str
    linked_event_id: Optional[str]
    flexibility: str
    minimum_allowed_amount: Optional[float]

    @field_validator("amount", "minimum_allowed_amount", mode="before")
    @classmethod
    def parse_optional_float(cls, v: object) -> Optional[float]:
        if v is None:
            return None
        s = str(v).strip()
        if not s or s.lower() == "none":
            return None
        return float(s)

    @field_validator("event_date", "settlement_date", mode="before")
    @classmethod
    def parse_date(cls, v: object) -> Optional[date]:
        if isinstance(v, date):
            return v
        s = str(v).strip()
        if not s:
            return None  # handled by model_validator below
        return date.fromisoformat(s)

    @model_validator(mode="after")
    def fill_missing_settlement_date(self) -> "FinancialEvent":
        if self.settlement_date is None:
            object.__setattr__(self, "settlement_date", self.event_date)
        if self.event_date is None:
            object.__setattr__(self, "event_date", self.settlement_date)
        return self

    @field_validator("linked_event_id", mode="before")
    @classmethod
    def parse_optional_str(cls, v: object) -> Optional[str]:
        if v is None:
            return None
        s = str(v).strip()
        return s if s and s.lower() != "none" else None


class ExchangeRate(BaseModel):
    rate_date: date
    from_currency: str
    to_currency: str
    rate: float

    @field_validator("rate_date", mode="before")
    @classmethod
    def parse_date(cls, v: object) -> date:
        if isinstance(v, date):
            return v
        return date.fromisoformat(str(v).strip())

    @field_validator("rate", mode="before")
    @classmethod
    def parse_float(cls, v: object) -> float:
        return float(str(v).strip())


class PaymentOption(BaseModel):
    payment_option_id: str
    request_id: str
    payment_method: str  # full_payment or installments
    payment_amount: float
    number_of_payments: int
    first_payment_date: date
    payment_frequency_days: Optional[int]
    financing_fee: float
    total_payable_amount: float

    @field_validator("first_payment_date", mode="before")
    @classmethod
    def parse_date(cls, v: object) -> date:
        if isinstance(v, date):
            return v
        return date.fromisoformat(str(v).strip())

    @field_validator("payment_frequency_days", mode="before")
    @classmethod
    def parse_optional_int(cls, v: object) -> Optional[int]:
        s = str(v).strip()
        if not s:
            return None
        return int(float(s))

    @field_validator("payment_amount", "financing_fee", "total_payable_amount", mode="before")
    @classmethod
    def parse_float(cls, v: object) -> float:
        return float(str(v).strip())

    @field_validator("number_of_payments", mode="before")
    @classmethod
    def parse_int(cls, v: object) -> int:
        return int(float(str(v).strip()))


class Message(BaseModel):
    message_id: str
    user_id: str
    request_id: Optional[str]
    related_event_id: Optional[str]
    sent_at: datetime
    source_type: str
    message_text: str

    @field_validator("request_id", "related_event_id", mode="before")
    @classmethod
    def parse_optional_str(cls, v: object) -> Optional[str]:
        s = str(v).strip()
        return s if s else None

    @field_validator("sent_at", mode="before")
    @classmethod
    def parse_datetime(cls, v: object) -> datetime:
        if isinstance(v, datetime):
            return v
        s = str(v).strip()
        # Handle trailing Z
        return datetime.fromisoformat(s.replace("Z", "+00:00"))


class ImageRecord(BaseModel):
    image_id: str
    user_id: str
    request_id: Optional[str]
    related_event_id: Optional[str]

    @field_validator("request_id", "related_event_id", mode="before")
    @classmethod
    def parse_optional_str(cls, v: object) -> Optional[str]:
        s = str(v).strip()
        return s if s else None


# ---------------------------------------------------------------------------
# Output row model
# ---------------------------------------------------------------------------

class OutputRow(BaseModel):
    request_id: str
    amount_safe_to_pay: float
    affordability_status: str
    recommended_payment_method: str
    payment_plan: str  # "none" or "YYYY-MM-DD:amount|..."
    earliest_date_for_full_payment: str  # "YYYY-MM-DD" or ""
    spending_changes_needed: str  # "none" or "stop:event_id|reduce_to:event_id:amount|..."
    decision_explanation: str

    VALID_AFFORDABILITY: ClassVar[frozenset] = frozenset(
        ["affordable_now", "affordable_with_plan", "affordable_later", "not_affordable"]
    )
    VALID_PAYMENT_METHOD: ClassVar[frozenset] = frozenset(
        ["full_payment", "partial_payment", "installments", "wait", "not_recommended"]
    )

    @model_validator(mode="after")
    def validate_constraints(self) -> "OutputRow":
        assert self.affordability_status in self.VALID_AFFORDABILITY, (
            f"Invalid affordability_status: {self.affordability_status}"
        )
        assert self.recommended_payment_method in self.VALID_PAYMENT_METHOD, (
            f"Invalid recommended_payment_method: {self.recommended_payment_method}"
        )
        assert self.amount_safe_to_pay >= 0, "amount_safe_to_pay must be >= 0"
        assert self.decision_explanation.strip(), "decision_explanation must not be empty"
        return self
