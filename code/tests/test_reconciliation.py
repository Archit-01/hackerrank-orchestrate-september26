"""
test_reconciliation.py — Unit tests for reconciliation pipeline.
Uses hand-built fixtures.
"""
import pytest
from datetime import date, datetime, timezone
from src.schema import FinancialEvent, FinancialProfile, Message, ImageRecord
from src.reconciliation import (
    resolve_linked_events,
    deduplicate_events,
    classify_recurring,
    apply_message_amendments,
)


def ev(event_id, status="settled", linked=None, ev_date=None, amount=100.0, category="utilities", direction="debit", flexibility="fixed"):
    return FinancialEvent(
        event_id=event_id,
        user_id="user_test",
        event_type="expense",
        description="test",
        category=category,
        direction=direction,
        amount=amount,
        currency="USD",
        event_date=ev_date or date(2024, 1, 1),
        settlement_date=ev_date or date(2024, 1, 1),
        status=status,
        linked_event_id=linked,
        flexibility=flexibility,
        minimum_allowed_amount=None,
    )


class TestResolveLinkedEvents:
    def test_cancellation_suppresses_parent(self):
        parent = ev("e1", status="settled")
        child = ev("e2", status="cancelled", linked="e1")
        result = resolve_linked_events([parent, child])
        event_ids = [e.event_id for e in result]
        assert "e1" not in event_ids
        assert "e2" not in event_ids

    def test_settled_child_suppresses_parent(self):
        parent = ev("e1", status="pending")
        child = ev("e2", status="settled", linked="e1")
        result = resolve_linked_events([parent, child])
        event_ids = [e.event_id for e in result]
        assert "e1" not in event_ids
        assert "e2" in event_ids  # child kept

    def test_newer_child_suppresses_parent(self):
        parent = ev("e1", status="pending", ev_date=date(2024, 1, 1))
        child = ev("e2", status="pending", linked="e1", ev_date=date(2024, 1, 10))
        result = resolve_linked_events([parent, child])
        event_ids = [e.event_id for e in result]
        assert "e1" not in event_ids

    def test_no_linked_event_unchanged(self):
        e1 = ev("e1")
        e2 = ev("e2")
        result = resolve_linked_events([e1, e2])
        assert len(result) == 2


class TestDeduplication:
    def test_duplicate_event_ids_kept_best(self):
        e1 = ev("e1", status="pending")
        e2 = ev("e1", status="settled")  # same ID, settled wins
        result = deduplicate_events([e1, e2])
        assert len(result) == 1
        assert result[0].status == "settled"

    def test_unique_events_all_kept(self):
        e1 = ev("e1")
        e2 = ev("e2")
        result = deduplicate_events([e1, e2])
        assert len(result) == 2


class TestClassifyRecurring:
    def test_monthly_recurring(self):
        events = [
            ev("e1", ev_date=date(2024, 1, 1), status="settled"),
            ev("e2", ev_date=date(2024, 2, 1), status="settled"),
            ev("e3", ev_date=date(2024, 3, 1), status="settled"),
        ]
        result = classify_recurring(events)
        assert result["e1"] is True
        assert result["e2"] is True

    def test_single_event_not_recurring(self):
        events = [ev("e1", ev_date=date(2024, 1, 1), status="settled")]
        result = classify_recurring(events)
        assert result["e1"] is False

    def test_irregular_events_not_recurring(self):
        events = [
            ev("e1", ev_date=date(2024, 1, 1), status="settled"),
            ev("e2", ev_date=date(2024, 6, 1), status="settled"),  # 150+ days gap
        ]
        result = classify_recurring(events)
        assert result["e1"] is False


class TestMessageAmendments:
    def _msg(self, related_event_id, text, intent_override=None):
        return Message(
            message_id="msg1",
            user_id="user_test",
            request_id=None,
            related_event_id=related_event_id,
            sent_at=datetime(2024, 1, 5, tzinfo=timezone.utc),
            source_type="bank",
            message_text=text,
        )

    def test_no_messages_unchanged(self):
        events = [ev("e1")]
        result = apply_message_amendments(events, [], use_llm=False)
        assert len(result) == 1

    def test_message_without_event_id_unchanged(self):
        events = [ev("e1")]
        msg = self._msg(None, "some message")
        result = apply_message_amendments(events, [msg], use_llm=False)
        assert len(result) == 1
