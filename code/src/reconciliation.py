"""
reconciliation.py — Resolve conflicts, apply messages/images, classify events.

Steps:
  1. Resolve linked_event_id chains (collapse related events per conflict rules).
  2. Apply message amendments (cancel/amend/delay from LLM classification).
  3. Fill blank amounts from image extractions.
  4. Deduplicate (prefer most recent, prefer settled).
  5. Classify events: recurring vs one-time, essential vs flexible.

Conflict resolution precedence (from spec):
  1. Explicit cancellation/settlement/amendment
  2. Newer record from same source
  3. Settled over estimate/forecast
  4. Financially safer interpretation
"""
from __future__ import annotations

import statistics
from collections import defaultdict
from datetime import date, timedelta
from typing import Dict, List, Optional, Set, Tuple

from .schema import FinancialEvent, FinancialProfile, ImageRecord, Message

# Minimum recurring instances to classify as recurring
MIN_RECURRING_INSTANCES = 2
# Max gap in days between occurrences to classify as recurring
MAX_RECURRING_GAP_DAYS = 45


# ---------------------------------------------------------------------------
# Step 1: Resolve linked_event_id chains
# ---------------------------------------------------------------------------

def resolve_linked_events(events: List[FinancialEvent]) -> List[FinancialEvent]:
    """
    For events connected via linked_event_id, keep only the authoritative record.

    Precedence:
      1. Status = cancelled → parent is cancelled (remove both or mark cancelled)
      2. Status = settled (amendment) → use the settled record
      3. Newer record from same source (by event_date)
      4. Settled over pending/scheduled
      5. Safer interpretation
    """
    # Build a map of event_id -> event
    ev_map: Dict[str, FinancialEvent] = {ev.event_id: ev for ev in events}

    # Build chains: parent_id -> list of child events
    children: Dict[str, List[FinancialEvent]] = defaultdict(list)
    for ev in events:
        if ev.linked_event_id and ev.linked_event_id in ev_map:
            children[ev.linked_event_id].append(ev)

    # Determine which event_ids to suppress
    suppressed: Set[str] = set()

    for parent_id, child_list in children.items():
        parent = ev_map.get(parent_id)
        if parent is None:
            continue

        for child in child_list:
            # Rule 1: explicit cancellation
            if child.status == "cancelled":
                suppressed.add(parent_id)
                suppressed.add(child.event_id)
                continue

            # Rule 2/3: amendment (child amends parent)
            if child.status == "settled":
                # Child settles/amends parent → suppress parent, keep child
                suppressed.add(parent_id)
                continue
            elif parent.status == "settled" and child.status != "settled":
                # Parent is settled, child is not → suppress child
                suppressed.add(child.event_id)
                continue
            else:
                # Newer record wins
                if child.event_date >= parent.event_date:
                    suppressed.add(parent_id)
                else:
                    suppressed.add(child.event_id)

    return [ev for ev in events if ev.event_id not in suppressed]


# ---------------------------------------------------------------------------
# Step 2: Apply message amendments
# ---------------------------------------------------------------------------

def apply_message_amendments(
    events: List[FinancialEvent],
    messages: List[Message],
    use_llm: bool = True,
) -> List[FinancialEvent]:
    """
    For each message with a related_event_id, classify intent and apply amendments.
    """
    # Index messages by related_event_id
    msg_by_event: Dict[str, List[Message]] = defaultdict(list)
    for msg in messages:
        if msg.related_event_id:
            msg_by_event[msg.related_event_id].append(msg)

    if not msg_by_event:
        return events

    # Build mutable copy
    ev_map: Dict[str, FinancialEvent] = {ev.event_id: ev for ev in events}
    suppressed: Set[str] = set()

    for event_id, msgs in msg_by_event.items():
        if event_id not in ev_map:
            continue
        ev = ev_map[event_id]

        # Sort messages by sent_at ascending; last message wins
        msgs_sorted = sorted(msgs, key=lambda m: m.sent_at)

        for msg in msgs_sorted:
            if use_llm:
                try:
                    from .llm.message_extraction import classify_message
                    event_context = (
                        f"{ev.event_type} {ev.category} {ev.direction} "
                        f"amount={ev.amount} currency={ev.currency} "
                        f"date={ev.settlement_date} status={ev.status}"
                    )
                    result = classify_message(msg.message_text, event_context)
                except Exception as e:
                    print(f"[reconciliation] LLM message classification failed: {e}")
                    result = {"intent": "irrelevant", "new_amount": None, "new_date": None}
            else:
                result = {"intent": "irrelevant", "new_amount": None, "new_date": None}

            intent = result.get("intent", "irrelevant")
            new_amount = result.get("new_amount")
            new_date = result.get("new_date")

            if intent == "cancel":
                suppressed.add(event_id)
            elif intent == "amend" and new_amount is not None:
                # Update event amount (create new object)
                ev = FinancialEvent(
                    **{**ev.model_dump(), "amount": new_amount}
                )
                ev_map[event_id] = ev
            elif intent == "delay" and new_date is not None:
                from datetime import date as date_type
                parsed_date = date_type.fromisoformat(new_date)
                ev = FinancialEvent(
                    **{**ev.model_dump(), "settlement_date": parsed_date, "event_date": parsed_date}
                )
                ev_map[event_id] = ev
            elif intent == "amend" and new_date is not None:
                from datetime import date as date_type
                parsed_date = date_type.fromisoformat(new_date)
                ev = FinancialEvent(
                    **{**ev.model_dump(), "settlement_date": parsed_date}
                )
                if new_amount is not None:
                    ev = FinancialEvent(**{**ev.model_dump(), "amount": new_amount})
                ev_map[event_id] = ev

    return [ev for eid, ev in ev_map.items() if eid not in suppressed]


# ---------------------------------------------------------------------------
# Step 3: Fill blank amounts from images
# ---------------------------------------------------------------------------

def fill_blank_amounts(
    events: List[FinancialEvent],
    images_by_event: Dict[str, ImageRecord],
    use_llm: bool = True,
) -> List[FinancialEvent]:
    """
    For events with amount=None, look up image and extract amount via VLM/OCR.
    Events that still have None amount after this step are excluded from forecasting.
    """
    result = []
    for ev in events:
        if ev.amount is not None:
            result.append(ev)
            continue

        img_rec = images_by_event.get(ev.event_id)
        if img_rec is None:
            print(f"[reconciliation] No image for blank-amount event {ev.event_id}. Skipping.")
            continue  # drop this event — cannot treat as zero

        if use_llm:
            try:
                from .llm.image_extraction import extract_amount_from_image
                extracted = extract_amount_from_image(img_rec.image_id, ev.currency)
                if extracted and extracted.get("amount") is not None:
                    new_ev = FinancialEvent(
                        **{**ev.model_dump(), "amount": extracted["amount"]}
                    )
                    result.append(new_ev)
                    print(f"[reconciliation] Filled {ev.event_id} amount={extracted['amount']} via {extracted.get('path_used')}")
                else:
                    print(f"[reconciliation] Could not extract amount for {ev.event_id}. Skipping.")
            except Exception as e:
                print(f"[reconciliation] Image extraction failed for {ev.event_id}: {e}. Skipping.")
        else:
            print(f"[reconciliation] LLM disabled; skipping blank-amount event {ev.event_id}.")

    return result


# ---------------------------------------------------------------------------
# Step 4: Deduplicate
# ---------------------------------------------------------------------------

def deduplicate_events(events: List[FinancialEvent]) -> List[FinancialEvent]:
    """
    Remove duplicate records (same event_id). Keep settled > pending > others.
    Within same status, keep latest event_date.
    """
    seen: Dict[str, FinancialEvent] = {}
    STATUS_RANK = {"settled": 0, "scheduled": 1, "pending": 2, "failed": 3, "cancelled": 4, "unrealized": 5}

    for ev in events:
        if ev.event_id not in seen:
            seen[ev.event_id] = ev
        else:
            existing = seen[ev.event_id]
            er = STATUS_RANK.get(existing.status, 99)
            nr = STATUS_RANK.get(ev.status, 99)
            if nr < er or (nr == er and ev.event_date > existing.event_date):
                seen[ev.event_id] = ev

    return list(seen.values())


# ---------------------------------------------------------------------------
# Step 5: Classify recurring vs one-time
# ---------------------------------------------------------------------------

def classify_recurring(events: List[FinancialEvent]) -> Dict[str, bool]:
    """
    Returns dict: event_id -> is_recurring (bool).

    Heuristic: if a user has ≥ MIN_RECURRING_INSTANCES settled/scheduled expenses
    in the same (category, direction) within MAX_RECURRING_GAP_DAYS of each other,
    those events are recurring.
    """
    # Group settled/scheduled debit/credit events by (user_id, category, direction)
    groups: Dict[Tuple, List[FinancialEvent]] = defaultdict(list)
    for ev in events:
        if ev.direction == "non_cash":
            continue
        if ev.status not in ("settled", "scheduled", "pending"):
            continue
        key = (ev.user_id, ev.category, ev.direction)
        groups[key].append(ev)

    is_recurring: Dict[str, bool] = {}

    for key, group in groups.items():
        # Sort by settlement_date
        sorted_group = sorted(group, key=lambda e: e.settlement_date)
        dates = [e.settlement_date for e in sorted_group]

        if len(dates) < MIN_RECURRING_INSTANCES:
            for ev in group:
                is_recurring.setdefault(ev.event_id, False)
            continue

        # Compute gaps between consecutive dates
        gaps = [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]
        if not gaps:
            for ev in group:
                is_recurring.setdefault(ev.event_id, False)
            continue

        median_gap = statistics.median(gaps)
        # Recurring if median gap <= MAX_RECURRING_GAP_DAYS and we have enough instances
        recurring = (
            len(dates) >= MIN_RECURRING_INSTANCES
            and 1 <= median_gap <= MAX_RECURRING_GAP_DAYS
        )

        for ev in group:
            is_recurring[ev.event_id] = recurring

    # Default all unclassified to False
    for ev in events:
        is_recurring.setdefault(ev.event_id, False)

    return is_recurring


# ---------------------------------------------------------------------------
# Public facade
# ---------------------------------------------------------------------------

def reconcile(
    events: List[FinancialEvent],
    messages: List[Message],
    images_by_event: Dict[str, ImageRecord],
    profile: FinancialProfile,
    use_llm: bool = True,
) -> Tuple[List[FinancialEvent], Dict[str, bool]]:
    """
    Full reconciliation pipeline for a user's events.

    Returns (reconciled_events, is_recurring_map).
    """
    # 1. Resolve linked_event_id chains
    events = resolve_linked_events(events)

    # 2. Deduplicate
    events = deduplicate_events(events)

    # 3. Fill blank amounts from images
    events = fill_blank_amounts(events, images_by_event, use_llm=use_llm)

    # 4. Apply message amendments
    events = apply_message_amendments(events, messages, use_llm=use_llm)

    # 5. Classify recurring
    is_recurring = classify_recurring(events)

    return events, is_recurring
