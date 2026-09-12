"""
pipeline.py — Orchestrates the full per-request processing pipeline.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .schema import (
    FinancialEvent,
    FinancialProfile,
    ImageRecord,
    Message,
    OutputRow,
    PaymentOption,
    RequestRow,
)
from .currency import build_rate_index, RateLookup
from .reconciliation import reconcile
from .decision_engine import decide
from .llm.explanation import generate_explanation
from .llm.usage_tracker import tracker

# Repo root
_REPO_ROOT = Path(__file__).resolve().parents[3]


def run_request(
    request: RequestRow,
    profile: FinancialProfile,
    user_events: List[FinancialEvent],
    messages: List[Message],
    images_by_event: Dict[str, ImageRecord],
    payment_options: List[PaymentOption],
    rate_index: RateLookup,
    use_llm: bool = True,
) -> OutputRow:
    """
    Process a single request and return an OutputRow.
    """
    # 1. Reconcile events
    reconciled_events, is_recurring = reconcile(
        events=user_events,
        messages=messages,
        images_by_event=images_by_event,
        profile=profile,
        use_llm=use_llm,
    )

    # 2. Deterministic decision
    result = decide(
        request=request,
        profile=profile,
        events=reconciled_events,
        is_recurring=is_recurring,
        payment_options=payment_options,
        rate_index=rate_index,
    )

    # 3. Generate explanation
    facts = {
        "request_id": request.request_id,
        "user_id": request.user_id,
        "request_type": request.request_type,
        "requested_amount": request.requested_amount,
        "request_date": request.request_date.isoformat(),
        "desired_completion_date": request.desired_completion_date.isoformat(),
        "home_currency": profile.home_currency,
        "current_available_balance": profile.current_available_balance,
        "minimum_balance_to_keep": profile.minimum_balance_to_keep,
        "amount_safe_to_pay": result.amount_safe_to_pay,
        "affordability_status": result.affordability_status,
        "recommended_payment_method": result.recommended_payment_method,
        "payment_plan": result.payment_plan,
        "earliest_date_for_full_payment": result.earliest_date_for_full_payment,
        "spending_changes_needed": result.spending_changes_needed,
    }

    if use_llm:
        try:
            explanation = generate_explanation(facts)
        except Exception as e:
            print(f"[pipeline] Explanation generation failed for {request.request_id}: {e}")
            explanation = _fallback_explanation(result, profile, request)
    else:
        explanation = _fallback_explanation(result, profile, request)

    return OutputRow(
        request_id=request.request_id,
        amount_safe_to_pay=result.amount_safe_to_pay,
        affordability_status=result.affordability_status,
        recommended_payment_method=result.recommended_payment_method,
        payment_plan=result.payment_plan,
        earliest_date_for_full_payment=result.earliest_date_for_full_payment,
        spending_changes_needed=result.spending_changes_needed,
        decision_explanation=explanation,
    )


def _fallback_explanation(result, profile: FinancialProfile, request: RequestRow) -> str:
    """Template-based explanation when LLM is unavailable."""
    cur = profile.home_currency
    min_b = profile.minimum_balance_to_keep
    amt = request.requested_amount
    safe = result.amount_safe_to_pay
    method = result.recommended_payment_method
    earliest = result.earliest_date_for_full_payment

    if method == "full_payment":
        return f"Pay {cur} {safe:,.2f} today. This keeps the {cur} {min_b:,.0f} minimum available over the next 90 days."
    elif method == "wait":
        return f"Pay {cur} {amt:,.2f} in full on {earliest}. Paying earlier would take the balance below the {cur} {min_b:,.0f} minimum."
    elif method == "not_recommended":
        return f"Do not make this payment. None of the available options keeps the {cur} {min_b:,.0f} minimum protected."
    elif method == "partial_payment":
        remaining = amt - safe
        return f"Pay {cur} {safe:,.2f} today and the remaining {cur} {remaining:,.2f} on {earliest}. This keeps the {cur} {min_b:,.0f} minimum protected."
    elif method == "installments":
        return f"Use the installment plan. This keeps the {cur} {min_b:,.0f} minimum available throughout the forecast period."
    return f"Based on your financial position, {cur} {safe:,.2f} is the safe amount to commit today."


def run_pipeline(
    requests: List[RequestRow],
    profiles: Dict[str, FinancialProfile],
    all_events: List[FinancialEvent],
    all_messages: List[Message],
    all_images: List[ImageRecord],
    all_payment_options: Dict[str, List[PaymentOption]],
    rate_index: RateLookup,
    use_llm: bool = True,
) -> List[OutputRow]:
    """
    Process all requests and return a list of OutputRow.
    """
    from .data_loader import events_by_user, messages_by_user, images_by_event as img_by_event_fn

    events_per_user = events_by_user(all_events)
    messages_per_user = messages_by_user(all_messages)
    images_map = img_by_event_fn(all_images)

    output_rows = []
    total = len(requests)

    for i, req in enumerate(requests, 1):
        print(f"[pipeline] Processing {req.request_id} ({i}/{total})...")
        try:
            profile = profiles.get(req.user_id)
            if profile is None:
                print(f"[pipeline] No profile found for {req.user_id}. Skipping.")
                continue

            user_events = events_per_user.get(req.user_id, [])
            user_messages = messages_per_user.get(req.user_id, [])

            # Also include request-level messages
            req_messages = [m for m in all_messages if m.request_id == req.request_id]
            combined_messages = list({m.message_id: m for m in user_messages + req_messages}.values())

            payment_opts = all_payment_options.get(req.request_id, [])

            row = run_request(
                request=req,
                profile=profile,
                user_events=user_events,
                messages=combined_messages,
                images_by_event=images_map,
                payment_options=payment_opts,
                rate_index=rate_index,
                use_llm=use_llm,
            )
            output_rows.append(row)

        except Exception as e:
            import traceback
            print(f"[pipeline] ERROR processing {req.request_id}: {e}")
            traceback.print_exc()
            # Emit a safe fallback row
            profile = profiles.get(req.user_id)
            output_rows.append(OutputRow(
                request_id=req.request_id,
                amount_safe_to_pay=0.0,
                affordability_status="not_affordable",
                recommended_payment_method="not_recommended",
                payment_plan="none",
                earliest_date_for_full_payment="",
                spending_changes_needed="none",
                decision_explanation=f"Processing error: {str(e)[:200]}",
            ))

    return output_rows
