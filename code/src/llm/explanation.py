"""
explanation.py — Generate a fluent, grounded decision_explanation.

Rules:
  1. The LLM is given a dict of all computed facts.
  2. The explanation must be 1-2 sentences, concise and grounded.
  3. Post-hoc numeric verifier: scan the output for numeric tokens.
     If any numeric token is NOT present in the facts, reject and revert to a
     safe template-based explanation.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Optional

from .client import chat_completion, GROQ_TEXT_MODEL
from .usage_tracker import tracker

_SYSTEM_PROMPT = """You are a financial advisor assistant writing short, clear explanations for payment decisions.

Rules:
- Write exactly 1-2 sentences.
- Only use numbers that appear in the facts provided to you. Do not introduce any other numbers.
- Do not add advice, caveats, or financial guidance beyond what the facts state.
- Be direct and factual.
- Do not mention the word "recommendation" or use passive voice excessively.

Format: plain text, no markdown, no bullet points.
"""


def _extract_numeric_tokens(text: str) -> set:
    """Extract all numeric tokens (integers and decimals) from text."""
    return set(re.findall(r"\b\d[\d,]*\.?\d*\b", text.replace(",", "")))


def _facts_to_numeric_tokens(facts: Dict[str, Any]) -> set:
    """Flatten all numeric values in the facts dict to their string representations."""
    tokens = set()
    for v in facts.values():
        if v is None:
            continue
        if isinstance(v, (int, float)):
            # Add both rounded and truncated forms to be lenient
            tokens.add(str(int(v)))
            tokens.add(f"{v:.2f}".rstrip("0").rstrip("."))
            tokens.add(str(round(v, 2)))
            tokens.add(str(v))
        elif isinstance(v, str):
            # Extract any numbers from string values
            tokens.update(re.findall(r"\d[\d,]*\.?\d*", v.replace(",", "")))
    return tokens


def _template_explanation(facts: Dict[str, Any]) -> str:
    """Safe template-based fallback explanation."""
    status = facts.get("affordability_status", "")
    method = facts.get("recommended_payment_method", "")
    currency = facts.get("home_currency", "")
    amount = facts.get("requested_amount", 0)
    safe = facts.get("amount_safe_to_pay", 0)
    min_bal = facts.get("minimum_balance_to_keep", 0)
    earliest = facts.get("earliest_date_for_full_payment", "")

    if status == "affordable_now":
        return (
            f"Pay {currency} {safe:,.2f} today. "
            f"This keeps the {currency} {min_bal:,.0f} minimum available."
        )
    elif method == "wait":
        return (
            f"Pay {currency} {amount:,.2f} in full on {earliest}. "
            f"Paying earlier would take the balance below the {currency} {min_bal:,.0f} minimum."
        )
    elif method == "not_recommended":
        return (
            f"Do not make this payment by the deadline. "
            f"None of the available options keeps the {currency} {min_bal:,.0f} minimum protected."
        )
    elif method == "partial_payment":
        return (
            f"Pay {currency} {safe:,.2f} today and the remaining "
            f"{currency} {amount - safe:,.2f} on {earliest}. "
            f"This keeps the {currency} {min_bal:,.0f} minimum protected."
        )
    elif method == "installments":
        plan = facts.get("payment_plan", "")
        return (
            f"Use installments per the selected payment plan. "
            f"This keeps the {currency} {min_bal:,.0f} minimum available."
        )
    else:
        return (
            f"Based on your financial position, {currency} {safe:,.2f} is safe to commit today, "
            f"maintaining the {currency} {min_bal:,.0f} minimum balance."
        )


def generate_explanation(facts: Dict[str, Any]) -> str:
    """
    Generate a fluent decision_explanation.

    Returns a verified explanation string (never contains hallucinated numbers).
    """
    # Build fact summary for the prompt
    fact_lines = []
    for k, v in facts.items():
        if v is not None and str(v) != "":
            fact_lines.append(f"- {k}: {v}")
    facts_text = "\n".join(fact_lines)

    user_prompt = (
        f"Write a 1-2 sentence explanation for this payment decision.\n\n"
        f"Facts:\n{facts_text}\n\n"
        f"Only use the numbers listed above. Do not introduce new numbers."
    )

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    content, in_tok, out_tok, latency = chat_completion(
        messages=messages,
        max_tokens=200,
        temperature=0.3,
    )
    tracker.record(
        model=GROQ_TEXT_MODEL,
        purpose="explanation",
        input_tokens=in_tok,
        output_tokens=out_tok,
        latency_ms=latency,
    )

    explanation = content.strip()

    # --- Post-hoc numeric verifier ---
    allowed_numbers = _facts_to_numeric_tokens(facts)
    generated_numbers = _extract_numeric_tokens(explanation)

    hallucinated = generated_numbers - allowed_numbers
    if hallucinated:
        print(
            f"[explanation] Hallucinated numbers detected: {hallucinated}. "
            f"Reverting to template."
        )
        explanation = _template_explanation(facts)

    if not explanation.strip():
        explanation = _template_explanation(facts)

    return explanation
