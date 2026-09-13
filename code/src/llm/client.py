"""
client.py — Groq client wrapper with model auto-discovery.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional, Tuple

from dotenv import load_dotenv

# Load .env from repo root
_repo_root = Path(__file__).resolve().parents[3]
load_dotenv(_repo_root / ".env")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_TEXT_MODEL = os.environ.get("GROQ_TEXT_MODEL", "")
GROQ_VISION_MODEL = os.environ.get("GROQ_VISION_MODEL", "")

# ---------------------------------------------------------------------------
# Model preference lists (ordered strongest → weakest)
# ---------------------------------------------------------------------------
PREFERRED_TEXT_MODELS = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "qwen/qwen3.6-27b",
    "qwen/qwen3.8-27b",
    "groq/compound",
    "llama-3.1-8b-instant",
    "llama3-8b-8192",
    "gemma2-9b-it",
    "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile",
    "llama3-70b-8192",
    "mixtral-8x7b-32768",
]

PREFERRED_VISION_MODELS = [
    "qwen/qwen3.8-27b",
    "llama-3.2-11b-vision-preview",
    "llama-3.2-90b-vision-preview",
    "llava-v1.5-7b-4096-preview",
]


def _get_client():
    """Lazy import groq to avoid import errors if not installed yet."""
    try:
        from groq import Groq  # type: ignore
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not set. Run: export GROQ_API_KEY=your_key")
        return Groq(api_key=GROQ_API_KEY)
    except ImportError:
        raise ImportError("groq package not installed. Run: pip install groq")


def auto_discover_models() -> Tuple[str, str]:
    """
    Query Groq /models endpoint and select:
      - strongest available general-purpose text model
      - strongest available vision model (or 'ocr_fallback' if none)

    Returns (text_model_id, vision_model_id).
    """
    client = _get_client()
    import time; time.sleep(2.1)
    available_ids = set()
    try:
        models_resp = client.models.list()
        for m in models_resp.data:
            available_ids.add(m.id)
    except Exception as e:
        print(f"[client] Warning: could not list models: {e}. Using defaults.")

    # Select text model
    text_model = PREFERRED_TEXT_MODELS[0]  # default
    for candidate in PREFERRED_TEXT_MODELS:
        if candidate in available_ids:
            text_model = candidate
            break

    # Select vision model
    vision_model = "ocr_fallback"
    for candidate in PREFERRED_VISION_MODELS:
        if candidate in available_ids:
            vision_model = candidate
            break

    return text_model, vision_model


def chat_completion(
    messages: list,
    model: Optional[str] = None,
    response_format: Optional[dict] = None,
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> Tuple[str, int, int, float]:
    """
    Call Groq chat completion.

    Returns (content_str, input_tokens, output_tokens, latency_ms).
    """
    client = _get_client()
    import time; time.sleep(2.1)
    
    if model:
        chosen_model = model
    elif GROQ_TEXT_MODEL:
        chosen_model = GROQ_TEXT_MODEL
    else:
        chosen_model, _ = auto_discover_models()

    kwargs: dict = {
        "model": chosen_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_format:
        kwargs["response_format"] = response_format

    t0 = time.perf_counter()
    import groq
    import re
    while True:
        try:
            resp = client.chat.completions.create(**kwargs)
            break
        except groq.RateLimitError as e:
            msg = str(e)
            m = re.search(r"Please try again in (\d+\.?\d*)s", msg)
            wait_time = max(float(m.group(1)) + 0.5, 15.0) if m else 15.0
            print(f"[client] Rate limit hit. Waiting {wait_time:.2f}s...")
            time.sleep(wait_time)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    content = resp.choices[0].message.content or ""
    in_tok = resp.usage.prompt_tokens if resp.usage else 0
    out_tok = resp.usage.completion_tokens if resp.usage else 0
    return content, in_tok, out_tok, latency_ms


def vision_completion(
    messages: list,
    model: Optional[str] = None,
    max_tokens: int = 256,
) -> Tuple[str, int, int, float]:
    """
    Call Groq vision (multimodal) completion.

    Returns (content_str, input_tokens, output_tokens, latency_ms).
    """
    client = _get_client()
    import time; time.sleep(2.1)
    
    if model:
        chosen_model = model
    elif GROQ_VISION_MODEL:
        chosen_model = GROQ_VISION_MODEL
    else:
        _, chosen_model = auto_discover_models()

    t0 = time.perf_counter()
    import groq
    import re
    while True:
        try:
            resp = client.chat.completions.create(
                model=chosen_model,
                messages=messages,
                max_tokens=max_tokens,
            )
            break
        except groq.RateLimitError as e:
            msg = str(e)
            m = re.search(r"Please try again in (\d+\.?\d*)s", msg)
            wait_time = max(float(m.group(1)) + 0.5, 15.0) if m else 15.0
            print(f"[client] Rate limit hit. Waiting {wait_time:.2f}s...")
            time.sleep(wait_time)
            
    latency_ms = (time.perf_counter() - t0) * 1000.0

    content = resp.choices[0].message.content or ""
    in_tok = resp.usage.prompt_tokens if resp.usage else 0
    out_tok = resp.usage.completion_tokens if resp.usage else 0
    return content, in_tok, out_tok, latency_ms
