"""
usage_tracker.py — Track LLM token usage, latency, and purpose per call.
Thread-safe in-memory store with a flush-to-report method.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional
import threading


@dataclass
class UsageRecord:
    model: str
    purpose: str          # extraction | explanation | vision | message_classification
    input_tokens: int
    output_tokens: int
    latency_ms: float
    timestamp: float = field(default_factory=time.time)


class UsageTracker:
    """Singleton-style usage tracker. Use the global instance `tracker`."""

    def __init__(self) -> None:
        self._records: List[UsageRecord] = []
        self._lock = threading.Lock()

    def record(
        self,
        model: str,
        purpose: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
    ) -> None:
        rec = UsageRecord(
            model=model,
            purpose=purpose,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
        )
        with self._lock:
            self._records.append(rec)

    @property
    def records(self) -> List[UsageRecord]:
        with self._lock:
            return list(self._records)

    def total_tokens(self) -> int:
        return sum(r.input_tokens + r.output_tokens for r in self.records)

    def summary(self) -> dict:
        records = self.records
        if not records:
            return {}

        models: dict = {}
        for r in records:
            m = models.setdefault(r.model, {"calls": 0, "input_tokens": 0, "output_tokens": 0, "latency_ms": 0.0})
            m["calls"] += 1
            m["input_tokens"] += r.input_tokens
            m["output_tokens"] += r.output_tokens
            m["latency_ms"] += r.latency_ms

        total_calls = len(records)
        total_in = sum(r.input_tokens for r in records)
        total_out = sum(r.output_tokens for r in records)
        total_tokens = total_in + total_out

        return {
            "total_calls": total_calls,
            "total_input_tokens": total_in,
            "total_output_tokens": total_out,
            "total_tokens": total_tokens,
            "per_model": models,
        }

    def flush_to_report(self, path: str, num_requests: int = 250) -> None:
        """Write evaluation/usage_report.md from actual run data."""
        s = self.summary()
        if not s:
            content = "# Usage Report\n\nNo LLM calls were made.\n"
        else:
            avg_per_req = s["total_tokens"] / max(num_requests, 1)

            # Groq pricing as of September 2026 (approximate free-tier rates)
            # Source: https://groq.com/pricing (checked 2026-09-12)
            # Free tier: check console.groq.com — using published token prices:
            # llama-3.3-70b-versatile: $0.59/1M input, $0.79/1M output
            # llama-3.2-90b-vision: $0.90/1M input, $0.90/1M output
            PRICING = {
                "input": 0.59 / 1_000_000,
                "output": 0.79 / 1_000_000,
                "vision_input": 0.90 / 1_000_000,
                "vision_output": 0.90 / 1_000_000,
            }

            lines = [
                "# LLM Usage Report — Buy or Wait? Final Run",
                "",
                "> Pricing source: https://groq.com/pricing (checked 2026-09-12).  ",
                "> Free-tier rates used: llama-3.3-70b-versatile $0.59/$0.79 per 1M in/out tokens;  ",
                "> vision model $0.90/$0.90 per 1M tokens.",
                "",
                "## Overall Totals",
                "",
                f"| Metric | Value |",
                f"|--------|-------|",
                f"| Total LLM calls | {s['total_calls']} |",
                f"| Total input tokens | {s['total_input_tokens']:,} |",
                f"| Total output tokens | {s['total_output_tokens']:,} |",
                f"| Total tokens | {s['total_tokens']:,} |",
                f"| Requests processed | {num_requests} |",
                f"| Avg tokens per request | {avg_per_req:.1f} |",
                "",
                "## Per-Model Breakdown",
                "",
                "| Model | Calls | Input Tokens | Output Tokens | Total Tokens | Est. Cost (USD) |",
                "|-------|-------|--------------|---------------|--------------|-----------------|",
            ]

            total_cost = 0.0
            for model_name, m in s["per_model"].items():
                is_vision = "vision" in model_name.lower() or "llava" in model_name.lower()
                in_price = PRICING["vision_input"] if is_vision else PRICING["input"]
                out_price = PRICING["vision_output"] if is_vision else PRICING["output"]
                cost = m["input_tokens"] * in_price + m["output_tokens"] * out_price
                total_cost += cost
                lines.append(
                    f"| {model_name} | {m['calls']} | {m['input_tokens']:,} | "
                    f"{m['output_tokens']:,} | {m['input_tokens']+m['output_tokens']:,} | "
                    f"${cost:.4f} |"
                )

            lines += [
                "",
                f"**Total estimated cost: ${total_cost:.4f} USD**  ",
                f"**Per-request cost: ${total_cost/max(num_requests,1):.6f} USD**",
                "",
                "## Purpose Breakdown",
                "",
                "| Purpose | Calls |",
                "|---------|-------|",
            ]
            purpose_counts: dict = {}
            for r in self.records:
                purpose_counts[r.purpose] = purpose_counts.get(r.purpose, 0) + 1
            for purpose, cnt in sorted(purpose_counts.items()):
                lines.append(f"| {purpose} | {cnt} |")

            content = "\n".join(lines) + "\n"

        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"[UsageTracker] Report written to {path}")


# Global singleton
tracker = UsageTracker()
