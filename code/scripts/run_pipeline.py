"""
run_pipeline.py — Entry point: reads dataset/requests.csv, writes output.csv.

Usage:
    python code/scripts/run_pipeline.py [--no-llm]

Options:
    --no-llm    Skip all LLM calls (use template explanations, skip image/message extraction).
                Useful for testing deterministic logic without a Groq API key.
"""
import sys
import os
import time
import argparse
from pathlib import Path

# Add code directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

_repo_root = Path(__file__).resolve().parents[2]
load_dotenv(_repo_root / ".env")

from src.data_loader import (
    load_requests,
    load_financial_profiles,
    load_financial_events,
    load_exchange_rates,
    load_payment_options,
    load_messages,
    load_images,
)
from src.currency import build_rate_index
from src.pipeline import run_pipeline
from src.output_writer import write_output
from src.llm.usage_tracker import tracker


def main():
    parser = argparse.ArgumentParser(description="Buy or Wait? Pipeline")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM calls")
    parser.add_argument("--dataset-dir", default=None, help="Path to dataset directory")
    parser.add_argument("--output", default=None, help="Output CSV path")
    parser.add_argument("--requests", default=None, help="Requests CSV path (default: dataset/requests.csv)")
    args = parser.parse_args()

    use_llm = not args.no_llm
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else (_repo_root / "dataset")
    output_path = Path(args.output) if args.output else (_repo_root / "output.csv")

    print("=" * 60)
    print("Buy or Wait? — Financial Decision Pipeline")
    print("=" * 60)
    print(f"Dataset dir : {dataset_dir}")
    print(f"Output path : {output_path}")
    print(f"LLM enabled : {use_llm}")
    print()

    t0 = time.perf_counter()

    # Load all data
    print("Loading data...")
    requests_csv = Path(args.requests) if args.requests else (dataset_dir / "requests.csv")
    requests = load_requests(requests_csv)
    profiles = load_financial_profiles(dataset_dir / "financial_profiles.csv")
    events = load_financial_events(dataset_dir / "financial_events.csv")
    rates = load_exchange_rates(dataset_dir / "exchange_rates.csv")
    payment_options = load_payment_options(dataset_dir / "request_payment_options.csv")
    messages = load_messages(dataset_dir / "messages.csv")
    images = load_images(dataset_dir / "images.csv")

    print(f"  Requests   : {len(requests)}")
    print(f"  Profiles   : {len(profiles)}")
    print(f"  Events     : {len(events)}")
    print(f"  Rates      : {len(rates)}")
    print(f"  Pay options: {sum(len(v) for v in payment_options.values())}")
    print(f"  Messages   : {len(messages)}")
    print(f"  Images     : {len(images)}")
    print()

    rate_index = build_rate_index(rates)

    # Run pipeline
    print("Running pipeline...")
    output_rows = run_pipeline(
        requests=requests,
        profiles=profiles,
        all_events=events,
        all_messages=messages,
        all_images=images,
        all_payment_options=payment_options,
        rate_index=rate_index,
        use_llm=use_llm,
    )

    # Write output
    write_output(output_rows, output_path)

    elapsed = time.perf_counter() - t0
    print(f"\nCompleted {len(output_rows)} rows in {elapsed:.1f}s")

    # Write usage report
    usage_report_path = _repo_root / "code" / "evaluation" / "usage_report.md"
    usage_report_path.parent.mkdir(parents=True, exist_ok=True)
    tracker.flush_to_report(str(usage_report_path), num_requests=len(requests))

    print(f"\nToken usage summary:")
    s = tracker.summary()
    if s:
        print(f"  Total calls : {s['total_calls']}")
        print(f"  Total tokens: {s['total_tokens']:,}")
    else:
        print("  No LLM calls made.")

    print(f"\nOutput written to: {output_path}")
    print(f"Usage report : {usage_report_path}")


if __name__ == "__main__":
    main()
