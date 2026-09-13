#!/usr/bin/env python3
import sys
import os
from pathlib import Path

def main():
    """
    Main entry point for HackerRank Orchestrate (Buy or Wait?).
    This script executes the full decision pipeline which:
    1. Loads the financial dataset (profiles, events, messages, rates)
    2. Reconciles events (extracts general messages, applies updates, handles links)
    3. Projects cash flow & balances 
    4. Executes the rule engine to determine affordability and payment plans
    5. Writes the final decisions to output.csv
    """
    # Ensure the code directory is in the python path
    code_dir = Path(__file__).parent.resolve()
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))
        
    try:
        from scripts.run_pipeline import main as run_pipeline
        print("============================================================")
        print("Starting HackerRank Orchestrate: Buy or Wait? Pipeline...")
        print("============================================================")
        run_pipeline()
    except Exception as e:
        print(f"Error running pipeline: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
