"""conftest.py — Ensure code/ is in sys.path for pytest."""
import sys
from pathlib import Path

# Add code/ directory to path so `src.*` imports work in tests
sys.path.insert(0, str(Path(__file__).resolve().parent / "code"))
