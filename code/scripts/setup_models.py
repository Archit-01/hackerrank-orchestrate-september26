"""
setup_models.py — Query Groq models list, select best text + vision models,
write chosen model names to .env.example and print them.

Run once before the pipeline:
    python code/scripts/setup_models.py
"""
import os
import sys
from pathlib import Path

# Add code directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

_repo_root = Path(__file__).resolve().parents[2]
load_dotenv(_repo_root / ".env")

from src.llm.client import auto_discover_models

def main():
    print("Querying Groq model catalog...")
    try:
        text_model, vision_model = auto_discover_models()
    except Exception as e:
        print(f"Error querying models: {e}")
        print("Make sure GROQ_API_KEY is set in .env")
        sys.exit(1)

    print(f"\nSelected models:")
    print(f"  GROQ_TEXT_MODEL   = {text_model}")
    print(f"  GROQ_VISION_MODEL = {vision_model}")

    # Write to .env (update existing values)
    env_path = _repo_root / ".env"
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()
        new_lines = []
        keys_set = set()
        for line in lines:
            if line.startswith("GROQ_TEXT_MODEL="):
                new_lines.append(f"GROQ_TEXT_MODEL={text_model}")
                keys_set.add("GROQ_TEXT_MODEL")
            elif line.startswith("GROQ_VISION_MODEL="):
                new_lines.append(f"GROQ_VISION_MODEL={vision_model}")
                keys_set.add("GROQ_VISION_MODEL")
            else:
                new_lines.append(line)
        if "GROQ_TEXT_MODEL" not in keys_set:
            new_lines.append(f"GROQ_TEXT_MODEL={text_model}")
        if "GROQ_VISION_MODEL" not in keys_set:
            new_lines.append(f"GROQ_VISION_MODEL={vision_model}")
        env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        print(f"\nUpdated {env_path}")
    else:
        print(f"\nWarning: {env_path} not found. Create it from code/.env.example first.")

    # Also update .env.example with chosen model names (no key)
    example_path = _repo_root / "code" / ".env.example"
    if example_path.exists():
        content = example_path.read_text(encoding="utf-8")
        content = content.replace("GROQ_TEXT_MODEL=", f"GROQ_TEXT_MODEL={text_model}")
        content = content.replace("GROQ_VISION_MODEL=", f"GROQ_VISION_MODEL={vision_model}")
        example_path.write_text(content, encoding="utf-8")
        print(f"Updated {example_path}")

    print("\nDone. Run the pipeline with:")
    print("  python code/scripts/run_pipeline.py")


if __name__ == "__main__":
    main()
