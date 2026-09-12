# Buy or Wait? — Solution README

Financial decision engine for the HackerRank Orchestrate (September 2026) hackathon.

---

## Architecture

```
Dataset CSVs  →  data_loader.py  →  reconciliation.py  →  forecast.py  →  decision_engine.py
                                                                                     ↓
                             LLM (Groq — explanation/image/message only)  →  pipeline.py
                                                                                     ↓
                                                                           output_writer.py → output.csv
```

### Key Design Principle
**The LLM never computes any numeric output.** All arithmetic (amount_safe_to_pay, earliest_date_for_full_payment, payment plans) is 100% deterministic Python. Groq is used only for:
- **(a) VLM image extraction**: extracting a dollar amount from a financial document image
- **(b) Message classification**: classifying a message as cancel/amend/delay/confirm/irrelevant (structured JSON output only)
- **(c) Explanation generation**: turning computed facts into fluent text — with a post-hoc numeric verifier that rejects any hallucinated number

### Module Responsibilities

| Module | Responsibility |
|--------|---------------|
| `src/schema.py` | Pydantic v2 models for all CSVs + output row |
| `src/data_loader.py` | Load and type-validate all dataset CSVs |
| `src/currency.py` | Dated FX conversion (dated rates only, never live) |
| `src/reconciliation.py` | Conflict resolution, message/image amendments, dedup, recurring classification |
| `src/forecast.py` | 90-day balance simulator (pure function) |
| `src/decision_engine.py` | Binary-search amount_safe_to_pay, linear-scan earliest_date, 6-rule plan ranking |
| `src/pipeline.py` | Orchestrates all modules per request |
| `src/output_writer.py` | Writes output.csv in exact required column order |
| `src/llm/client.py` | Groq client wrapper + model auto-discovery |
| `src/llm/usage_tracker.py` | Token/cost/latency logging |
| `src/llm/image_extraction.py` | VLM or OCR fallback for blank amounts |
| `src/llm/message_extraction.py` | Message → structured JSON intent |
| `src/llm/explanation.py` | Facts → fluent explanation + anti-hallucination verifier |

---

## How Models Were Chosen

Run `python code/scripts/setup_models.py` to auto-select:
- **GROQ_TEXT_MODEL**: Selected from `PREFERRED_TEXT_MODELS` list by querying Groq's `/models` endpoint — strongest available model with JSON support
- **GROQ_VISION_MODEL**: Selected from `PREFERRED_VISION_MODELS` list; falls back to `"ocr_fallback"` if no vision model is available

Default preferences (strongest → weakest):
- Text: `llama-3.3-70b-versatile` → `llama-3.1-70b-versatile` → `llama3-70b-8192` → ...
- Vision: `llama-3.2-90b-vision-preview` → `llama-3.2-11b-vision-preview` → `llava-v1.5-7b-4096-preview`

---

## Setup

### Prerequisites
- Python 3.10+
- Groq API key from https://console.groq.com

### Steps

```bash
# 1. Clone the repo (already done for the hackathon fork)
git clone https://github.com/Archit-01/hackerrank-orchestrate-september26.git
cd hackerrank-orchestrate-september26

# 2. Run setup script (creates venv + installs requirements)
bash code/setup.sh

# 3. Activate the venv
source venv/bin/activate   # Linux/macOS
# Windows: venv\Scripts\activate

# 4. Create .env from template
cp code/.env.example .env
# Edit .env and add your GROQ_API_KEY

# 5. Auto-discover and save best available Groq models
python code/scripts/setup_models.py

# 6. Run the pipeline (produces output.csv at repo root)
python code/scripts/run_pipeline.py

# Optional: run without LLM calls (no API key needed, uses template explanations)
python code/scripts/run_pipeline.py --no-llm
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `GROQ_API_KEY` | Your Groq API key (required for LLM calls) |
| `GROQ_TEXT_MODEL` | Auto-populated by `setup_models.py` |
| `GROQ_VISION_MODEL` | Auto-populated by `setup_models.py` |

---

## Running Evaluation

```bash
# Score against the 25 sample requests (ground truth)
python code/evaluation/evaluate.py

# Validate output.csv against all spec constraints (must show 0 violations)
python code/evaluation/validate_output.py

# Run unit tests
python -m pytest code/tests/ -v
```

---

## How to Reproduce

```bash
# Fresh reproduction from scratch:
bash code/setup.sh
source venv/bin/activate
cp code/.env.example .env && echo "GROQ_API_KEY=<your_key>" >> .env
python code/scripts/setup_models.py
python code/scripts/run_pipeline.py
python code/evaluation/validate_output.py
```

---

## Token Usage Report

After running the pipeline, `code/evaluation/usage_report.md` is auto-generated with:
- Per-model call counts, input/output token counts
- Total tokens, average per request
- Estimated cost using Groq's published pricing

---

## Submission Files

| File | Location |
|------|----------|
| `code.zip` | Zip of `code/` directory (excluding venv, __pycache__) |
| `output.csv` | At repo root |
| `log.txt` | At repo root (chat transcript) |

Submit at: https://www.hackerrank.com/contests/hackerrank-orchestrate-september26/challenges/buy-or-wait/submission
