#!/usr/bin/env bash
# setup.sh — Creates venv and installs requirements for Buy or Wait?
set -e

echo "=== Buy or Wait? Setup ==="

# Create virtual environment
python3 -m venv venv
echo "Virtual environment created at ./venv"

# Activate
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip --quiet

# Install requirements
pip install -r code/requirements.txt

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Copy .env.example to .env and add your GROQ_API_KEY:"
echo "       cp code/.env.example .env"
echo "       echo 'GROQ_API_KEY=your_key_here' >> .env"
echo ""
echo "  2. Run model auto-discovery (writes chosen models to .env.example):"
echo "       source venv/bin/activate"
echo "       python code/scripts/setup_models.py"
echo ""
echo "  3. Run the full pipeline to produce output.csv:"
echo "       python code/scripts/run_pipeline.py"
echo ""
echo "  4. Evaluate against sample_requests.csv:"
echo "       python code/evaluation/evaluate.py"
echo ""
echo "  5. Validate the output (must show zero violations):"
echo "       python code/evaluation/validate_output.py"
echo ""
echo "Windows note: Replace 'source venv/bin/activate' with 'venv\\Scripts\\activate'"
