#!/usr/bin/env bash
# Launch the food-chain chemostat Streamlit demo (Git Bash / macOS / Linux).
# Usage:  ./run_demo.sh
set -e
cd "$(dirname "$0")"

echo "Checking dependencies..."
if ! python -c "import streamlit, plotly, scipy, numpy" 2>/dev/null; then
    echo "Installing requirements..."
    python -m pip install -r requirements.txt
fi

echo "Starting demo at http://localhost:8501  (press Ctrl+C to stop)"
exec python -m streamlit run app.py --server.port 8501
