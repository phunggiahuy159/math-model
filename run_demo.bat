@echo off
REM Launch the food-chain chemostat Streamlit demo (Windows).
REM Usage: double-click this file, or run  run_demo.bat  from a terminal.

setlocal
cd /d "%~dp0"

echo Checking dependencies...
python -c "import streamlit, plotly, scipy, numpy" 2>nul
if errorlevel 1 (
    echo Installing requirements...
    python -m pip install -r requirements.txt
)

echo Starting demo at http://localhost:8501  (press Ctrl+C to stop)
python -m streamlit run app.py --server.port 8501

endlocal
