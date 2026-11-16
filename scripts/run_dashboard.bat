@echo off
title XAUUSD AI Trading Dashboard
cd /d "%~dp0\.."
call .venv\Scripts\activate.bat
echo Starting Streamlit Trading Dashboard...
streamlit run dashboard/app.py --server.port 8501 --server.headless false
pause
