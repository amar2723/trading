@echo off
title XAUUSD AI Scheduled Retraining
cd /d "%~dp0\.."
call .venv\Scripts\activate.bat
echo Running scheduled model retraining pipeline...
python ops/scheduled_retraining.py
echo Retraining complete.
timeout /t 5
