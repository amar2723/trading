@echo off
title XAUUSD AI Live Signal Scanner
cd /d "%~dp0\.."
call .venv\Scripts\activate.bat
echo Starting Live Scanner with auto-recovery...
:loop
python main.py --symbol XAUUSD --timeframe M5 --session OVERLAP --trend-alignment --avoid-chop
echo Scanner exited or crashed with error code %errorlevel%. Restarting in 10 seconds...
timeout /t 10 /nobreak
goto loop
