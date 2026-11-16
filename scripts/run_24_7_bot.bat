@echo off
title XAUUSD 24/7 AI Live Trader Daemon
cd /d "%~dp0\.."
echo =======================================================
echo    STARTING 24/7 AUTONOMOUS LIVE TRADING ENGINE        
echo =======================================================
call .venv\Scripts\activate.bat

:loop
echo [%date% %time%] Starting autonomous live trader...
.venv\Scripts\python scripts\autonomous_live_trader.py
echo [%date% %time%] Bot stopped or crashed with code %errorlevel%. Auto-restarting in 5 seconds...
timeout /t 5 /nobreak
goto loop
