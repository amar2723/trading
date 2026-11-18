@echo off
title XAUUSD 24/7 AI Live Trader Daemon
cd /d "%~dp0\.."
echo =======================================================
echo    STARTING 24/7 AUTONOMOUS LIVE TRADING ENGINE        
echo =======================================================
call .venv\Scripts\activate.bat

:loop
echo [%date% %time%] Starting autonomous live trader...
.venv\Scripts\python -u scripts\autonomous_live_trader.py
echo [%date% %time%] Bot stopped or crashed with code %errorlevel%. Auto-restarting in 5 seconds...
ping 127.0.0.1 -n 6 >nul
goto loop
