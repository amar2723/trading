@echo off
title Telegram Bot Controller - @AMARREDDYBOT
cd /d "%~dp0\.."
call .venv\Scripts\activate.bat
echo Starting Two-Way Telegram Controller for @AMARREDDYBOT...
python scripts/telegram_control_bot.py
pause
