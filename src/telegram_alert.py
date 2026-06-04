from __future__ import annotations

import os

import requests
from dotenv import load_dotenv


load_dotenv()


def format_signal_message(symbol: str, signal: dict) -> str:
    return (
        f"{signal['signal']} {symbol}\n\n"
        f"Entry: {signal.get('entry')}\n"
        f"SL: {signal.get('sl')}\n"
        f"TP1: {signal.get('tp1')}\n\n"
        f"TP2/Liquidity: {signal.get('tp2')}\n"
        f"Trailing Stop Ref: {signal.get('trailing_stop')}\n\n"
        f"Reason: {signal.get('reason', '')}\n"
        f"Confidence: {signal.get('confidence', 0):.0f}%"
    )


def send_telegram_alert(symbol: str, signal: dict) -> bool:
    token = os.getenv("BOT_TOKEN")
    chat_id = os.getenv("CHAT_ID")
    if not token or not chat_id:
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    message = format_signal_message(symbol, signal)
    response = requests.post(url, json={"chat_id": chat_id, "text": message}, timeout=10)
    response.raise_for_status()
    return True
