from __future__ import annotations

"""
VIP Interactive Telegram Trading Bot for @AMARREDDYBOT.
Unified with 24/7 Master Production Engine:
- Instant Next-Trade Signals with Exact Entry, Stop Loss, TP1, TP2, and Lot Size
- Institutional Pre-Sweep Liquidity Pool Watchlist (BSL & SSL)
- Session Timing Windows (UTC & IST) and Dynamic Sweep Arrival ETA
- Exclusive Liquidity Sweep Strategy (Generic chop trades 100% muted)
"""

import logging
import os
import sys
import time
from pathlib import Path
import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from src.live_data import fetch_live_candles
from scripts.autonomous_live_trader import (
    ProductionRuntimeManager,
    analyze_with_6_master_upgrades,
    format_card,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("TelegramVIPBot")


def run_telegram_vip_bot():
    token = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("BOT_TOKEN")
    if not token:
        print("[!] Error: TELEGRAM_BOT_TOKEN is not set in .env")
        return

    mgr = ProductionRuntimeManager()
    last_update_id = 0
    base_url = f"https://api.telegram.org/bot{token}"

    print(f"[*] Starting Telegram VIP Instant Trade Listener for @AMARREDDYBOT...")

    while True:
        try:
            url = f"{base_url}/getUpdates?offset={last_update_id + 1}&timeout=10"
            resp = requests.get(url, timeout=15).json()
            if not resp.get("ok"):
                time.sleep(2)
                continue

            for update in resp.get("result", []):
                last_update_id = update["update_id"]
                msg = update.get("message", {})
                chat_id = msg.get("chat", {}).get("id")
                text = msg.get("text", "").strip()

                if not chat_id or not text:
                    continue

                cmd = text.lower()
                df = fetch_live_candles("XAUUSD", "M5", 50)
                setup = analyze_with_6_master_upgrades(df, mgr)

                if cmd.startswith("/start"):
                    reply = (
                        "👑 *VIP XAUUSD Institutional Trading Assistant Online!*\n\n"
                        "Upgraded with Pre-Sweep Liquidity Pool Mapping & Dynamic Arrival ETA.\n\n"
                        "Commands:\n"
                        "⚡ `/trade` or `/signal` - Live Institutional Liquidity Setup & Arrival ETA\n"
                        "📊 `/status` - Current Market Regime, Session Killzone & Timing\n"
                        "ℹ️ `/help` - Rules of engagement and execution guide"
                    )
                    send_msg(base_url, chat_id, reply)
                    send_msg(base_url, chat_id, format_card(setup, "LIVE MARKET SETUP"))

                elif any(cmd.startswith(c) for c in ["/signal", "/trade", "/next", "/entry"]):
                    reply = format_card(setup, "INSTANT MASTER SETUP")
                    send_msg(base_url, chat_id, reply)

                elif cmd.startswith("/status") or cmd.startswith("/bias"):
                    reply = format_card(setup, "LIVE MARKET PULSE")
                    send_msg(base_url, chat_id, reply)

                elif cmd.startswith("/help"):
                    reply = (
                        "📖 *VIP Execution Rules:*\n\n"
                        "1. **Exclusive Liquidity Sweeps**: We ONLY trade confirmed sweeps of resting high/low liquidity pools.\n"
                        "2. **The 1% Rule**: Risk at most 1% of your account per position.\n"
                        "3. **TP1 De-Risking**: When TP1 is hit, close 50% of position and move SL to Entry (Break-Even).\n"
                        "4. **No News Trading**: Stand aside 30 minutes before and after Red-Folder USD events."
                    )
                    send_msg(base_url, chat_id, reply)

        except Exception as exc:
            logger.error("Polling error: %s", exc)
            time.sleep(3)


def send_msg(base_url: str, chat_id: int, text: str):
    requests.post(
        f"{base_url}/sendMessage",
        json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
        timeout=10,
    )


if __name__ == "__main__":
    run_telegram_vip_bot()

