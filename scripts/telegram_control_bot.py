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
    format_macro_card,
    format_news_card,
    format_liquidity_card,
    format_pnl_card,
    format_telemetry_pulse,
    get_institutional_keyboard,
    broker,
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

    print(f"[*] Starting Telegram VIP Institutional Trading Desk for @AMARREDDYBOT...")

    while True:
        try:
            url = f"{base_url}/getUpdates?offset={last_update_id + 1}&timeout=10"
            resp = requests.get(url, timeout=15).json()
            if not resp.get("ok"):
                time.sleep(2)
                continue

            for update in resp.get("result", []):
                last_update_id = update["update_id"]

                # 1. Handle Inline Button Clicks
                if "callback_query" in update:
                    cq = update["callback_query"]
                    cq_id = cq["id"]
                    chat_id = cq.get("message", {}).get("chat", {}).get("id")
                    c_data = cq.get("data", "")

                    try:
                        requests.post(f"{base_url}/answerCallbackQuery", json={"callback_query_id": cq_id}, timeout=5)
                    except Exception:
                        pass

                    if not chat_id:
                        continue

                    df = fetch_live_candles("XAUUSD", "M5", 50)
                    setup = analyze_with_6_master_upgrades(df, mgr)

                    if c_data == "cmd_signal":
                        send_msg(base_url, chat_id, format_card(setup, "INSTANT MASTER SETUP"), reply_markup=get_institutional_keyboard())
                    elif c_data == "cmd_pulse":
                        send_msg(base_url, chat_id, format_telemetry_pulse(setup, broker, mgr), reply_markup=get_institutional_keyboard())
                    elif c_data == "cmd_liquidity":
                        send_msg(base_url, chat_id, format_liquidity_card(setup.get("watchlist", {})), reply_markup=get_institutional_keyboard())
                    elif c_data == "cmd_macro":
                        send_msg(base_url, chat_id, format_macro_card(setup.get("macro_snapshot")), reply_markup=get_institutional_keyboard())
                    elif c_data == "cmd_news":
                        send_msg(base_url, chat_id, format_news_card(setup.get("news_status")), reply_markup=get_institutional_keyboard())
                    elif c_data == "cmd_pnl":
                        send_msg(base_url, chat_id, format_pnl_card(broker, mgr), reply_markup=get_institutional_keyboard())
                    continue

                # 2. Handle Text Commands
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
                        "👑 *VIP XAUUSD JPMorgan Institutional Desk Online!*\n\n"
                        "Upgraded with Macro Compass (DXY & US10Y), News Blackout & HTF Pools.\n\n"
                        "Tap the quick buttons below or use commands:\n"
                        "⚡ `/trade` or `/signal` - Live Institutional Liquidity Setup & Arrival ETA\n"
                        "🧭 `/macro` - DXY & US 10Y Yield Concordance Regime\n"
                        "📅 `/news` - High-Impact Red-Folder Economic Calendar\n"
                        "🎯 `/liquidity` - PDH, PDL, PWH, PWL & Resting Stop-Run Pools\n"
                        "📈 `/pnl` - Verified Win Rate & Performance Summary\n"
                        "📊 `/status` - Current Market Regime & Killzone\n"
                        "ℹ️ `/help` - Rules of engagement and execution guide"
                    )
                    send_msg(base_url, chat_id, reply, reply_markup=get_institutional_keyboard())
                    send_msg(base_url, chat_id, format_card(setup, "LIVE MARKET SETUP"), reply_markup=get_institutional_keyboard())

                elif any(cmd.startswith(c) for c in ["/signal", "/trade", "/next", "/entry"]):
                    reply = format_card(setup, "INSTANT MASTER SETUP")
                    send_msg(base_url, chat_id, reply, reply_markup=get_institutional_keyboard())

                elif cmd.startswith("/macro"):
                    reply = format_macro_card(setup.get("macro_snapshot"))
                    send_msg(base_url, chat_id, reply, reply_markup=get_institutional_keyboard())

                elif cmd.startswith("/news"):
                    reply = format_news_card(setup.get("news_status"))
                    send_msg(base_url, chat_id, reply, reply_markup=get_institutional_keyboard())

                elif cmd.startswith("/liquidity"):
                    reply = format_liquidity_card(setup.get("watchlist", {}))
                    send_msg(base_url, chat_id, reply, reply_markup=get_institutional_keyboard())

                elif cmd.startswith("/pnl"):
                    reply = format_pnl_card(broker, mgr)
                    send_msg(base_url, chat_id, reply, reply_markup=get_institutional_keyboard())

                elif cmd.startswith("/status") or cmd.startswith("/bias"):
                    reply = format_card(setup, "LIVE MARKET PULSE")
                    send_msg(base_url, chat_id, reply, reply_markup=get_institutional_keyboard())

                elif cmd.startswith("/help"):
                    reply = (
                        "📖 *JPMorgan Institutional Execution Rules:*\n\n"
                        "1. **Exclusive Liquidity Sweeps**: We ONLY trade confirmed sweeps of resting high/low liquidity pools.\n"
                        "2. **Macro Concordance Alignment**: Gold shorts are prohibited during strong dollar/yield sell-offs.\n"
                        "3. **Strict News Blackout**: Stand aside 30 minutes before and after Tier-1 USD news events.\n"
                        "4. **Volatility-Targeted Sizing**: Lot sizes dynamically scale down when ATR spikes.\n"
                        "5. **TP1 De-Risking**: Close 50% at TP1 and advance SL to Entry (Break-Even).\n"
                        "6. **Strict 1.0% Risk Ceiling**: Capital is never risked above 1.0% per trade."
                    )
                    send_msg(base_url, chat_id, reply, reply_markup=get_institutional_keyboard())

        except Exception as exc:
            logger.error("Polling error: %s", exc)
            time.sleep(3)


def send_msg(base_url: str, chat_id: int, text: str, reply_markup: dict | None = None):
    try:
        payload = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}
        if reply_markup is not None:
            payload["reply_markup"] = reply_markup
        r = requests.post(f"{base_url}/sendMessage", json=payload, timeout=10)
        if r.status_code == 400:
            # Fallback to plain text if Markdown parsing fails
            plain_payload = {"chat_id": chat_id, "text": text}
            if reply_markup is not None:
                plain_payload["reply_markup"] = reply_markup
            requests.post(f"{base_url}/sendMessage", json=plain_payload, timeout=10)
    except Exception as e:
        logger.error("send_msg error: %s", e)


if __name__ == "__main__":
    run_telegram_vip_bot()

