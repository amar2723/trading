from __future__ import annotations

"""
Telegram Connection & Diagnostic Tool for @AMARREDDYBOT.

Checks:
1. Reads token from .env
2. Validates bot identity with Telegram API (getMe)
3. Detects incoming chat ID from recent messages (getUpdates)
4. Sends a test trade alert to confirm full connectivity
"""

import os
import sys
from pathlib import Path
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")


def test_telegram_connection():
    token = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID") or os.getenv("CHAT_ID")

    print("==================================================")
    print("      TELEGRAM BOT CONNECTION CHECKER             ")
    print("==================================================")

    if not token:
        print("[!] No TELEGRAM_BOT_TOKEN found in .env!")
        print("Please paste your Bot API Token from @BotFather into .env:")
        print("TELEGRAM_BOT_TOKEN=your_token_here\n")
        return

    # 1. Verify Bot Token
    try:
        me_url = f"https://api.telegram.org/bot{token}/getMe"
        resp = requests.get(me_url, timeout=10).json()
        if not resp.get("ok"):
            print(f"[X] Invalid Token error: {resp.get('description')}")
            return
        bot_info = resp.get("result", {})
        bot_username = bot_info.get("username")
        print(f"[OK] Successfully authenticated with Telegram as: @{bot_username}")
    except Exception as exc:
        print(f"[X] Network connection error: {exc}")
        return

    # 2. Check or auto-detect Chat ID
    if not chat_id:
        print("\n[*] No TELEGRAM_CHAT_ID found in .env. Checking for messages sent to the bot...")
        try:
            updates_url = f"https://api.telegram.org/bot{token}/getUpdates"
            updates = requests.get(updates_url, timeout=10).json().get("result", [])
            if updates:
                last_msg = updates[-1].get("message", {})
                detected_chat = last_msg.get("chat", {}).get("id")
                first_name = last_msg.get("chat", {}).get("first_name", "Trader")
                if detected_chat:
                    print(f"[+] Found message from {first_name}! Chat ID: {detected_chat}")
                    chat_id = str(detected_chat)
            else:
                print("[!] No messages received yet.")
                print(f"Open Telegram, search for @{bot_username}, and tap START or send a message (e.g. 'hello').")
                print("Then run this script again to auto-detect your Chat ID!\n")
                return
        except Exception as exc:
            print(f"[X] Error fetching updates: {exc}")
            return

    # 3. Send test alert
    if chat_id:
        print(f"\n[*] Sending test trading signal to Chat ID {chat_id}...")
        test_message = (
            "🔔 *XAUUSD AI Trading Assistant Connected!*\n\n"
            "✅ *Bot*: @" + bot_username + "\n"
            "📈 *Status*: Active & Listening\n"
            "🎯 *Engine*: Institutional Confluence Engine (ICE)\n"
            "⚡ *Target Asset*: XAUUSD (Gold M5)\n\n"
            "Your trading system is now linked to your Telegram account!"
        )
        send_url = f"https://api.telegram.org/bot{token}/sendMessage"
        send_resp = requests.post(send_url, json={"chat_id": chat_id, "text": test_message, "parse_mode": "Markdown"}, timeout=10).json()
        if send_resp.get("ok"):
            print("[SUCCESS] Test alert sent to your phone! Check your Telegram app.")
        else:
            print(f"[X] Failed to send message: {send_resp.get('description')}")


if __name__ == "__main__":
    test_telegram_connection()
