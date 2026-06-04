from __future__ import annotations

import logging
from dataclasses import dataclass

import requests


logger = logging.getLogger(__name__)


@dataclass
class AlertConfig:
    desktop: bool = False
    telegram_token: str | None = None
    telegram_chat_id: str | None = None
    discord_webhook: str | None = None


class AlertManager:
    def __init__(self, config: AlertConfig | None = None):
        self.config = config or AlertConfig()

    def format_alert(self, signal: dict, symbol: str = "XAUUSD") -> str:
        return (
            f"{signal['signal']} SIGNAL\n"
            f"Symbol: {symbol}\n"
            f"Entry: {signal.get('entry')}\n"
            f"SL: {signal.get('sl')}\n"
            f"TP1: {signal.get('tp1')}\n"
            f"TP2: {signal.get('tp2')}\n"
            f"Confidence: {signal.get('confidence', 0):.1f}%"
        )

    def send(self, signal: dict, symbol: str = "XAUUSD") -> None:
        message = self.format_alert(signal, symbol)
        print(message)
        logger.info("Alert sent:\n%s", message)
        self._desktop(message)
        self._telegram(message)
        self._discord(message)

    def _desktop(self, message: str) -> None:
        if not self.config.desktop:
            return
        try:
            from plyer import notification

            notification.notify(title="Trading Signal", message=message, timeout=8)
        except Exception as exc:
            logger.warning("Desktop notification skipped: %s", exc)

    def _telegram(self, message: str) -> None:
        if not self.config.telegram_token or not self.config.telegram_chat_id:
            return
        try:
            url = f"https://api.telegram.org/bot{self.config.telegram_token}/sendMessage"
            requests.post(url, json={"chat_id": self.config.telegram_chat_id, "text": message}, timeout=10)
        except Exception as exc:
            logger.warning("Telegram alert skipped: %s", exc)

    def _discord(self, message: str) -> None:
        if not self.config.discord_webhook:
            return
        try:
            requests.post(self.config.discord_webhook, json={"content": message}, timeout=10)
        except Exception as exc:
            logger.warning("Discord alert skipped: %s", exc)
