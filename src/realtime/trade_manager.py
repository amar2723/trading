from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path


logger = logging.getLogger(__name__)


@dataclass
class LiveRiskConfig:
    min_confidence: float = 70.0
    max_spread: float = 80.0
    max_atr: float = 25.0
    daily_loss_limit: float = 300.0
    cooldown_candles: int = 1


class TradeManager:
    def __init__(self, config: LiveRiskConfig | None = None, log_path: str | Path = "logs/signals.csv"):
        self.config = config or LiveRiskConfig()
        self.log_path = Path(log_path)
        self.last_signal_key: tuple[str, str] | None = None
        self.daily_pnl = 0.0
        self._ensure_log()

    def risk_filter(self, signal: dict) -> tuple[bool, str]:
        if signal["signal"] == "HOLD":
            return False, "hold"
        if signal.get("confidence", 0) < self.config.min_confidence:
            return False, "confidence_below_threshold"
        if signal.get("spread") is not None and signal["spread"] > self.config.max_spread:
            return False, "spread_too_high"
        if signal.get("atr") is not None and signal["atr"] > self.config.max_atr:
            return False, "atr_too_high"
        if self.daily_pnl <= -abs(self.config.daily_loss_limit):
            return False, "daily_loss_limit_hit"
        key = (signal.get("timestamp", ""), signal.get("signal", ""))
        if key == self.last_signal_key:
            return False, "duplicate_signal"
        return True, "accepted"

    def accept_signal(self, signal: dict) -> bool:
        ok, reason = self.risk_filter(signal)
        signal["result"] = reason
        self.log_signal(signal)
        if ok:
            self.last_signal_key = (signal.get("timestamp", ""), signal.get("signal", ""))
        return ok

    def log_signal(self, signal: dict) -> None:
        self._ensure_log()
        with self.log_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._fields())
            writer.writerow({field: signal.get(field) for field in self._fields()})
        logger.info("Signal logged: %s %s", signal.get("signal"), signal.get("result"))

    def _ensure_log(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_path.exists():
            with self.log_path.open("w", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=self._fields()).writeheader()

    @staticmethod
    def _fields() -> list[str]:
        return ["timestamp", "signal", "entry", "sl", "tp1", "tp2", "risk_reward", "confidence", "spread", "atr", "result"]
