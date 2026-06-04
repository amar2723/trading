from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from src.realtime.alert_manager import AlertConfig, AlertManager
from src.realtime.live_data_feed import LiveDataFeed
from src.realtime.mt5_connector import MT5Connector, MT5LoginConfig
from src.realtime.signal_engine import SignalConfig, SignalEngine
from src.realtime.trade_manager import LiveRiskConfig, TradeManager


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


@dataclass
class RealtimeConfig:
    symbol: str = "XAUUSD"
    primary_timeframe: str = "M5"
    timeframes: tuple[str, ...] = ("M1", "M5", "M15", "H1")
    bars: int = 500
    model_path: str = "models/xgboost_model.pkl"
    poll_seconds: int = 5
    confidence_threshold: float = 70.0
    max_spread: float = 80.0
    max_atr: float = 25.0
    daily_loss_limit: float = 300.0


def load_config(path: str | Path = "config/realtime_config.json") -> RealtimeConfig:
    config_path = Path(path)
    if not config_path.exists():
        return RealtimeConfig()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if "timeframes" in payload:
        payload["timeframes"] = tuple(payload["timeframes"])
    return RealtimeConfig(**payload)


class RealtimePipeline:
    def __init__(self, config: RealtimeConfig | None = None, connector: MT5Connector | None = None):
        self.config = config or RealtimeConfig()
        self.connector = connector or MT5Connector()
        self.feed = LiveDataFeed(self.connector, self.config.symbol, self.config.timeframes, self.config.bars)
        self.signal_engine = SignalEngine(
            self.config.model_path if Path(self.config.model_path).exists() else None,
            SignalConfig(self.config.confidence_threshold, self.config.max_spread, self.config.max_atr),
        )
        self.trade_manager = TradeManager(LiveRiskConfig(self.config.confidence_threshold, self.config.max_spread, self.config.max_atr, self.config.daily_loss_limit))
        self.alert_manager = AlertManager(AlertConfig())

    def start(self, login: MT5LoginConfig | None = None) -> None:
        if not self.connector.initialize_mt5():
            raise RuntimeError("Could not initialize MT5")
        if login and not self.connector.login(login):
            raise RuntimeError("Could not login to MT5")
        if not self.connector.select_symbol(self.config.symbol):
            raise RuntimeError(f"Could not select symbol {self.config.symbol}")
        if not self.connector.verify_connection():
            raise RuntimeError("MT5 connection verification failed")

    def stop(self) -> None:
        self.connector.shutdown()

    def run_once(self) -> dict:
        updates = self.feed.refresh()
        if not updates.get(self.config.primary_timeframe, False):
            return {"signal": "HOLD", "result": "no_new_candle"}
        data = self.feed.get_latest(self.config.primary_timeframe)
        signal = self.signal_engine.latest_signal(data)
        accepted = self.trade_manager.accept_signal(signal)
        if accepted:
            self.alert_manager.send(signal, self.config.symbol)
        return signal

    def run_forever(self) -> None:
        self.start()
        try:
            while True:
                try:
                    self.run_once()
                except Exception as exc:
                    logger.exception("Realtime loop error: %s", exc)
                time.sleep(self.config.poll_seconds)
        finally:
            self.stop()
