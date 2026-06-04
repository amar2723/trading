from __future__ import annotations

import pandas as pd

from src.realtime.alert_manager import AlertManager
from src.realtime.live_data_feed import LiveDataFeed
from src.realtime.signal_engine import SignalConfig, SignalEngine
from src.realtime.trade_manager import LiveRiskConfig, TradeManager


class FakeConnector:
    def __init__(self):
        self.calls = 0

    def get_rates(self, symbol: str, timeframe: str, bars: int = 500) -> pd.DataFrame:
        self.calls += 1
        return candle_frame()


def candle_frame(rows: int = 80) -> pd.DataFrame:
    close = [2000 + i * 0.2 for i in range(rows)]
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=rows, freq="5min"),
            "open": [value - 0.1 for value in close],
            "high": [value + 0.4 for value in close],
            "low": [value - 0.4 for value in close],
            "close": close,
            "tick_volume": [100 + i for i in range(rows)],
            "spread": [20 for _ in range(rows)],
        }
    )
    return df


def test_live_data_feed_detects_new_candle():
    feed = LiveDataFeed(FakeConnector(), symbol="XAUUSD", timeframes=("M5",), bars=80)
    updates = feed.refresh()
    assert updates["M5"] is True
    assert "M5" in feed.latest_candles


def test_signal_generation_hold_when_confidence_low():
    engine = SignalEngine(config=SignalConfig(confidence_threshold=99.0))
    signal = engine.latest_signal(candle_frame())
    assert signal["signal"] == "HOLD"
    assert "confidence" in signal


def test_alert_format():
    manager = AlertManager()
    message = manager.format_alert(
        {"signal": "BUY", "entry": 2000, "sl": 1995, "tp1": 2005, "tp2": 2010, "confidence": 82.0},
        "XAUUSD",
    )
    assert "BUY SIGNAL" in message
    assert "Symbol: XAUUSD" in message
    assert "Confidence: 82.0%" in message


def test_trade_logging_and_duplicate_filter(tmp_path):
    log_path = tmp_path / "signals.csv"
    manager = TradeManager(LiveRiskConfig(min_confidence=70), log_path)
    signal = {
        "timestamp": "2024-01-01 00:00:00",
        "signal": "BUY",
        "entry": 2000,
        "sl": 1995,
        "tp1": 2005,
        "tp2": 2010,
        "risk_reward": 1.0,
        "confidence": 80,
        "spread": 20,
        "atr": 2,
    }
    assert manager.accept_signal(signal)
    assert not manager.accept_signal(signal.copy())
    content = log_path.read_text(encoding="utf-8")
    assert "BUY" in content
    assert "duplicate_signal" in content
