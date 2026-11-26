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


def test_fetch_live_spot_quote_swissquote_and_fallback(monkeypatch):
    from src import live_data

    class FakeResp:
        def __init__(self, status_code, payload):
            self.status_code = status_code
            self._payload = payload

        def json(self):
            return self._payload

    class FakeSession:
        def __init__(self, mode):
            self.mode = mode

        def get(self, url, timeout=3):
            if "swissquote" in url:
                if self.mode == "swissquote":
                    return FakeResp(200, [{"spreadProfilePrices": [{"bid": 3570.90, "ask": 3571.50}]}])
                return FakeResp(503, {})
            if "gold-api" in url:
                return FakeResp(200, {"price": 3571.20})
            return FakeResp(404, {})

    monkeypatch.setattr(live_data, "get_web_session", lambda: FakeSession("swissquote"))
    q1 = live_data.fetch_live_spot_quote("XAUUSD")
    assert q1 is not None
    assert q1["source"] == "SWISSQUOTE_SPOT_XAUUSD"
    assert q1["bid"] == 3570.90
    assert q1["ask"] == 3571.50
    assert q1["mid"] == 3571.20

    monkeypatch.setattr(live_data, "get_web_session", lambda: FakeSession("fallback"))
    q2 = live_data.fetch_live_spot_quote("XAUUSD")
    assert q2 is not None
    assert q2["source"] == "GOLD_API_SPOT_XAUUSD"
    assert q2["mid"] == 3571.20


def test_fetch_live_candles_survives_frozen_futures_with_live_spot(monkeypatch):
    from datetime import datetime, timedelta, timezone
    from src import live_data

    now_utc = datetime.now(timezone.utc)
    # Simulate Yahoo GC=F bars frozen 3 hours ago at 3600.0
    frozen_start = int((now_utc - timedelta(hours=3)).timestamp() // 300) * 300
    ts_list = [frozen_start - 300 * (29 - i) for i in range(30)]
    yahoo_payload = {
        "chart": {
            "result": [{
                "meta": {"regularMarketPrice": 3600.0, "regularMarketTime": frozen_start},
                "timestamp": ts_list,
                "indicators": {
                    "quote": [{
                        "open": [3600.0] * 30,
                        "high": [3602.0] * 30,
                        "low": [3598.0] * 30,
                        "close": [3600.0] * 30,
                        "volume": [100.0] * 30,
                    }]
                },
            }]
        }
    }

    class FakeResp:
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return yahoo_payload

    class FakeSession:
        def get(self, url, timeout=5):
            return FakeResp()

    monkeypatch.setattr(live_data, "get_web_session", lambda: FakeSession())
    monkeypatch.setattr(live_data, "is_terminal_process_active", lambda: False)
    monkeypatch.setattr(live_data, "fetch_live_spot_quote", lambda sym="XAUUSD": {
        "bid": 3570.80, "ask": 3571.40, "mid": 3571.10, "spread_pips": 6.0,
        "ts": now_utc.timestamp(), "source": "SWISSQUOTE_SPOT_XAUUSD",
    })
    live_data._SPOT_BAR_CACHE.clear()

    df = live_data.fetch_live_candles("XAUUSD", "M5", bars=50)
    assert not df.empty
    assert df["data_source"].iloc[-1] == "LIVE_WEB_SPOT_XAUUSD"
    assert df.attrs["regular_market_price"] == 3571.10
    assert abs(float(df["close"].iloc[-1]) - 3571.10) < 0.05
