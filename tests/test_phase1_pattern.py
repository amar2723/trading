from __future__ import annotations

import pandas as pd

from src.pattern_detector import detect_pattern


def test_bullish_sweep_pattern():
    df = _with_warmup(
        {"timestamp": "2024-01-01 00:00:00", "open": 10, "high": 11, "low": 9, "close": 9.5},
        {"timestamp": "2024-01-01 00:05:00", "open": 9.4, "high": 12, "low": 8.8, "close": 11.5},
    )
    signal = detect_pattern(df, use_closed_candle=False)
    assert signal.signal == "BUY"
    assert signal.entry == 11.5
    assert signal.sl == 8.8
    assert signal.debug["bull_score"] >= 3


def test_bearish_sweep_pattern():
    df = _with_warmup(
        {"timestamp": "2024-01-01 00:00:00", "open": 10, "high": 11, "low": 9, "close": 10.5},
        {"timestamp": "2024-01-01 00:05:00", "open": 10.6, "high": 11.2, "low": 8, "close": 8.5},
    )
    signal = detect_pattern(df, use_closed_candle=False)
    assert signal.signal == "SELL"
    assert signal.entry == 8.5
    assert signal.sl == 11.2
    assert signal.debug["bear_score"] >= 3


def test_no_pattern():
    df = _with_warmup(
        {"timestamp": "2024-01-01 00:00:00", "open": 10, "high": 11, "low": 9, "close": 9.5},
        {"timestamp": "2024-01-01 00:05:00", "open": 9.6, "high": 10, "low": 9.2, "close": 9.8},
    )
    assert detect_pattern(df, use_closed_candle=False).signal == "NONE"


def _with_warmup(previous: dict, current: dict) -> pd.DataFrame:
    warmup = [
        {
            "timestamp": f"2023-12-31 23:{minute:02d}:00",
            "open": 10.0,
            "high": 10.6,
            "low": 9.8,
            "close": 10.2,
        }
        for minute in range(20)
    ]
    return pd.DataFrame(warmup + [previous, current])
