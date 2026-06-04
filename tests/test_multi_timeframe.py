from __future__ import annotations

import pandas as pd

from src.multi_timeframe import analyze_h1_trend, confirm_mtf_entry, validate_m15_structure


def candles(rows: int = 250) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=rows, freq="5min"),
            "open": [100 + i * 0.1 for i in range(rows)],
            "high": [101 + i * 0.1 for i in range(rows)],
            "low": [99 + i * 0.1 for i in range(rows)],
            "close": [100.5 + i * 0.1 for i in range(rows)],
            "volume": [100] * rows,
        }
    )


def test_mtf_modules_return_schema():
    h1 = candles()
    m15 = candles()
    m5 = candles()
    trend = analyze_h1_trend(h1)
    structure = validate_m15_structure(m15)
    signal = confirm_mtf_entry(trend, structure, m5)
    assert trend.trend_direction in {"Bullish", "Bearish", "Neutral"}
    assert isinstance(structure.bullish_structure_valid, bool)
    assert signal.signal in {"BUY", "SELL", "NO TRADE"}
