from __future__ import annotations

import pandas as pd

from src.market_filters import add_filter_features, passes_chop_filter, passes_session_filter, passes_trend_filter


def test_session_filter():
    assert passes_session_filter("2024-01-01 08:00:00", "LONDON")
    assert not passes_session_filter("2024-01-01 22:00:00", "LONDON")


def test_trend_filter():
    row = pd.Series({"trend_direction": "BULLISH"})
    assert passes_trend_filter(row, "BUY", True)
    assert not passes_trend_filter(row, "SELL", True)


def test_chop_filter_and_features():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=30, freq="5min"),
            "open": range(30),
            "high": [x + 2 for x in range(30)],
            "low": [x - 2 for x in range(30)],
            "close": [x + 1 for x in range(30)],
        }
    )
    features = add_filter_features(df)
    assert "trend_direction" in features.columns
    assert passes_chop_filter(pd.Series({"chop_ratio": 5.0}), True, 4.0)
