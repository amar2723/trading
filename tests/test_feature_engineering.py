from __future__ import annotations

import pandas as pd

from src.feature_engineering.feature_pipeline import build_features
from src.feature_engineering.indicators import atr, ema, rsi
from src.feature_engineering.market_structure import add_trend_features, detect_swings


def sample_ohlcv(rows: int = 80) -> pd.DataFrame:
    close = [2000 + i + (2 if i % 10 == 0 else 0) for i in range(rows)]
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=rows, freq="5min"),
            "open": [value - 0.5 for value in close],
            "high": [value + 1.0 for value in close],
            "low": [value - 1.0 for value in close],
            "close": close,
            "tick_volume": [100 + i for i in range(rows)],
            "spread": [20 for _ in range(rows)],
        }
    )


def test_ema_returns_expected_first_value():
    series = pd.Series([10.0, 11.0, 12.0])
    result = ema(series, 20)
    assert result.iloc[0] == 10.0
    assert result.iloc[-1] > result.iloc[0]


def test_rsi_is_bounded():
    result = rsi(pd.Series(range(1, 40)), 14)
    assert ((result >= 0) & (result <= 100)).all()


def test_atr_is_positive():
    result = atr(sample_ohlcv(), 14)
    assert (result > 0).all()


def test_swing_detection_marks_known_high_and_low():
    df = pd.DataFrame(
        {
            "high": [1, 2, 5, 2, 1, 2, 3],
            "low": [1, 0, -1, 0, 1, 0, 1],
        }
    )
    result = detect_swings(df, lookback=2)
    assert bool(result.loc[2, "swing_high"])
    assert bool(result.loc[2, "swing_low"])


def test_trend_detection_bullish_alignment():
    df = sample_ohlcv()
    features = build_features(df)
    trended = add_trend_features(features)
    assert "trend_direction" in trended.columns
    assert set(trended["ema_alignment"].unique()).issubset({"bullish", "bearish", "neutral"})
