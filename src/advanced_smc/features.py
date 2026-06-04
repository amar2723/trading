from __future__ import annotations

import numpy as np
import pandas as pd


def add_core_features(df: pd.DataFrame, swing_lookback: int = 3, atr_period: int = 14) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out["body"] = (out["close"] - out["open"]).abs()
    out["range"] = out["high"] - out["low"]
    out["upper_wick"] = out["high"] - out[["open", "close"]].max(axis=1)
    out["lower_wick"] = out[["open", "close"]].min(axis=1) - out["low"]
    out["avg_body_20"] = out["body"].rolling(20, min_periods=1).mean()

    tr = pd.concat(
        [
            out["high"] - out["low"],
            (out["high"] - out["close"].shift()).abs(),
            (out["low"] - out["close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["atr"] = tr.rolling(atr_period, min_periods=1).mean()
    out["ema20"] = out["close"].ewm(span=20, adjust=False).mean()
    out["ema50"] = out["close"].ewm(span=50, adjust=False).mean()
    out["trend_direction"] = np.select(
        [out["ema20"] > out["ema50"], out["ema20"] < out["ema50"]],
        ["BULLISH", "BEARISH"],
        default="NEUTRAL",
    )

    window = swing_lookback * 2 + 1
    out["swing_high"] = out["high"].eq(out["high"].rolling(window, center=True, min_periods=window).max()).fillna(False)
    out["swing_low"] = out["low"].eq(out["low"].rolling(window, center=True, min_periods=window).min()).fillna(False)
    out["swing_high_price"] = np.where(out["swing_high"], out["high"], np.nan)
    out["swing_low_price"] = np.where(out["swing_low"], out["low"], np.nan)
    out["prev_swing_high"] = pd.Series(out["swing_high_price"]).ffill().shift()
    out["prev_swing_low"] = pd.Series(out["swing_low_price"]).ffill().shift()
    out["higher_high"] = out["swing_high"] & (out["high"] > out["prev_swing_high"])
    out["lower_high"] = out["swing_high"] & (out["high"] < out["prev_swing_high"])
    out["higher_low"] = out["swing_low"] & (out["low"] > out["prev_swing_low"])
    out["lower_low"] = out["swing_low"] & (out["low"] < out["prev_swing_low"])
    return out
