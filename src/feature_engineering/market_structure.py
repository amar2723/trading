from __future__ import annotations

import numpy as np
import pandas as pd


def detect_swings(df: pd.DataFrame, lookback: int = 3) -> pd.DataFrame:
    """Detect local highs/lows and store their price coordinates."""
    out = df.copy()
    window = lookback * 2 + 1
    out["swing_high"] = out["high"].eq(out["high"].rolling(window, center=True, min_periods=window).max()).fillna(False)
    out["swing_low"] = out["low"].eq(out["low"].rolling(window, center=True, min_periods=window).min()).fillna(False)
    out["pivot_high"] = out["swing_high"]
    out["pivot_low"] = out["swing_low"]
    out["swing_high_price"] = np.where(out["swing_high"], out["high"], np.nan)
    out["swing_low_price"] = np.where(out["swing_low"], out["low"], np.nan)
    out["pivot_high_price"] = out["swing_high_price"]
    out["pivot_low_price"] = out["swing_low_price"]
    return out


def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    bullish = (out["ema20"] > out["ema50"]) & (out["ema50"] > out["ema200"])
    bearish = (out["ema20"] < out["ema50"]) & (out["ema50"] < out["ema200"])
    out["ema_alignment"] = np.select([bullish, bearish], ["bullish", "bearish"], default="neutral")
    out["trend_direction"] = out["ema_alignment"]
    out["trend_strength"] = (out["ema20"] - out["ema200"]).abs() / out["atr"].replace(0, np.nan)
    return out


def add_market_structure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Classify new pivots into HH/HL/LH/LL and carry the latest structure state."""
    out = df.copy()
    previous_swing_high = out["swing_high_price"].ffill().shift()
    previous_swing_low = out["swing_low_price"].ffill().shift()
    out["higher_high"] = out["swing_high"] & (out["high"] > previous_swing_high)
    out["lower_high"] = out["swing_high"] & (out["high"] < previous_swing_high)
    out["higher_low"] = out["swing_low"] & (out["low"] > previous_swing_low)
    out["lower_low"] = out["swing_low"] & (out["low"] < previous_swing_low)

    labels = np.select(
        [out["higher_high"], out["higher_low"], out["lower_high"], out["lower_low"]],
        ["HH", "HL", "LH", "LL"],
        default=None,
    )
    out["structure_event"] = pd.Series(labels, index=out.index).replace("None", np.nan)
    out["structure_state"] = out["structure_event"].ffill().fillna("unknown")
    return out
