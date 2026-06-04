from __future__ import annotations

import numpy as np
import pandas as pd


def add_candle_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create candle anatomy and common candle-pattern features."""
    out = df.copy()
    out["range_size"] = out["high"] - out["low"]
    out["body_size"] = (out["close"] - out["open"]).abs()
    out["body_percentage"] = out["body_size"] / out["range_size"].replace(0, np.nan)
    out["upper_wick"] = out["high"] - out[["open", "close"]].max(axis=1)
    out["lower_wick"] = out[["open", "close"]].min(axis=1) - out["low"]
    out["wick_ratio"] = out["upper_wick"] / out["lower_wick"].replace(0, np.nan)
    out["bullish_candle"] = out["close"] > out["open"]
    out["bearish_candle"] = out["close"] < out["open"]

    prev_open = out["open"].shift()
    prev_close = out["close"].shift()
    current_body_high = out[["open", "close"]].max(axis=1)
    current_body_low = out[["open", "close"]].min(axis=1)
    prev_body_high = pd.concat([prev_open, prev_close], axis=1).max(axis=1)
    prev_body_low = pd.concat([prev_open, prev_close], axis=1).min(axis=1)
    out["engulfing_pattern"] = (current_body_high > prev_body_high) & (current_body_low < prev_body_low)

    out["inside_bar"] = (out["high"] < out["high"].shift()) & (out["low"] > out["low"].shift())
    out["outside_bar"] = (out["high"] > out["high"].shift()) & (out["low"] < out["low"].shift())

    out["candle_body"] = out["body_size"]
    out["body_to_range"] = out["body_percentage"]
    out["volatility"] = out["close"].pct_change().rolling(20, min_periods=1).std()
    out["volume_change"] = out[_volume_col(out)].pct_change().replace([np.inf, -np.inf], np.nan)
    out["range"] = out["range_size"]
    return out


def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["rolling_std"] = out["close"].pct_change().rolling(20, min_periods=1).std()
    out["volatility_ratio"] = out["range_size"] / out["atr"].replace(0, np.nan)
    return out


def _volume_col(df: pd.DataFrame) -> str:
    return "tick_volume" if "tick_volume" in df.columns else "volume"
