from __future__ import annotations

import numpy as np
import pandas as pd


def _previous_swing_low(df: pd.DataFrame) -> pd.Series:
    if "swing_low_price" in df.columns:
        return df["swing_low_price"].replace(0, np.nan).ffill().shift()
    if "swing_low" in df.columns:
        return df["low"].where(df["swing_low"].astype(bool)).ffill().shift()
    return df["low"].shift().rolling(20, min_periods=1).min()


def _previous_swing_high(df: pd.DataFrame) -> pd.Series:
    if "swing_high_price" in df.columns:
        return df["swing_high_price"].replace(0, np.nan).ffill().shift()
    if "swing_high" in df.columns:
        return df["high"].where(df["swing_high"].astype(bool)).ffill().shift()
    return df["high"].shift().rolling(20, min_periods=1).max()


def detect_liquidity_sweeps(df: pd.DataFrame) -> pd.DataFrame:
    """Detect stop runs through previous swing highs/lows followed by reclaim/rejection."""
    out = df.copy()
    out["previous_swing_low"] = _previous_swing_low(out)
    out["previous_swing_high"] = _previous_swing_high(out)

    bullish = (out["low"] < out["previous_swing_low"]) & (out["close"] > out["previous_swing_low"])
    bearish = (out["high"] > out["previous_swing_high"]) & (out["close"] < out["previous_swing_high"])

    out["bullish_liquidity_sweep"] = bullish.astype(int)
    out["bearish_liquidity_sweep"] = bearish.astype(int)
    out["sweep_price"] = np.select(
        [bullish, bearish],
        [out["previous_swing_low"], out["previous_swing_high"]],
        default=np.nan,
    )
    out["sweep_time"] = pd.NaT
    if "timestamp" in out.columns:
        out.loc[bullish | bearish, "sweep_time"] = out.loc[bullish | bearish, "timestamp"]
    elif "time" in out.columns:
        out.loc[bullish | bearish, "sweep_time"] = out.loc[bullish | bearish, "time"]

    out["sweep_low"] = np.where(bullish, out["low"], np.nan)
    out["sweep_high"] = np.where(bearish, out["high"], np.nan)
    return out
