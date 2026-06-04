from __future__ import annotations

import numpy as np
import pandas as pd


def add_liquidity(df: pd.DataFrame, lookback: int = 50, equal_tolerance_atr: float = 0.15) -> pd.DataFrame:
    out = df.copy()
    out["buy_side_liquidity"] = out["high"].where(out["swing_high"]).ffill()
    out["sell_side_liquidity"] = out["low"].where(out["swing_low"]).ffill()

    tolerance = out["atr"] * equal_tolerance_atr
    recent_high = out["high"].shift().rolling(lookback, min_periods=1).max()
    recent_low = out["low"].shift().rolling(lookback, min_periods=1).min()
    out["equal_highs"] = (out["high"] - recent_high).abs() <= tolerance
    out["equal_lows"] = (out["low"] - recent_low).abs() <= tolerance
    out["high_cluster"] = out["high"].shift().rolling(lookback, min_periods=1).quantile(0.9)
    out["low_cluster"] = out["low"].shift().rolling(lookback, min_periods=1).quantile(0.1)

    out["sell_side_liquidity_sweep"] = (out["low"] < out["prev_swing_low"]) & (out["close"] > out["prev_swing_low"])
    out["buy_side_liquidity_sweep"] = (out["high"] > out["prev_swing_high"]) & (out["close"] < out["prev_swing_high"])
    out["swept_low"] = np.where(out["sell_side_liquidity_sweep"], out["low"], np.nan)
    out["swept_high"] = np.where(out["buy_side_liquidity_sweep"], out["high"], np.nan)
    out["recent_swept_low"] = pd.Series(out["swept_low"]).ffill()
    out["recent_swept_high"] = pd.Series(out["swept_high"]).ffill()
    return out
