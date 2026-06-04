from __future__ import annotations

import numpy as np
import pandas as pd


def map_liquidity(df: pd.DataFrame, lookback: int = 50, tolerance_atr: float = 0.15) -> pd.DataFrame:
    out = df.copy()
    tolerance = out["atr"] * tolerance_atr
    recent_high = out["high"].shift().rolling(lookback, min_periods=1).max()
    recent_low = out["low"].shift().rolling(lookback, min_periods=1).min()

    out["equal_highs"] = (out["high"] - recent_high).abs() <= tolerance
    out["equal_lows"] = (out["low"] - recent_low).abs() <= tolerance
    out["swing_high_liquidity"] = out["high"].where(out["swing_high"]).ffill()
    out["swing_low_liquidity"] = out["low"].where(out["swing_low"]).ffill()
    out["high_liquidity_cluster"] = out["high"].shift().rolling(lookback, min_periods=1).quantile(0.9)
    out["low_liquidity_cluster"] = out["low"].shift().rolling(lookback, min_periods=1).quantile(0.1)

    out["nearest_buy_liquidity"] = out[["swing_high_liquidity", "high_liquidity_cluster"]].max(axis=1)
    out["nearest_sell_liquidity"] = out[["swing_low_liquidity", "low_liquidity_cluster"]].min(axis=1)
    out["next_buy_liquidity"] = out["high"].where(out["swing_high"]).bfill()
    out["next_sell_liquidity"] = out["low"].where(out["swing_low"]).bfill()
    out["major_swing_high_liquidity"] = out["high"].shift().rolling(lookback * 2, min_periods=1).max()
    out["major_swing_low_liquidity"] = out["low"].shift().rolling(lookback * 2, min_periods=1).min()
    return out


def liquidity_quality(row: pd.Series, side: str) -> bool:
    entry = row.get("close")
    if side == "BUY":
        target = row.get("nearest_buy_liquidity")
        return pd.notna(target) and pd.notna(entry) and target > entry
    target = row.get("nearest_sell_liquidity")
    return pd.notna(target) and pd.notna(entry) and target < entry
