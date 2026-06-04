from __future__ import annotations

import numpy as np
import pandas as pd


def detect_mss(df: pd.DataFrame, sweep_window: int = 20) -> pd.DataFrame:
    """Detect market structure shifts after liquidity sweeps."""
    out = df.copy()

    lower_high_price = out["high"].where(out.get("lower_high", False).astype(bool) if hasattr(out.get("lower_high", False), "astype") else False)
    higher_low_price = out["low"].where(out.get("higher_low", False).astype(bool) if hasattr(out.get("higher_low", False), "astype") else False)

    out["recent_lower_high"] = lower_high_price.ffill().shift()
    out["recent_higher_low"] = higher_low_price.ffill().shift()

    # Fallback to previous structure coordinates when HH/HL/LH/LL labels are sparse.
    if "previous_structure_high" in out.columns:
        out["recent_lower_high"] = out["recent_lower_high"].fillna(out["previous_structure_high"])
    if "previous_structure_low" in out.columns:
        out["recent_higher_low"] = out["recent_higher_low"].fillna(out["previous_structure_low"])

    after_bullish_sweep = out["bullish_liquidity_sweep"].rolling(sweep_window, min_periods=1).max().astype(bool)
    after_bearish_sweep = out["bearish_liquidity_sweep"].rolling(sweep_window, min_periods=1).max().astype(bool)

    out["bullish_mss"] = (after_bullish_sweep & (out["close"] > out["recent_lower_high"])).fillna(False).astype(int)
    out["bearish_mss"] = (after_bearish_sweep & (out["close"] < out["recent_higher_low"])).fillna(False).astype(int)
    return out
