from __future__ import annotations

import numpy as np
import pandas as pd


def analyze_structure(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["previous_structure_high"] = out["swing_high_price"].ffill().shift()
    out["previous_structure_low"] = out["swing_low_price"].ffill().shift()
    out["HH"] = out["swing_high"] & (out["high"] > out["previous_structure_high"])
    out["LH"] = out["swing_high"] & (out["high"] < out["previous_structure_high"])
    out["HL"] = out["swing_low"] & (out["low"] > out["previous_structure_low"])
    out["LL"] = out["swing_low"] & (out["low"] < out["previous_structure_low"])
    out["bullish_structure"] = out["HH"].rolling(20, min_periods=1).max().astype(bool) & out["HL"].rolling(20, min_periods=1).max().astype(bool)
    out["bearish_structure"] = out["LH"].rolling(20, min_periods=1).max().astype(bool) & out["LL"].rolling(20, min_periods=1).max().astype(bool)
    out["trend_direction"] = np.select(
        [out["bullish_structure"], out["bearish_structure"], out["ema20"] > out["ema50"], out["ema20"] < out["ema50"]],
        ["BULLISH", "BEARISH", "BULLISH", "BEARISH"],
        default="NEUTRAL",
    )
    out["close_above_structure"] = out["close"] > out["previous_structure_high"]
    out["close_below_structure"] = out["close"] < out["previous_structure_low"]
    return out
