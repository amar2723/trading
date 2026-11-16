from __future__ import annotations

"""
Institutional Absorption & Delta Divergence Detector.

Identifies when Smart Money absorbs retail market orders at key liquidity zones.
"""

import numpy as np
import pandas as pd
from src.order_flow.volume_delta import compute_volume_delta


def detect_absorption(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    Detects Institutional Absorption:
    1. Bullish Absorption: Price swept a recent low, but Buyer Dominance > 55%
       or CVD creates a bullish divergence (Higher Low on CVD while Lower Low on Price).
    2. Bearish Absorption: Price swept a recent high, but Seller Dominance > 55%
       or CVD creates a bearish divergence (Lower High on CVD while Higher High on Price).
    """
    out = compute_volume_delta(df, rolling_window=lookback)
    
    rolling_low = out["low"].shift(1).rolling(lookback, min_periods=5).min()
    rolling_high = out["high"].shift(1).rolling(lookback, min_periods=5).max()
    
    cvd_rolling_low = out["cvd"].shift(1).rolling(lookback, min_periods=5).min()
    cvd_rolling_high = out["cvd"].shift(1).rolling(lookback, min_periods=5).max()
    
    # 1. Sweep of Price Extreme
    price_swept_low = out["low"] < rolling_low
    price_swept_high = out["high"] > rolling_high
    
    # 2. Delta Divergence at the Sweep
    # Price pierced low, but CVD did NOT pierce its low (Smart money absorbed selling!)
    bullish_cvd_div = price_swept_low & (out["cvd"] >= cvd_rolling_low)
    # Price pierced high, but CVD did NOT pierce its high (Smart money absorbed buying!)
    bearish_cvd_div = price_swept_high & (out["cvd"] <= cvd_rolling_high)
    
    # 3. High Volume Absorption Wick
    candle_range = (out["high"] - out["low"]).replace(0, 0.00001)
    lower_wick = (np.minimum(out["open"], out["close"]) - out["low"]) / candle_range
    upper_wick = (out["high"] - np.maximum(out["open"], out["close"])) / candle_range
    
    bullish_wick_absorption = (lower_wick >= 0.45) & (out["buyer_dominance_pct"] >= 52.0)
    bearish_wick_absorption = (upper_wick >= 0.45) & (out["seller_dominance_pct"] >= 52.0)
    
    out["bullish_absorption"] = (price_swept_low & bullish_cvd_div) | bullish_wick_absorption
    out["bearish_absorption"] = (price_swept_high & bearish_cvd_div) | bearish_wick_absorption
    
    return out
