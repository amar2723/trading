from __future__ import annotations

"""
Market Situation & Regime Classifier.

Analyzes current market conditions:
- TRENDING_BULLISH
- TRENDING_BEARISH
- RANGING_CONSOLIDATION
- VOLATILITY_EXPANSION (News/High Risk)
"""

from typing import Literal
import numpy as np
import pandas as pd


MarketRegime = Literal[
    "TRENDING_BULLISH",
    "TRENDING_BEARISH",
    "RANGING_CONSOLIDATION",
    "VOLATILITY_EXPANSION",
]


def classify_market_regime(df: pd.DataFrame, ema_fast: int = 20, ema_slow: int = 50, atr_period: int = 14) -> pd.DataFrame:
    """
    Classifies each bar into one of 4 dynamic market regimes.
    """
    out = df.copy()
    
    # 1. EMAs
    out["ema_fast"] = out["close"].ewm(span=ema_fast, adjust=False).mean()
    out["ema_slow"] = out["close"].ewm(span=ema_slow, adjust=False).mean()
    out["ema_trend_slope"] = (out["ema_fast"] - out["ema_slow"]) / out["close"] * 1000.0
    
    # 2. Average True Range & Volatility Ratio
    high_low = out["high"] - out["low"]
    high_close = (out["high"] - out["close"].shift()).abs()
    low_close = (out["low"] - out["close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    out["atr"] = true_range.rolling(atr_period, min_periods=1).mean()
    avg_atr_long = out["atr"].rolling(50, min_periods=10).mean()
    out["volatility_expansion_ratio"] = out["atr"] / avg_atr_long.replace(0, 0.00001)
    
    # 3. Choppiness Index (0 to 100)
    # Higher (> 61.8) = heavy consolidation/chop; Lower (< 38.2) = strong directional trend
    range_high = out["high"].rolling(14).max()
    range_low = out["low"].rolling(14).min()
    atr_sum = true_range.rolling(14).sum()
    out["chop_index"] = 100 * np.log10(atr_sum / (range_high - range_low).replace(0, 0.0001)) / np.log10(14)
    out["chop_index"] = out["chop_index"].fillna(50.0).clip(0, 100)
    
    # 4. Regime Assignment
    regimes = []
    for _, row in out.iterrows():
        vol_ratio = row.get("volatility_expansion_ratio", 1.0)
        chop = row.get("chop_index", 50.0)
        fast = row.get("ema_fast", 0.0)
        slow = row.get("ema_slow", 0.0)
        price = row.get("close", 0.0)
        
        # News / High Risk spike
        if vol_ratio > 2.4:
            regimes.append("VOLATILITY_EXPANSION")
        # Heavy chop / Ranging
        elif chop > 58.0 or abs(fast - slow) / max(price, 1) < 0.0005:
            regimes.append("RANGING_CONSOLIDATION")
        # Bullish trend
        elif fast > slow and price > slow:
            regimes.append("TRENDING_BULLISH")
        # Bearish trend
        elif fast < slow and price < slow:
            regimes.append("TRENDING_BEARISH")
        else:
            regimes.append("RANGING_CONSOLIDATION")
            
    out["market_regime"] = regimes
    return out
