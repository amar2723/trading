from __future__ import annotations

"""
Order Flow & Buyer/Seller Pressure Engine.

Computes:
1. Estimated Buyer Volume vs Seller Volume (Tick Delta)
2. Cumulative Volume Delta (CVD)
3. Buyer/Seller Dominance Ratio (0 - 100%)
4. Volume Climax & Institutional Effort-vs-Result
"""

import numpy as np
import pandas as pd


def compute_volume_delta(df: pd.DataFrame, rolling_window: int = 20) -> pd.DataFrame:
    """
    Estimates Buyer Volume vs Seller Volume and Cumulative Delta (CVD).
    
    Using Volume Spread Analysis (VSA) principles:
    - Candle range = High - Low
    - Upper wick = High - max(Open, Close) (Seller defense)
    - Lower wick = min(Open, Close) - Low (Buyer defense)
    - Close location within the bar defines the net delta fraction.
    """
    out = df.copy()
    vol_col = "tick_volume" if "tick_volume" in out.columns else "volume" if "volume" in out.columns else None
    if vol_col is None:
        out["volume"] = 1000
        vol_col = "volume"

    candle_range = (out["high"] - out["low"]).replace(0, 0.00001)
    
    # Buyer fraction: how close the finish was to the high vs the low
    # When close == high -> 100% buyer dominance
    # When close == low  -> 100% seller dominance
    close_position = (out["close"] - out["low"]) / candle_range
    buyer_fraction = np.clip(close_position, 0.0, 1.0)
    seller_fraction = 1.0 - buyer_fraction
    
    total_vol = out[vol_col].astype(float)
    out["buyer_volume"] = total_vol * buyer_fraction
    out["seller_volume"] = total_vol * seller_fraction
    out["volume_delta"] = out["buyer_volume"] - out["seller_volume"]
    
    # Cumulative Volume Delta (CVD)
    out["cvd"] = out["volume_delta"].cumsum()
    out["cvd_rolling"] = out["volume_delta"].rolling(rolling_window, min_periods=1).sum()
    
    # Buyer Dominance Ratio (50% is equilibrium)
    rolling_buy = out["buyer_volume"].rolling(rolling_window, min_periods=1).sum()
    rolling_sell = out["seller_volume"].rolling(rolling_window, min_periods=1).sum()
    total_rolling = (rolling_buy + rolling_sell).replace(0, 0.00001)
    
    out["buyer_dominance_pct"] = (rolling_buy / total_rolling) * 100.0
    out["seller_dominance_pct"] = (rolling_sell / total_rolling) * 100.0
    
    # Institutional Volume Spike (Ultra High Volume = 2.0x average)
    avg_vol = total_vol.rolling(rolling_window, min_periods=1).mean()
    out["is_volume_climax"] = total_vol > (avg_vol * 2.0)
    out["is_volume_dryup"] = total_vol < (avg_vol * 0.5)
    
    return out
