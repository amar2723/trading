from __future__ import annotations

import numpy as np
import pandas as pd


def generate_entries(df: pd.DataFrame) -> pd.DataFrame:
    """Create BUY/SELL/HOLD entries from SMC pattern confirmations."""
    out = df.copy()
    time_col = "timestamp" if "timestamp" in out.columns else "time" if "time" in out.columns else None

    buy = (
        out.get("bullish_liquidity_sweep", 0).astype(bool)
        & out.get("bullish_mss", 0).astype(bool)
        & out.get("bullish_bos", 0).astype(bool)
    )
    sell = (
        out.get("bearish_liquidity_sweep", 0).astype(bool)
        & out.get("bearish_mss", 0).astype(bool)
        & out.get("bearish_bos", 0).astype(bool)
    )

    out["entry_type"] = np.select([buy, sell], ["BUY", "SELL"], default="HOLD")
    out["entry_price"] = np.where(buy | sell, out["close"], np.nan)
    out["entry_time"] = pd.NaT
    if time_col:
        out.loc[buy | sell, "entry_time"] = out.loc[buy | sell, time_col]
    out["target"] = np.select([buy, sell], [1, -1], default=0).astype(int)
    return out
