from __future__ import annotations

import numpy as np
import pandas as pd


def detect_bos(df: pd.DataFrame) -> pd.DataFrame:
    """Detect close-based breaks beyond the latest known structure high/low."""
    out = df.copy()
    structure_high = out.get("swing_high_price", out["high"].where(out.get("swing_high", False))).replace(0, np.nan)
    structure_low = out.get("swing_low_price", out["low"].where(out.get("swing_low", False))).replace(0, np.nan)

    out["previous_structure_high"] = structure_high.ffill().shift()
    out["previous_structure_low"] = structure_low.ffill().shift()
    out["bullish_bos"] = (out["close"] > out["previous_structure_high"]).fillna(False).astype(int)
    out["bearish_bos"] = (out["close"] < out["previous_structure_low"]).fillna(False).astype(int)
    return out
