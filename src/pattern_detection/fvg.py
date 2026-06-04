from __future__ import annotations

import numpy as np
import pandas as pd


def detect_fvg(df: pd.DataFrame) -> pd.DataFrame:
    """Detect three-candle fair value gaps."""
    out = df.copy()
    candle1_high = out["high"].shift(2)
    candle1_low = out["low"].shift(2)
    candle3_low = out["low"]
    candle3_high = out["high"]

    bullish = candle1_high < candle3_low
    bearish = candle1_low > candle3_high

    out["bullish_fvg"] = bullish.fillna(False).astype(int)
    out["bearish_fvg"] = bearish.fillna(False).astype(int)
    out["fvg_top"] = np.select([bullish, bearish], [candle3_low, candle1_low], default=np.nan)
    out["fvg_bottom"] = np.select([bullish, bearish], [candle1_high, candle3_high], default=np.nan)
    return out
