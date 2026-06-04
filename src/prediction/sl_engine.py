from __future__ import annotations

import pandas as pd


def calculate_stop_loss(row: pd.Series, signal: str, atr_buffer: float = 0.25) -> float | None:
    atr = float(row.get("atr") or 0)
    if signal == "BUY":
        swept = row.get("recent_swept_low")
        base = swept if pd.notna(swept) else row.get("low")
        return float(base) - atr * atr_buffer if pd.notna(base) else None
    if signal == "SELL":
        swept = row.get("recent_swept_high")
        base = swept if pd.notna(swept) else row.get("high")
        return float(base) + atr * atr_buffer if pd.notna(base) else None
    return None
