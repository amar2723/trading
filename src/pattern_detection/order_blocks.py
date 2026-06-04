from __future__ import annotations

import numpy as np
import pandas as pd


def add_displacement(df: pd.DataFrame, lookback: int = 10, multiplier: float = 1.8) -> pd.DataFrame:
    out = df.copy()
    body = out["body_size"] if "body_size" in out.columns else (out["close"] - out["open"]).abs()
    avg_body = body.shift().rolling(lookback, min_periods=1).mean()
    displacement = body > multiplier * avg_body
    out["avg_body_last_10"] = avg_body
    out["displacement"] = displacement.fillna(False).astype(int)
    out["displacement_direction"] = np.select(
        [displacement & (out["close"] > out["open"]), displacement & (out["close"] < out["open"])],
        ["bullish", "bearish"],
        default="none",
    )
    out["bullish_displacement"] = ((out["displacement"] == 1) & (out["displacement_direction"] == "bullish")).astype(int)
    out["bearish_displacement"] = ((out["displacement"] == 1) & (out["displacement_direction"] == "bearish")).astype(int)
    return out


def detect_order_blocks(df: pd.DataFrame, search_back: int = 10) -> pd.DataFrame:
    """Find the last opposite candle before a displacement candle."""
    out = add_displacement(df)
    out["bullish_ob"] = 0
    out["bearish_ob"] = 0
    out["ob_high"] = np.nan
    out["ob_low"] = np.nan
    out["ob_time"] = pd.NaT

    time_col = "timestamp" if "timestamp" in out.columns else "time" if "time" in out.columns else None

    for i in range(len(out)):
        direction = out.at[i, "displacement_direction"]
        if direction not in {"bullish", "bearish"}:
            continue

        start = max(0, i - search_back)
        previous = out.iloc[start:i].copy()
        if direction == "bullish":
            candidates = previous[previous["close"] < previous["open"]]
            flag = "bullish_ob"
        else:
            candidates = previous[previous["close"] > previous["open"]]
            flag = "bearish_ob"

        if candidates.empty:
            continue

        ob_idx = candidates.index[-1]
        out.at[i, flag] = 1
        out.at[i, "ob_high"] = out.at[ob_idx, "high"]
        out.at[i, "ob_low"] = out.at[ob_idx, "low"]
        if time_col:
            out.at[i, "ob_time"] = out.at[ob_idx, time_col]

    return out
