from __future__ import annotations

import numpy as np
import pandas as pd


def add_displacement(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close_near_high = (out["high"] - out["close"]) <= out["range"] * 0.25
    close_near_low = (out["close"] - out["low"]) <= out["range"] * 0.25
    strong = (out["body"] > 1.5 * out["avg_body_20"]) & (out["range"] > out["atr"])
    out["bullish_displacement"] = strong & (out["close"] > out["open"]) & close_near_high
    out["bearish_displacement"] = strong & (out["close"] < out["open"]) & close_near_low
    return out


def add_mss_bos(df: pd.DataFrame, sweep_window: int = 20) -> pd.DataFrame:
    out = df.copy()
    recent_low_sweep = out["sell_side_liquidity_sweep"].rolling(sweep_window, min_periods=1).max().astype(bool)
    recent_high_sweep = out["buy_side_liquidity_sweep"].rolling(sweep_window, min_periods=1).max().astype(bool)
    out["bullish_bos"] = out["close"] > out["prev_swing_high"]
    out["bearish_bos"] = out["close"] < out["prev_swing_low"]
    out["bullish_mss"] = recent_low_sweep & out["bullish_bos"]
    out["bearish_mss"] = recent_high_sweep & out["bearish_bos"]
    return out


def add_order_blocks(df: pd.DataFrame, search_back: int = 10) -> pd.DataFrame:
    out = df.copy()
    out["bullish_ob"] = False
    out["bearish_ob"] = False
    out["ob_high"] = np.nan
    out["ob_low"] = np.nan
    for i in range(len(out)):
        if out.at[i, "bullish_displacement"]:
            prev = out.iloc[max(0, i - search_back):i]
            candidates = prev[prev["close"] < prev["open"]]
            if not candidates.empty:
                ob = candidates.iloc[-1]
                out.at[i, "bullish_ob"] = True
                out.at[i, "ob_high"] = ob["high"]
                out.at[i, "ob_low"] = ob["low"]
        if out.at[i, "bearish_displacement"]:
            prev = out.iloc[max(0, i - search_back):i]
            candidates = prev[prev["close"] > prev["open"]]
            if not candidates.empty:
                ob = candidates.iloc[-1]
                out.at[i, "bearish_ob"] = True
                out.at[i, "ob_high"] = ob["high"]
                out.at[i, "ob_low"] = ob["low"]
    out["last_bullish_ob_high"] = out["ob_high"].where(out["bullish_ob"]).ffill()
    out["last_bullish_ob_low"] = out["ob_low"].where(out["bullish_ob"]).ffill()
    out["last_bearish_ob_high"] = out["ob_high"].where(out["bearish_ob"]).ffill()
    out["last_bearish_ob_low"] = out["ob_low"].where(out["bearish_ob"]).ffill()
    return out


def add_fvg(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["bullish_fvg"] = out["high"].shift(2) < out["low"]
    out["bearish_fvg"] = out["low"].shift(2) > out["high"]
    out["fvg_bottom"] = np.select([out["bullish_fvg"], out["bearish_fvg"]], [out["high"].shift(2), out["high"]], default=np.nan)
    out["fvg_top"] = np.select([out["bullish_fvg"], out["bearish_fvg"]], [out["low"], out["low"].shift(2)], default=np.nan)
    out["fvg_gap_size"] = (out["fvg_top"] - out["fvg_bottom"]).abs()
    out["last_bullish_fvg_top"] = out["fvg_top"].where(out["bullish_fvg"]).ffill()
    out["last_bullish_fvg_bottom"] = out["fvg_bottom"].where(out["bullish_fvg"]).ffill()
    out["last_bearish_fvg_top"] = out["fvg_top"].where(out["bearish_fvg"]).ffill()
    out["last_bearish_fvg_bottom"] = out["fvg_bottom"].where(out["bearish_fvg"]).ffill()
    return out


def add_retests(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["bullish_ob_retest"] = (out["low"] <= out["last_bullish_ob_high"]) & (out["close"] >= out["last_bullish_ob_low"])
    out["bearish_ob_retest"] = (out["high"] >= out["last_bearish_ob_low"]) & (out["close"] <= out["last_bearish_ob_high"])
    out["bullish_fvg_retest"] = (out["low"] <= out["last_bullish_fvg_top"]) & (out["close"] >= out["last_bullish_fvg_bottom"])
    out["bearish_fvg_retest"] = (out["high"] >= out["last_bearish_fvg_bottom"]) & (out["close"] <= out["last_bearish_fvg_top"])
    return out
