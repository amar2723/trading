from __future__ import annotations

import numpy as np
import pandas as pd


def _swing_highs(df: pd.DataFrame, lookback: int) -> pd.Series:
    highs = df["high"]
    return highs.eq(highs.rolling(lookback * 2 + 1, center=True).max()).fillna(False)


def _swing_lows(df: pd.DataFrame, lookback: int) -> pd.Series:
    lows = df["low"]
    return lows.eq(lows.rolling(lookback * 2 + 1, center=True).min()).fillna(False)


def add_market_concepts(
    df: pd.DataFrame,
    swing_lookback: int = 3,
    sweep_lookback: int = 20,
    displacement_atr_mult: float = 1.2,
    fvg_min_atr_mult: float = 0.15,
) -> pd.DataFrame:
    out = df.copy()
    out["swing_high"] = _swing_highs(out, swing_lookback)
    out["swing_low"] = _swing_lows(out, swing_lookback)
    out["prev_swing_high"] = out["high"].where(out["swing_high"]).ffill().shift()
    out["prev_swing_low"] = out["low"].where(out["swing_low"]).ffill().shift()

    rolling_high = out["high"].shift().rolling(sweep_lookback).max()
    rolling_low = out["low"].shift().rolling(sweep_lookback).min()
    out["bullish_liquidity_sweep"] = (out["low"] < rolling_low) & (out["close"] > rolling_low)
    out["bearish_liquidity_sweep"] = (out["high"] > rolling_high) & (out["close"] < rolling_high)
    out["sweep_low"] = np.where(out["bullish_liquidity_sweep"], out["low"], np.nan)
    out["sweep_high"] = np.where(out["bearish_liquidity_sweep"], out["high"], np.nan)

    out["bullish_bos"] = out["close"] > out["prev_swing_high"]
    out["bearish_bos"] = out["close"] < out["prev_swing_low"]

    recent_bear_sweep = out["bearish_liquidity_sweep"].rolling(sweep_lookback).max().fillna(0).astype(bool)
    recent_bull_sweep = out["bullish_liquidity_sweep"].rolling(sweep_lookback).max().fillna(0).astype(bool)
    out["bullish_mss"] = out["bullish_bos"] & recent_bull_sweep
    out["bearish_mss"] = out["bearish_bos"] & recent_bear_sweep

    out["displacement_candle"] = out["candle_body"] > (out["atr"] * displacement_atr_mult)
    out["bullish_displacement"] = out["displacement_candle"] & (out["close"] > out["open"])
    out["bearish_displacement"] = out["displacement_candle"] & (out["close"] < out["open"])

    fvg_min = out["atr"] * fvg_min_atr_mult
    out["bullish_fvg"] = out["low"] > out["high"].shift(2)
    out["bearish_fvg"] = out["high"] < out["low"].shift(2)
    out["bullish_fvg_size"] = np.where(out["bullish_fvg"], out["low"] - out["high"].shift(2), 0.0)
    out["bearish_fvg_size"] = np.where(out["bearish_fvg"], out["low"].shift(2) - out["high"], 0.0)
    out["bullish_fvg"] = out["bullish_fvg"] & (out["bullish_fvg_size"] >= fvg_min)
    out["bearish_fvg"] = out["bearish_fvg"] & (out["bearish_fvg_size"] >= fvg_min)

    out["support"] = out["low"].where(out["swing_low"]).ffill()
    out["resistance"] = out["high"].where(out["swing_high"]).ffill()
    out["nearest_liquidity_low"] = out["low"].where(out["swing_low"] | out["bullish_liquidity_sweep"]).ffill()
    out["nearest_liquidity_high"] = out["high"].where(out["swing_high"] | out["bearish_liquidity_sweep"]).ffill()
    out["next_liquidity_high"] = out["high"].where(out["swing_high"]).bfill()
    out["next_liquidity_low"] = out["low"].where(out["swing_low"]).bfill()
    return out
