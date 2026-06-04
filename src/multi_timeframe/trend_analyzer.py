from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.advanced_smc.features import add_core_features


@dataclass
class TrendState:
    trend_direction: str
    trend_strength: float
    ema_alignment: str
    has_hh_hl: bool
    has_lh_ll: bool


def analyze_h1_trend(h1_candles: pd.DataFrame) -> TrendState:
    data = add_core_features(h1_candles)
    return trend_state_from_prepared(data, len(data) - 1)


def trend_state_from_prepared(data: pd.DataFrame, idx: int) -> TrendState:
    latest = data.iloc[idx]
    recent = data.iloc[max(0, idx - 80): idx + 1]
    has_hh_hl = bool(recent["higher_high"].any() and recent["higher_low"].any())
    has_lh_ll = bool(recent["lower_high"].any() and recent["lower_low"].any())

    bullish_ema = latest["ema20"] > latest["ema50"] > latest["close"] * 0 or latest["ema20"] > latest["ema50"]
    bearish_ema = latest["ema20"] < latest["ema50"]
    # EMA200 is not produced by add_core_features, so calculate here for the strict alignment.
    ema200 = data["close"].ewm(span=200, adjust=False).mean().iloc[idx]
    bullish_ema = latest["ema20"] > latest["ema50"] > ema200
    bearish_ema = latest["ema20"] < latest["ema50"] < ema200

    atr = latest.get("atr") or 1
    strength = abs(float(latest["ema20"] - ema200)) / float(atr) if atr else 0.0
    if bullish_ema and has_hh_hl:
        direction = "Bullish"
        alignment = "EMA20 > EMA50 > EMA200"
    elif bearish_ema and has_lh_ll:
        direction = "Bearish"
        alignment = "EMA20 < EMA50 < EMA200"
    else:
        direction = "Neutral"
        alignment = "Mixed"
    return TrendState(direction, float(strength), alignment, has_hh_hl, has_lh_ll)
