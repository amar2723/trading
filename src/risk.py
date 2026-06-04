from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class TradePlan:
    signal: str
    confidence: float
    entry: float | None
    sl: float | None
    tp1: float | None
    tp2: float | None
    risk_reward: float | None


def build_trade_plan(row: pd.Series, signal: str, confidence: float) -> TradePlan:
    if signal == "HOLD":
        return TradePlan(signal, confidence, None, None, None, None, None)

    entry = float(row["close"])
    if signal == "BUY":
        sl = float(row.get("sweep_low") if pd.notna(row.get("sweep_low")) else row["low"] - row["atr"])
        tp1 = float(row.get("resistance") if pd.notna(row.get("resistance")) and row.get("resistance") > entry else entry + row["atr"] * 2)
        tp2 = float(row.get("next_liquidity_high") if pd.notna(row.get("next_liquidity_high")) and row.get("next_liquidity_high") > tp1 else entry + row["atr"] * 3)
        risk = entry - sl
        reward = tp1 - entry
    else:
        sl = float(row.get("sweep_high") if pd.notna(row.get("sweep_high")) else row["high"] + row["atr"])
        tp1 = float(row.get("support") if pd.notna(row.get("support")) and row.get("support") < entry else entry - row["atr"] * 2)
        tp2 = float(row.get("next_liquidity_low") if pd.notna(row.get("next_liquidity_low")) and row.get("next_liquidity_low") < tp1 else entry - row["atr"] * 3)
        risk = sl - entry
        reward = entry - tp1
    rr = reward / risk if risk > 0 else None
    return TradePlan(signal, confidence, entry, sl, tp1, tp2, rr)
