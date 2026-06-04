from __future__ import annotations

import pandas as pd


def calculate_take_profits(row: pd.Series, signal: str, entry: float, stop_loss: float) -> dict:
    risk = abs(entry - stop_loss)
    if risk <= 0:
        return {"tp1": None, "tp2": None, "tp3": None, "rr_ratio": None}
    if signal == "BUY":
        tp1 = _above(row.get("nearest_buy_liquidity"), entry, entry + 1.5 * risk)
        tp2 = _above(row.get("next_buy_liquidity"), tp1, entry + 2.5 * risk)
        tp3 = _above(row.get("major_swing_high_liquidity"), tp2, entry + 3.5 * risk)
        rr = (tp1 - entry) / risk
    elif signal == "SELL":
        tp1 = _below(row.get("nearest_sell_liquidity"), entry, entry - 1.5 * risk)
        tp2 = _below(row.get("next_sell_liquidity"), tp1, entry - 2.5 * risk)
        tp3 = _below(row.get("major_swing_low_liquidity"), tp2, entry - 3.5 * risk)
        rr = (entry - tp1) / risk
    else:
        return {"tp1": None, "tp2": None, "tp3": None, "rr_ratio": None}
    return {"tp1": float(tp1), "tp2": float(tp2), "tp3": float(tp3), "rr_ratio": float(rr)}


def _above(value, threshold: float, fallback: float) -> float:
    return float(value) if pd.notna(value) and float(value) > threshold else float(fallback)


def _below(value, threshold: float, fallback: float) -> float:
    return float(value) if pd.notna(value) and float(value) < threshold else float(fallback)
