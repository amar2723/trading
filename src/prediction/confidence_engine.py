from __future__ import annotations

import pandas as pd

from src.prediction.liquidity_mapper import liquidity_quality


def calculate_confidence(row: pd.Series, signal: str) -> tuple[float, list[str]]:
    if signal not in {"BUY", "SELL"}:
        return 0.0, []
    reasons = []
    score = 0.0
    if signal == "BUY":
        checks = [
            ("Liquidity Sweep", bool(row.get("sell_side_liquidity_sweep")), 25),
            ("MSS", bool(row.get("bullish_mss")), 25),
            ("Displacement", bool(row.get("bullish_displacement")), 20),
            ("Trend Alignment", row.get("trend_direction") == "BULLISH", 15),
            ("Liquidity Target Quality", liquidity_quality(row, "BUY"), 15),
        ]
    else:
        checks = [
            ("Liquidity Sweep", bool(row.get("buy_side_liquidity_sweep")), 25),
            ("MSS", bool(row.get("bearish_mss")), 25),
            ("Displacement", bool(row.get("bearish_displacement")), 20),
            ("Trend Alignment", row.get("trend_direction") == "BEARISH", 15),
            ("Liquidity Target Quality", liquidity_quality(row, "SELL"), 15),
        ]
    for label, passed, points in checks:
        if passed:
            score += points
            reasons.append(label)
    return min(score, 100.0), reasons
