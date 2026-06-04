from __future__ import annotations

import pandas as pd


def evaluate_entry(row: pd.Series) -> tuple[str, list[str]]:
    buy_reasons = []
    if bool(row.get("sell_side_liquidity_sweep")):
        buy_reasons.append("Sell-side liquidity sweep")
    if bool(row.get("bullish_mss")):
        buy_reasons.append("Bullish MSS")
    if bool(row.get("bullish_displacement")):
        buy_reasons.append("Bullish displacement")
    if bool(row.get("close_above_structure")):
        buy_reasons.append("Close above structure")

    sell_reasons = []
    if bool(row.get("buy_side_liquidity_sweep")):
        sell_reasons.append("Buy-side liquidity sweep")
    if bool(row.get("bearish_mss")):
        sell_reasons.append("Bearish MSS")
    if bool(row.get("bearish_displacement")):
        sell_reasons.append("Bearish displacement")
    if bool(row.get("close_below_structure")):
        sell_reasons.append("Close below structure")

    buy_valid = len(buy_reasons) == 4
    sell_valid = len(sell_reasons) == 4
    if buy_valid and not sell_valid:
        return "BUY", buy_reasons
    if sell_valid and not buy_valid:
        return "SELL", sell_reasons
    return "NO TRADE", buy_reasons if len(buy_reasons) >= len(sell_reasons) else sell_reasons
