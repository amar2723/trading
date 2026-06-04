from __future__ import annotations

import numpy as np
import pandas as pd


LOOKAHEAD_WINDOWS = (20, 50, 100)


def simulate_trades(df: pd.DataFrame, lookahead: int = 100) -> pd.DataFrame:
    """Simulate each generated trade and label TP/SL order plus MFE/MAE metrics."""
    out = df.copy()
    out["trade_result"] = "NO_TRADE"
    out["exit_price"] = np.nan
    out["exit_index"] = np.nan
    out["hit_window"] = np.nan
    out["max_favorable_excursion"] = np.nan
    out["max_adverse_excursion"] = np.nan
    out["max_profit_points"] = np.nan
    out["max_drawdown_points"] = np.nan

    for window in LOOKAHEAD_WINDOWS:
        out[f"result_{window}"] = "NO_TRADE"

    for i, row in out.iterrows():
        if row.get("entry_type") not in {"BUY", "SELL"}:
            continue
        if pd.isna(row.get("entry_price")) or pd.isna(row.get("sl_price")) or pd.isna(row.get("tp1")):
            continue

        future = out.iloc[i + 1 : i + 1 + lookahead]
        if future.empty:
            continue

        direction = 1 if row["entry_type"] == "BUY" else -1
        high_move = (future["high"] - row["entry_price"]) * direction
        low_move = (future["low"] - row["entry_price"]) * direction
        favorable = high_move if direction == 1 else low_move
        adverse = low_move if direction == 1 else high_move

        mfe = float(favorable.max())
        mae = float(adverse.min())
        out.at[i, "max_favorable_excursion"] = mfe
        out.at[i, "max_adverse_excursion"] = mae
        out.at[i, "max_profit_points"] = mfe
        out.at[i, "max_drawdown_points"] = abs(mae)

        result, exit_price, exit_index, hit_window = _simulate_single(row, future)
        out.at[i, "trade_result"] = result
        out.at[i, "exit_price"] = exit_price
        out.at[i, "exit_index"] = exit_index
        out.at[i, "hit_window"] = hit_window

        for window in LOOKAHEAD_WINDOWS:
            partial_future = out.iloc[i + 1 : i + 1 + window]
            partial_result, _, _, _ = _simulate_single(row, partial_future)
            out.at[i, f"result_{window}"] = partial_result

    return out


def _simulate_single(row: pd.Series, future: pd.DataFrame) -> tuple[str, float | None, int | None, int | None]:
    if future.empty:
        return "NO_EXIT", None, None, None

    entry = float(row["entry_price"])
    sl = float(row["sl_price"])
    tp1 = float(row["tp1"])
    tp2 = float(row["tp2"]) if pd.notna(row.get("tp2")) else np.nan
    side = row["entry_type"]

    tp1_hit = False
    for step, (idx, candle) in enumerate(future.iterrows(), start=1):
        if side == "BUY":
            sl_hit = candle["low"] <= sl
            tp1_now = candle["high"] >= tp1
            tp2_now = pd.notna(tp2) and candle["high"] >= tp2
        else:
            sl_hit = candle["high"] >= sl
            tp1_now = candle["low"] <= tp1
            tp2_now = pd.notna(tp2) and candle["low"] <= tp2

        if sl_hit and not tp1_hit:
            return "LOSS", sl, int(idx), step
        if tp2_now:
            return "WIN", tp2, int(idx), step
        if tp1_now:
            tp1_hit = True
            if pd.isna(tp2):
                return "WIN", tp1, int(idx), step
        if sl_hit and tp1_hit:
            return "PARTIAL", tp1, int(idx), step

    if tp1_hit:
        return "PARTIAL", tp1, None, None
    return "NO_EXIT", entry, None, None
