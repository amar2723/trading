from __future__ import annotations

import numpy as np
import pandas as pd


def add_stop_loss(df: pd.DataFrame) -> pd.DataFrame:
    """Attach stop-loss levels and risk points to generated entries."""
    out = df.copy()
    buy = out["entry_type"].eq("BUY")
    sell = out["entry_type"].eq("SELL")

    sweep_low = out.get("sweep_low", pd.Series(np.nan, index=out.index))
    sweep_high = out.get("sweep_high", pd.Series(np.nan, index=out.index))
    atr = out.get("atr", pd.Series(0.0, index=out.index)).replace(0, np.nan)

    buy_sl = sweep_low.where(sweep_low.notna(), out["low"] - atr)
    sell_sl = sweep_high.where(sweep_high.notna(), out["high"] + atr)
    out["sl_price"] = np.select([buy, sell], [buy_sl, sell_sl], default=np.nan)
    out["risk_points"] = np.select(
        [buy, sell],
        [out["entry_price"] - out["sl_price"], out["sl_price"] - out["entry_price"]],
        default=np.nan,
    )
    out.loc[out["risk_points"] <= 0, ["sl_price", "risk_points"]] = np.nan
    return out


def add_take_profit(df: pd.DataFrame) -> pd.DataFrame:
    """Attach TP1/TP2 levels from nearby liquidity zones with ATR fallbacks."""
    out = df.copy()
    buy = out["entry_type"].eq("BUY")
    sell = out["entry_type"].eq("SELL")
    atr = out.get("atr", pd.Series(0.0, index=out.index)).replace(0, np.nan).fillna(out["close"].abs() * 0.001)

    resistance = out.get("resistance", pd.Series(np.nan, index=out.index))
    support = out.get("support", pd.Series(np.nan, index=out.index))
    next_high = out.get("next_liquidity_high", pd.Series(np.nan, index=out.index))
    next_low = out.get("next_liquidity_low", pd.Series(np.nan, index=out.index))
    previous_high = out.get("previous_swing_high", pd.Series(np.nan, index=out.index))
    previous_low = out.get("previous_swing_low", pd.Series(np.nan, index=out.index))

    buy_tp1 = _first_valid_above(out["entry_price"], [resistance, previous_high], out["entry_price"] + 2 * atr)
    buy_tp2 = _first_valid_above(out["entry_price"], [next_high], out["entry_price"] + 3 * atr)
    sell_tp1 = _first_valid_below(out["entry_price"], [support, previous_low], out["entry_price"] - 2 * atr)
    sell_tp2 = _first_valid_below(out["entry_price"], [next_low], out["entry_price"] - 3 * atr)

    out["tp1"] = np.select([buy, sell], [buy_tp1, sell_tp1], default=np.nan)
    out["tp2"] = np.select([buy, sell], [buy_tp2, sell_tp2], default=np.nan)
    out["reward_points"] = np.select(
        [buy, sell],
        [out["tp1"] - out["entry_price"], out["entry_price"] - out["tp1"]],
        default=np.nan,
    )
    out["risk_reward_ratio"] = out["reward_points"] / out["risk_points"].replace(0, np.nan)
    out["risk_reward"] = out["risk_reward_ratio"]
    return out


def generate_sl_tp(df: pd.DataFrame) -> pd.DataFrame:
    return add_take_profit(add_stop_loss(df))


def _first_valid_above(entry: pd.Series, candidates: list[pd.Series], fallback: pd.Series) -> pd.Series:
    result = fallback.copy()
    for candidate in reversed(candidates):
        result = candidate.where(candidate > entry, result)
    return result


def _first_valid_below(entry: pd.Series, candidates: list[pd.Series], fallback: pd.Series) -> pd.Series:
    result = fallback.copy()
    for candidate in reversed(candidates):
        result = candidate.where(candidate < entry, result)
    return result
