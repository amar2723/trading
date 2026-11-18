"""
Evaluates the true production-gated strategy (Value Pullback + Donchian + ML filter)
under realistic capital sizing ($1,000 equity, so 0.01 lot risks ~0.5%, well within the 1.0% limit).
Uses conservative same-candle execution, realistic spread ($0.25) and slippage ($0.05).
"""

from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.rigorous_baseline_recalculation import (
    load_and_prepare_market_data,
    CONTRACT_SIZE,
    LOT_SIZE,
    BENCHMARK_SPREAD_USD,
    BENCHMARK_SLIPPAGE_USD,
    TOTAL_COST_PER_TRADE_USD
)

ROOT = Path(__file__).resolve().parents[1]

def run():
    df = load_and_prepare_market_data()
    n = len(df)
    split_idx = int(n * 0.8)

    ensemble = joblib.load(ROOT / "models/xauusd_master_ensemble.pkl")

    in_pos = False
    side = ""
    entry_p = 0.0
    sl_p = 0.0
    tp1_p = 0.0
    be_trig_p = 0.0
    be_active = False
    entry_idx = 0
    risk_dist_usd = 0.0
    initial_risk_usd = 0.0
    strategy_used = ""
    current_equity = 1000.0

    trade_ledger = []

    for i in range(50, n - 1):
        c_close = df["close"].iloc[i]
        c_open = df["open"].iloc[i]
        c_high = df["high"].iloc[i]
        c_low = df["low"].iloc[i]
        c_atr = df["atr"].iloc[i] if pd.notnull(df["atr"].iloc[i]) else 3.5

        if in_pos:
            resolved = False
            outcome = ""
            exit_price = 0.0
            effective_sl = entry_p if be_active else sl_p

            if side == "BUY":
                if c_low <= effective_sl:
                    resolved = True
                    exit_price = effective_sl - BENCHMARK_SLIPPAGE_USD
                    gross_pnl = (exit_price - entry_p) * LOT_SIZE * CONTRACT_SIZE
                    net_pnl = gross_pnl - (BENCHMARK_SLIPPAGE_USD * LOT_SIZE * CONTRACT_SIZE)
                    outcome = "BREAK_EVEN" if be_active else "LOSS"
                elif c_high >= tp1_p:
                    resolved = True
                    exit_price = tp1_p - BENCHMARK_SLIPPAGE_USD
                    gross_pnl = (exit_price - entry_p) * LOT_SIZE * CONTRACT_SIZE
                    net_pnl = gross_pnl - (BENCHMARK_SLIPPAGE_USD * LOT_SIZE * CONTRACT_SIZE)
                    outcome = "WIN"
                elif i - entry_idx >= 24:
                    resolved = True
                    exit_price = c_close - (BENCHMARK_SPREAD_USD / 2) - BENCHMARK_SLIPPAGE_USD
                    net_pnl = (exit_price - entry_p) * LOT_SIZE * CONTRACT_SIZE
                    outcome = "WIN" if net_pnl > 0 else "LOSS"

                if not resolved and c_high >= be_trig_p:
                    be_active = True

            elif side == "SELL":
                if c_high >= effective_sl:
                    resolved = True
                    exit_price = effective_sl + BENCHMARK_SLIPPAGE_USD
                    gross_pnl = (entry_p - exit_price) * LOT_SIZE * CONTRACT_SIZE
                    net_pnl = gross_pnl - (BENCHMARK_SLIPPAGE_USD * LOT_SIZE * CONTRACT_SIZE)
                    outcome = "BREAK_EVEN" if be_active else "LOSS"
                elif c_low <= tp1_p:
                    resolved = True
                    exit_price = tp1_p + BENCHMARK_SLIPPAGE_USD
                    gross_pnl = (entry_p - exit_price) * LOT_SIZE * CONTRACT_SIZE
                    net_pnl = gross_pnl - (BENCHMARK_SLIPPAGE_USD * LOT_SIZE * CONTRACT_SIZE)
                    outcome = "WIN"
                elif i - entry_idx >= 24:
                    resolved = True
                    exit_price = c_close + (BENCHMARK_SPREAD_USD / 2) + BENCHMARK_SLIPPAGE_USD
                    net_pnl = (entry_p - exit_price) * LOT_SIZE * CONTRACT_SIZE
                    outcome = "WIN" if net_pnl > 0 else "LOSS"

                if not resolved and c_low <= be_trig_p:
                    be_active = True

            if resolved:
                r_multiple = round(net_pnl / max(0.01, initial_risk_usd), 3)
                current_equity += net_pnl
                trade_ledger.append({
                    "entry_bar": entry_idx,
                    "exit_bar": i,
                    "entry_time": str(df["timestamp"].iloc[entry_idx]),
                    "exit_time": str(df["timestamp"].iloc[i]),
                    "is_test_set": entry_idx >= split_idx,
                    "strategy": strategy_used,
                    "side": side,
                    "entry_price": round(entry_p, 2),
                    "exit_price": round(exit_price, 2),
                    "initial_sl": round(sl_p, 2),
                    "initial_tp": round(tp1_p, 2),
                    "outcome": outcome,
                    "initial_risk_usd": round(initial_risk_usd, 2),
                    "net_pnl_usd": round(net_pnl, 2),
                    "r_multiple": r_multiple,
                    "ending_equity": round(current_equity, 2),
                })
                in_pos = False

        if not in_pos:
            h1_bull = df["h1_trend_bias"].iloc[i] == 1
            h1_bear = df["h1_trend_bias"].iloc[i] == -1
            m5_e20 = df["ema_20"].iloc[i]
            m5_e50 = df["ema_50"].iloc[i]
            m5_rsi = df["rsi"].iloc[i]

            cand_risk_dist = max(c_atr * 1.5, 4.50)

            is_pullback_buy = h1_bull and (min(m5_e20, m5_e50) - 1.5 <= c_close <= max(m5_e20, m5_e50) + 1.0) and (38.0 <= m5_rsi <= 62.0) and (c_close >= c_open)
            is_pullback_sell = h1_bear and (min(m5_e20, m5_e50) - 1.0 <= c_close <= max(m5_e20, m5_e50) + 1.5) and (38.0 <= m5_rsi <= 62.0) and (c_close <= c_open)
            is_donch_buy = (c_close > df["donch_h20"].iloc[i]) and h1_bull and (m5_rsi <= 68.0)
            is_donch_sell = (c_close < df["donch_l20"].iloc[i]) and h1_bear and (m5_rsi >= 32.0)

            action = "HOLD"
            strat = "HOLD"
            if is_pullback_buy: action, strat = "BUY", "H1_PULLBACK"
            elif is_pullback_sell: action, strat = "SELL", "H1_PULLBACK"
            elif is_donch_buy: action, strat = "BUY", "DONCHIAN"
            elif is_donch_sell: action, strat = "SELL", "DONCHIAN"

            if action != "HOLD":
                feat_row = pd.DataFrame([{
                    "htf_trend_bias": df["h1_trend_bias"].iloc[i],
                    "h1_dist_ema50": df["h1_dist_ema50"].iloc[i],
                    "dist_ema_50": (c_close - df["ema_50"].iloc[i]) / df["ema_50"].iloc[i] * 100,
                    "is_prime_session": df["is_prime"].iloc[i],
                    "is_asian_chop": df["is_asian"].iloc[i],
                    "atr_pct": (c_atr / c_close) * 100,
                    "volatility_expansion": 1.0,
                    "discount_depth": 0.5,
                    "sweep_high_confirmed": 0,
                    "sweep_low_confirmed": 0,
                    "signed_tick_volume_proxy": 0.0,
                    "cvd_slope": 0.0,
                    "signed_buyer_proxy_pct": df["buyer_pct"].iloc[i],
                    "rsi": df["rsi"].iloc[i],
                }])
                p_xgb = float(ensemble["xgb_model"].predict_proba(feat_row)[0][1])
                p_lgb = float(ensemble["lgb_model"].predict_proba(feat_row)[0][1])
                win_prob = (p_xgb * 0.55) + (p_lgb * 0.45)
                expected_R = (win_prob * 1.5) - ((1.0 - win_prob) * 1.0) - (TOTAL_COST_PER_TRADE_USD / (cand_risk_dist * LOT_SIZE * CONTRACT_SIZE))

                asian_shield = df["is_asian"].iloc[i] == 1

                if expected_R >= 0.15 and not asian_shield:
                    in_pos = True
                    side = action
                    strategy_used = strat
                    risk_dist_usd = cand_risk_dist
                    entry_idx = i
                    be_active = False
                    initial_risk_usd = cand_risk_dist * LOT_SIZE * CONTRACT_SIZE

                    next_open = df["close"].iloc[i]
                    if action == "BUY":
                        entry_p = next_open + (BENCHMARK_SPREAD_USD / 2) + BENCHMARK_SLIPPAGE_USD
                        sl_p = entry_p - risk_dist_usd
                        tp1_p = entry_p + (risk_dist_usd * 1.5)
                        be_trig_p = entry_p + max(risk_dist_usd * 0.75, 3.50)
                    else:
                        entry_p = next_open - (BENCHMARK_SPREAD_USD / 2) - BENCHMARK_SLIPPAGE_USD
                        sl_p = entry_p + risk_dist_usd
                        tp1_p = entry_p - (risk_dist_usd * 1.5)
                        be_trig_p = entry_p - max(risk_dist_usd * 0.75, 3.50)

    df_res = pd.DataFrame(trade_ledger)
    df_res.to_csv(ROOT / "reports/baseline_trade_ledger_capitalized_gated.csv", index=False)
    
    total = len(df_res)
    wins = df_res[df_res["outcome"] == "WIN"]
    losses = df_res[df_res["outcome"] == "LOSS"]
    bes = df_res[df_res["outcome"] == "BREAK_EVEN"]

    gp = wins["net_pnl_usd"].sum()
    gl = abs(losses["net_pnl_usd"].sum()) + abs(bes["net_pnl_usd"].clip(upper=0).sum())
    pf = round(gp / gl, 2) if gl > 0 else 0.0
    net_pnl = df_res["net_pnl_usd"].sum()
    net_R = df_res["r_multiple"].sum()
    exp_R = round(net_R / total, 3) if total > 0 else 0.0

    cum_pnl = df_res["net_pnl_usd"].cumsum()
    max_dd = round((cum_pnl.cummax() - cum_pnl).max(), 2)

    oos = df_res[df_res["is_test_set"]]
    oos_total = len(oos)
    oos_wins = oos[oos["outcome"] == "WIN"]
    oos_losses = oos[oos["outcome"] == "LOSS"]
    oos_bes = oos[oos["outcome"] == "BREAK_EVEN"]
    oos_gp = oos_wins["net_pnl_usd"].sum()
    oos_gl = abs(oos_losses["net_pnl_usd"].sum()) + abs(oos_bes["net_pnl_usd"].clip(upper=0).sum())
    oos_pf = round(oos_gp / oos_gl, 2) if oos_gl > 0 else 0.0
    oos_pnl = oos["net_pnl_usd"].sum()
    oos_exp_R = round(oos["r_multiple"].mean(), 3) if oos_total > 0 else 0.0
    oos_cum_pnl = oos["net_pnl_usd"].cumsum()
    oos_dd = round((oos_cum_pnl.cummax() - oos_cum_pnl).max(), 2) if oos_total > 0 else 0.0

    res = {
        "overall": {
            "total_trades": total,
            "wins": len(wins),
            "losses": len(losses),
            "break_evens": len(bes),
            "win_rate_pct": round(len(wins) / total * 100, 2),
            "loss_rate_pct": round(len(losses) / total * 100, 2),
            "break_even_pct": round(len(bes) / total * 100, 2),
            "profit_factor": pf,
            "net_pnl_usd": round(net_pnl, 2),
            "net_expectancy_R": exp_R,
            "max_drawdown_usd": max_dd,
        },
        "unseen_oos": {
            "total_trades": oos_total,
            "wins": len(oos_wins),
            "losses": len(oos_losses),
            "break_evens": len(oos_bes),
            "win_rate_pct": round(len(oos_wins) / oos_total * 100, 2) if oos_total > 0 else 0.0,
            "loss_rate_pct": round(len(oos_losses) / oos_total * 100, 2) if oos_total > 0 else 0.0,
            "break_even_pct": round(len(oos_bes) / oos_total * 100, 2) if oos_total > 0 else 0.0,
            "profit_factor": oos_pf,
            "net_pnl_usd": round(oos_pnl, 2),
            "net_expectancy_R": oos_exp_R,
            "max_drawdown_usd": oos_dd,
        }
    }

    out_file = ROOT / "reports/baseline_capitalized_gated_report.json"
    with open(out_file, "w") as f:
        json.dump(res, f, indent=2)

    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    run()
