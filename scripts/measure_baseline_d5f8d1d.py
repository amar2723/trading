"""
Script to measure the exact, rigorous baseline performance of the d5f8d1d codebase
on the actual historical dataset:
- data/raw/XAUUSD_M5_60d_max.csv (13,854 bars)
- data/raw/XAUUSD_1H_2years.csv (13,755 bars)
Uses the exact d5f8d1d logic with realistic execution (spread + slippage).
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[1]

def run_baseline_evaluation():
    m5_path = ROOT / "data/raw/XAUUSD_M5_60d_max.csv"
    h1_path = ROOT / "data/raw/XAUUSD_1H_2years.csv"

    df_m5 = pd.read_csv(m5_path)
    df_h1 = pd.read_csv(h1_path)
    df_m5["timestamp"] = pd.to_datetime(df_m5["timestamp"])
    df_h1["timestamp"] = pd.to_datetime(df_h1["timestamp"])

    df_m5 = df_m5.sort_values("timestamp").reset_index(drop=True)
    df_h1 = df_h1.sort_values("timestamp").reset_index(drop=True)

    # 1. Higher Timeframe Alignment (Causal)
    df_h1["h1_ema_50"] = df_h1["close"].ewm(span=50).mean()
    df_h1["h1_ema_200"] = df_h1["close"].ewm(span=200).mean()
    df_h1["htf_trend_bias"] = np.where(df_h1["close"] > df_h1["h1_ema_200"], 1, -1)
    df_h1["htf_trend_bias"] = df_h1["htf_trend_bias"].shift(1)
    df_h1["h1_dist_ema50"] = ((df_h1["close"] - df_h1["h1_ema_50"]) / df_h1["h1_ema_50"] * 100).shift(1)

    df_h1_sub = df_h1[["timestamp", "htf_trend_bias", "h1_dist_ema50"]].dropna()
    df_m5 = pd.merge_asof(df_m5, df_h1_sub, on="timestamp", direction="backward")

    close = df_m5["close"].values
    high = df_m5["high"].values
    low = df_m5["low"].values
    vol = df_m5["volume"].values
    timestamps = df_m5["timestamp"]
    n = len(df_m5)

    # Technical Indicators matching d5f8d1d
    ema_20 = pd.Series(close).ewm(span=20).mean().values
    ema_50 = pd.Series(close).ewm(span=50).mean().values

    # Session hours
    hours = timestamps.dt.hour.values
    is_prime = np.isin(hours, [7, 8, 9, 10, 12, 13, 14, 15, 16])
    is_asian = np.isin(hours, [0, 1, 2, 3, 4, 5, 6])

    # ATR
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(14).mean().values

    # Rolling highs/lows
    roll_h20 = pd.Series(high).rolling(20).max().shift(1).values
    roll_l20 = pd.Series(low).rolling(20).min().shift(1).values

    # RSI
    delta_s = pd.Series(close).diff()
    gain = delta_s.where(delta_s > 0, 0.0).rolling(14).mean()
    loss = (-delta_s.where(delta_s < 0, 0.0)).rolling(14).mean()
    rs = gain / np.maximum(loss, 1e-6)
    rsi = (100 - (100 / (1 + rs))).values

    # Order flow signed proxy
    p_diff = np.diff(close, prepend=close[0])
    v_buy = pd.Series(np.where(p_diff >= 0, vol, 0.0)).rolling(14).sum().values
    v_tot = pd.Series(vol).rolling(14).sum().values
    buyer_pct = (v_buy / np.maximum(v_tot, 1.0)) * 100

    # Load master ensemble if available
    ensemble_path = ROOT / "models/xauusd_master_ensemble.pkl"
    ensemble = None
    if ensemble_path.exists():
        try:
            ensemble = joblib.load(ensemble_path)
        except Exception:
            pass

    # Execution Simulation Constants (Realistic Octa Broker)
    SPREAD = 0.25      # 25 pips ($0.25 on gold)
    SLIPPAGE = 0.05    # 5 pips ($0.05 on gold)
    TOTAL_COST = SPREAD + SLIPPAGE # 30 pips / $0.30

    trades = []
    
    # Chronological Split: 80% Train / 20% Unseen Test
    test_start_idx = int(n * 0.8)

    # We evaluate sequentially. If in position, we hold until exit.
    in_position = False
    pos_side = None
    pos_entry = 0.0
    pos_sl = 0.0
    pos_tp1 = 0.0
    pos_be_trig = 0.0
    pos_be_active = False
    pos_entry_bar = 0
    pos_risk_dist = 0.0
    pos_session = ""
    pos_month = ""

    for i in range(50, n - 1):
        c_time = timestamps.iloc[i]
        c_close = close[i]
        c_high = high[i]
        c_low = low[i]
        c_atr = atr[i] if not np.isnan(atr[i]) else 3.5

        # Check existing position exits first on this bar
        if in_position:
            # Bar resolution
            outcome = None
            exit_price = 0.0
            r_mult = 0.0

            if pos_side == "BUY":
                # Check BE activation
                if c_high >= pos_be_trig:
                    pos_be_active = True

                cur_sl = (pos_entry + TOTAL_COST) if pos_be_active else pos_sl

                # Conservative same-bar tie-breaker: check SL first
                if c_low <= cur_sl:
                    outcome = "BREAK_EVEN" if pos_be_active else "LOSS"
                    exit_price = cur_sl
                    r_mult = 0.0 if pos_be_active else -1.0
                elif c_high >= pos_tp1:
                    outcome = "WIN"
                    exit_price = pos_tp1
                    r_mult = 1.5
                elif i - pos_entry_bar >= 24: # 2-hour timeout
                    exit_price = c_close - TOTAL_COST
                    pnl_raw = exit_price - pos_entry
                    r_mult = pnl_raw / pos_risk_dist
                    outcome = "WIN" if r_mult > 0 else "LOSS"

            elif pos_side == "SELL":
                if c_low <= pos_be_trig:
                    pos_be_active = True

                cur_sl = (pos_entry - TOTAL_COST) if pos_be_active else pos_sl

                if c_high >= cur_sl:
                    outcome = "BREAK_EVEN" if pos_be_active else "LOSS"
                    exit_price = cur_sl
                    r_mult = 0.0 if pos_be_active else -1.0
                elif c_low <= pos_tp1:
                    outcome = "WIN"
                    exit_price = pos_tp1
                    r_mult = 1.5
                elif i - pos_entry_bar >= 24:
                    exit_price = c_close + TOTAL_COST
                    pnl_raw = pos_entry - exit_price
                    r_mult = pnl_raw / pos_risk_dist
                    outcome = "WIN" if r_mult > 0 else "LOSS"

            if outcome is not None:
                pnl_usd = r_mult * pos_risk_dist * 1.0 # 0.01 lot = $1/pt
                trades.append({
                    "entry_bar": pos_entry_bar,
                    "exit_bar": i,
                    "is_test_set": pos_entry_bar >= test_start_idx,
                    "timestamp": str(timestamps.iloc[pos_entry_bar]),
                    "side": pos_side,
                    "entry": pos_entry,
                    "exit": exit_price,
                    "outcome": outcome,
                    "r_mult": r_mult,
                    "pnl_usd": pnl_usd,
                    "session": pos_session,
                    "month": pos_month,
                })
                in_position = False

        # If flat, evaluate d5f8d1d signal generation
        if not in_position:
            # Multi-factor score in d5f8d1d
            long_score = 0
            short_score = 0

            if c_close > ema_20[i]:
                long_score += 2
            else:
                short_score += 2

            if ema_20[i] > ema_50[i]:
                long_score += 2
            else:
                short_score += 2

            if buyer_pct[i] > 50.0:
                long_score += 2
            else:
                short_score += 2

            span_20 = max(roll_h20[i] - roll_l20[i], 0.1)
            sw_low = (low[i] < roll_l20[i]) and (close[i] > roll_l20[i])
            sw_high = (high[i] > roll_h20[i]) and (close[i] < roll_h20[i])

            if sw_low or (c_close <= roll_l20[i] + 2.0 and rsi[i] <= 40.0):
                long_score += 3
            if sw_high or (c_close >= roll_h20[i] - 2.0 and rsi[i] >= 60.0):
                short_score += 3

            if rsi[i] >= 70.0:
                long_score -= 4
            if rsi[i] <= 30.0:
                short_score -= 4

            # d5f8d1d Action determination
            action = "BUY" if long_score >= short_score else "SELL"
            
            # Risk sizing
            risk_dist = round(max(c_atr * 1.25, 3.20), 2)

            # Realistic entry executed on next bar open with cost
            next_open = close[i] # close of signal bar = open of execution
            sess_str = "LONDON" if is_prime[i] and hours[i] <= 11 else ("NY" if is_prime[i] else ("ASIAN" if is_asian[i] else "OTHER"))

            in_position = True
            pos_side = action
            pos_risk_dist = risk_dist
            pos_entry_bar = i
            pos_session = sess_str
            pos_month = c_time.strftime("%Y-%m")

            if action == "BUY":
                pos_entry = round(next_open + TOTAL_COST, 2)
                pos_sl = round(pos_entry - risk_dist, 2)
                pos_tp1 = round(pos_entry + (risk_dist * 1.5), 2)
                pos_be_trig = round(pos_entry + 2.00, 2)
            else:
                pos_entry = round(next_open - TOTAL_COST, 2)
                pos_sl = round(pos_entry + risk_dist, 2)
                pos_tp1 = round(pos_entry - (risk_dist * 1.5), 2)
                pos_be_trig = round(pos_entry - 2.00, 2)

    df_trades = pd.DataFrame(trades)
    
    # Compute metrics for overall and unseen test set
    def calc_metrics(sub_df, name):
        if len(sub_df) == 0:
            return {}
        wins = sub_df[sub_df["outcome"] == "WIN"]
        losses = sub_df[sub_df["outcome"] == "LOSS"]
        bes = sub_df[sub_df["outcome"] == "BREAK_EVEN"]

        total_trades = len(sub_df)
        win_rate = (len(wins) / total_trades) * 100
        loss_rate = (len(losses) / total_trades) * 100
        be_rate = (len(bes) / total_trades) * 100

        gross_profit = wins["pnl_usd"].sum()
        gross_loss = abs(losses["pnl_usd"].sum())
        profit_factor = round(gross_profit / max(1e-6, gross_loss), 2)
        net_pnl = sub_df["pnl_usd"].sum()
        net_R = sub_df["r_mult"].sum()
        exp_R = round(net_R / total_trades, 3)

        # Drawdown in USD and R
        cum_pnl = sub_df["pnl_usd"].cumsum()
        peak = cum_pnl.cummax()
        dd = peak - cum_pnl
        max_dd = round(dd.max(), 2)

        cum_r = sub_df["r_mult"].cumsum()
        r_peak = cum_r.cummax()
        max_dd_r = round((r_peak - cum_r).max(), 2)

        return {
            "name": name,
            "total_trades": total_trades,
            "win_rate_pct": round(win_rate, 2),
            "loss_rate_pct": round(loss_rate, 2),
            "break_even_pct": round(be_rate, 2),
            "profit_factor": profit_factor,
            "net_pnl_usd": round(net_pnl, 2),
            "net_expectancy_R": exp_R,
            "total_R": round(net_R, 2),
            "max_drawdown_usd": max_dd,
            "max_drawdown_R": max_dd_r,
            "avg_win_usd": round(wins["pnl_usd"].mean(), 2) if len(wins) > 0 else 0.0,
            "avg_loss_usd": round(abs(losses["pnl_usd"].mean()), 2) if len(losses) > 0 else 0.0,
        }

    overall_metrics = calc_metrics(df_trades, "Full Dataset (13,854 M5 Bars)")
    unseen_metrics = calc_metrics(df_trades[df_trades["is_test_set"]], "Unseen Test Set (Latest 2,771 M5 Bars)")
    
    # Session breakdown on test set
    test_df = df_trades[df_trades["is_test_set"]]
    session_res = {}
    for sess, grp in test_df.groupby("session"):
        session_res[sess] = calc_metrics(grp, sess)

    # Monthly breakdown
    monthly_res = {}
    for m, grp in df_trades.groupby("month"):
        monthly_res[m] = calc_metrics(grp, m)

    res = {
        "overall": overall_metrics,
        "unseen_test": unseen_metrics,
        "session_breakdown_unseen": session_res,
        "monthly_breakdown": monthly_res
    }

    out_file = ROOT / "reports/baseline_d5f8d1d_report.json"
    with open(out_file, "w") as f:
        json.dump(res, f, indent=2)

    print("=" * 60)
    print("      MEASURED TRUE BASELINE OF d5f8d1d SYSTEM")
    print("=" * 60)
    print(f"Total Completed Trades:      {overall_metrics['total_trades']}")
    print(f"Overall Win Rate:            {overall_metrics['win_rate_pct']}%")
    print(f"Overall Break-Even Rate:     {overall_metrics['break_even_pct']}%")
    print(f"Overall Profit Factor:       {overall_metrics['profit_factor']}")
    print(f"Overall Net Expectancy (R):  {overall_metrics['net_expectancy_R']:+.3f}R")
    print(f"Overall Total Net PnL (USD): ${overall_metrics['net_pnl_usd']:+.2f}")
    print(f"Overall Max Drawdown:        ${overall_metrics['max_drawdown_usd']:.2f} ({overall_metrics['max_drawdown_R']:.1f}R)")
    print("-" * 60)
    print(f"UNSEEN TEST SET (20% OOS):")
    print(f"  Trades:                    {unseen_metrics['total_trades']}")
    print(f"  Win Rate:                  {unseen_metrics['win_rate_pct']}%")
    print(f"  Profit Factor:             {unseen_metrics['profit_factor']}")
    print(f"  Net Expectancy (R):        {unseen_metrics['net_expectancy_R']:+.3f}R")
    print(f"  Net PnL (USD):             ${unseen_metrics['net_pnl_usd']:+.2f}")
    print(f"  Max Drawdown:              ${unseen_metrics['max_drawdown_usd']:.2f}")
    print("=" * 60)

if __name__ == "__main__":
    run_baseline_evaluation()
