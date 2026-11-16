from __future__ import annotations

"""
Demo Money & Algorithm Comparison Tester for XAUUSD Trading Bot.

Compares:
1. Advanced SMC Quant Engine (70% & 60% confidence thresholds)
2. ICT Silver Bullet (YouTube Killzone Strategy)
3. London Open Asian Judas Swing Strategy
4. Standard Two-Candle Pattern Detector

Simulates realistic execution on a $10,000 demo account with:
- $100 risk per trade (1% risk)
- Spread: 25 points (0.25 on Gold)
- Commission & slippage modeled
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from src.advanced_smc import AdvancedSMCEngine
from src.strategies.viral_algos import ViralSMCEnsembleEngine
from src.strategies.institutional_confluence_engine import InstitutionalConfluenceEngine
from trade_evaluator import simulate_trade


def run_demo_test(csv_path: str = "data/raw/XAUUSD_M5_realistic.csv", initial_balance: float = 10000.0, risk_pct: float = 1.0):
    risk_per_trade = initial_balance * (risk_pct / 100.0)
    df = pd.read_csv(csv_path)
    print(f"\n=======================================================")
    print(f"  XAUUSD BOT DEMO MONEY TESTING REPORT ($10,000 ACCOUNT)")
    print(f"=======================================================")
    print(f"Data file: {csv_path} | Total Bars: {len(df)}")
    print(f"Risk Per Trade: ${risk_per_trade:.2f} ({risk_pct}%)\n")

    # 1. Institutional Confluence Engine (Regime + Order Flow + SMC)
    print("--- 1. Institutional Confluence Engine (Regime + Buyer/Seller Flow + SMC) ---")
    ice_engine = InstitutionalConfluenceEngine("XAUUSD", smc_confidence=70.0, min_dom_pct=52.0)
    ice_data = ice_engine.enrich(df)
    ice_trades = []
    last_ice_idx = -20
    for idx in range(30, len(ice_data) - 20):
        if idx - last_ice_idx < 8:
            continue
        row = ice_data.iloc[idx]
        sig = ice_engine.evaluate_bar(row)
        if sig.signal in ["BUY", "SELL"]:
            future = ice_data.iloc[idx + 1 : idx + 21]
            outcome, exit_p, exit_t, rr = simulate_trade(sig.signal, future, sig.sl, sig.tp1, sig.tp2)
            ice_trades.append({"outcome": outcome, "rr": rr, "regime": sig.regime})
            last_ice_idx = idx

    ice_df = pd.DataFrame(ice_trades)
    if not ice_df.empty:
        wins = len(ice_df[ice_df["outcome"].isin(["TP1", "TP2"])])
        losses = len(ice_df[ice_df["outcome"] == "SL"])
        no_hit = len(ice_df[ice_df["outcome"] == "NO_HIT"])
        resolved = wins + losses
        wr = (wins / resolved * 100) if resolved else 0
        net_rr = ice_df["rr"].sum()
        pnl = net_rr * risk_per_trade
        print(f"Trades: {len(ice_df)} | Wins: {wins} | Losses: {losses} | No Hit: {no_hit}")
        print(f"Win Rate (resolved): {wr:.1f}%")
        print(f"Net Profit: ${pnl:.2f} ({pnl/initial_balance*100:+.2f}%)\n")

    # 2. Advanced SMC Quant Engine (70% Confidence)
    print("--- 2. Advanced SMC Quant Engine (Confidence >= 70%) ---")
    engine70 = AdvancedSMCEngine("XAUUSD", min_confidence=70.0)
    enriched = engine70.enrich(df)
    trades70 = []
    for idx in range(30, len(enriched) - 20):
        sig = engine70._signal_from_row(enriched.iloc[idx])
        if sig.signal in ["BUY", "SELL"]:
            future = enriched.iloc[idx + 1 : idx + 21]
            outcome, exit_p, exit_t, rr = simulate_trade(sig.signal, future, sig.stop_loss, sig.tp1, sig.tp2)
            trades70.append({"outcome": outcome, "rr": rr})
    
    tdf70 = pd.DataFrame(trades70)
    if not tdf70.empty:
        wins = len(tdf70[tdf70["outcome"].isin(["TP1", "TP2"])])
        losses = len(tdf70[tdf70["outcome"] == "SL"])
        no_hit = len(tdf70[tdf70["outcome"] == "NO_HIT"])
        resolved = wins + losses
        wr = (wins / resolved * 100) if resolved else 0
        net_rr = tdf70["rr"].sum()
        pnl = net_rr * risk_per_trade
        print(f"Trades: {len(tdf70)} | Wins: {wins} | Losses: {losses} | No Hit: {no_hit}")
        print(f"Win Rate (resolved): {wr:.1f}%")
        print(f"Net Profit: ${pnl:.2f} ({pnl/initial_balance*100:+.2f}%)\n")

    # 3. Advanced SMC Quant Engine (60% Confidence)
    print("--- 3. Advanced SMC Quant Engine (Confidence >= 60%) ---")
    engine60 = AdvancedSMCEngine("XAUUSD", min_confidence=60.0)
    trades60 = []
    for idx in range(30, len(enriched) - 20):
        sig = engine60._signal_from_row(enriched.iloc[idx])
        if sig.signal in ["BUY", "SELL"]:
            future = enriched.iloc[idx + 1 : idx + 21]
            outcome, exit_p, exit_t, rr = simulate_trade(sig.signal, future, sig.stop_loss, sig.tp1, sig.tp2)
            trades60.append({"outcome": outcome, "rr": rr})
    
    tdf60 = pd.DataFrame(trades60)
    if not tdf60.empty:
        wins = len(tdf60[tdf60["outcome"].isin(["TP1", "TP2"])])
        losses = len(tdf60[tdf60["outcome"] == "SL"])
        no_hit = len(tdf60[tdf60["outcome"] == "NO_HIT"])
        resolved = wins + losses
        wr = (wins / resolved * 100) if resolved else 0
        net_rr = tdf60["rr"].sum()
        pnl = net_rr * risk_per_trade
        print(f"Trades: {len(tdf60)} | Wins: {wins} | Losses: {losses} | No Hit: {no_hit}")
        print(f"Win Rate (resolved): {wr:.1f}%")
        print(f"Net Profit: ${pnl:.2f} ({pnl/initial_balance*100:+.2f}%)\n")

    # 3. Viral YouTube Strategies (Silver Bullet & Asian Judas Swing)
    print("--- 3. Viral Social Media Strategies (Raw Silver Bullet & Judas Swing) ---")
    viral_engine = ViralSMCEnsembleEngine("XAUUSD")
    v_data = viral_engine.analyze(df)
    v_trades = []
    for idx in range(30, len(v_data) - 20):
        row = v_data.iloc[idx]
        sig = viral_engine.get_signal_from_row(row)
        if sig.signal in ["BUY", "SELL"]:
            future = v_data.iloc[idx + 1 : idx + 21]
            outcome, exit_p, exit_t, rr = simulate_trade(sig.signal, future, sig.sl, sig.tp1, sig.tp2)
            v_trades.append({"strategy": sig.strategy_name, "outcome": outcome, "rr": rr})
    
    v_df = pd.DataFrame(v_trades)
    if not v_df.empty:
        wins = len(v_df[v_df["outcome"].isin(["TP1", "TP2"])])
        losses = len(v_df[v_df["outcome"] == "SL"])
        resolved = wins + losses
        wr = (wins / resolved * 100) if resolved else 0
        net_rr = v_df["rr"].sum()
        pnl = net_rr * risk_per_trade
        print(f"Trades: {len(v_df)} | Wins: {wins} | Losses: {losses}")
        print(f"Win Rate (resolved): {wr:.1f}%")
        print(f"Net Profit: ${pnl:.2f} ({pnl/initial_balance*100:+.2f}%)")
        print("Note: Raw viral strategies without MSS confirmation suffer severe stopouts during continuous Gold trends.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run demo money simulation.")
    parser.add_argument("--csv", default="data/raw/XAUUSD_M5_realistic.csv")
    parser.add_argument("--balance", type=float, default=10000.0)
    parser.add_argument("--risk", type=float, default=1.0)
    args = parser.parse_args()
    run_demo_test(args.csv, args.balance, args.risk)
