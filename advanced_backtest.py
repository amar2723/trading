from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.advanced_smc import AdvancedSMCEngine
from src.live_data import fetch_live_candles
from trade_evaluator import build_metrics, simulate_trade


PERIODS = [1000, 5000, 10000]


def backtest(candles: pd.DataFrame, symbol: str, min_confidence: float = 70.0, lookahead: int = 20) -> tuple[pd.DataFrame, dict]:
    engine = AdvancedSMCEngine(symbol, min_confidence)
    enriched = engine.enrich(candles)
    trades = []
    for idx in range(100, len(enriched) - lookahead):
        signal = engine._signal_from_row(enriched.iloc[idx])
        if signal.signal not in {"BUY", "SELL"}:
            continue
        future = enriched.iloc[idx + 1:idx + 1 + lookahead]
        outcome, exit_price, exit_time, rr = simulate_trade(signal.signal, future, signal.stop_loss, signal.tp1, signal.tp2)
        trades.append({**signal.to_dict(), "outcome": outcome, "exit_price": exit_price, "exit_time": exit_time, "rr": rr})
    df = pd.DataFrame(trades)
    metrics = build_metrics(df.rename(columns={"risk_reward": "risk_reward_ratio"})) if not df.empty else build_metrics(pd.DataFrame())
    rr = pd.to_numeric(df["rr"], errors="coerce").fillna(0) if not df.empty else pd.Series(dtype=float)
    metrics["sharpe_ratio"] = float(rr.mean() / rr.std()) if len(rr) > 1 and rr.std() else 0.0
    return df, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest advanced SMC engine.")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--timeframe", default="M5")
    parser.add_argument("--output-dir", default="reports/advanced_smc")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for bars in PERIODS:
        candles = fetch_live_candles(args.symbol, args.timeframe, bars)
        trades, metrics = backtest(candles, args.symbol)
        trades.to_csv(output / f"advanced_trades_{bars}.csv", index=False)
        rows.append({"candles": bars, **metrics})
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "advanced_backtest_summary.csv", index=False)
    (output / "advanced_backtest_report.html").write_text(summary.to_html(index=False), encoding="utf-8")
    print(json.dumps(rows, indent=2, default=str))


if __name__ == "__main__":
    main()
