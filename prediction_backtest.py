from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.live_data import fetch_live_candles
from src.prediction import PredictionPipeline
from trade_evaluator import build_metrics, simulate_trade


PERIODS = [1000, 5000, 10000]


def backtest_predictions(candles: pd.DataFrame, min_confidence: float = 70.0, lookahead: int = 20) -> tuple[pd.DataFrame, dict]:
    pipeline = PredictionPipeline(min_confidence)
    enriched = pipeline.prepare(candles)
    trades = []
    for idx in range(100, len(enriched) - lookahead):
        prediction = pipeline.predict_row(enriched.iloc[idx])
        if prediction["signal"] not in {"BUY", "SELL"}:
            continue
        future = enriched.iloc[idx + 1 : idx + 1 + lookahead]
        outcome, exit_price, exit_time, rr = simulate_trade(
            prediction["signal"],
            future,
            prediction["stop_loss"],
            prediction["tp1"],
            prediction["tp2"],
        )
        trades.append({**prediction, "outcome": outcome, "exit_price": exit_price, "exit_time": exit_time, "rr": rr})
    trade_df = pd.DataFrame(trades)
    metrics = build_metrics(trade_df) if not trade_df.empty else build_metrics(pd.DataFrame())
    metrics["prediction_accuracy"] = metrics["win_rate"]
    rr = pd.to_numeric(trade_df["rr"], errors="coerce").fillna(0) if not trade_df.empty else pd.Series(dtype=float)
    metrics["sharpe_ratio"] = float(rr.mean() / rr.std()) if len(rr) > 1 and rr.std() else 0.0
    return trade_df, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest prediction pipeline.")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--timeframe", default="M5")
    parser.add_argument("--output-dir", default="reports/prediction")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for bars in PERIODS:
        candles = fetch_live_candles(args.symbol, args.timeframe, bars)
        trades, metrics = backtest_predictions(candles)
        trades.to_csv(output / f"prediction_trades_{bars}.csv", index=False)
        rows.append({"candles": bars, **metrics})
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "prediction_backtest_summary.csv", index=False)
    (output / "prediction_backtest_report.html").write_text(summary.to_html(index=False), encoding="utf-8")
    print(json.dumps(rows, indent=2, default=str))


if __name__ == "__main__":
    main()
