from __future__ import annotations

import argparse
import json

import pandas as pd

from src.feature_engineering.indicators import add_features
from src.pattern_detection.concepts import add_market_concepts
from src.signals import generate_trade_plan


def max_drawdown(equity: list[float]) -> float:
    peak = equity[0] if equity else 0.0
    max_dd = 0.0
    for value in equity:
        peak = max(peak, value)
        max_dd = min(max_dd, value - peak)
    return float(max_dd)


def run_backtest(
    csv_path: str,
    model_path: str | None = None,
    spread_points: float = 25.0,
    slippage_points: float = 5.0,
    commission: float = 7.0,
    point: float = 0.01,
    lot_size: float = 1.0,
) -> dict:
    df = pd.read_csv(csv_path, parse_dates=["time"])
    df = add_market_concepts(add_features(df)).dropna().reset_index(drop=True)
    trades = []
    equity = [0.0]
    costs = (spread_points + slippage_points) * point * lot_size + commission

    for i in range(len(df) - 1):
        plan = generate_trade_plan(df.iloc[i], model_path)
        if plan.signal == "HOLD" or not plan.entry or not plan.sl or not plan.tp1:
            continue
        future = df.iloc[i + 1 : i + 61]
        pnl = 0.0
        exit_price = plan.entry
        outcome = "TIMEOUT"
        for _, candle in future.iterrows():
            if plan.signal == "BUY":
                if candle["low"] <= plan.sl:
                    exit_price = plan.sl
                    pnl = exit_price - plan.entry
                    outcome = "SL"
                    break
                if candle["high"] >= plan.tp1:
                    exit_price = plan.tp1
                    pnl = exit_price - plan.entry
                    outcome = "TP"
                    break
            else:
                if candle["high"] >= plan.sl:
                    exit_price = plan.sl
                    pnl = plan.entry - exit_price
                    outcome = "SL"
                    break
                if candle["low"] <= plan.tp1:
                    exit_price = plan.tp1
                    pnl = plan.entry - exit_price
                    outcome = "TP"
                    break
        net = pnl * lot_size - costs
        equity.append(equity[-1] + net)
        trades.append(
            {
                "time": str(df.iloc[i]["time"]),
                "signal": plan.signal,
                "entry": plan.entry,
                "sl": plan.sl,
                "tp1": plan.tp1,
                "tp2": plan.tp2,
                "exit": exit_price,
                "outcome": outcome,
                "net_pnl": net,
                "confidence": plan.confidence,
            }
        )

    wins = [t for t in trades if t["net_pnl"] > 0]
    losses = [t for t in trades if t["net_pnl"] < 0]
    gross_profit = sum(t["net_pnl"] for t in wins)
    gross_loss = abs(sum(t["net_pnl"] for t in losses))
    return {
        "trades": trades,
        "metrics": {
            "total_trades": len(trades),
            "win_rate": len(wins) / len(trades) * 100 if trades else 0.0,
            "profit_factor": gross_profit / gross_loss if gross_loss else None,
            "max_drawdown": max_drawdown(equity),
            "net_profit": equity[-1] if equity else 0.0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--model-path")
    args = parser.parse_args()
    print(json.dumps(run_backtest(args.csv, args.model_path), indent=2))


if __name__ == "__main__":
    main()
