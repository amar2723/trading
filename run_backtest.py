from __future__ import annotations

import argparse
import json

from src.backtesting.backtest_pipeline import run_backtest_pipeline
from src.backtesting.backtester import ExecutionConfig
from src.backtesting.risk_manager import RiskConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Module 6 backtesting engine.")
    parser.add_argument("--model", help="Trained model bundle path")
    parser.add_argument("--data", required=True, help="Historical/labeled/prediction-ready CSV")
    parser.add_argument("--output", default="reports/backtest")
    parser.add_argument("--spread-points", type=float, default=25.0)
    parser.add_argument("--slippage-points", type=float, default=5.0)
    parser.add_argument("--commission", type=float, default=7.0)
    parser.add_argument("--execution-delay", type=int, default=1)
    parser.add_argument("--initial-balance", type=float, default=10_000.0)
    parser.add_argument("--sizing-mode", choices=["fixed_lot", "fixed_dollar_risk", "percent_risk"], default="percent_risk")
    parser.add_argument("--percent-risk", type=float, default=1.0)
    args = parser.parse_args()

    execution = ExecutionConfig(args.spread_points, args.commission, args.slippage_points, args.execution_delay)
    risk = RiskConfig(initial_balance=args.initial_balance, sizing_mode=args.sizing_mode, percent_risk=args.percent_risk)
    result = run_backtest_pipeline(args.data, args.model, args.output, execution, risk)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
