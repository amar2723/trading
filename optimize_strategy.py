from __future__ import annotations

import argparse
import json

from src.optimization.strategy_optimizer import OptimizationWeights, StrategyOptimizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize strategy parameters for risk-adjusted performance.")
    parser.add_argument("--data", required=True, help="Labeled/prediction-ready CSV")
    parser.add_argument("--model", help="Model bundle path")
    parser.add_argument("--output", default="reports/optimization")
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--profit-factor-weight", type=float, default=1.5)
    parser.add_argument("--expectancy-weight", type=float, default=1.0)
    parser.add_argument("--sharpe-weight", type=float, default=1.2)
    parser.add_argument("--drawdown-weight", type=float, default=1.4)
    parser.add_argument("--ruin-weight", type=float, default=2.0)
    args = parser.parse_args()

    weights = OptimizationWeights(
        profit_factor=args.profit_factor_weight,
        expectancy=args.expectancy_weight,
        sharpe_ratio=args.sharpe_weight,
        maximum_drawdown=args.drawdown_weight,
        risk_of_ruin=args.ruin_weight,
    )
    result = StrategyOptimizer(args.data, args.model, args.output, weights).optimize(args.trials)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
