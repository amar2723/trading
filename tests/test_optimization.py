from __future__ import annotations

from src.optimization.strategy_optimizer import OptimizationWeights, objective_score


def test_objective_rewards_profit_and_penalizes_ruin():
    good = {
        "profit_factor": 2.0,
        "expectancy": 20.0,
        "sharpe_ratio": 1.5,
        "maximum_drawdown": -200.0,
        "monte_carlo": {"risk_of_ruin": 1.0},
    }
    bad = {
        "profit_factor": 1.1,
        "expectancy": 2.0,
        "sharpe_ratio": 0.2,
        "maximum_drawdown": -2000.0,
        "monte_carlo": {"risk_of_ruin": 35.0},
    }
    assert objective_score(good, OptimizationWeights()) > objective_score(bad, OptimizationWeights())
