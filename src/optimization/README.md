# Strategy Optimization

This module searches for parameters that:

- maximize Profit Factor
- maximize Expectancy
- maximize Sharpe Ratio
- minimize Maximum Drawdown
- minimize Risk of Ruin

## Usage

```bash
python optimize_strategy.py --data data/labeled/XAUUSD_M5_labeled.csv --model models/xgboost_model.pkl --trials 100
```

Without model predictions:

```bash
python optimize_strategy.py --data data/labeled/XAUUSD_M5_labeled.csv --trials 100
```

## Objective

The optimizer uses a weighted score:

```text
score =
  profit_factor_weight * profit_factor
  + expectancy_weight * expectancy
  + sharpe_weight * sharpe_ratio
  - drawdown_weight * abs(maximum_drawdown)
  - ruin_weight * risk_of_ruin
```

Defaults favor survival strongly, so high returns with ugly drawdowns are penalized.

## Optimized Parameters

- spread points
- slippage points
- commission
- execution delay
- TP1 close fraction
- percent risk
- max daily loss
- max consecutive losses
- max drawdown protection

## Outputs

```text
reports/optimization/best_optimization.json
reports/optimization/optimization_trials.csv
reports/optimization/trial_*/metrics.json
```
