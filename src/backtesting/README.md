# Module 6: Backtesting Engine

This module simulates trade execution from historical data, pattern signals, and optional model predictions.

## Files

- `backtester.py`: Trade execution engine with BUY/SELL/HOLD, TP1/TP2, costs, delay, and partial exits.
- `portfolio.py`: Balance, equity, closed PnL, trade log, and curve tracking.
- `risk_manager.py`: Position sizing and risk controls.
- `performance.py`: Performance, trade-analysis, Monte Carlo, and walk-forward helpers.
- `equity_curve.py`: Equity/balance/drawdown curves, returns, and Plotly reports.
- `backtest_pipeline.py`: End-to-end pipeline.
- `run_backtest.py`: Root-level CLI.

## Usage

Run from the `trading_ai` directory:

```bash
python run_backtest.py --model models/xgboost_model.pkl --data data/labeled/XAUUSD_M5_labeled.csv
```

Without a model, the engine uses `predicted_signal` or `entry_type` from the data:

```bash
python run_backtest.py --data data/labeled/XAUUSD_M5_labeled.csv
```

## Outputs

```text
reports/backtest/trade_log.csv
reports/backtest/equity_curve.csv
reports/backtest/metrics.json
reports/backtest/monthly_returns.csv
reports/backtest/yearly_returns.csv
reports/backtest/equity_curve.html
reports/backtest/drawdown_report.html
reports/backtest/trade_analysis.html
reports/backtest/performance_dashboard.html
reports/backtest/performance_report.html
```

## Risk Controls

The risk manager supports:

- fixed lot
- fixed dollar risk
- percent risk
- daily loss limit
- maximum consecutive losses
- maximum open positions
- maximum drawdown pause

## Market Conditions

Configurable execution assumptions:

- spread
- commission
- slippage
- execution delay
- point size
- TP1 partial close fraction
- move stop to breakeven after TP1

## Tests

```bash
python -m pytest tests/test_backtesting.py --basetemp .pytest_tmp
```
