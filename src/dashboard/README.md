# Module 8: Dashboard and Monitoring

This module provides a Streamlit dashboard for the full XAUUSD trading AI pipeline.

## Files

- `data_loader.py`: Safe CSV/JSON/report loading helpers.
- `charts.py`: Plotly charts for candles, patterns, equity, drawdown, trades, and feature importance.
- `components.py`: Streamlit UI sections and metric cards.
- `dashboard_app.py`: Main Streamlit dashboard.
- `dashboard/app.py`: Root dashboard entry point.

## Usage

Run from the `trading_ai` directory:

```bash
streamlit run dashboard/app.py
```

## Dashboard Tabs

- `Market`: Candlestick chart with liquidity sweeps, MSS, BOS, and order block markers.
- `Live Signals`: Latest and historical signal log from `logs/signals.csv`.
- `Backtest`: Metrics, equity curve, drawdown, trade PnL, and trade log.
- `Model Reports`: Training metrics and top feature importance.

## Expected Inputs

The dashboard reads existing pipeline outputs:

```text
data/raw/*.csv
data/processed/*.csv
data/patterns/*.csv
logs/signals.csv
reports/metrics.json
reports/feature_importance.csv
reports/backtest/*
```

Missing files are handled gracefully with empty tables/charts.

## Tests

```bash
python -m pytest tests/test_dashboard.py --basetemp .pytest_tmp
```
