# XAUUSD AI Trading Assistant

Python trading assistant for XAUUSD using MetaTrader5 OHLCV data, technical features, smart-money style concept detection, XGBoost confidence scoring, backtesting, realtime paper trading, and a Streamlit dashboard.

This is research software, not financial advice. Run it in paper mode first and validate every assumption against your broker's XAUUSD symbol, spread, commission, and execution model.

## Project Structure

```text
trading_ai/
  data/
    raw/
    processed/
    labels/
  models/
  src/
    data_ingestion/
    feature_engineering/
    pattern_detection/
    labeling/
    training/
  backtesting/
  dashboard/
  realtime/
  requirements.txt
  main.py
```

## Install

```bash
cd trading_ai
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

MetaTrader5 must be installed and logged in on the same Windows machine.

## Fetch Data

```bash
python main.py fetch --symbol XAUUSD --timeframe M1 --bars 5000 --output data/raw/XAUUSD_M1.csv
python main.py fetch --symbol XAUUSD --timeframe M5 --bars 5000 --output data/raw/XAUUSD_M5.csv
python main.py fetch --symbol XAUUSD --timeframe M15 --bars 5000 --output data/raw/XAUUSD_M15.csv
```

## Generate Latest Signal

```bash
python main.py signal --csv data/raw/XAUUSD_M5.csv
```

With trained ML confidence:

```bash
python main.py signal --csv data/raw/XAUUSD_M5.csv --model-path models/xauusd_xgboost.joblib
```

Output includes:

```text
Signal: BUY/SELL/HOLD
Confidence %
Entry
SL
TP1
TP2
Risk Reward Ratio
```

## Train XGBoost

```bash
python -m src.training.train_xgboost --csv data/raw/XAUUSD_M5.csv --model-dir models
```

Training uses:

- XGBoost classifier
- TimeSeriesSplit
- TP-before-SL target labels
- feature importance CSV
- SHAP values when SHAP runs successfully in the environment

Artifacts:

- `models/xauusd_xgboost.joblib`
- `models/feature_importance.csv`
- `models/training_metrics.json`
- `models/shap_values.joblib` or `models/shap_warning.txt`

## Backtest

```bash
python backtesting/backtest.py --csv data/raw/XAUUSD_M5.csv --model-path models/xauusd_xgboost.joblib
```

The backtest includes spread, slippage, and commission. Metrics include win rate, profit factor, max drawdown, and net profit.

## Paper Trading

```bash
python -m realtime.paper_trader --symbol XAUUSD --timeframe M5 --model-path models/xauusd_xgboost.joblib
```

On every new candle it fetches MT5 data, recalculates features and concepts, generates a signal, and records paper trades in memory.

## Dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard shows live/latest signals from CSV data, candlestick chart, liquidity sweeps, MSS, BOS, FVG markers, concept table, trade history from backtests, and backtest metrics.

## Trading Logic

BUY requires:

- Bullish liquidity sweep
- Bullish MSS
- Bullish BOS
- Bullish displacement candle

SELL requires:

- Bearish liquidity sweep
- Bearish MSS
- Bearish BOS
- Bearish displacement candle

Otherwise the assistant returns HOLD.

Risk management:

- BUY stop loss is below the sweep low when available
- SELL stop loss is above the sweep high when available
- TP1 targets nearest liquidity/support-resistance zone
- TP2 targets the next liquidity zone or ATR extension fallback

## Notes

Concept detection is intentionally deterministic and transparent. You can tune sensitivity in `src/config.py`, especially `swing_lookback`, `sweep_lookback`, `displacement_atr_mult`, and `fvg_min_atr_mult`.
