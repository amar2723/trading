# Phase 1: Real-Time Pattern Proof Setup

Phase 1 ignores AI, CNN, XGBoost, and model training. The only goal is proving Python can connect to MetaTrader 5 and fetch live XAUUSD candles.

## 1. Verify Python

```bash
python --version
```

## 2. Install MetaTrader 5

Install MetaTrader 5, open a demo account, and make sure your broker offers `XAUUSD`.

Keep MT5 open and logged in before running the Python scripts.

## 3. Create Virtual Environment

From the `trading_ai` folder:

```bash
python -m venv venv
venv\Scripts\activate
pip install pandas numpy MetaTrader5 streamlit plotly python-telegram-bot
pip freeze > requirements.txt
```

## 4. Test MT5 Connection

```bash
python src/mt5_connector.py
```

Expected:

```text
Connected
```

## 5. Fetch Live Gold Data

```bash
python src/live_data.py --symbol XAUUSD --timeframe M5 --bars 100
```

Expected columns:

```text
timestamp
open
high
low
close
volume
```

If this prints live candles, Python is officially connected to the market.
