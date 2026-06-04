# Module 7: Real-Time Trading Engine

This module connects to MetaTrader5, watches for newly closed candles, builds features and Smart Money Concepts patterns, loads the trained model, generates BUY/SELL/HOLD signals, applies risk filters, sends alerts, and logs signals.

## Files

- `mt5_connector.py`: MT5 lifecycle, login, connection checks, rates, and ticks.
- `live_data_feed.py`: Fetches M1, M5, M15, and H1 candles and detects new closed candles.
- `signal_engine.py`: Builds realtime features, detects patterns, loads model probabilities, and creates signal levels.
- `trade_manager.py`: Risk filters, duplicate-signal prevention, and `logs/signals.csv` logging.
- `alert_manager.py`: Console, log, desktop, Telegram, and Discord alerts.
- `realtime_pipeline.py`: End-to-end realtime orchestration.
- `run_live.py`: Root-level CLI.
- `config/realtime_config.json`: Runtime configuration.

## Usage

Run from the `trading_ai` directory:

```bash
python run_live.py
```

Run one polling cycle:

```bash
python run_live.py --once
```

## Signal Rules

BUY requires:

- confidence above threshold
- bullish liquidity sweep
- bullish MSS
- bullish BOS

SELL requires:

- confidence above threshold
- bearish liquidity sweep
- bearish MSS
- bearish BOS

Otherwise the engine outputs HOLD.

## Risk Filters

Signals are rejected when:

- confidence is below `70%` by default
- spread is too high
- ATR is too high
- daily loss limit is hit
- the signal duplicates the last signal on the same candle

## Alert Format

```text
BUY SIGNAL
Symbol: XAUUSD
Entry: xxxx
SL: xxxx
TP1: xxxx
TP2: xxxx
Confidence: xx%
```

## Trade Log

Signals are saved to:

```text
logs/signals.csv
```

## Tests

```bash
python -m pytest tests/test_realtime.py --basetemp .pytest_tmp
```
