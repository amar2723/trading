# Module 4: Automatic Trade Labeling Engine

This module converts Smart Money Concepts pattern data into a labeled training dataset.

## Files

- `entry_generator.py`: Creates BUY, SELL, and HOLD entries.
- `sl_tp_generator.py`: Generates stop loss, TP1, TP2, risk, reward, and RR.
- `trade_simulator.py`: Simulates forward candles over 20, 50, and 100 candle windows.
- `label_generator.py`: Builds labels, train/validation/test splits, and reports.
- `generate_labels.py`: Root-level CLI script.

## Usage

Run from the `trading_ai` directory:

```bash
python generate_labels.py --input data/patterns/XAUUSD_M5_patterns.csv --output data/labeled/XAUUSD_M5_labeled.csv
```

This writes:

```text
data/labeled/XAUUSD_M5_labeled.csv
data/labeled/train.csv
data/labeled/validation.csv
data/labeled/test.csv
```

## Labels

Directional target:

- `BUY = 1`
- `SELL = -1`
- `HOLD = 0`

Binary target:

- `profitable_trade = 1` for `WIN` or `PARTIAL`
- `losing_trade = 0/1`, where `1` means `LOSS`

## Trade Results

The simulator looks ahead up to 100 candles and also stores results for 20 and 50 candle windows:

- `WIN`: TP2 hit, or TP1 hit when TP2 is unavailable
- `LOSS`: SL hit before TP1
- `PARTIAL`: TP1 hit before SL, but TP2 not reached
- `NO_EXIT`: neither TP nor SL hit in the window

## Report Metrics

The report includes:

- total trades
- win rate
- loss rate
- average RR
- average profit
- average loss

## Tests

```bash
python -m pytest tests/test_labeling.py --basetemp .pytest_tmp
```
