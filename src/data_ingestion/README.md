# Module 1: Historical Data Collection

This module collects, loads, validates, and saves historical XAUUSD OHLCV data.

## Files

- `mt5_loader.py`: Fetches historical data from MetaTrader5.
- `csv_loader.py`: Loads CSV data, validates columns, and handles missing values.
- `data_validator.py`: Checks missing rows, duplicates, bad timestamps, nulls, and negative prices.
- `logging_utils.py`: Shared logging setup for this module.
- `collect_data.py`: Command-line entry point in the project root.

## Output Schema

All loaders return a pandas DataFrame with:

```text
timestamp
open
high
low
close
tick_volume
spread
```

## Collect MT5 Data

Run from the `trading_ai` directory:

```bash
python collect_data.py --symbol XAUUSD --timeframe M5 --start 2023-01-01 --end 2025-01-01
```

The output file is saved to:

```text
data/raw/XAUUSD_M5.csv
```

Supported timeframes:

```text
M1
M5
M15
```

## CSV Loading

```python
from src.data_ingestion.csv_loader import load_csv_data

df = load_csv_data("data/raw/XAUUSD_M5.csv")
```

The CSV loader normalizes common column aliases such as `time` to `timestamp` and `volume` to `tick_volume`, drops invalid rows, sorts by timestamp, and validates the result.

## Validation

```python
from src.data_ingestion.data_validator import validate_data

report = validate_data(df)
print(report.to_dict())
```

The validation report includes:

- missing required columns
- missing rows inferred from timestamp frequency
- duplicate timestamp rows
- incorrect timestamps
- null values by column
- negative or zero price rows
- final `is_valid` status

## Tests

Run from the `trading_ai` directory:

```bash
pytest tests/test_data_validator.py tests/test_csv_loader.py
```

## Notes

MetaTrader5 collection requires the MT5 terminal to be installed, running, and logged into the broker account on the same machine.
