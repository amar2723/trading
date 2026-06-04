# Module 2: Feature Engineering

This module converts clean OHLCV candles into technical, candle, volatility, swing, trend, and market-structure features.

## Files

- `indicators.py`: EMA20, EMA50, EMA200, RSI(14), ATR(14), MACD, VWAP, Bollinger Bands, and volume moving average.
- `candle_features.py`: Candle body, wick, engulfing, inside-bar, outside-bar, and volatility helpers.
- `market_structure.py`: Swing/pivot detection, trend state, and HH/HL/LH/LL structure state.
- `feature_pipeline.py`: End-to-end `build_features(df)` pipeline with data-quality handling.
- `generate_features.py`: Root-level CLI script.

## Usage

Run from the `trading_ai` directory:

```bash
python generate_features.py --input data/raw/XAUUSD_M5.csv --output data/processed/XAUUSD_M5_features.csv
```

## Python API

```python
import pandas as pd
from src.feature_engineering.feature_pipeline import build_features

raw = pd.read_csv("data/raw/XAUUSD_M5.csv")
features = build_features(raw)
```

## Output

The output includes the original OHLCV columns plus:

- EMA20, EMA50, EMA200
- RSI, ATR, MACD, VWAP, Bollinger Bands
- volume moving average
- body size, body percentage, upper wick, lower wick, wick ratio
- bullish/bearish candle flags
- engulfing, inside bar, outside bar
- rolling standard deviation, range size, volatility ratio
- swing high/low and pivot high/low coordinates
- trend direction, trend strength, EMA alignment
- higher high, higher low, lower high, lower low, structure state

## Data Quality

The feature pipeline:

- normalizes OHLCV columns through Module 1 cleaning
- repairs missing candles using the inferred timestamp frequency
- caps extreme OHLCV outliers with a robust median absolute deviation rule
- replaces infinite values
- forward/back-fills numeric feature gaps

## Tests

```bash
python -m pytest tests/test_feature_engineering.py --basetemp .pytest_tmp
```
