# Module 3: Smart Money Concepts Detection Engine

This module detects Smart Money Concepts patterns from a feature-engineered OHLCV DataFrame.

## Files

- `liquidity_sweep.py`: Bullish and bearish liquidity sweep detection.
- `mss.py`: Market Structure Shift detection after sweeps.
- `bos.py`: Break of Structure detection.
- `fvg.py`: Fair Value Gap detection.
- `order_blocks.py`: Displacement candle and order block detection.
- `pattern_pipeline.py`: End-to-end pattern pipeline, CSV saving, and Plotly visualization.
- `detect_patterns.py`: Root-level command-line script.

## Usage

Run from the `trading_ai` directory:

```bash
python detect_patterns.py --input data/processed/XAUUSD_M5_features.csv --output data/patterns/XAUUSD_M5_patterns.csv
```

## Python API

```python
import pandas as pd
from src.pattern_detection.pattern_pipeline import detect_patterns, plot_patterns

features = pd.read_csv("data/processed/XAUUSD_M5_features.csv")
patterns = detect_patterns(features)
fig = plot_patterns(patterns)
fig.show()
```

## Pattern Output

The pipeline adds:

- `bullish_liquidity_sweep`, `bearish_liquidity_sweep`
- `sweep_price`, `sweep_time`
- `bullish_mss`, `bearish_mss`
- `bullish_bos`, `bearish_bos`
- `bullish_fvg`, `bearish_fvg`, `fvg_top`, `fvg_bottom`
- `displacement`, `displacement_direction`
- `bullish_ob`, `bearish_ob`, `ob_high`, `ob_low`
- `premium_zone`, `discount_zone`, `equilibrium`

## Tests

```bash
python -m pytest tests/test_pattern_detection.py --basetemp .pytest_tmp
```
