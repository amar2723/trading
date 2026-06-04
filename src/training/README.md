# Module 5: Machine Learning Training Engine

This module trains profitability models from labeled trade datasets.

## Models

- RandomForestClassifier
- XGBoostClassifier as the main model, with a sklearn fallback if XGBoost is unavailable
- LightGBMClassifier, with a sklearn fallback if LightGBM is unavailable

## Targets

Primary target:

```text
profitable_trade
1 = profitable
0 = losing
```

Secondary target:

```text
trade_direction / target
BUY = 1
SELL = -1
HOLD = 0
```

## Usage

Run from the `trading_ai` directory:

```bash
python train_model.py --data data/labeled/XAUUSD_M5_labeled.csv
```

With Optuna:

```bash
python train_model.py --data data/labeled/XAUUSD_M5_labeled.csv --optimize
```

## Outputs

Models:

```text
models/xgboost_model.pkl
models/lightgbm_model.pkl
models/random_forest.pkl
models/feature_columns.pkl
models/scaler.pkl
```

Reports:

```text
reports/metrics.json
reports/training_report.html
reports/feature_importance.csv
reports/feature_importance.png
reports/feature_importance.html
reports/shap_report.html
reports/probability_output.csv
```

## Probability Output

The pipeline creates:

- `buy_probability`
- `sell_probability`
- `hold_probability`
- `confidence_score`

The confidence score combines model probability with available pattern agreement.

## Testing

```bash
python -m pytest tests/test_training.py --basetemp .pytest_tmp
```
