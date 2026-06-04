from __future__ import annotations

import joblib
import pandas as pd

from src.training.feature_selection import load_labeled_data, prepare_features, select_feature_columns
from src.training.train_pipeline import predict_probabilities, train_all_models


def labeled_frame(rows: int = 36) -> pd.DataFrame:
    data = []
    for i in range(rows):
        is_buy = i % 3 == 0
        is_sell = i % 3 == 1
        profitable = 1 if i % 4 in {0, 1} else 0
        data.append(
            {
                "timestamp": pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=5 * i),
                "open": 2000 + i,
                "high": 2002 + i,
                "low": 1999 + i,
                "close": 2001 + i,
                "ema20": 2000 + i,
                "ema50": 1998 + i,
                "ema200": 1990 + i,
                "rsi": 45 + (i % 20),
                "atr": 2.5,
                "body_size": 1.0 + (i % 4),
                "rolling_std": 0.01,
                "volatility_ratio": 1.1,
                "bullish_liquidity_sweep": int(is_buy),
                "bullish_mss": int(is_buy),
                "bullish_bos": int(is_buy),
                "bearish_liquidity_sweep": int(is_sell),
                "bearish_mss": int(is_sell),
                "bearish_bos": int(is_sell),
                "bullish_fvg": int(is_buy),
                "bearish_fvg": int(is_sell),
                "bullish_ob": int(is_buy),
                "bearish_ob": int(is_sell),
                "risk_reward_ratio": 1.5 + (i % 3) * 0.2,
                "entry_type": "BUY" if is_buy else "SELL" if is_sell else "HOLD",
                "target": 1 if is_buy else -1 if is_sell else 0,
                "entry_price": 2001 + i,
                "exit_price": 2003 + i if profitable else 1999 + i,
                "profitable_trade": profitable,
                "losing_trade": int(not profitable),
                "trade_result": "WIN" if profitable else "LOSS",
            }
        )
    return pd.DataFrame(data)


def test_feature_loading_excludes_labels(tmp_path):
    path = tmp_path / "labeled.csv"
    labeled_frame().to_csv(path, index=False)
    df = load_labeled_data(path)
    columns = select_feature_columns(df)
    assert "profitable_trade" not in columns
    assert "timestamp" not in columns
    X, y, direction, selected = prepare_features(df)
    assert len(X) == len(y) == len(direction)
    assert selected == columns


def test_training_prediction_probability_and_model_saving(tmp_path):
    data_path = tmp_path / "labeled.csv"
    model_dir = tmp_path / "models"
    report_dir = tmp_path / "reports"
    labeled_frame().to_csv(data_path, index=False)

    result = train_all_models(data_path, model_dir=model_dir, report_dir=report_dir)
    assert (model_dir / "xgboost_model.pkl").exists()
    assert (model_dir / "lightgbm_model.pkl").exists()
    assert (model_dir / "random_forest.pkl").exists()
    assert (model_dir / "feature_columns.pkl").exists()
    assert (model_dir / "scaler.pkl").exists()
    assert (report_dir / "metrics.json").exists()

    bundle = joblib.load(model_dir / "xgboost_model.pkl")
    probabilities = predict_probabilities(bundle, labeled_frame())
    assert {"buy_probability", "sell_probability", "hold_probability", "confidence_score"}.issubset(probabilities.columns)
    assert probabilities["confidence_score"].between(0, 100).all()
