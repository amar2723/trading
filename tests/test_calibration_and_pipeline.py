import pytest
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from src.feature_pipeline import (
    FEATURE_COLUMNS,
    compute_indicator_dataframe,
    extract_features_from_row,
)
from src.ml_meta_filter import MLMetaFilter, FilterDecision

ROOT = Path(__file__).resolve().parents[1]
ENSEMBLE_PATH = ROOT / "models/xauusd_master_ensemble.pkl"


def test_shared_feature_pipeline_consistency():
    # Synthetic M5 candles
    dates = pd.date_range("2026-11-18 09:00", periods=50, freq="5min")
    df = pd.DataFrame({
        "timestamp": dates,
        "open": np.linspace(2600.0, 2620.0, 50),
        "high": np.linspace(2602.0, 2622.0, 50),
        "low": np.linspace(2599.0, 2619.0, 50),
        "close": np.linspace(2601.0, 2621.0, 50),
        "tick_volume": [200.0] * 50,
    })

    df_proc = compute_indicator_dataframe(df)
    assert "ema_20" in df_proc.columns
    assert "ema_50" in df_proc.columns
    assert "atr" in df_proc.columns
    assert "rsi" in df_proc.columns
    assert "buyer_pct" in df_proc.columns
    assert "h1_trend_bias" in df_proc.columns

    # Extract feature row
    last_row = df_proc.iloc[-1].to_dict()
    feat_df = extract_features_from_row(last_row, action="BUY", strategy_name="H1_PULLBACK")

    assert list(feat_df.columns) == FEATURE_COLUMNS
    assert feat_df["side_code"].iloc[0] == 1.0
    assert feat_df["is_pullback"].iloc[0] == 1.0
    assert feat_df["is_donchian"].iloc[0] == 0.0
    assert not feat_df.isna().any().any()


def test_calibrator_integration_consistency():
    assert ENSEMBLE_PATH.exists(), "Trained model ensemble must exist"
    model_dict = joblib.load(ENSEMBLE_PATH)
    assert "calibrator" in model_dict, "Ensemble must contain calibrator"
    calibrator = model_dict["calibrator"]
    scaler = model_dict["scaler"]
    xgb_m = model_dict["xgb_model"]
    lgb_m = model_dict["lgb_model"]

    # Generate dummy input row
    dummy_dict = {col: 0.5 for col in FEATURE_COLUMNS}
    dummy_dict["side_code"] = 1.0
    dummy_dict["is_pullback"] = 1.0
    feat_row = pd.DataFrame([dummy_dict], columns=FEATURE_COLUMNS)

    X_scaled = scaler.transform(feat_row)
    p_xgb = float(xgb_m.predict_proba(X_scaled)[0][1])
    p_lgb = float(lgb_m.predict_proba(X_scaled)[0][1])
    p_raw = (p_xgb * 0.55) + (p_lgb * 0.45)
    p_cal_expected = round(float(np.clip(calibrator.predict(np.array([p_raw]))[0], 0.0, 1.0)), 4)

    # Now evaluate through MLMetaFilter
    filter_obj = MLMetaFilter(model_path=ENSEMBLE_PATH)
    
    # Create test M5 candles
    dates = pd.date_range("2026-11-18 09:00", periods=50, freq="5min")
    df_test = pd.DataFrame({
        "timestamp": dates,
        "open": [2600.0] * 50,
        "high": [2602.0] * 50,
        "low": [2598.0] * 50,
        "close": [2600.0] * 50,
        "tick_volume": [200.0] * 50,
    })
    df_test = compute_indicator_dataframe(df_test)

    decision = filter_obj.evaluate_candidate(
        candidate_action="BUY",
        strategy_name="H1_PULLBACK",
        df_m5=df_test,
        h1_trend_bias=1,
        h1_dist_ema50=0.0,
        risk_usd=4.50,
    )

    assert isinstance(decision, FilterDecision)
    assert 0.0 <= decision.predicted_win_prob <= 1.0
    # Probability must reflect calibrator transformation, not uncalibrated raw score
    assert decision.confidence_pct == round(decision.predicted_win_prob * 100, 1)
