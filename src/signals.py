from __future__ import annotations

import joblib
import pandas as pd

from src.risk import TradePlan, build_trade_plan
from src.training.train_xgboost import MODEL_FEATURES


def rule_signal(row: pd.Series) -> str:
    buy = (
        bool(row.get("bullish_liquidity_sweep"))
        and bool(row.get("bullish_mss"))
        and bool(row.get("bullish_bos"))
        and bool(row.get("bullish_displacement"))
    )
    sell = (
        bool(row.get("bearish_liquidity_sweep"))
        and bool(row.get("bearish_mss"))
        and bool(row.get("bearish_bos"))
        and bool(row.get("bearish_displacement"))
    )
    if buy:
        return "BUY"
    if sell:
        return "SELL"
    return "HOLD"


def ml_confidence(row: pd.Series, model_path: str | None = None) -> float:
    if not model_path:
        return 50.0
    bundle = joblib.load(model_path)
    model = bundle["model"]
    features = bundle.get("features", MODEL_FEATURES)
    X = pd.DataFrame([{f: float(row.get(f, 0.0)) for f in features}])
    return float(model.predict_proba(X)[0, 1] * 100)


def generate_trade_plan(row: pd.Series, model_path: str | None = None) -> TradePlan:
    signal = rule_signal(row)
    confidence = ml_confidence(row, model_path) if signal != "HOLD" else 0.0
    return build_trade_plan(row, signal, confidence)
