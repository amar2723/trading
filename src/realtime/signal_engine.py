from __future__ import annotations

import logging
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd

from src.feature_engineering.feature_pipeline import build_features
from src.labeling.sl_tp_generator import generate_sl_tp
from src.pattern_detection.pattern_pipeline import detect_patterns
from src.training.train_pipeline import predict_probabilities


logger = logging.getLogger(__name__)


@dataclass
class SignalConfig:
    confidence_threshold: float = 70.0
    max_spread: float = 80.0
    max_atr: float = 25.0


class SignalEngine:
    def __init__(self, model_path: str | None = None, config: SignalConfig | None = None):
        self.model_path = model_path
        self.config = config or SignalConfig()
        self.model_bundle = joblib.load(model_path) if model_path else None

    def process_candles(self, df: pd.DataFrame) -> pd.DataFrame:
        features = build_features(df)
        patterns = detect_patterns(features)
        if self.model_bundle is not None:
            probabilities = predict_probabilities(self.model_bundle, patterns)
            patterns = pd.concat([patterns.reset_index(drop=True), probabilities], axis=1)
        else:
            patterns["buy_probability"] = _pattern_probability(patterns, "BUY")
            patterns["sell_probability"] = _pattern_probability(patterns, "SELL")
            patterns["hold_probability"] = 100 - patterns[["buy_probability", "sell_probability"]].max(axis=1)
            patterns["confidence_score"] = patterns[["buy_probability", "sell_probability", "hold_probability"]].max(axis=1)
        return generate_sl_tp(_attach_entries(patterns))

    def latest_signal(self, df: pd.DataFrame) -> dict:
        enriched = self.process_candles(df)
        row = enriched.iloc[-1]
        signal = self._decide(row)
        return {
            "timestamp": str(row.get("timestamp", "")),
            "signal": signal,
            "entry": _safe_float(row.get("entry_price") if signal != "HOLD" else row.get("close")),
            "sl": _safe_float(row.get("sl_price")),
            "tp1": _safe_float(row.get("tp1")),
            "tp2": _safe_float(row.get("tp2")),
            "risk_reward": _safe_float(row.get("risk_reward_ratio")),
            "confidence": _safe_float(row.get("confidence_score", 0.0)),
            "spread": _safe_float(row.get("spread", 0.0)),
            "atr": _safe_float(row.get("atr", 0.0)),
            "raw": row.to_dict(),
        }

    def _decide(self, row: pd.Series) -> str:
        confidence = float(row.get("confidence_score", 0.0))
        if confidence < self.config.confidence_threshold:
            return "HOLD"
        buy = bool(row.get("bullish_liquidity_sweep", 0)) and bool(row.get("bullish_mss", 0)) and bool(row.get("bullish_bos", 0))
        sell = bool(row.get("bearish_liquidity_sweep", 0)) and bool(row.get("bearish_mss", 0)) and bool(row.get("bearish_bos", 0))
        if buy and row.get("buy_probability", 0) >= row.get("sell_probability", 0):
            return "BUY"
        if sell and row.get("sell_probability", 0) >= row.get("buy_probability", 0):
            return "SELL"
        return "HOLD"


def _attach_entries(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    buy = out.get("bullish_liquidity_sweep", 0).astype(bool) & out.get("bullish_mss", 0).astype(bool) & out.get("bullish_bos", 0).astype(bool)
    sell = out.get("bearish_liquidity_sweep", 0).astype(bool) & out.get("bearish_mss", 0).astype(bool) & out.get("bearish_bos", 0).astype(bool)
    out["entry_type"] = np.select([buy, sell], ["BUY", "SELL"], default="HOLD")
    out["entry_price"] = np.where(buy | sell, out["close"], np.nan)
    out["entry_time"] = out.get("timestamp", pd.Series(pd.NaT, index=out.index))
    return out


def _pattern_probability(df: pd.DataFrame, side: str) -> pd.Series:
    if side == "BUY":
        columns = ["bullish_liquidity_sweep", "bullish_mss", "bullish_bos", "bullish_fvg", "bullish_ob"]
    else:
        columns = ["bearish_liquidity_sweep", "bearish_mss", "bearish_bos", "bearish_fvg", "bearish_ob"]
    available = [column for column in columns if column in df.columns]
    if not available:
        return pd.Series(0.0, index=df.index)
    return df[available].fillna(0).astype(float).mean(axis=1) * 100


def _safe_float(value) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None
