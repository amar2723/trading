from __future__ import annotations

import logging

import pandas as pd

from src.advanced_smc.features import add_core_features
from src.advanced_smc.liquidity import add_liquidity
from src.advanced_smc.patterns import add_displacement, add_mss_bos
from src.prediction.confidence_engine import calculate_confidence
from src.prediction.entry_engine import evaluate_entry
from src.prediction.liquidity_mapper import map_liquidity
from src.prediction.sl_engine import calculate_stop_loss
from src.prediction.structure_analyzer import analyze_structure
from src.prediction.tp_engine import calculate_take_profits


logger = logging.getLogger(__name__)


class PredictionPipeline:
    def __init__(self, min_confidence: float = 70.0):
        self.min_confidence = min_confidence

    def prepare(self, candles: pd.DataFrame) -> pd.DataFrame:
        data = add_core_features(candles)
        data = add_liquidity(data)
        data = add_displacement(data)
        data = add_mss_bos(data)
        data = analyze_structure(data)
        data = map_liquidity(data)
        return data

    def predict(self, candles: pd.DataFrame, use_closed_candle: bool = True) -> dict:
        data = self.prepare(candles)
        row = data.iloc[-2] if use_closed_candle and len(data) > 1 else data.iloc[-1]
        return self.predict_row(row)

    def predict_row(self, row: pd.Series) -> dict:
        signal, entry_reasons = evaluate_entry(row)
        confidence, confidence_reasons = calculate_confidence(row, signal)
        if confidence < self.min_confidence:
            signal = "NO TRADE"
        entry = float(row["close"]) if signal in {"BUY", "SELL"} else None
        stop_loss = calculate_stop_loss(row, signal) if entry is not None else None
        tps = calculate_take_profits(row, signal, entry, stop_loss) if entry is not None and stop_loss is not None else {}
        return {
            "signal": signal,
            "confidence": confidence,
            "entry": entry,
            "stop_loss": stop_loss,
            "tp1": tps.get("tp1"),
            "tp2": tps.get("tp2"),
            "tp3": tps.get("tp3"),
            "rr_ratio": tps.get("rr_ratio"),
            "reason": entry_reasons + [reason for reason in confidence_reasons if reason not in entry_reasons],
            "nearest_buy_liquidity": _safe(row.get("nearest_buy_liquidity")),
            "nearest_sell_liquidity": _safe(row.get("nearest_sell_liquidity")),
            "trend_direction": row.get("trend_direction"),
            "timestamp": str(row.get("timestamp", "")),
        }


def run_prediction(candles: pd.DataFrame, min_confidence: float = 70.0) -> dict:
    return PredictionPipeline(min_confidence).predict(candles)


def _safe(value):
    return None if pd.isna(value) else float(value)
