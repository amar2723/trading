from __future__ import annotations

import logging

import pandas as pd

from src.advanced_smc.features import add_core_features
from src.advanced_smc.liquidity import add_liquidity
from src.advanced_smc.models import TradeSignal
from src.advanced_smc.patterns import add_displacement, add_fvg, add_mss_bos, add_order_blocks, add_retests


logger = logging.getLogger(__name__)


class AdvancedSMCEngine:
    def __init__(self, symbol: str = "XAUUSD", min_confidence: float = 70.0):
        self.symbol = symbol
        self.min_confidence = min_confidence

    def enrich(self, candles: pd.DataFrame) -> pd.DataFrame:
        data = add_core_features(candles)
        data = add_liquidity(data)
        data = add_displacement(data)
        data = add_mss_bos(data)
        data = add_order_blocks(data)
        data = add_fvg(data)
        data = add_retests(data)
        return data

    def predict(self, candles: pd.DataFrame, use_closed_candle: bool = True) -> TradeSignal:
        data = self.enrich(candles)
        row = data.iloc[-2] if use_closed_candle and len(data) > 1 else data.iloc[-1]
        return self._signal_from_row(row)

    def _signal_from_row(self, row: pd.Series) -> TradeSignal:
        buy_reasons, buy_conf = self._score_buy(row)
        sell_reasons, sell_conf = self._score_sell(row)
        timestamp = str(row.get("timestamp", ""))
        if buy_conf >= self.min_confidence and buy_conf >= sell_conf:
            return self._build_trade("BUY", row, buy_conf, buy_reasons, timestamp)
        if sell_conf >= self.min_confidence:
            return self._build_trade("SELL", row, sell_conf, sell_reasons, timestamp)
        return TradeSignal(self.symbol, "NO TRADE", None, None, None, None, None, None, max(buy_conf, sell_conf), ["Confidence below threshold"], timestamp)

    def _score_buy(self, row: pd.Series) -> tuple[list[str], float]:
        reasons, score = [], 0.0
        if bool(row.get("sell_side_liquidity_sweep")):
            reasons.append("Sell Side Liquidity Sweep")
            score += 25
        if bool(row.get("bullish_mss")):
            reasons.append("Bullish MSS")
            score += 25
        if bool(row.get("bullish_displacement")):
            reasons.append("Bullish Displacement")
            score += 20
        if bool(row.get("bullish_ob_retest")):
            reasons.append("Bullish Order Block Retest")
            score += 15
        if bool(row.get("bullish_fvg_retest")):
            reasons.append("Bullish FVG Retest")
            score += 15
        return reasons, min(score, 100.0)

    def _score_sell(self, row: pd.Series) -> tuple[list[str], float]:
        reasons, score = [], 0.0
        if bool(row.get("buy_side_liquidity_sweep")):
            reasons.append("Buy Side Liquidity Sweep")
            score += 25
        if bool(row.get("bearish_mss")):
            reasons.append("Bearish MSS")
            score += 25
        if bool(row.get("bearish_displacement")):
            reasons.append("Bearish Displacement")
            score += 20
        if bool(row.get("bearish_ob_retest")):
            reasons.append("Bearish Order Block Retest")
            score += 15
        if bool(row.get("bearish_fvg_retest")):
            reasons.append("Bearish FVG Retest")
            score += 15
        return reasons, min(score, 100.0)

    def _build_trade(self, side: str, row: pd.Series, confidence: float, reasons: list[str], timestamp: str) -> TradeSignal:
        entry = float(row["close"])
        atr = float(row.get("atr") or 0)
        if side == "BUY":
            swept = row.get("recent_swept_low")
            sl = float(swept if pd.notna(swept) else row["low"]) - atr * 0.25
            risk = entry - sl
            targets = self._buy_targets(row, entry, risk)
        else:
            swept = row.get("recent_swept_high")
            sl = float(swept if pd.notna(swept) else row["high"]) + atr * 0.25
            risk = sl - entry
            targets = self._sell_targets(row, entry, risk)
        rr = (targets[0] - entry) / risk if side == "BUY" and risk > 0 else (entry - targets[0]) / risk if risk > 0 else None
        return TradeSignal(self.symbol, side, entry, sl, targets[0], targets[1], targets[2], rr, confidence, reasons, timestamp)

    def _buy_targets(self, row: pd.Series, entry: float, risk: float) -> tuple[float, float, float]:
        nearest = row.get("buy_side_liquidity")
        tp1 = float(nearest) if pd.notna(nearest) and nearest > entry else entry + 1.5 * risk
        opposite = row.get("high_cluster")
        tp2 = float(opposite) if pd.notna(opposite) and opposite > tp1 else entry + 2.5 * risk
        major = row.get("prev_swing_high")
        tp3 = float(major) if pd.notna(major) and major > tp2 else entry + 3.5 * risk
        return tp1, tp2, tp3

    def _sell_targets(self, row: pd.Series, entry: float, risk: float) -> tuple[float, float, float]:
        nearest = row.get("sell_side_liquidity")
        tp1 = float(nearest) if pd.notna(nearest) and nearest < entry else entry - 1.5 * risk
        opposite = row.get("low_cluster")
        tp2 = float(opposite) if pd.notna(opposite) and opposite < tp1 else entry - 2.5 * risk
        major = row.get("prev_swing_low")
        tp3 = float(major) if pd.notna(major) and major < tp2 else entry - 3.5 * risk
        return tp1, tp2, tp3
