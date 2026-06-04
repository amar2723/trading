from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from src.advanced_smc.features import add_core_features
from src.advanced_smc.liquidity import add_liquidity
from src.advanced_smc.patterns import add_displacement, add_mss_bos
from src.multi_timeframe.structure_validator import StructureState
from src.multi_timeframe.trend_analyzer import TrendState


@dataclass
class MultiTimeframeSignal:
    signal: str
    confidence: float
    entry: float | None
    stop_loss: float | None
    tp1: float | None
    tp2: float | None
    tp3: float | None
    rr_ratio: float | None
    reasons: list[str]
    h1_trend: str
    m15_bullish_structure: bool
    m15_bearish_structure: bool
    timestamp: str

    def to_dict(self) -> dict:
        return asdict(self)


def prepare_m5(candles: pd.DataFrame) -> pd.DataFrame:
    data = add_core_features(candles)
    data = add_liquidity(data)
    data = add_displacement(data)
    data = add_mss_bos(data)
    data["close_above_structure"] = data["close"] > data["prev_swing_high"]
    data["close_below_structure"] = data["close"] < data["prev_swing_low"]
    data["nearest_buy_liquidity"] = data["high"].where(data["swing_high"]).ffill()
    data["nearest_sell_liquidity"] = data["low"].where(data["swing_low"]).ffill()
    data["next_buy_liquidity"] = data["high"].where(data["swing_high"]).bfill()
    data["next_sell_liquidity"] = data["low"].where(data["swing_low"]).bfill()
    data["major_buy_liquidity"] = data["high"].rolling(100, min_periods=1).max()
    data["major_sell_liquidity"] = data["low"].rolling(100, min_periods=1).min()
    return data


def confirm_mtf_entry(
    h1: TrendState,
    m15: StructureState,
    m5_candles: pd.DataFrame,
    min_confidence: float = 75.0,
    use_closed_candle: bool = True,
) -> MultiTimeframeSignal:
    data = prepare_m5(m5_candles)
    row = data.iloc[-2] if use_closed_candle and len(data) > 1 else data.iloc[-1]
    return confirm_mtf_row(h1, m15, row, min_confidence)


def confirm_mtf_row(
    h1: TrendState,
    m15: StructureState,
    row: pd.Series,
    min_confidence: float = 75.0,
) -> MultiTimeframeSignal:
    timestamp = str(row.get("timestamp", ""))
    buy_reasons = []
    if h1.trend_direction == "Bullish":
        buy_reasons.append("H1 Trend Alignment")
    if m15.bullish_mss:
        buy_reasons.append("M15 Bullish MSS")
    if m15.bullish_bos:
        buy_reasons.append("M15 Bullish BOS")
    if bool(row.get("sell_side_liquidity_sweep")):
        buy_reasons.append("M5 Sell Side Liquidity Sweep")
    if bool(row.get("bullish_displacement")):
        buy_reasons.append("M5 Bullish Displacement")
    if bool(row.get("close_above_structure")):
        buy_reasons.append("M5 Close Above Structure")

    sell_reasons = []
    if h1.trend_direction == "Bearish":
        sell_reasons.append("H1 Trend Alignment")
    if m15.bearish_mss:
        sell_reasons.append("M15 Bearish MSS")
    if m15.bearish_bos:
        sell_reasons.append("M15 Bearish BOS")
    if bool(row.get("buy_side_liquidity_sweep")):
        sell_reasons.append("M5 Buy Side Liquidity Sweep")
    if bool(row.get("bearish_displacement")):
        sell_reasons.append("M5 Bearish Displacement")
    if bool(row.get("close_below_structure")):
        sell_reasons.append("M5 Close Below Structure")

    buy_confidence = _confidence("BUY", h1, m15, row)
    sell_confidence = _confidence("SELL", h1, m15, row)
    if buy_confidence >= min_confidence and len(buy_reasons) >= 5 and buy_confidence >= sell_confidence:
        return _build_signal("BUY", row, buy_confidence, buy_reasons, h1, m15, timestamp)
    if sell_confidence >= min_confidence and len(sell_reasons) >= 5:
        return _build_signal("SELL", row, sell_confidence, sell_reasons, h1, m15, timestamp)
    return MultiTimeframeSignal("NO TRADE", max(buy_confidence, sell_confidence), None, None, None, None, None, None, ["MTF conditions incomplete"], h1.trend_direction, m15.bullish_structure_valid, m15.bearish_structure_valid, timestamp)


def _confidence(side: str, h1: TrendState, m15: StructureState, row: pd.Series) -> float:
    score = 0.0
    if (side == "BUY" and h1.trend_direction == "Bullish") or (side == "SELL" and h1.trend_direction == "Bearish"):
        score += 20
    if (side == "BUY" and m15.bullish_mss) or (side == "SELL" and m15.bearish_mss):
        score += 20
    if (side == "BUY" and m15.bullish_bos) or (side == "SELL" and m15.bearish_bos):
        score += 15
    if (side == "BUY" and bool(row.get("sell_side_liquidity_sweep"))) or (side == "SELL" and bool(row.get("buy_side_liquidity_sweep"))):
        score += 20
    if (side == "BUY" and bool(row.get("bullish_displacement"))) or (side == "SELL" and bool(row.get("bearish_displacement"))):
        score += 15
    if _target_quality(side, row):
        score += 10
    return score


def _target_quality(side: str, row: pd.Series) -> bool:
    if side == "BUY":
        return pd.notna(row.get("nearest_buy_liquidity")) and row["nearest_buy_liquidity"] > row["close"]
    return pd.notna(row.get("nearest_sell_liquidity")) and row["nearest_sell_liquidity"] < row["close"]


def _build_signal(side: str, row: pd.Series, confidence: float, reasons: list[str], h1: TrendState, m15: StructureState, timestamp: str) -> MultiTimeframeSignal:
    entry = float(row["close"])
    atr = float(row.get("atr") or 0)
    if side == "BUY":
        base = row.get("recent_swept_low") if pd.notna(row.get("recent_swept_low")) else row["low"]
        sl = float(base) - atr * 0.25
        risk = entry - sl
        tp1 = _above(row.get("nearest_buy_liquidity"), entry, entry + 1.5 * risk)
        if tp1 - entry < 1.5 * risk:
            return MultiTimeframeSignal("NO TRADE", confidence, None, None, None, None, None, None, ["Rejected: TP1 < 1.5R"], h1.trend_direction, m15.bullish_structure_valid, m15.bearish_structure_valid, timestamp)
        tp2 = _above(row.get("next_buy_liquidity"), tp1, entry + 2.5 * risk)
        tp3 = _above(row.get("major_buy_liquidity"), tp2, entry + 3.5 * risk)
        rr = (tp1 - entry) / risk if risk > 0 else None
    else:
        base = row.get("recent_swept_high") if pd.notna(row.get("recent_swept_high")) else row["high"]
        sl = float(base) + atr * 0.25
        risk = sl - entry
        tp1 = _below(row.get("nearest_sell_liquidity"), entry, entry - 1.5 * risk)
        if entry - tp1 < 1.5 * risk:
            return MultiTimeframeSignal("NO TRADE", confidence, None, None, None, None, None, None, ["Rejected: TP1 < 1.5R"], h1.trend_direction, m15.bullish_structure_valid, m15.bearish_structure_valid, timestamp)
        tp2 = _below(row.get("next_sell_liquidity"), tp1, entry - 2.5 * risk)
        tp3 = _below(row.get("major_sell_liquidity"), tp2, entry - 3.5 * risk)
        rr = (entry - tp1) / risk if risk > 0 else None
    return MultiTimeframeSignal(side, confidence, entry, sl, tp1, tp2, tp3, rr, reasons, h1.trend_direction, m15.bullish_structure_valid, m15.bearish_structure_valid, timestamp)


def _above(value, threshold: float, fallback: float) -> float:
    return float(value) if pd.notna(value) and float(value) > threshold else float(fallback)


def _below(value, threshold: float, fallback: float) -> float:
    return float(value) if pd.notna(value) and float(value) < threshold else float(fallback)
