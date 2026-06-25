from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import BinaryIO

import numpy as np
import pandas as pd
from PIL import Image


@dataclass
class ImageTradeAnalysis:
    signal: str
    confidence: float
    bias: str
    trend: str
    green_pressure: float
    red_pressure: float
    recent_pressure: str
    entry: float | None
    stop_loss: float | None
    tp1: float | None
    tp2: float | None
    tp3: float | None
    risk_reward: float | None
    live_price_source: str
    reason: list[str]
    warning: str

    def to_dict(self) -> dict:
        return asdict(self)


def analyze_chart_image(
    image_file: BinaryIO,
    candles: pd.DataFrame | None = None,
    live_tick: dict | None = None,
) -> ImageTradeAnalysis:
    """
    Analyze a chart screenshot with simple color/shape heuristics.

    This is intentionally conservative. A screenshot does not contain reliable
    executable OHLC prices, so price levels are only generated when live candle
    data is also supplied.
    """
    image = Image.open(image_file).convert("RGB")
    # Keep analysis fast and robust on large screenshots.
    image.thumbnail((1200, 800))
    arr = np.asarray(image).astype(np.int16)

    red_mask, green_mask = _candle_color_masks(arr)
    active_mask = red_mask | green_mask
    active_count = int(active_mask.sum())
    if active_count < 50:
        return ImageTradeAnalysis(
            signal="NO TRADE",
            confidence=0.0,
            bias="UNKNOWN",
            trend="UNKNOWN",
            green_pressure=0.0,
            red_pressure=0.0,
            recent_pressure="UNKNOWN",
            entry=None,
            stop_loss=None,
            tp1=None,
            tp2=None,
            tp3=None,
            risk_reward=None,
            live_price_source="none",
            reason=["Not enough candle-like red/green pixels detected"],
            warning=_warning(),
        )

    green_pressure = float(green_mask.sum() / active_count)
    red_pressure = float(red_mask.sum() / active_count)
    trend = _estimate_trend(active_mask)
    recent_pressure = _recent_pressure(red_mask, green_mask)
    signal, confidence, reasons = _classify(green_pressure, red_pressure, trend, recent_pressure)
    levels = _levels_from_live_candles(signal, candles, live_tick)

    return ImageTradeAnalysis(
        signal=signal,
        confidence=confidence,
        bias="BULLISH" if green_pressure > red_pressure else "BEARISH",
        trend=trend,
        green_pressure=green_pressure,
        red_pressure=red_pressure,
        recent_pressure=recent_pressure,
        entry=levels["entry"],
        stop_loss=levels["stop_loss"],
        tp1=levels["tp1"],
        tp2=levels["tp2"],
        tp3=levels["tp3"],
        risk_reward=levels["risk_reward"],
        live_price_source=levels["live_price_source"],
        reason=reasons,
        warning=_warning(),
    )


def _candle_color_masks(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    red = arr[:, :, 0]
    green = arr[:, :, 1]
    blue = arr[:, :, 2]
    # Covers common MT5/TradingView green/red candles while ignoring white text/grid.
    green_mask = (green > 95) & (green > red * 1.25) & (green > blue * 1.15)
    red_mask = (red > 110) & (red > green * 1.20) & (red > blue * 1.10)
    return red_mask, green_mask


def _estimate_trend(active_mask: np.ndarray) -> str:
    height, width = active_mask.shape
    chunks = np.array_split(np.arange(width), 3)
    y_means = []
    for cols in chunks:
        ys, _ = np.where(active_mask[:, cols])
        y_means.append(float(ys.mean()) if len(ys) else np.nan)
    if any(np.isnan(y_means)):
        return "RANGE"
    # In image coordinates, lower y means higher price on the chart.
    delta = y_means[0] - y_means[-1]
    if delta > height * 0.04:
        return "UP"
    if delta < -height * 0.04:
        return "DOWN"
    return "RANGE"


def _recent_pressure(red_mask: np.ndarray, green_mask: np.ndarray) -> str:
    width = red_mask.shape[1]
    start = int(width * 0.66)
    recent_red = int(red_mask[:, start:].sum())
    recent_green = int(green_mask[:, start:].sum())
    total = recent_red + recent_green
    if total < 20:
        return "UNKNOWN"
    ratio = recent_green / total
    if ratio >= 0.58:
        return "BULLISH"
    if ratio <= 0.42:
        return "BEARISH"
    return "MIXED"


def _classify(green_pressure: float, red_pressure: float, trend: str, recent_pressure: str) -> tuple[str, float, list[str]]:
    bull_score = 0
    bear_score = 0
    reasons: list[str] = []

    if green_pressure >= 0.56:
        bull_score += 25
        reasons.append("More bullish candle pixels than bearish")
    if red_pressure >= 0.56:
        bear_score += 25
        reasons.append("More bearish candle pixels than bullish")
    if trend == "UP":
        bull_score += 25
        reasons.append("Visual trend slopes upward")
    elif trend == "DOWN":
        bear_score += 25
        reasons.append("Visual trend slopes downward")
    if recent_pressure == "BULLISH":
        bull_score += 25
        reasons.append("Recent right-side candles show bullish pressure")
    elif recent_pressure == "BEARISH":
        bear_score += 25
        reasons.append("Recent right-side candles show bearish pressure")

    if bull_score >= 50 and bull_score > bear_score:
        return "BUY", float(min(85, bull_score)), reasons
    if bear_score >= 50 and bear_score > bull_score:
        return "SELL", float(min(85, bear_score)), reasons
    return "NO TRADE", float(max(bull_score, bear_score)), reasons or ["Mixed screenshot evidence"]


def _levels_from_live_candles(signal: str, candles: pd.DataFrame | None, live_tick: dict | None) -> dict[str, float | str | None]:
    empty = {
        "entry": None,
        "stop_loss": None,
        "tp1": None,
        "tp2": None,
        "tp3": None,
        "risk_reward": None,
        "live_price_source": "none",
    }
    if signal not in {"BUY", "SELL"} or candles is None or candles.empty or len(candles) < 20:
        return empty

    data = candles.copy()
    entry, source = _live_entry(signal, data, live_tick)
    atr = _atr(data)
    recent = data.tail(20)
    if signal == "BUY":
        stop_loss = float(recent["low"].min()) - atr * 0.25
        risk = entry - stop_loss
        if risk <= 0:
            return empty
        return {
            "entry": entry,
            "stop_loss": stop_loss,
            "tp1": entry + risk * 1.5,
            "tp2": entry + risk * 2.5,
            "tp3": entry + risk * 3.5,
            "risk_reward": 1.5,
            "live_price_source": source,
        }

    stop_loss = float(recent["high"].max()) + atr * 0.25
    risk = stop_loss - entry
    if risk <= 0:
        return empty
    return {
        "entry": entry,
        "stop_loss": stop_loss,
        "tp1": entry - risk * 1.5,
        "tp2": entry - risk * 2.5,
        "tp3": entry - risk * 3.5,
        "risk_reward": 1.5,
        "live_price_source": source,
    }


def _live_entry(signal: str, data: pd.DataFrame, live_tick: dict | None) -> tuple[float, str]:
    if live_tick:
        if signal == "BUY" and live_tick.get("ask"):
            return float(live_tick["ask"]), "live ask"
        if signal == "SELL" and live_tick.get("bid"):
            return float(live_tick["bid"]), "live bid"
    return float(data.iloc[-1]["close"]), "latest candle close"


def _atr(data: pd.DataFrame, period: int = 14) -> float:
    high_low = data["high"] - data["low"]
    high_close = (data["high"] - data["close"].shift()).abs()
    low_close = (data["low"] - data["close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    value = float(true_range.tail(period).mean())
    return value if value > 0 else 0.0


def _warning() -> str:
    return "Screenshot analysis is approximate. Use live MT5 candle data and risk controls before any demo decision."
