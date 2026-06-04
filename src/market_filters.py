from __future__ import annotations

import pandas as pd


SESSION_WINDOWS = {
    "ALL": None,
    "LONDON": (7, 12),
    "NEW_YORK": (13, 20),
    "OVERLAP": (13, 16),
}


def add_filter_features(candles: pd.DataFrame, atr_period: int = 14, chop_lookback: int = 20) -> pd.DataFrame:
    data = candles.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    data["ema20"] = data["close"].ewm(span=20, adjust=False).mean()
    data["ema50"] = data["close"].ewm(span=50, adjust=False).mean()
    data["trend_direction"] = "NEUTRAL"
    data.loc[data["ema20"] > data["ema50"], "trend_direction"] = "BULLISH"
    data.loc[data["ema20"] < data["ema50"], "trend_direction"] = "BEARISH"

    true_range = pd.concat(
        [
            data["high"] - data["low"],
            (data["high"] - data["close"].shift()).abs(),
            (data["low"] - data["close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    data["atr"] = true_range.rolling(atr_period, min_periods=1).mean()
    data["range_high"] = data["high"].rolling(chop_lookback, min_periods=1).max()
    data["range_low"] = data["low"].rolling(chop_lookback, min_periods=1).min()
    data["range_width"] = data["range_high"] - data["range_low"]
    data["chop_ratio"] = data["range_width"] / data["atr"].replace(0, pd.NA)
    return data


def passes_session_filter(timestamp, session: str) -> bool:
    session = session.upper()
    window = SESSION_WINDOWS.get(session)
    if window is None:
        return True
    ts = pd.to_datetime(timestamp, errors="coerce")
    if pd.isna(ts):
        return False
    start, end = window
    return start <= ts.hour < end


def passes_trend_filter(row: pd.Series, signal: str, enabled: bool) -> bool:
    if not enabled:
        return True
    trend = row.get("trend_direction", "NEUTRAL")
    if signal == "BUY":
        return trend == "BULLISH"
    if signal == "SELL":
        return trend == "BEARISH"
    return False


def passes_chop_filter(row: pd.Series, enabled: bool, min_chop_ratio: float = 4.0) -> bool:
    if not enabled:
        return True
    ratio = row.get("chop_ratio")
    try:
        return float(ratio) >= min_chop_ratio
    except Exception:
        return False
