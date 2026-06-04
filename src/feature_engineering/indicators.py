from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average using the standard trading span formula."""
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index with simple rolling gains and losses."""
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period, min_periods=period).mean()
    loss = -delta.clip(upper=0).rolling(period, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    value = 100 - (100 / (1 + rs))
    return value.fillna(50.0)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range."""
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(period, min_periods=1).mean()


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    return pd.DataFrame(
        {
            "macd": macd_line,
            "macd_signal": signal_line,
            "macd_histogram": macd_line - signal_line,
        }
    )


def vwap(df: pd.DataFrame) -> pd.Series:
    volume_col = "tick_volume" if "tick_volume" in df.columns else "volume"
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    cumulative_volume = df[volume_col].replace(0, np.nan).cumsum()
    return (typical_price * df[volume_col]).cumsum() / cumulative_volume


def bollinger_bands(series: pd.Series, period: int = 20, std_mult: float = 2.0) -> pd.DataFrame:
    middle = series.rolling(period, min_periods=1).mean()
    std = series.rolling(period, min_periods=1).std(ddof=0)
    return pd.DataFrame(
        {
            "bb_middle": middle,
            "bb_upper": middle + std * std_mult,
            "bb_lower": middle - std * std_mult,
            "bb_width": (std * std_mult * 2) / middle.replace(0, np.nan),
        }
    )


def volume_moving_average(df: pd.DataFrame, period: int = 20) -> pd.Series:
    volume_col = "tick_volume" if "tick_volume" in df.columns else "volume"
    return df[volume_col].rolling(period, min_periods=1).mean()


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ema20"] = ema(out["close"], 20)
    out["ema50"] = ema(out["close"], 50)
    out["ema200"] = ema(out["close"], 200)
    out["rsi"] = rsi(out["close"], 14)
    out["atr"] = atr(out, 14)
    out = pd.concat([out, macd(out["close"])], axis=1)
    out["vwap"] = vwap(out)
    out = pd.concat([out, bollinger_bands(out["close"])], axis=1)
    out["volume_ma"] = volume_moving_average(out, 20)
    return out


def add_features(df: pd.DataFrame, atr_period: int = 14, rsi_period: int = 14) -> pd.DataFrame:
    """Backward-compatible wrapper used by earlier modules."""
    from src.feature_engineering.feature_pipeline import build_features

    return build_features(df)


FEATURE_COLUMNS = [
    "ema20",
    "ema50",
    "ema200",
    "rsi",
    "atr",
    "macd",
    "macd_signal",
    "macd_histogram",
    "vwap",
    "bb_middle",
    "bb_upper",
    "bb_lower",
    "bb_width",
    "volume_ma",
    "body_size",
    "body_percentage",
    "upper_wick",
    "lower_wick",
    "wick_ratio",
    "range_size",
    "rolling_std",
    "volatility_ratio",
    "trend_strength",
]
