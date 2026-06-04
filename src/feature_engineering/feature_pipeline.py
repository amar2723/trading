from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data_ingestion.data_validator import clean_data
from src.feature_engineering.candle_features import add_candle_features, add_volatility_features
from src.feature_engineering.indicators import add_technical_indicators
from src.feature_engineering.market_structure import add_market_structure_features, add_trend_features, detect_swings


def repair_missing_candles(df: pd.DataFrame) -> pd.DataFrame:
    """Reindex to the inferred candle interval and forward-fill OHLC gaps conservatively."""
    data = clean_data(df)
    if len(data) < 3:
        return data

    diffs = data["timestamp"].sort_values().diff().dropna()
    frequency = diffs.mode().iloc[0] if not diffs.empty else None
    if frequency is None or frequency <= pd.Timedelta(0):
        return data

    data = data.set_index("timestamp").sort_index()
    full_index = pd.date_range(data.index.min(), data.index.max(), freq=frequency)
    data = data.reindex(full_index)
    data.index.name = "timestamp"
    data["close"] = data["close"].ffill()
    for column in ["open", "high", "low"]:
        data[column] = data[column].fillna(data["close"])
    for column in ["tick_volume", "spread"]:
        if column in data.columns:
            data[column] = data[column].fillna(0)
    return data.reset_index()


def cap_outliers(df: pd.DataFrame, columns: list[str] | None = None, z_limit: float = 8.0) -> pd.DataFrame:
    """Winsorize extreme numeric values using a robust median absolute deviation rule."""
    out = df.copy()
    columns = columns or ["open", "high", "low", "close", "tick_volume", "spread"]
    for column in [c for c in columns if c in out.columns]:
        series = pd.to_numeric(out[column], errors="coerce")
        median = series.median()
        mad = (series - median).abs().median()
        if pd.isna(mad) or mad == 0:
            continue
        lower = median - z_limit * 1.4826 * mad
        upper = median + z_limit * 1.4826 * mad
        out[column] = series.clip(lower, upper)
    return out


def finalize_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.replace([np.inf, -np.inf], np.nan)
    bool_cols = out.select_dtypes(include=["bool"]).columns
    numeric_cols = out.select_dtypes(include=["number"]).columns
    out[list(bool_cols)] = out[list(bool_cols)].fillna(False)
    out[list(numeric_cols)] = out[list(numeric_cols)].ffill().bfill().fillna(0)
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build all Module 2 features from an OHLCV DataFrame."""
    data = repair_missing_candles(df)
    if "volume" not in data.columns and "tick_volume" in data.columns:
        data["volume"] = data["tick_volume"]
    data = cap_outliers(data)
    data = add_technical_indicators(data)
    data = add_candle_features(data)
    data = add_volatility_features(data)
    data = detect_swings(data)
    data = add_trend_features(data)
    data = add_market_structure_features(data)
    return finalize_features(data)


def save_features(df: pd.DataFrame, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
