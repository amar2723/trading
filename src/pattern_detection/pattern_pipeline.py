from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.pattern_detection.bos import detect_bos
from src.pattern_detection.fvg import detect_fvg
from src.pattern_detection.liquidity_sweep import detect_liquidity_sweeps
from src.pattern_detection.mss import detect_mss
from src.pattern_detection.order_blocks import detect_order_blocks


PATTERN_COLUMNS = [
    "bullish_liquidity_sweep",
    "bearish_liquidity_sweep",
    "sweep_price",
    "sweep_time",
    "bullish_mss",
    "bearish_mss",
    "bullish_bos",
    "bearish_bos",
    "bullish_fvg",
    "bearish_fvg",
    "fvg_top",
    "fvg_bottom",
    "displacement",
    "displacement_direction",
    "bullish_ob",
    "bearish_ob",
    "ob_high",
    "ob_low",
    "premium_zone",
    "discount_zone",
    "equilibrium",
]


def add_signal_zones(df: pd.DataFrame, lookback: int = 50) -> pd.DataFrame:
    out = df.copy()
    range_high = out["high"].rolling(lookback, min_periods=1).max()
    range_low = out["low"].rolling(lookback, min_periods=1).min()
    out["equilibrium"] = (range_high + range_low) / 2
    out["premium_zone"] = range_high - ((range_high - range_low) * 0.25)
    out["discount_zone"] = range_low + ((range_high - range_low) * 0.25)
    return out


def detect_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full Smart Money Concepts detection pipeline."""
    out = df.copy()
    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out = detect_liquidity_sweeps(out)
    out = detect_bos(out)
    out = detect_mss(out)
    out = detect_fvg(out)
    out = detect_order_blocks(out)
    out = add_signal_zones(out)
    return _finalize_patterns(out)


def _finalize_patterns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.replace([np.inf, -np.inf], np.nan)
    flag_cols = [
        "bullish_liquidity_sweep",
        "bearish_liquidity_sweep",
        "bullish_mss",
        "bearish_mss",
        "bullish_bos",
        "bearish_bos",
        "bullish_fvg",
        "bearish_fvg",
        "displacement",
        "bullish_displacement",
        "bearish_displacement",
        "bullish_ob",
        "bearish_ob",
    ]
    for column in flag_cols:
        if column in out.columns:
            out[column] = out[column].fillna(0).astype(int)
    return out


def save_patterns(df: pd.DataFrame, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def plot_patterns(df: pd.DataFrame) -> go.Figure:
    """Create an interactive Plotly chart with SMC pattern markers."""
    time_col = "timestamp" if "timestamp" in df.columns else "time"
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df[time_col],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name="Candles",
            )
        ]
    )

    _add_marker(fig, df, time_col, "bullish_liquidity_sweep", "low", "Bullish Sweep")
    _add_marker(fig, df, time_col, "bearish_liquidity_sweep", "high", "Bearish Sweep")
    _add_marker(fig, df, time_col, "bullish_mss", "close", "Bullish MSS")
    _add_marker(fig, df, time_col, "bearish_mss", "close", "Bearish MSS")
    _add_marker(fig, df, time_col, "bullish_bos", "close", "Bullish BOS")
    _add_marker(fig, df, time_col, "bearish_bos", "close", "Bearish BOS")
    _add_marker(fig, df, time_col, "bullish_ob", "ob_low", "Bullish OB")
    _add_marker(fig, df, time_col, "bearish_ob", "ob_high", "Bearish OB")

    for _, row in df[df.get("bullish_fvg", 0).astype(bool) | df.get("bearish_fvg", 0).astype(bool)].iterrows():
        fig.add_shape(
            type="rect",
            x0=row[time_col],
            x1=row[time_col],
            y0=row["fvg_bottom"],
            y1=row["fvg_top"],
            line={"width": 1},
            fillcolor="rgba(90, 140, 255, 0.18)",
        )

    fig.update_layout(height=720, xaxis_rangeslider_visible=False, title="Smart Money Concepts")
    return fig


def _add_marker(fig: go.Figure, df: pd.DataFrame, time_col: str, flag_col: str, price_col: str, name: str) -> None:
    if flag_col not in df.columns or price_col not in df.columns:
        return
    points = df[df[flag_col].astype(bool)]
    if points.empty:
        return
    fig.add_trace(go.Scatter(x=points[time_col], y=points[price_col], mode="markers", name=name))
