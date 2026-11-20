from __future__ import annotations

"""
Shared Feature Engineering and Multi-Timeframe Alignment Pipeline for XAUUSD.
Guarantees 100% mathematical consistency across:
1. Historical dataset generation and feature extraction
2. Purged walk-forward model training
3. Event-driven backtesting
4. Autonomous live inference

Features:
- side_code: +1 (BUY) or -1 (SELL)
- is_pullback: 1 if pullback strategy, 0 otherwise
- is_donchian: 1 if Donchian breakout, 0 otherwise
- htf_trend_bias: +1 (bullish) or -1 (bearish) from completed H1 bars (or documented M5 proxy)
- h1_dist_ema50: % distance between price and H1 EMA 50
- dist_ema_20: % distance between price and M5 EMA 20
- dist_ema_50: % distance between price and M5 EMA 50
- rsi: 14-period Relative Strength Index
- atr_pct: 14-period ATR as % of price
- buyer_pct: 14-period causal signed tick volume ratio (or neutral 50.0 if volume unavailable)
- is_prime_session: 1 if London (07-10 UTC) or NY (12-16 UTC), 0 otherwise
- is_asian_chop: 1 if Asian session (00-06 UTC), 0 otherwise
"""

import logging
from dataclasses import dataclass
from typing import Literal
import numpy as np
import pandas as pd

logger = logging.getLogger("FeaturePipeline")

FEATURE_COLUMNS = [
    "side_code",
    "is_pullback",
    "is_donchian",
    "htf_trend_bias",
    "h1_dist_ema50",
    "dist_ema_20",
    "dist_ema_50",
    "rsi",
    "atr_pct",
    "buyer_pct",
    "is_prime_session",
    "is_asian_chop",
]


def compute_indicator_dataframe(
    df_m5: pd.DataFrame,
    df_h1: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Computes all technical indicators causally on the full M5 DataFrame.
    If df_h1 is provided, aligns completed H1 candles causally using merge_asof.
    If df_h1 is None, uses an explicit 240-period M5 EMA proxy (labeled M5_PROXY_240).
    """
    df = df_m5.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)

    # Base M5 Indicators
    df["ema_20"] = df["close"].ewm(span=20).mean()
    df["ema_50"] = df["close"].ewm(span=50).mean()
    df["donch_h20"] = df["high"].rolling(20).max().shift(1)
    df["donch_l20"] = df["low"].rolling(20).min().shift(1)

    # ATR (14-period True Range)
    tr = np.maximum(
        df["high"] - df["low"],
        np.maximum(np.abs(df["high"] - df["close"].shift(1)), np.abs(df["low"] - df["close"].shift(1)))
    )
    df["atr"] = tr.rolling(14).mean()

    # RSI (14-period)
    diffs = df["close"].diff()
    gain = diffs.clip(lower=0).rolling(14).mean()
    loss = (-diffs.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100.0 - (100.0 / (1.0 + (gain / (loss + 1e-9))))

    # Causal Signed Tick Volume Ratio (14-bar rolling)
    vol_col = "tick_volume" if "tick_volume" in df.columns else ("volume" if "volume" in df.columns else None)
    if vol_col and df[vol_col].sum() > 0:
        vol = df[vol_col].values
        close = df["close"].values
        n = len(df)
        buyer_pct_arr = np.full(n, 50.0)
        for idx in range(15, n):
            p_diff = np.diff(close[idx - 14 : idx + 1])
            v_deltas = vol[idx - 13 : idx + 1]
            b_vol = np.sum(v_deltas[p_diff >= 0])
            s_vol = np.sum(v_deltas[p_diff < 0])
            tot = b_vol + s_vol
            buyer_pct_arr[idx] = (b_vol / tot * 100.0) if tot > 0 else 50.0
        df["buyer_pct"] = buyer_pct_arr
    else:
        df["buyer_pct"] = 50.0

    # Multi-Timeframe H1 Macro Trend Alignment
    if df_h1 is not None and len(df_h1) >= 2:
        h1 = df_h1.copy()
        if "timestamp" in h1.columns:
            h1["timestamp"] = pd.to_datetime(h1["timestamp"])
            h1 = h1.sort_values("timestamp").reset_index(drop=True)

        # An H1 bar timestamp T represents interval [T, T + 1 hour).
        # It is strictly completed at T + 1 hour.
        h1["h1_completed_at"] = h1["timestamp"] + pd.Timedelta(hours=1)
        h1["h1_close"] = h1["close"].astype(float)
        h1["h1_ema20"] = h1["h1_close"].ewm(span=20).mean()
        h1["h1_ema50"] = h1["h1_close"].ewm(span=50).mean()
        h1["h1_trend_bias"] = np.where(h1["h1_close"] > h1["h1_ema50"], 1, -1)

        # Drop prior H1 columns if present to avoid _x / _y collisions during merge_asof
        cols_to_drop = [c for c in ["h1_trend_bias", "h1_dist_ema50", "h1_close", "h1_completed_at", "h1_ema20", "h1_ema50"] if c in df.columns]
        df_clean = df.drop(columns=cols_to_drop) if cols_to_drop else df

        # Causal as-of merge (guarantees zero future leakage from incomplete H1 bars)
        merged = pd.merge_asof(
            df_clean,
            h1[["h1_completed_at", "h1_close", "h1_ema20", "h1_ema50", "h1_trend_bias"]],
            left_on="timestamp",
            right_on="h1_completed_at",
            direction="backward",
        )
        df["h1_trend_bias"] = merged["h1_trend_bias"].fillna(1).astype(int)
        df["h1_dist_ema50"] = ((merged["close"] - merged["h1_ema50"]) / merged["h1_ema50"] * 100.0).fillna(0.0)
        df["h1_close"] = merged["h1_close"]
        df["h1_completed_at"] = merged["h1_completed_at"]

        # Preserve provenance identity of H1 source
        is_futures = False
        if "is_futures_proxy" in h1.columns and bool(h1["is_futures_proxy"].iloc[-1]):
            is_futures = True
        elif "data_source" in h1.columns and any("FUTURES" in str(x).upper() for x in h1["data_source"]):
            is_futures = True
        elif "is_broker_spot" in h1.columns and not bool(h1["is_broker_spot"].iloc[-1]):
            is_futures = True

        df["h1_source"] = "FUTURES_H1_MERGE_ASOF" if is_futures else "BROKER_H1_MERGE_ASOF"
    elif "h1_trend_bias" in df.columns and "h1_dist_ema50" in df.columns:
        # Preserve previously merged / computed causal H1 features (do not overwrite with M5 proxy)
        if "h1_source" not in df.columns:
            df["h1_source"] = "PRESERVED_EXISTING_H1"
    else:
        # Documented M5 Proxy (240-period EMA = 20 hours)
        h1_trend = df["close"].ewm(span=240).mean()
        df["h1_trend_bias"] = np.where(df["close"] > h1_trend, 1, -1)
        df["h1_dist_ema50"] = (df["close"] - h1_trend) / h1_trend * 100.0
        df["h1_source"] = "M5_PROXY_240_EMA"

    # Percentage distances & Session flags
    df["dist_ema_20"] = (df["close"] - df["ema_20"]) / df["ema_20"] * 100.0
    df["dist_ema_50"] = (df["close"] - df["ema_50"]) / df["ema_50"] * 100.0
    df["atr_pct"] = (df["atr"] / df["close"]) * 100.0

    hours = df["timestamp"].dt.hour
    df["is_prime_session"] = np.where((hours.between(7, 10)) | (hours.between(12, 16)), 1, 0)
    df["is_asian_chop"] = np.where(hours.between(0, 6), 1, 0)

    return df


def extract_features_from_row(
    row: pd.Series | dict,
    action: str,
    strategy_name: str,
) -> pd.DataFrame:
    """
    Extracts the exact standardized feature vector matching FEATURE_COLUMNS
    for a single candidate signal row.
    """
    side_code = 1 if action == "BUY" else (-1 if action == "SELL" else 0)
    is_pullback = 1 if "PULLBACK" in strategy_name.upper() else 0
    is_donchian = 1 if "DONCHIAN" in strategy_name.upper() else 0

    feat_dict = {
        "side_code": float(side_code),
        "is_pullback": float(is_pullback),
        "is_donchian": float(is_donchian),
        "htf_trend_bias": float(row.get("h1_trend_bias", 1)),
        "h1_dist_ema50": float(row.get("h1_dist_ema50", 0.0)),
        "dist_ema_20": float(row.get("dist_ema_20", 0.0)),
        "dist_ema_50": float(row.get("dist_ema_50", 0.0)),
        "rsi": float(row.get("rsi", 50.0)),
        "atr_pct": float(row.get("atr_pct", 0.15)),
        "buyer_pct": float(row.get("buyer_pct", 50.0)),
        "is_prime_session": float(row.get("is_prime_session", 0)),
        "is_asian_chop": float(row.get("is_asian_chop", 0)),
    }
    return pd.DataFrame([feat_dict], columns=FEATURE_COLUMNS)
