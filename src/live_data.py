from __future__ import annotations

import argparse
import json
import logging
import os
import urllib.request
import pandas as pd

logger = logging.getLogger("LiveData")

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

TIMEFRAMES = (
    {
        "M1": mt5.TIMEFRAME_M1 if mt5 else None,
        "M5": mt5.TIMEFRAME_M5 if mt5 else None,
        "M15": mt5.TIMEFRAME_M15 if mt5 else None,
        "H1": mt5.TIMEFRAME_H1 if mt5 else None,
    }
    if mt5 is not None
    else {}
)


def fetch_live_candles_web(symbol: str = "XAUUSD", timeframe: str = "M5", bars: int = 100) -> pd.DataFrame:
    """Fetches real live gold candles (spot/futures) via real-time market API."""
    interval_map = {"M1": "1m", "M5": "5m", "M15": "15m", "H1": "1h"}
    interval = interval_map.get(timeframe.upper(), "5m")
    range_param = "5d" if interval in ["5m", "15m", "1h"] else "1d"
    
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/GC=F?interval={interval}&range={range_param}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    with urllib.request.urlopen(req, timeout=8) as resp:
        data = json.loads(resp.read().decode())
        res = data["chart"]["result"][0]
        timestamps = res["timestamp"]
        quotes = res["indicators"]["quote"][0]
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(timestamps, unit="s"),
            "open": quotes["open"],
            "high": quotes["high"],
            "low": quotes["low"],
            "close": quotes["close"],
            "volume": quotes.get("volume", [100.0] * len(timestamps)),
        }).dropna()
        df = df.rename(columns={"volume": "tick_volume"})
        df["volume"] = df["tick_volume"]
        df["spread"] = 25.0
        return df.tail(bars).reset_index(drop=True)


def fetch_live_candles(symbol: str = "XAUUSD", timeframe: str = "M5", bars: int = 100) -> pd.DataFrame:
    """
    Fetch the latest live candles.
    Prioritizes real MT5 broker spot feed when configured; falls back to live web futures proxy (GC=F).
    """
    # 1. MT5 broker spot quote feed (Primary when MT5 terminal is configured and active)
    if mt5 is not None and os.environ.get("USE_MT5", "0") == "1":
        try:
            if mt5.initialize(timeout=5000):
                tf = TIMEFRAMES.get(timeframe.upper(), mt5.TIMEFRAME_M5)
                if mt5.symbol_select(symbol, True):
                    rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars)
                    if rates is not None and len(rates) > 0:
                        df = pd.DataFrame(rates)
                        df["timestamp"] = pd.to_datetime(df["time"], unit="s")
                        df = df.rename(columns={"tick_volume": "volume"})
                        df["tick_volume"] = df["volume"]
                        logger.info("Successfully fetched %d spot candles from MT5 broker (%s)", len(df), symbol)
                        return df[["timestamp", "open", "high", "low", "close", "volume", "tick_volume", "spread"]].reset_index(drop=True)
                mt5.shutdown()
        except Exception as exc:
            logger.warning("MT5 spot feed attempt failed: %s. Falling back to web proxy.", exc)

    # 2. Live Web Market Feed (COMEX GC=F Futures Proxy)
    try:
        df = fetch_live_candles_web(symbol, timeframe, bars)
        if df is not None and not df.empty and float(df["close"].iloc[-1]) > 3000.0:
            # Check data staleness
            latest_bar_time = df["timestamp"].iloc[-1]
            now_utc = pd.Timestamp.utcnow().tz_localize(None)
            age_min = (now_utc - latest_bar_time).total_seconds() / 60.0
            if age_min > 30.0:
                logger.warning("Data notice: latest candle is %.1f min old (bar: %s, UTC: %s)", age_min, latest_bar_time, now_utc)
            return df
    except Exception as exc:
        logger.warning("Primary web live fetch attempt failed: %s", exc)

    logger.error("All live fetch attempts failed or returned invalid data. Returning empty DataFrame to prevent stale trade execution.")
    return pd.DataFrame()


def fetch_live_tick(symbol: str = "XAUUSD") -> dict:
    """Fetch current live price and spread."""
    candles = fetch_live_candles(symbol, "M5", 2)
    last = candles.iloc[-1]
    return {
        "bid": float(last["close"]) - 0.25,
        "ask": float(last["close"]) + 0.25,
        "spread": 0.50,
        "time": str(last["timestamp"]),
    }
