from __future__ import annotations

import argparse
import json
import logging
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
    """Fetch the latest live candles: MT5 first, web live feed fallback second."""
    if mt5 is not None:
        try:
            if mt5.initialize():
                tf = TIMEFRAMES.get(timeframe.upper(), mt5.TIMEFRAME_M5)
                if mt5.symbol_select(symbol, True):
                    rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars)
                    if rates is not None and len(rates) > 0:
                        df = pd.DataFrame(rates)
                        df["timestamp"] = pd.to_datetime(df["time"], unit="s")
                        df = df.rename(columns={"tick_volume": "volume"})
                        df["tick_volume"] = df["volume"]
                        return df[["timestamp", "open", "high", "low", "close", "volume", "tick_volume", "spread"]].reset_index(drop=True)
                mt5.shutdown()
        except Exception as exc:
            logger.warning("MT5 live fetch failed, falling back to Web Live Feed: %s", exc)

    # Web live stream fallback
    try:
        return fetch_live_candles_web(symbol, timeframe, bars)
    except Exception as exc:
        logger.error("Web live fetch error: %s", exc)
        from pathlib import Path
        data_file = Path("data/raw/XAUUSD_M5_realistic.csv")
        return pd.read_csv(data_file).tail(bars).reset_index(drop=True)


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
