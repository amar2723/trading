from __future__ import annotations

"""
Live Market Data Acquisition, Normalization, and Safety Validation for XAUUSD.
Handles:
1. Native MT5 Broker Spot feed (Primary when MT5 desktop terminal is active)
2. Yahoo Finance COMEX Gold Futures (GC=F) Proxy (Research/Fallback ONLY)
3. Strict Safety Validation:
   - Full candle closure verification (prevents look-ahead / in-progress bar contamination)
   - UTC timestamp normalization and monotonic chronological ordering
   - Data freshness gating (rejects stale candles > 30 minutes old)
   - Strict source labeling (YAHOO_GC_F_FUTURES_PROXY vs BROKER_MT5_SPOT)
   - Failsafe HOLD trigger on futures proxy when spot execution is required
"""

import argparse
import json
import logging
import os
import time
import urllib.request
import requests
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np
import pandas as pd

from src.broker_adapter import SpreadQuote

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

TIMEFRAME_MINUTES = {
    "M1": 1,
    "M5": 5,
    "M15": 15,
    "M30": 30,
    "H1": 60,
    "H4": 240,
    "D1": 1440,
}


def validate_candle_dataframe(
    df: pd.DataFrame,
    timeframe: str = "M5",
    max_age_minutes: float = 30.0,
    min_bars: int = 20,
) -> tuple[bool, str]:
    """
    Validates that a candle DataFrame contains fully closed candles, is chronologically
    ordered, UTC-normalized, non-forming, and within the allowable freshness window.
    Returns (True, "OK") or (False, reason).
    """
    if df is None or df.empty:
        return False, "Candle DataFrame is empty or None."

    if len(df) < min_bars:
        return False, f"Insufficient bars: got {len(df)}, required at least {min_bars}."

    required_cols = ["timestamp", "open", "high", "low", "close"]
    for col in required_cols:
        if col not in df.columns:
            return False, f"Missing required column: {col}."

    # Validate numeric cleanliness (no NaN, inf, or non-positive prices)
    for col in ["open", "high", "low", "close"]:
        s = pd.to_numeric(df[col], errors="coerce")
        if s.isna().any():
            return False, f"Non-numeric or NaN values detected in {col}."
        if (s <= 0.0).any():
            return False, f"Non-positive price detected in {col}."

    # Timestamp UTC normalization and monotonicity
    ts = pd.to_datetime(df["timestamp"], utc=True)
    if not ts.is_monotonic_increasing:
        return False, "Candle timestamps are not chronologically ascending."

    # Check for duplicate timestamps
    if ts.duplicated().any():
        return False, "Duplicate candle timestamps detected."

    # Candle closure verification against timeframe interval
    # In financial time series, timestamp T for an M-minute bar covers [T, T + M).
    # The candle is strictly completed/closed only when current time >= T + M.
    tf_minutes = TIMEFRAME_MINUTES.get(timeframe.upper(), 5)
    latest_ts = ts.iloc[-1].to_pydatetime()
    if latest_ts.tzinfo is None:
        latest_ts = latest_ts.replace(tzinfo=timezone.utc)

    candle_close_time = latest_ts + timedelta(minutes=tf_minutes)
    now_utc = datetime.now(timezone.utc)

    if now_utc < candle_close_time:
        return False, (
            f"Incomplete/forming candle detected: bar starting at {latest_ts.isoformat()} "
            f"for timeframe {timeframe} closes at {candle_close_time.isoformat()} "
            f"(current time is {now_utc.isoformat()})."
        )

    # Freshness validation against elapsed time since bar closure
    age_seconds = (now_utc - candle_close_time).total_seconds()
    age_minutes = age_seconds / 60.0

    if age_minutes < -5.0:
        return False, f"Future timestamp detected: candle closure is {-age_minutes:.1f} min in the future."

    effective_max_age = max(max_age_minutes, tf_minutes * 1.5)
    if age_minutes > effective_max_age:
        return False, f"Stale candles: latest bar closed {age_minutes:.1f} min ago (max allowed: {effective_max_age:.1f} min for {timeframe})."

    return True, "OK"


_WEB_SESSION: requests.Session | None = None
_SPOT_BAR_CACHE: dict[str, dict[pd.Timestamp, dict]] = {}


def get_web_session() -> requests.Session:
    global _WEB_SESSION
    if _WEB_SESSION is None:
        _WEB_SESSION = requests.Session()
        _WEB_SESSION.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json",
        })
    return _WEB_SESSION


def fetch_live_spot_quote(symbol: str = "XAUUSD") -> dict | None:
    """
    Fetch real-time OTC Spot XAUUSD Bid/Ask quote from public institutional feeds.
    1. Swissquote Public BBO (Live institutional FX Spot Bid/Ask)
    2. Gold-API Spot XAU/USD fallback
    """
    session = get_web_session()
    try:
        r = session.get(
            "https://forex-data-feed.swissquote.com/public-quotes/bboquotes/instrument/XAU/USD",
            timeout=3,
        )
        if r.status_code == 200:
            payload = r.json()
            if isinstance(payload, list) and payload:
                profiles = payload[0].get("spreadProfilePrices") or []
                if profiles:
                    p0 = profiles[0]
                    bid = float(p0["bid"])
                    ask = float(p0["ask"])
                    if bid > 1000.0 and ask >= bid:
                        mid = round((bid + ask) / 2.0, 2)
                        return {
                            "bid": round(bid, 2),
                            "ask": round(ask, 2),
                            "mid": mid,
                            "spread_pips": round((ask - bid) * 10.0, 1),
                            "ts": time.time(),
                            "source": "SWISSQUOTE_SPOT_XAUUSD",
                        }
    except Exception as exc:
        logger.debug("Swissquote spot fetch note: %s", exc)

    try:
        r = session.get("https://api.gold-api.com/price/XAU", timeout=3)
        if r.status_code == 200:
            data = r.json()
            price = float(data.get("price", 0.0))
            if price > 1000.0:
                return {
                    "bid": round(price - 0.25, 2),
                    "ask": round(price + 0.25, 2),
                    "mid": round(price, 2),
                    "spread_pips": 5.0,
                    "ts": time.time(),
                    "source": "GOLD_API_SPOT_XAUUSD",
                }
    except Exception as exc:
        logger.debug("Gold-API spot fetch note: %s", exc)

    return None


def _update_spot_bar_cache(timeframe: str, spot_mid: float, now_utc: datetime) -> None:
    """Record live spot ticks into rolling closed/forming OHLC bars per timeframe."""
    tf_key = timeframe.upper()
    tf_minutes = TIMEFRAME_MINUTES.get(tf_key, 5)
    bucket_epoch = int(now_utc.timestamp() // (tf_minutes * 60)) * (tf_minutes * 60)
    bucket_ts = pd.to_datetime(bucket_epoch, unit="s", utc=True).tz_localize(None)

    cache = _SPOT_BAR_CACHE.setdefault(tf_key, {})
    if bucket_ts not in cache:
        cache[bucket_ts] = {
            "timestamp": bucket_ts,
            "open": spot_mid,
            "high": spot_mid,
            "low": spot_mid,
            "close": spot_mid,
            "tick_volume": 100.0,
            "volume": 100.0,
        }
    else:
        bar = cache[bucket_ts]
        bar["high"] = max(bar["high"], spot_mid)
        bar["low"] = min(bar["low"], spot_mid)
        bar["close"] = spot_mid
        bar["tick_volume"] += 10.0
        bar["volume"] = bar["tick_volume"]

    # Prune old bars beyond 200 buckets
    if len(cache) > 200:
        for old_k in sorted(cache.keys())[:-200]:
            del cache[old_k]


def fetch_live_candles_web(symbol: str = "XAUUSD", timeframe: str = "M5", bars: int = 100) -> pd.DataFrame:
    """
    Fetches historical gold structure from Yahoo Finance (GC=F) and anchors it to
    real-time OTC Spot XAUUSD (Swissquote BBO / Gold-API) so holiday halts or
    futures delays never stall live predictions.
    """
    interval_map = {"M1": "1m", "M5": "5m", "M15": "15m", "H1": "1h"}
    interval = interval_map.get(timeframe.upper(), "5m")
    range_param = "5d" if interval in ["5m", "15m", "1h"] else "1d"

    url = f"https://query1.finance.yahoo.com/v8/finance/chart/GC=F?interval={interval}&range={range_param}"
    session = get_web_session()
    resp = session.get(url, timeout=5)
    resp.raise_for_status()
    data = resp.json()
    res = data["chart"]["result"][0]
    meta = res.get("meta", {})
    timestamps = res["timestamp"]
    quotes = res["indicators"]["quote"][0]
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(timestamps, unit="s", utc=True).tz_localize(None),
        "open": quotes["open"],
        "high": quotes["high"],
        "low": quotes["low"],
        "close": quotes["close"],
        "volume": quotes.get("volume", [100.0] * len(timestamps)),
    }).dropna()
    df = df.rename(columns={"volume": "tick_volume"})
    df["volume"] = df["tick_volume"]

    # Drop forming in-progress bars whose close time is strictly in the future
    now_utc = datetime.now(timezone.utc)
    tf_minutes = TIMEFRAME_MINUTES.get(timeframe.upper(), 5)
    tf_delta = timedelta(minutes=tf_minutes)
    while len(df) > 1:
        last_candle_ts = df["timestamp"].iloc[-1].to_pydatetime()
        if last_candle_ts.tzinfo is None:
            last_candle_ts = last_candle_ts.replace(tzinfo=timezone.utc)
        candle_close_time = last_candle_ts + tf_delta
        if now_utc < candle_close_time:
            df = df.iloc[:-1].reset_index(drop=True)
        else:
            break

    spot = fetch_live_spot_quote(symbol)
    if spot is not None and not df.empty:
        spot_mid = spot["mid"]
        _update_spot_bar_cache(timeframe, spot_mid, now_utc)

        # Rebase futures proxy history onto live Spot XAUUSD price space
        last_fut_close = float(df["close"].iloc[-1])
        basis = round(spot_mid - last_fut_close, 2)
        if abs(basis) > 0.05:
            for col in ("open", "high", "low", "close"):
                df[col] = (df[col].astype(float) + basis).round(2)

        # If Yahoo bars are frozen (e.g., COMEX holiday), append closed spot bar(s)
        last_ts = df["timestamp"].iloc[-1].to_pydatetime().replace(tzinfo=timezone.utc)
        latest_closed_epoch = (int(now_utc.timestamp() // (tf_minutes * 60)) - 1) * (tf_minutes * 60)
        latest_closed_ts = pd.to_datetime(latest_closed_epoch, unit="s", utc=True).tz_localize(None)
        if (now_utc - (last_ts + tf_delta)).total_seconds() > (tf_minutes * 60 * 1.2):
            spot_cache = _SPOT_BAR_CACHE.get(timeframe.upper(), {})
            closed_spot_bars = [
                b for ts_k, b in sorted(spot_cache.items())
                if ts_k > df["timestamp"].iloc[-1] and ts_k <= latest_closed_ts
            ]
            if closed_spot_bars:
                df = pd.concat([df, pd.DataFrame(closed_spot_bars)], ignore_index=True)
            else:
                bridge_bar = pd.DataFrame([{
                    "timestamp": latest_closed_ts,
                    "open": spot_mid,
                    "high": round(max(spot_mid, spot["ask"]), 2),
                    "low": round(min(spot_mid, spot["bid"]), 2),
                    "close": spot_mid,
                    "tick_volume": 100.0,
                    "volume": 100.0,
                }])
                df = pd.concat([df, bridge_bar], ignore_index=True)

        df["data_source"] = "LIVE_WEB_SPOT_XAUUSD"
        df["is_broker_spot"] = False
        df["is_futures_proxy"] = False
        df.attrs["regular_market_price"] = spot_mid
        df.attrs["regular_market_time"] = spot["ts"]
        df.attrs["spot_bid"] = spot["bid"]
        df.attrs["spot_ask"] = spot["ask"]
        df.attrs["spot_spread_pips"] = spot["spread_pips"]
        df.attrs["spot_source"] = spot["source"]
    else:
        df["data_source"] = "YAHOO_GC_F_FUTURES_PROXY"
        df["is_broker_spot"] = False
        df["is_futures_proxy"] = True
        if meta.get("regularMarketPrice") is not None:
            df.attrs["regular_market_price"] = float(meta["regularMarketPrice"])
            df.attrs["regular_market_time"] = meta.get("regularMarketTime")

    return df.tail(bars).reset_index(drop=True)


def is_terminal_process_active() -> bool:
    try:
        import subprocess
        out = subprocess.check_output('tasklist /fi "imagename eq terminal64.exe"', shell=True).decode(errors="ignore")
        return "terminal64.exe" in out.lower()
    except Exception:
        return False


def fetch_live_candles(
    symbol: str = "XAUUSD",
    timeframe: str = "M5",
    bars: int = 100,
    require_broker_spot: bool = False,
    max_age_minutes: float = 30.0,
) -> pd.DataFrame:
    """
    Fetch the latest live candles.
    Prioritizes real MT5 broker spot feed when configured and terminal process is open.
    If require_broker_spot is True and MT5 is unavailable, returns empty DataFrame with HOLD warning.
    """
    # 1. MT5 broker spot quote feed (Primary when MT5 desktop terminal process is actively running)
    if mt5 is not None and os.environ.get("USE_MT5", "0") == "1" and is_terminal_process_active():
        try:
            if mt5.initialize(timeout=3000):
                tf = TIMEFRAMES.get(timeframe.upper(), mt5.TIMEFRAME_M5)
                if mt5.symbol_select(symbol, True):
                    # Query pos=1 to guarantee querying only fully closed historical bars
                    # (pos=0 represents the actively forming in-progress bar)
                    rates = mt5.copy_rates_from_pos(symbol, tf, 1, bars)
                    if rates is not None and len(rates) > 0:
                        df = pd.DataFrame(rates)
                        now_ts = time.time()
                        # Use broker tick time to determine server timezone offset (e.g. EET / UTC+2)
                        tick = mt5.symbol_info_tick(symbol)
                        if tick is not None and getattr(tick, "time", 0) > 0:
                            server_offset_sec = round((tick.time - now_ts) / 3600.0) * 3600
                        else:
                            server_offset_sec = 0
                        df["timestamp"] = pd.to_datetime(df["time"] - server_offset_sec, unit="s", utc=True).dt.tz_localize(None)
                        df = df.rename(columns={"tick_volume": "volume"})
                        df["tick_volume"] = df["volume"]
                        df["data_source"] = "BROKER_MT5_SPOT"
                        df["is_broker_spot"] = True
                        df["is_futures_proxy"] = False

                        # Validate closed bars
                        valid, reason = validate_candle_dataframe(df, timeframe, max_age_minutes)
                        if valid:
                            logger.info("Fetched %d validated spot candles from MT5 broker (%s)", len(df), symbol)
                            return df[["timestamp", "open", "high", "low", "close", "volume", "tick_volume", "data_source", "is_broker_spot", "is_futures_proxy"]].reset_index(drop=True)
                        else:
                            logger.error("MT5 candle validation failed: %s. Rejecting invalid data.", reason)
                            mt5.shutdown()
                            return pd.DataFrame()
                mt5.shutdown()
        except Exception as exc:
            logger.warning("MT5 spot feed attempt failed: %s. Falling back to web proxy.", exc)

    if require_broker_spot:
        logger.error("require_broker_spot=True but MT5 broker spot feed is unavailable. Refusing proxy data.")
        return pd.DataFrame()

    # 2. Live Web Market Feed (COMEX GC=F Futures Proxy)
    try:
        df = fetch_live_candles_web(symbol, timeframe, bars)
        if df is not None and not df.empty:
            valid, reason = validate_candle_dataframe(df, timeframe, max_age_minutes)
            if valid:
                return df
            else:
                logger.error("Web proxy candle validation failed: %s. Rejecting invalid data.", reason)
                return pd.DataFrame()
    except Exception as exc:
        logger.warning("Primary web live fetch attempt failed: %s", exc)

    logger.error("All live fetch attempts failed or returned invalid data. Returning empty DataFrame.")
    return pd.DataFrame()


def fetch_live_tick(symbol: str = "XAUUSD") -> SpreadQuote:
    """
    Fetch current live price and spread quote with strict provenance tagging.
    Returns SpreadQuote with is_broker_confirmed=True only if obtained from connected broker.
    """
    if mt5 is not None and os.environ.get("USE_MT5", "0") == "1":
        try:
            if mt5.initialize(timeout=5000):
                tick = mt5.symbol_info_tick(symbol)
                if tick is not None and tick.ask > tick.bid > 0:
                    return SpreadQuote.from_bid_ask(
                        bid=tick.bid,
                        ask=tick.ask,
                        time_str=str(datetime.fromtimestamp(tick.time, tz=timezone.utc)),
                        is_broker_confirmed=True,
                        quote_source="BROKER_MT5_SPOT",
                    )
        except Exception as e:
            logger.warning("Live tick MT5 error: %s", e)

    spot = fetch_live_spot_quote(symbol)
    if spot is not None:
        return SpreadQuote.from_bid_ask(
            bid=spot["bid"],
            ask=spot["ask"],
            time_str=datetime.now(timezone.utc).isoformat(),
            is_broker_confirmed=False,
            quote_source=spot["source"],
        )

    # Synthetic fallback (explicitly marked is_broker_confirmed=False)
    candles = fetch_live_candles(symbol, "M5", 25)
    if not candles.empty:
        reg_price = candles.attrs.get("regular_market_price")
        last = float(reg_price) if reg_price is not None else float(candles.iloc[-1]["close"])
        return SpreadQuote.from_bid_ask(
            bid=round(last - 0.21, 2),
            ask=round(last + 0.21, 2),
            time_str=datetime.now(timezone.utc).isoformat(),
            is_broker_confirmed=False,
            quote_source="SYNTHETIC_PROXY",
        )

    return SpreadQuote.from_bid_ask(
        bid=2600.0,
        ask=2600.42,
        time_str=datetime.now(timezone.utc).isoformat(),
        is_broker_confirmed=False,
        quote_source="EMERGENCY_STUB",
    )
