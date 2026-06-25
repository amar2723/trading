from __future__ import annotations

import argparse

import pandas as pd

try:
    import MetaTrader5 as mt5
except ImportError:  # Streamlit Cloud/Linux cannot load the MT5 terminal package.
    mt5 = None

TIMEFRAMES = (
    {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "H1": mt5.TIMEFRAME_H1,
    }
    if mt5 is not None
    else {}
)


def fetch_live_candles(symbol: str = "XAUUSD", timeframe: str = "M5", bars: int = 100) -> pd.DataFrame:
    """Fetch the latest candles from MetaTrader 5."""
    if mt5 is None:
        raise RuntimeError(
            "MetaTrader5 is not available in this environment. "
            "Run locally on Windows with MT5 open, or deploy on a Windows VPS."
        )
    tf = TIMEFRAMES.get(timeframe.upper())
    if tf is None:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")

    try:
        if not mt5.symbol_select(symbol, True):
            raise RuntimeError(f"Could not select {symbol}: {mt5.last_error()}")

        rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"No data returned for {symbol}: {mt5.last_error()}")

        df = pd.DataFrame(rates)
        df["timestamp"] = pd.to_datetime(df["time"], unit="s")
        df = df.rename(columns={"tick_volume": "volume"})
        return df[["timestamp", "open", "high", "low", "close", "volume"]]
    finally:
        mt5.shutdown()


def fetch_live_tick(symbol: str = "XAUUSD") -> dict:
    """Fetch the current live tick from MetaTrader 5."""
    if mt5 is None:
        raise RuntimeError(
            "MetaTrader5 is not available in this environment. "
            "Run locally on Windows with MT5 open, or deploy on a Windows VPS."
        )
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")

    try:
        if not mt5.symbol_select(symbol, True):
            raise RuntimeError(f"Could not select {symbol}: {mt5.last_error()}")
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise RuntimeError(f"No tick returned for {symbol}: {mt5.last_error()}")
        return {
            "time": pd.to_datetime(tick.time, unit="s"),
            "bid": float(tick.bid),
            "ask": float(tick.ask),
            "last": float(tick.last),
            "volume": float(tick.volume),
        }
    finally:
        mt5.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch live MT5 gold candles.")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--timeframe", default="M5", choices=["M1", "M5", "M15", "H1"])
    parser.add_argument("--bars", type=int, default=100)
    args = parser.parse_args()

    candles = fetch_live_candles(args.symbol, args.timeframe, args.bars)
    print(candles.tail(20).to_string(index=False))


if __name__ == "__main__":
    main()
