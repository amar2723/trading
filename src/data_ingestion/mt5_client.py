from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd


TIMEFRAME_MAP = {
    "M1": "TIMEFRAME_M1",
    "M5": "TIMEFRAME_M5",
    "M15": "TIMEFRAME_M15",
}


@dataclass
class MT5Client:
    symbol: str = "XAUUSD"

    def _mt5(self):
        try:
            import MetaTrader5 as mt5
        except ImportError as exc:
            raise RuntimeError("Install MetaTrader5 and run on a machine with the MT5 terminal.") from exc
        return mt5

    def initialize(self) -> None:
        mt5 = self._mt5()
        if not mt5.initialize():
            raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
        if not mt5.symbol_select(self.symbol, True):
            raise RuntimeError(f"Could not select symbol {self.symbol}: {mt5.last_error()}")

    def shutdown(self) -> None:
        self._mt5().shutdown()

    def fetch_ohlcv(self, timeframe: str, bars: int = 3000) -> pd.DataFrame:
        mt5 = self._mt5()
        tf_attr = TIMEFRAME_MAP.get(timeframe.upper())
        if tf_attr is None:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        rates = mt5.copy_rates_from_pos(self.symbol, getattr(mt5, tf_attr), 0, bars)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"No rates returned for {self.symbol} {timeframe}: {mt5.last_error()}")
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df = df.rename(columns={"tick_volume": "volume"})
        keep = ["time", "open", "high", "low", "close", "volume", "spread", "real_volume"]
        return df[[c for c in keep if c in df.columns]].sort_values("time").reset_index(drop=True)

    def latest_tick(self) -> dict:
        tick = self._mt5().symbol_info_tick(self.symbol)
        if tick is None:
            return {}
        return tick._asdict()


def save_ohlcv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)


def load_ohlcv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
    return df
