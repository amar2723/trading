from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

import pandas as pd


logger = logging.getLogger(__name__)


TIMEFRAME_MAP = {
    "M1": "TIMEFRAME_M1",
    "M5": "TIMEFRAME_M5",
    "M15": "TIMEFRAME_M15",
    "H1": "TIMEFRAME_H1",
}


@dataclass
class MT5LoginConfig:
    login_id: int | None = None
    password: str | None = None
    server: str | None = None


class MT5Connector:
    """Thin MetaTrader5 wrapper with explicit connection lifecycle."""

    def __init__(self):
        self.mt5 = None

    def _load_mt5(self):
        if self.mt5 is not None:
            return self.mt5
        try:
            import MetaTrader5 as mt5
        except ImportError as exc:
            raise RuntimeError("MetaTrader5 package is required for realtime trading.") from exc
        self.mt5 = mt5
        return self.mt5

    def initialize_mt5(self) -> bool:
        mt5 = self._load_mt5()
        ok = bool(mt5.initialize())
        if not ok:
            logger.error("MT5 initialize failed: %s", mt5.last_error())
        return ok

    def login(self, config: MT5LoginConfig) -> bool:
        mt5 = self._load_mt5()
        if config.login_id is None:
            return True
        ok = bool(mt5.login(config.login_id, password=config.password, server=config.server))
        if not ok:
            logger.error("MT5 login failed: %s", mt5.last_error())
        return ok

    def shutdown(self) -> None:
        if self.mt5 is not None:
            self.mt5.shutdown()

    def verify_connection(self) -> bool:
        mt5 = self._load_mt5()
        info = mt5.terminal_info()
        account = mt5.account_info()
        return info is not None and account is not None

    def select_symbol(self, symbol: str) -> bool:
        mt5 = self._load_mt5()
        return bool(mt5.symbol_select(symbol, True))

    def get_rates(self, symbol: str, timeframe: str, bars: int = 500) -> pd.DataFrame:
        mt5 = self._load_mt5()
        key = timeframe.upper()
        if key not in TIMEFRAME_MAP:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        rates = mt5.copy_rates_from_pos(symbol, getattr(mt5, TIMEFRAME_MAP[key]), 0, bars)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"No MT5 rates returned for {symbol} {key}: {mt5.last_error()}")
        data = pd.DataFrame(rates)
        data["timestamp"] = pd.to_datetime(data["time"], unit="s")
        if "tick_volume" not in data.columns and "volume" in data.columns:
            data["tick_volume"] = data["volume"]
        return data[["timestamp", "open", "high", "low", "close", "tick_volume", "spread"]].sort_values("timestamp").reset_index(drop=True)

    def get_tick(self, symbol: str) -> dict:
        tick = self._load_mt5().symbol_info_tick(symbol)
        return tick._asdict() if tick is not None else {}
