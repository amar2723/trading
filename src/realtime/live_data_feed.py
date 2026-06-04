from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from src.realtime.mt5_connector import MT5Connector


logger = logging.getLogger(__name__)


@dataclass
class LiveDataFeed:
    connector: MT5Connector
    symbol: str = "XAUUSD"
    timeframes: tuple[str, ...] = ("M1", "M5", "M15", "H1")
    bars: int = 500
    latest_candles: dict[str, pd.DataFrame] = field(default_factory=dict)
    last_closed_time: dict[str, pd.Timestamp] = field(default_factory=dict)

    def refresh(self) -> dict[str, bool]:
        """Fetch latest candles and return which timeframes have a new closed candle."""
        updates: dict[str, bool] = {}
        for timeframe in self.timeframes:
            data = self.connector.get_rates(self.symbol, timeframe, self.bars)
            self.latest_candles[timeframe] = data
            closed_time = self._closed_candle_time(data)
            updates[timeframe] = closed_time is not None and self.last_closed_time.get(timeframe) != closed_time
            if updates[timeframe]:
                self.last_closed_time[timeframe] = closed_time
                logger.info("New %s candle closed at %s", timeframe, closed_time)
        return updates

    def get_latest(self, timeframe: str) -> pd.DataFrame:
        if timeframe not in self.latest_candles:
            self.refresh()
        return self.latest_candles[timeframe]

    @staticmethod
    def _closed_candle_time(df: pd.DataFrame) -> pd.Timestamp | None:
        if df.empty:
            return None
        if len(df) >= 2:
            return pd.Timestamp(df.iloc[-2]["timestamp"])
        return pd.Timestamp(df.iloc[-1]["timestamp"])
