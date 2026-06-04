from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from src.data_ingestion.data_validator import REQUIRED_COLUMNS, clean_data, validate_data
from src.data_ingestion.logging_utils import get_logger


TIMEFRAME_MAP = {
    "M1": "TIMEFRAME_M1",
    "M5": "TIMEFRAME_M5",
    "M15": "TIMEFRAME_M15",
}


def _parse_date(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value)


def _load_mt5():
    try:
        import MetaTrader5 as mt5
    except ImportError as exc:
        raise RuntimeError("MetaTrader5 package is required for MT5 data collection.") from exc
    return mt5


def get_historical_data(symbol: str, timeframe: str, start_date: str | datetime, end_date: str | datetime) -> pd.DataFrame:
    logger = get_logger(__name__)
    mt5 = _load_mt5()
    timeframe_key = timeframe.upper()

    if timeframe_key not in TIMEFRAME_MAP:
        raise ValueError(f"Unsupported timeframe '{timeframe}'. Use one of: {', '.join(TIMEFRAME_MAP)}")

    start = _parse_date(start_date)
    end = _parse_date(end_date)
    if start >= end:
        raise ValueError("start_date must be earlier than end_date")

    logger.info("Initializing MT5 for %s %s from %s to %s", symbol, timeframe_key, start, end)
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")

    try:
        if not mt5.symbol_select(symbol, True):
            raise RuntimeError(f"Could not select symbol {symbol}: {mt5.last_error()}")

        rates = mt5.copy_rates_range(symbol, getattr(mt5, TIMEFRAME_MAP[timeframe_key]), start, end)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"No historical data returned for {symbol} {timeframe_key}: {mt5.last_error()}")

        df = pd.DataFrame(rates)
        df["timestamp"] = pd.to_datetime(df["time"], unit="s")
        df = df.rename(columns={"tick_volume": "tick_volume"})
        keep = ["timestamp", "open", "high", "low", "close", "tick_volume", "spread"]
        cleaned = clean_data(df[keep])
        report = validate_data(cleaned)
        if not report.is_valid:
            logger.warning("MT5 data validation issues: %s", report.to_dict())
        return cleaned[REQUIRED_COLUMNS]
    finally:
        mt5.shutdown()
        logger.info("MT5 shutdown complete")


def save_raw_data(df: pd.DataFrame, symbol: str, timeframe: str, output_dir: str | Path = "data/raw") -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    file_path = output_path / f"{symbol}_{timeframe.upper()}.csv"
    df.to_csv(file_path, index=False)
    get_logger(__name__).info("Saved %s rows to %s", len(df), file_path)
    return file_path
