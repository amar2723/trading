from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import pandas as pd

from src.data_ingestion.logging_utils import get_logger


REQUIRED_COLUMNS = ["timestamp", "open", "high", "low", "close", "tick_volume", "spread"]
PRICE_COLUMNS = ["open", "high", "low", "close"]


@dataclass
class ValidationReport:
    rows: int
    missing_required_columns: list[str]
    duplicate_rows: int
    null_values: dict[str, int]
    negative_price_rows: int
    incorrect_timestamp_rows: int
    missing_rows: int
    is_valid: bool

    def to_dict(self) -> dict:
        return asdict(self)


def normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename_map = {
        "time": "timestamp",
        "date": "timestamp",
        "datetime": "timestamp",
        "volume": "tick_volume",
        "tickvol": "tick_volume",
    }
    out = out.rename(columns={k: v for k, v in rename_map.items() if k in out.columns})
    return out


def infer_frequency(timestamps: pd.Series) -> pd.Timedelta | None:
    sorted_ts = timestamps.dropna().sort_values()
    if len(sorted_ts) < 3:
        return None
    diffs = sorted_ts.diff().dropna()
    if diffs.empty:
        return None
    return diffs.mode().iloc[0]


def validate_data(df: pd.DataFrame, required_columns: Iterable[str] = REQUIRED_COLUMNS) -> ValidationReport:
    logger = get_logger(__name__)
    required = list(required_columns)
    data = normalize_ohlcv_columns(df)

    missing_required_columns = [column for column in required if column not in data.columns]
    if missing_required_columns:
        report = ValidationReport(
            rows=len(data),
            missing_required_columns=missing_required_columns,
            duplicate_rows=0,
            null_values={},
            negative_price_rows=0,
            incorrect_timestamp_rows=len(data),
            missing_rows=0,
            is_valid=False,
        )
        logger.warning("Validation failed: missing columns %s", missing_required_columns)
        return report

    timestamp = pd.to_datetime(data["timestamp"], errors="coerce")
    incorrect_timestamp_rows = int(timestamp.isna().sum())
    duplicate_rows = int(data.duplicated(subset=["timestamp"]).sum())
    null_values = {column: int(data[column].isna().sum()) for column in required}

    numeric_prices = data[PRICE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    negative_price_rows = int((numeric_prices <= 0).any(axis=1).sum())

    missing_rows = 0
    clean_ts = timestamp.dropna().drop_duplicates().sort_values()
    frequency = infer_frequency(clean_ts)
    if frequency is not None and len(clean_ts) >= 2:
        expected = pd.date_range(clean_ts.iloc[0], clean_ts.iloc[-1], freq=frequency)
        missing_rows = int(len(expected.difference(pd.DatetimeIndex(clean_ts))))

    is_valid = (
        not missing_required_columns
        and duplicate_rows == 0
        and incorrect_timestamp_rows == 0
        and all(value == 0 for value in null_values.values())
        and negative_price_rows == 0
    )

    report = ValidationReport(
        rows=len(data),
        missing_required_columns=missing_required_columns,
        duplicate_rows=duplicate_rows,
        null_values=null_values,
        negative_price_rows=negative_price_rows,
        incorrect_timestamp_rows=incorrect_timestamp_rows,
        missing_rows=missing_rows,
        is_valid=is_valid,
    )
    logger.info("Validation report: %s", report.to_dict())
    return report


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    data = normalize_ohlcv_columns(df)
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    for column in ["open", "high", "low", "close", "tick_volume", "spread"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=REQUIRED_COLUMNS)
    data = data[(data[PRICE_COLUMNS] > 0).all(axis=1)]
    data = data.drop_duplicates(subset=["timestamp"], keep="last")
    return data.sort_values("timestamp").reset_index(drop=True)
