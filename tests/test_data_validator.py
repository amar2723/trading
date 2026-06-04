from __future__ import annotations

import pandas as pd

from src.data_ingestion.data_validator import clean_data, validate_data


def sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="5min"),
            "open": [2000.0, 2001.0, 2002.0],
            "high": [2002.0, 2003.0, 2004.0],
            "low": [1999.0, 2000.0, 2001.0],
            "close": [2001.0, 2002.0, 2003.0],
            "tick_volume": [100, 120, 130],
            "spread": [20, 20, 25],
        }
    )


def test_validate_clean_data_is_valid():
    report = validate_data(sample_frame())
    assert report.is_valid
    assert report.duplicate_rows == 0
    assert report.negative_price_rows == 0


def test_validate_detects_duplicates_and_negative_prices():
    df = pd.concat([sample_frame(), sample_frame().iloc[[1]]], ignore_index=True)
    df.loc[0, "close"] = -1
    report = validate_data(df)
    assert not report.is_valid
    assert report.duplicate_rows == 1
    assert report.negative_price_rows == 1


def test_clean_data_removes_bad_rows_and_sorts():
    df = sample_frame().iloc[::-1].copy()
    df.loc[0, "open"] = None
    cleaned = clean_data(df)
    assert len(cleaned) == 2
    assert cleaned["timestamp"].is_monotonic_increasing
