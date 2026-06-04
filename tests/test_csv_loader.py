from __future__ import annotations

import pandas as pd

from src.data_ingestion.csv_loader import load_csv_data


def test_load_csv_data_handles_missing_values(tmp_path):
    csv_path = tmp_path / "xauusd.csv"
    pd.DataFrame(
        {
            "timestamp": ["2024-01-01 00:00:00", "2024-01-01 00:05:00"],
            "open": [2000.0, None],
            "high": [2002.0, 2003.0],
            "low": [1999.0, 2000.0],
            "close": [2001.0, 2002.0],
            "tick_volume": [100, 120],
            "spread": [20, 20],
        }
    ).to_csv(csv_path, index=False)

    loaded = load_csv_data(csv_path)
    assert len(loaded) == 1
    assert list(loaded.columns) == ["timestamp", "open", "high", "low", "close", "tick_volume", "spread"]
