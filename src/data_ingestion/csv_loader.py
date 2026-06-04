from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_ingestion.data_validator import REQUIRED_COLUMNS, clean_data, validate_data
from src.data_ingestion.logging_utils import get_logger


def load_csv_data(path: str | Path, strict: bool = True) -> pd.DataFrame:
    logger = get_logger(__name__)
    csv_path = Path(path)
    logger.info("Loading CSV data from %s", csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    cleaned = clean_data(df)
    report = validate_data(cleaned)

    if strict and not report.is_valid:
        raise ValueError(f"CSV validation failed: {report.to_dict()}")

    logger.info("Loaded %s cleaned rows from %s", len(cleaned), csv_path)
    return cleaned[REQUIRED_COLUMNS]
