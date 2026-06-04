from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data_ingestion.data_validator import validate_data
from src.data_ingestion.mt5_loader import get_historical_data, save_raw_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect historical MT5 OHLCV data.")
    parser.add_argument("--symbol", required=True, help="Trading symbol, e.g. XAUUSD")
    parser.add_argument("--timeframe", required=True, choices=["M1", "M5", "M15"])
    parser.add_argument("--start", required=True, help="Start date, e.g. 2023-01-01")
    parser.add_argument("--end", required=True, help="End date, e.g. 2025-01-01")
    parser.add_argument("--output-dir", default="data/raw")
    args = parser.parse_args()

    df = get_historical_data(args.symbol, args.timeframe, args.start, args.end)
    report = validate_data(df)
    saved_path = save_raw_data(df, args.symbol, args.timeframe, Path(args.output_dir))

    print(json.dumps({"saved_path": str(saved_path), "validation": report.to_dict()}, indent=2))


if __name__ == "__main__":
    main()
