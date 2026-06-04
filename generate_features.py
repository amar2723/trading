from __future__ import annotations

import argparse
import json

import pandas as pd

from src.feature_engineering.feature_pipeline import build_features, save_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate engineered trading features.")
    parser.add_argument("--input", required=True, help="Input OHLCV CSV path")
    parser.add_argument("--output", required=True, help="Output feature CSV path")
    args = parser.parse_args()

    raw = pd.read_csv(args.input)
    features = build_features(raw)
    saved_path = save_features(features, args.output)
    print(json.dumps({"saved_path": str(saved_path), "rows": len(features), "columns": len(features.columns)}, indent=2))


if __name__ == "__main__":
    main()
