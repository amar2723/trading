from __future__ import annotations

import argparse
import json

import pandas as pd

from src.pattern_detection.pattern_pipeline import detect_patterns, save_patterns


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect Smart Money Concepts patterns.")
    parser.add_argument("--input", required=True, help="Feature CSV path")
    parser.add_argument("--output", required=True, help="Output pattern CSV path")
    args = parser.parse_args()

    features = pd.read_csv(args.input)
    patterns = detect_patterns(features)
    saved_path = save_patterns(patterns, args.output)
    print(json.dumps({"saved_path": str(saved_path), "rows": len(patterns), "columns": len(patterns.columns)}, indent=2))


if __name__ == "__main__":
    main()
