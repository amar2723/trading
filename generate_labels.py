from __future__ import annotations

import argparse
import json

import pandas as pd

from src.labeling.label_generator import build_report, generate_labels, save_labeled_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate labeled training data from pattern data.")
    parser.add_argument("--input", required=True, help="Pattern CSV path")
    parser.add_argument("--output", required=True, help="Output labeled CSV path")
    parser.add_argument("--no-splits", action="store_true", help="Do not create train/validation/test CSVs")
    args = parser.parse_args()

    patterns = pd.read_csv(args.input)
    labeled = generate_labels(patterns)
    paths = save_labeled_dataset(labeled, args.output, create_splits=not args.no_splits)
    report = build_report(labeled)
    print(json.dumps({"paths": {k: str(v) for k, v in paths.items()}, "report": report}, indent=2))


if __name__ == "__main__":
    main()
