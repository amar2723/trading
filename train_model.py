from __future__ import annotations

import argparse
import json

from src.training.train_pipeline import train_all_models


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ML models on labeled XAUUSD data.")
    parser.add_argument("--data", required=True, help="Path to labeled CSV")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--optimize", action="store_true", help="Run Optuna hyperparameter optimization")
    args = parser.parse_args()
    result = train_all_models(args.data, args.models_dir, args.reports_dir, optimize=args.optimize)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
