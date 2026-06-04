from __future__ import annotations

import argparse
import json

from sklearn.ensemble import GradientBoostingClassifier

from src.training.feature_selection import select_feature_columns


MODEL_FEATURES: list[str] = []


def create_xgboost(**params):
    """Create the main XGBoost model, falling back to sklearn if xgboost is unavailable."""
    try:
        from xgboost import XGBClassifier

        defaults = {
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.03,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "random_state": 42,
        }
        defaults.update(params)
        return XGBClassifier(**defaults)
    except Exception:
        fallback = {"n_estimators": params.get("n_estimators", 200), "learning_rate": params.get("learning_rate", 0.03), "max_depth": params.get("max_depth", 3), "random_state": 42}
        return GradientBoostingClassifier(**fallback)


def train(csv_path: str, model_dir: str = "models") -> dict:
    """Backward-compatible entry point for earlier project commands."""
    from src.training.train_pipeline import train_all_models

    return train_all_models(csv_path, model_dir=model_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the XGBoost-centered model suite.")
    parser.add_argument("--csv", required=True, help="Labeled CSV path")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--report-dir", default="reports")
    args = parser.parse_args()
    from src.training.train_pipeline import train_all_models

    print(json.dumps(train_all_models(args.csv, args.model_dir, args.report_dir), indent=2, default=str))


if __name__ == "__main__":
    main()
