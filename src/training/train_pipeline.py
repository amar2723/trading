from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.training.evaluate import classification_metrics, save_metrics, trading_metrics, write_html_report
from src.training.feature_selection import prepare_features, save_feature_artifacts
from src.training.shap_analysis import generate_shap_reports
from src.training.train_lightgbm import create_lightgbm
from src.training.train_random_forest import create_random_forest
from src.training.train_xgboost import create_xgboost


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


MODEL_BUILDERS = {
    "random_forest": create_random_forest,
    "xgboost": create_xgboost,
    "lightgbm": create_lightgbm,
}


def train_all_models(data_path: str | Path, model_dir: str | Path = "models", report_dir: str | Path = "reports", optimize: bool = False) -> dict:
    from src.training.feature_selection import load_labeled_data

    df = load_labeled_data(data_path)
    X, y, direction, feature_columns = prepare_features(df)
    version = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    model_path = Path(model_dir)
    report_path = Path(report_dir)
    model_path.mkdir(parents=True, exist_ok=True)
    report_path.mkdir(parents=True, exist_ok=True)

    split_metrics = {}
    trained_models = {}
    tscv = TimeSeriesSplit(n_splits=5)
    for name, builder in MODEL_BUILDERS.items():
        logger.info("Training %s", name)
        params = optimize_params(builder, X, y) if optimize and name in {"xgboost", "lightgbm"} else {}
        fold_scores = []
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
            model = builder(**params)
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            probabilities = _positive_probability(model, X.iloc[test_idx])
            metrics = classification_metrics(y.iloc[test_idx], probabilities)
            metrics["fold"] = fold
            fold_scores.append(metrics)
        final_model = builder(**params)
        final_model.fit(X, y)
        trained_models[name] = final_model
        split_metrics[name] = fold_scores

    artifact_names = {
        "xgboost": "xgboost_model.pkl",
        "lightgbm": "lightgbm_model.pkl",
        "random_forest": "random_forest.pkl",
    }
    for name, model in trained_models.items():
        joblib.dump({"model": model, "version": version, "feature_columns": feature_columns}, model_path / artifact_names[name])
        joblib.dump({"model": model, "version": version, "feature_columns": feature_columns}, model_path / f"{name}_{version}.pkl")

    _, _, scaler = _fit_scaler(X)
    save_feature_artifacts(feature_columns, scaler, model_path)

    importance = feature_importance(trained_models["xgboost"], feature_columns)
    importance.to_csv(report_path / "feature_importance.csv", index=False)
    importance.to_csv(model_path / "feature_importance.csv", index=False)
    write_feature_importance_png(importance, report_path / "feature_importance.png")
    write_html_report("Feature Importance", {"top_50": importance.head(50).to_dict(orient="records")}, report_path / "feature_importance.html")

    model_probabilities = _positive_probability(trained_models["xgboost"], X)
    metrics = {
        "model_version": version,
        "classification": split_metrics,
        "trading": trading_metrics(df),
        "feature_count": len(feature_columns),
        "row_count": len(df),
    }
    save_metrics(metrics, report_path)
    write_html_report("Training Report", metrics, report_path / "training_report.html")
    shap_status = generate_shap_reports(trained_models["xgboost"], X, report_path)
    metrics["shap"] = shap_status

    probabilities = direction_probabilities(trained_models["xgboost"], X, direction, df)
    probabilities.to_csv(report_path / "probability_output.csv", index=False)
    return {"version": version, "models": list(trained_models), "reports": str(report_path), "model_dir": str(model_path), "metrics": metrics}


def optimize_params(builder, X: pd.DataFrame, y: pd.Series) -> dict:
    try:
        import optuna

        def objective(trial):
            params = {
                "max_depth": trial.suggest_int("max_depth", 2, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            }
            scores = []
            for train_idx, test_idx in TimeSeriesSplit(n_splits=3).split(X):
                model = builder(**params)
                model.fit(X.iloc[train_idx], y.iloc[train_idx])
                scores.append(classification_metrics(y.iloc[test_idx], _positive_probability(model, X.iloc[test_idx]))["f1"])
            return float(np.mean(scores))

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=20, show_progress_bar=False)
        return study.best_params
    except Exception as exc:
        logger.warning("Optuna optimization skipped: %s", exc)
        return {}


def predict_probabilities(model_bundle: dict, rows: pd.DataFrame, pattern_weight: float = 0.2) -> pd.DataFrame:
    model = model_bundle["model"]
    feature_columns = model_bundle["feature_columns"]
    X = rows[feature_columns].replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
    proba = _positive_probability(model, X)
    direction = rows.get("target", pd.Series(0, index=rows.index)).fillna(0).astype(int)
    return direction_probabilities_from_profit(proba, direction, rows, pattern_weight)


def direction_probabilities(model, X: pd.DataFrame, direction: pd.Series, df: pd.DataFrame) -> pd.DataFrame:
    return direction_probabilities_from_profit(_positive_probability(model, X), direction, df)


def direction_probabilities_from_profit(profit_probability: np.ndarray, direction: pd.Series, df: pd.DataFrame, pattern_weight: float = 0.2) -> pd.DataFrame:
    buy_pattern = _pattern_agreement(df, "BUY")
    sell_pattern = _pattern_agreement(df, "SELL")
    buy = np.where(direction.eq(1), profit_probability, 0.0) * (1 - pattern_weight) + buy_pattern * pattern_weight
    sell = np.where(direction.eq(-1), profit_probability, 0.0) * (1 - pattern_weight) + sell_pattern * pattern_weight
    hold = np.clip(1 - np.maximum(buy, sell), 0, 1)
    total = buy + sell + hold
    result = pd.DataFrame({"buy_probability": buy / total * 100, "sell_probability": sell / total * 100, "hold_probability": hold / total * 100})
    result["confidence_score"] = result[["buy_probability", "sell_probability", "hold_probability"]].max(axis=1)
    return result


def feature_importance(model, feature_columns: list[str]) -> pd.DataFrame:
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = np.abs(model.coef_).ravel()
    else:
        values = np.zeros(len(feature_columns))
    return pd.DataFrame({"feature": feature_columns, "importance": values}).sort_values("importance", ascending=False).head(50)


def write_feature_importance_png(importance: pd.DataFrame, output_path: str | Path) -> None:
    try:
        import matplotlib.pyplot as plt

        top = importance.head(25).iloc[::-1]
        plt.figure(figsize=(10, 8))
        plt.barh(top["feature"], top["importance"])
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    except Exception as exc:
        Path(str(output_path) + ".txt").write_text(str(exc), encoding="utf-8")


def _positive_probability(model, X: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(X)
    if probabilities.shape[1] == 1:
        return np.zeros(len(X)) if model.classes_[0] == 0 else np.ones(len(X))
    positive_index = list(model.classes_).index(1)
    return probabilities[:, positive_index]


def _pattern_agreement(df: pd.DataFrame, side: str) -> np.ndarray:
    if side == "BUY":
        cols = ["bullish_liquidity_sweep", "bullish_mss", "bullish_bos", "bullish_fvg", "bullish_ob"]
    else:
        cols = ["bearish_liquidity_sweep", "bearish_mss", "bearish_bos", "bearish_fvg", "bearish_ob"]
    available = [c for c in cols if c in df.columns]
    if not available:
        return np.zeros(len(df))
    return df[available].fillna(0).astype(float).mean(axis=1).to_numpy()


def _fit_scaler(X: pd.DataFrame):
    from src.training.feature_selection import scale_features

    return scale_features(X)
