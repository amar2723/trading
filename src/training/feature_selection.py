from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


PRIMARY_TARGET = "profitable_trade"
SECONDARY_TARGET = "target"

EXCLUDE_COLUMNS = {
    "timestamp",
    "time",
    "entry_time",
    "sweep_time",
    "ob_time",
    "exit_index",
    "exit_price",
    "hit_window",
    "trade_result",
    "result_20",
    "result_50",
    "result_100",
    "profitable_trade",
    "losing_trade",
    "target",
    "entry_type",
    "displacement_direction",
    "structure_state",
    "structure_event",
}

FUTURE_COLUMNS = {
    "max_favorable_excursion",
    "max_adverse_excursion",
    "max_profit_points",
    "max_drawdown_points",
}


def load_labeled_data(path: str | Path) -> pd.DataFrame:
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Labeled dataset not found: {data_path}")
    df = pd.read_csv(data_path)
    if PRIMARY_TARGET not in df.columns:
        raise ValueError(f"Missing primary target column: {PRIMARY_TARGET}")
    return df


def build_trade_direction(df: pd.DataFrame) -> pd.Series:
    if "target" in df.columns:
        return df["target"].fillna(0).astype(int)
    if "entry_type" not in df.columns:
        return pd.Series(0, index=df.index, dtype=int)
    return df["entry_type"].map({"BUY": 1, "SELL": -1, "HOLD": 0}).fillna(0).astype(int)


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = EXCLUDE_COLUMNS | FUTURE_COLUMNS
    candidates = []
    for column in df.columns:
        if column in excluded:
            continue
        if pd.api.types.is_bool_dtype(df[column]) or pd.api.types.is_numeric_dtype(df[column]):
            candidates.append(column)
    if not candidates:
        raise ValueError("No numeric feature columns found after exclusions.")
    return candidates


def prepare_features(df: pd.DataFrame, feature_columns: list[str] | None = None) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str]]:
    columns = feature_columns or select_feature_columns(df)
    X = df[columns].replace([np.inf, -np.inf], np.nan).copy()
    X = X.ffill().bfill().fillna(0.0).astype(float)
    y = df[PRIMARY_TARGET].fillna(0).astype(int)
    direction = build_trade_direction(df)
    return X, y, direction, columns


def scale_features(X_train: pd.DataFrame, X_other: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame | None, StandardScaler]:
    scaler = StandardScaler()
    train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    other_scaled = None
    if X_other is not None:
        other_scaled = pd.DataFrame(scaler.transform(X_other), columns=X_other.columns, index=X_other.index)
    return train_scaled, other_scaled, scaler


def save_feature_artifacts(feature_columns: list[str], scaler: StandardScaler, model_dir: str | Path) -> None:
    path = Path(model_dir)
    path.mkdir(parents=True, exist_ok=True)
    joblib.dump(feature_columns, path / "feature_columns.pkl")
    joblib.dump(scaler, path / "scaler.pkl")
