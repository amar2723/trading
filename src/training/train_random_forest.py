from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier


def create_random_forest(**params) -> RandomForestClassifier:
    defaults = {
        "n_estimators": 300,
        "max_depth": 8,
        "min_samples_leaf": 3,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
    }
    defaults.update(params)
    return RandomForestClassifier(**defaults)
