from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier


def create_lightgbm(**params):
    """Create LightGBM when installed, otherwise use sklearn histogram boosting."""
    try:
        from lightgbm import LGBMClassifier

        defaults = {
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.03,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "random_state": 42,
            "verbosity": -1,
        }
        defaults.update(params)
        return LGBMClassifier(**defaults)
    except Exception:
        fallback = {"max_iter": params.get("n_estimators", 200), "learning_rate": params.get("learning_rate", 0.03), "max_leaf_nodes": 31, "random_state": 42}
        return HistGradientBoostingClassifier(**fallback)
