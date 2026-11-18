from __future__ import annotations

"""
Machine Learning Meta-Filter for XAUUSD (Gold).
Uses Dual XGBoost + LightGBM Ensemble as a strict gatekeeper:
Evaluates candidate signals from deterministic strategies to decide whether
to TAKE or ABSTAIN based on expected return E[R].
"""

from dataclasses import dataclass
import logging
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from src.feature_pipeline import (
    FEATURE_COLUMNS,
    compute_indicator_dataframe,
    extract_features_from_row,
)

logger = logging.getLogger("MLMetaFilter")

ROOT = Path(__file__).resolve().parents[1]
ENSEMBLE_PATH = ROOT / "models/xauusd_master_ensemble.pkl"


@dataclass
class FilterDecision:
    decision: str  # "TAKE" or "ABSTAIN"
    predicted_win_prob: float
    expected_R: float
    confidence_pct: float
    filter_reason: str


class MLMetaFilter:
    def __init__(self, model_path: Path | str = ENSEMBLE_PATH):
        self.model_path = Path(model_path) if model_path is not None else ENSEMBLE_PATH
        self.ensemble = self._load_model()

    def _load_model(self) -> dict | None:
        if self.model_path.exists():
            try:
                model_dict = joblib.load(self.model_path)
                logger.info("Successfully loaded ML ensemble from %s", self.model_path)
                return model_dict
            except Exception as e:
                logger.error("Failed to load ML model from %s: %s", self.model_path, e)
                return None
        logger.warning("ML model path %s does not exist.", self.model_path)
        return None

    def evaluate_candidate(
        self,
        candidate_action: str,
        strategy_name: str,
        df_m5: pd.DataFrame,
        h1_trend_bias: int = 1,
        h1_dist_ema50: float = 0.0,
        risk_usd: float = 4.50,
        min_expected_R: float = 0.15,
    ) -> FilterDecision:
        if candidate_action == "HOLD":
            return FilterDecision(
                decision="ABSTAIN",
                predicted_win_prob=0.0,
                expected_R=0.0,
                confidence_pct=0.0,
                filter_reason="Candidate action is HOLD."
            )

        if self.ensemble is None:
            return FilterDecision(
                decision="ABSTAIN",
                predicted_win_prob=0.0,
                expected_R=-1.0,
                confidence_pct=0.0,
                filter_reason="ABSTAIN: ML model ensemble is unavailable or not loaded."
            )

        # Build standard feature row using shared feature pipeline
        if "ema_20" not in df_m5.columns or "atr" not in df_m5.columns or "rsi" not in df_m5.columns:
            df_proc = compute_indicator_dataframe(df_m5)
        else:
            df_proc = df_m5

        last_row = df_proc.iloc[-1].to_dict()
        if h1_trend_bias is not None:
            last_row["h1_trend_bias"] = h1_trend_bias
        if h1_dist_ema50 is not None:
            last_row["h1_dist_ema50"] = h1_dist_ema50

        feat_row = extract_features_from_row(last_row, candidate_action, strategy_name)

        try:
            scaler = self.ensemble.get("scaler")
            feat_cols = self.ensemble.get("feature_cols", FEATURE_COLUMNS)
            missing_cols = [c for c in feat_cols if c not in feat_row.columns]
            if missing_cols:
                raise ValueError(f"Feature row missing required model columns: {missing_cols}")

            X_in = feat_row[feat_cols]
            if scaler is not None:
                X_in = scaler.transform(X_in)

            # Compute raw probabilities from ensemble
            p_xgb = float(self.ensemble["xgb_model"].predict_proba(X_in)[0][1])
            p_lgb = float(self.ensemble["lgb_model"].predict_proba(X_in)[0][1])
            p_raw = (p_xgb * 0.55) + (p_lgb * 0.45)

            # Apply saved calibrator if present
            calibrator = self.ensemble.get("calibrator")
            if calibrator is not None:
                try:
                    p_cal = float(calibrator.predict(np.array([p_raw]))[0])
                    win_prob = round(float(np.clip(p_cal, 0.0, 1.0)), 4)
                except Exception as cal_err:
                    logger.warning("Calibrator inference error (%s); falling back to raw ensemble probability.", cal_err)
                    win_prob = round(float(p_raw), 4)
            else:
                win_prob = round(float(p_raw), 4)
        except Exception as e:
            logger.error("ML inference error: %s. Failsafe ABSTAIN triggered.", e)
            return FilterDecision(
                decision="ABSTAIN",
                predicted_win_prob=0.0,
                expected_R=-1.0,
                confidence_pct=0.0,
                filter_reason=f"ABSTAIN: ML inference error ({e})."
            )

        # Realistic Expected return with spread & slippage drag ($0.35 round-trip cost)
        cost_drag_R = 0.35 / max(1.0, risk_usd)
        expected_R = round((win_prob * 1.5) - ((1.0 - win_prob) * 1.0) - cost_drag_R, 3)
        confidence_pct = round(win_prob * 100, 1)

        # Gate: Approve ONLY if positive expectancy hurdle is cleared
        if expected_R >= min_expected_R:
            return FilterDecision(
                decision="TAKE",
                predicted_win_prob=win_prob,
                expected_R=expected_R,
                confidence_pct=confidence_pct,
                filter_reason=f"Approved by ML Meta-Filter (Win Prob: {win_prob*100:.1f}%, Expected R: +{expected_R:.2f}R)."
            )
        else:
            return FilterDecision(
                decision="ABSTAIN",
                predicted_win_prob=win_prob,
                expected_R=expected_R,
                confidence_pct=confidence_pct,
                filter_reason=f"Vetoed by ML Meta-Filter (Expected R: {expected_R:+.2f}R below required hurdle +{min_expected_R:.2f}R)."
            )
