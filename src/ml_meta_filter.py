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

        close = df_m5["close"].values
        high = df_m5["high"].values
        low = df_m5["low"].values
        vol = df_m5["tick_volume"].values if "tick_volume" in df_m5 else df_m5["volume"].values
        current_price = float(close[-1])

        # EMAs
        if "ema_20" in df_m5.columns and "ema_50" in df_m5.columns:
            ema20 = float(df_m5["ema_20"].iloc[-1])
            ema50 = float(df_m5["ema_50"].iloc[-1])
        else:
            ema20 = float(pd.Series(close).ewm(span=20).mean().iloc[-1])
            ema50 = float(pd.Series(close).ewm(span=50).mean().iloc[-1])

        dist_ema20 = (current_price - ema20) / max(1.0, ema20) * 100.0
        dist_ema50 = (current_price - ema50) / max(1.0, ema50) * 100.0

        # Session
        hour = pd.to_datetime(df_m5["timestamp"].iloc[-1]).hour
        is_prime = 1 if (7 <= hour <= 10 or 12 <= hour <= 16) else 0
        is_asian = 1 if (0 <= hour <= 6) else 0

        # ATR
        if "atr" in df_m5.columns:
            atr = float(df_m5["atr"].iloc[-1])
        else:
            tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
            atr = float(pd.Series(tr).rolling(14).mean().iloc[-1])
        atr_pct = (atr / max(1.0, current_price)) * 100.0

        # Signed buyer volume
        p_diff = np.diff(close[-15:])
        v_deltas = vol[-14:]
        b_vol = np.sum(v_deltas[p_diff >= 0])
        s_vol = np.sum(v_deltas[p_diff < 0])
        buyer_pct = (b_vol / max(1.0, b_vol + s_vol)) * 100.0

        # RSI
        if "rsi" in df_m5.columns:
            rsi = float(df_m5["rsi"].iloc[-1])
        else:
            diffs = pd.Series(close).diff()
            gain = diffs.clip(lower=0).rolling(14).mean()
            loss = (-diffs.clip(upper=0)).rolling(14).mean()
            rsi = float((100.0 - (100.0 / (1.0 + (gain / (loss + 1e-9))))).iloc[-1])

        feat_row = pd.DataFrame([{
            "side_code": 1 if candidate_action == "BUY" else -1,
            "is_pullback": 1 if "PULLBACK" in strategy_name else 0,
            "is_donchian": 1 if "DONCHIAN" in strategy_name else 0,
            "htf_trend_bias": h1_trend_bias,
            "h1_dist_ema50": h1_dist_ema50,
            "dist_ema_20": dist_ema20,
            "dist_ema_50": dist_ema50,
            "rsi": rsi,
            "atr_pct": atr_pct,
            "buyer_pct": buyer_pct,
            "is_prime_session": is_prime,
            "is_asian_chop": is_asian,
        }])

        try:
            scaler = self.ensemble.get("scaler")
            feat_cols = self.ensemble.get("feature_cols", [])
            missing_cols = [c for c in feat_cols if c not in feat_row.columns]
            if missing_cols:
                raise ValueError(f"Feature row missing required model columns: {missing_cols}")

            X_in = feat_row[feat_cols]
            if scaler is not None:
                X_in = scaler.transform(X_in)

            # Check for probability calibration wrapper or base models
            if "calibrated_ensemble" in self.ensemble and self.ensemble["calibrated_ensemble"] is not None:
                win_prob = float(self.ensemble["calibrated_ensemble"].predict_proba(X_in)[0][1])
            else:
                p_xgb = float(self.ensemble["xgb_model"].predict_proba(X_in)[0][1])
                p_lgb = float(self.ensemble["lgb_model"].predict_proba(X_in)[0][1])
                win_prob = (p_xgb * 0.55) + (p_lgb * 0.45)

            win_prob = round(float(win_prob), 4)
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
