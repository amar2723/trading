from __future__ import annotations

"""
Machine Learning Meta-Filter for XAUUSD (Gold).
Uses Dual XGBoost + LightGBM Ensemble as a strict gatekeeper:
Evaluates candidate signals from deterministic strategies to decide whether
to TAKE or ABSTAIN based on expected return E[R].
"""

from dataclasses import dataclass
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

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
                return joblib.load(self.model_path)
            except Exception:
                return None
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
        if candidate_action == "HOLD" or self.ensemble is None:
            return FilterDecision(
                decision="ABSTAIN",
                predicted_win_prob=0.30,
                expected_R=-0.50,
                confidence_pct=50.0,
                filter_reason="Candidate action is HOLD or model unavailable."
            )

        close = df_m5["close"].values
        high = df_m5["high"].values
        low = df_m5["low"].values
        vol = df_m5["tick_volume"].values if "tick_volume" in df_m5 else df_m5["volume"].values
        current_price = float(close[-1])

        # EMAs
        ema20 = float(pd.Series(close).ewm(span=20).mean().iloc[-1])
        ema50 = float(pd.Series(close).ewm(span=50).mean().iloc[-1])
        dist_ema20 = (current_price - ema20) / ema20 * 100
        dist_ema50 = (current_price - ema50) / ema50 * 100

        # Session
        hour = pd.to_datetime(df_m5["timestamp"].iloc[-1]).hour
        is_prime = 1 if (7 <= hour <= 10 or 12 <= hour <= 16) else 0
        is_asian = 1 if (0 <= hour <= 6) else 0

        # ATR
        tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
        atr = float(pd.Series(tr).rolling(14).mean().iloc[-1])
        atr_pct = (atr / current_price) * 100

        # Signed buyer volume
        p_diff = np.diff(close[-15:])
        v_deltas = vol[-14:]
        b_vol = np.sum(v_deltas[p_diff >= 0])
        s_vol = np.sum(v_deltas[p_diff < 0])
        buyer_pct = (b_vol / max(1.0, b_vol + s_vol)) * 100

        # RSI
        diffs = pd.Series(close).diff()
        gain = diffs.clip(lower=0).rolling(14).mean()
        loss = (-diffs.clip(upper=0)).rolling(14).mean()
        rsi = float((100 - (100 / (1 + (gain / (loss + 1e-9))))).iloc[-1])

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
            feat_cols = self.ensemble["feature_cols"]
            X_in = feat_row[feat_cols]
            if scaler is not None:
                X_in = scaler.transform(X_in)

            p_xgb = float(self.ensemble["xgb_model"].predict_proba(X_in)[0][1])
            p_lgb = float(self.ensemble["lgb_model"].predict_proba(X_in)[0][1])
            win_prob = round((p_xgb * 0.55) + (p_lgb * 0.45), 3)
        except Exception as e:
            logger.warning("ML inference note: %s", e)
            win_prob = 0.35

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
