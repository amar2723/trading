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
    def __init__(self, model_path: Path = ENSEMBLE_PATH):
        self.model_path = model_path
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
        df_m5: pd.DataFrame,
        h1_trend_bias: int = 1,
        h1_dist_ema50: float = 0.0,
        min_expected_R: float = 0.05,
    ) -> FilterDecision:
        if candidate_action == "HOLD" or self.ensemble is None:
            return FilterDecision(
                decision="ABSTAIN",
                predicted_win_prob=0.35,
                expected_R=-0.47,
                confidence_pct=50.0,
                filter_reason="No candidate signal or model unavailable."
            )

        close = df_m5["close"].values
        high = df_m5["high"].values
        low = df_m5["low"].values
        vol = df_m5["tick_volume"].values if "tick_volume" in df_m5 else df_m5["volume"].values
        current_price = float(close[-1])

        # Indicator values
        ema50 = float(pd.Series(close).ewm(span=50).mean().iloc[-1])
        dist_ema50 = (current_price - ema50) / ema50 * 100

        hour = pd.to_datetime(df_m5["timestamp"].iloc[-1]).hour
        is_prime = 1 if (7 <= hour <= 10 or 12 <= hour <= 16) else 0
        is_asian = 1 if (0 <= hour <= 6) else 0

        tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
        atr = float(pd.Series(tr).rolling(14).mean().iloc[-1])
        atr_pct = (atr / current_price) * 100
        vol_exp = float((pd.Series(tr).rolling(14).mean() / pd.Series(tr).rolling(50).mean()).iloc[-1])

        recent_h20 = float(pd.Series(high).rolling(20).max().shift(1).iloc[-1])
        recent_l20 = float(pd.Series(low).rolling(20).min().shift(1).iloc[-1])
        span = max(0.1, recent_h20 - recent_l20)
        disc_depth = (current_price - recent_l20) / span

        sweep_h = 1 if (high[-1] > recent_h20 and close[-1] < recent_h20) else 0
        sweep_l = 1 if (low[-1] < recent_l20 and close[-1] > recent_l20) else 0

        p_deltas = np.diff(close[-15:])
        v_deltas = vol[-14:]
        signed_v = float(p_deltas[-1] >= 0) * vol[-1] - float(p_deltas[-1] < 0) * vol[-1]
        cvd_sl = float(np.sum(np.where(p_deltas >= 0, v_deltas, -v_deltas)))

        b_vol = np.sum(v_deltas[p_deltas >= 0])
        s_vol = np.sum(v_deltas[p_deltas < 0])
        tot = max(1.0, b_vol + s_vol)
        buyer_pct = (b_vol / tot) * 100

        diffs = pd.Series(close).diff()
        gain = diffs.clip(lower=0).rolling(14).mean()
        loss = (-diffs.clip(upper=0)).rolling(14).mean()
        rsi = float((100 - (100 / (1 + (gain / (loss + 1e-9))))).iloc[-1])

        feat_row = pd.DataFrame([{
            "htf_trend_bias": h1_trend_bias,
            "h1_dist_ema50": h1_dist_ema50,
            "dist_ema_50": dist_ema50,
            "is_prime_session": is_prime,
            "is_asian_chop": is_asian,
            "atr_pct": atr_pct,
            "volatility_expansion": vol_exp,
            "discount_depth": disc_depth,
            "sweep_high_confirmed": sweep_h,
            "sweep_low_confirmed": sweep_l,
            "signed_tick_volume_proxy": signed_v,
            "cvd_slope": cvd_sl,
            "signed_buyer_proxy_pct": buyer_pct,
            "rsi": rsi,
        }])

        try:
            p_xgb = float(self.ensemble["xgb_model"].predict_proba(feat_row)[0][1])
            p_lgb = float(self.ensemble["lgb_model"].predict_proba(feat_row)[0][1])
            win_prob = round((p_xgb * 0.55) + (p_lgb * 0.45), 3)
        except Exception:
            win_prob = 0.40

        # Expected return calculation based on 1:1.5 R:R:
        # E[R] = (WinProb * 1.5R) - ((1 - WinProb) * 1.0R) - Costs (0.05R spread/comm)
        expected_R = round((win_prob * 1.5) - ((1.0 - win_prob) * 1.0) - 0.05, 3)
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
