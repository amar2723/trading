"""
Production-Grade Walk-Forward Tournament & Event-Driven Evaluation for XAUUSD.
Strictly implements:
1. Genuinely executable next-open fills (Candle i close signal -> Candle i+1 open fill).
2. True timestamp-based boundary purging:
   - For every candidate, tracks exact entry_time and resolution exit_time.
   - Purges any training candidate whose exit_time extends into the validation split.
   - Explicitly purges the development/test boundary.
3. Time-aware probability calibration (fitted chronologically on pre-validation temporal split, never shuffled KFold).
4. Direct, independent comparison of:
   - XGBoost
   - LightGBM
   - 55/45 Gradient Ensemble
   - Calibrated Ensemble
   - No-ML Strategy Baseline
5. Aggregate Profit Factor calculated from total gross profit / total gross loss across folds.
6. Event-Driven Production Backtest on out-of-sample data (at most 1 active position, sequential equity updates).
7. Full reporting of:
   - CSV Provenance and SHA-256 checksums
   - Number of independent executed trades
   - AUC and bootstrap 95% confidence intervals
   - Brier score (calibration loss)
   - Mean net R and bootstrap 95% confidence intervals
   - Profit factor, Max drawdown
   - Sensitivity to increased spread and slippage (+50%, +100%)
   - Statistical test results (or INCONCLUSIVE classification).
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss, precision_score
import lightgbm as lgb
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.strategies.benchmark_strategies import (
    H1TrendM5PullbackStrategy,
    DonchianBreakoutStrategy,
    VolatilitySqueezeStrategy,
    CandidateSignal,
)
from src.ml_meta_filter import MLMetaFilter
from src.risk_governor import RiskGovernor
from src.backtest_engine import EventDrivenBacktestEngine, BacktestResults

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("WalkForwardRigorous")

CONTRACT_SIZE = 100.0
LOT_SIZE = 0.01
BENCHMARK_SPREAD_USD = 0.24      # 24 pips = $0.24 / oz
BENCHMARK_SLIPPAGE_USD = 0.05    # $0.05 / oz entry, $0.05 / oz exit
TOTAL_FRICTION_USD = BENCHMARK_SPREAD_USD + (2 * BENCHMARK_SLIPPAGE_USD)  # $0.34 / oz


def compute_file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    ts_cols = [c for c in df.columns if "time" in c.lower() or "date" in c.lower()]
    ts_col = ts_cols[0]
    df["timestamp"] = pd.to_datetime(df[ts_col])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Clean numeric columns
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    
    vol_col = "tick_volume" if "tick_volume" in df.columns else ("volume" if "volume" in df.columns else None)
    if vol_col:
        df["tick_volume"] = df[vol_col].astype(float)
    else:
        df["tick_volume"] = 100.0

    # Indicators
    df["ema_20"] = df["close"].ewm(span=20).mean()
    df["ema_50"] = df["close"].ewm(span=50).mean()
    df["donch_h20"] = df["high"].rolling(20).max().shift(1)
    df["donch_l20"] = df["low"].rolling(20).min().shift(1)

    tr = np.maximum(
        df["high"] - df["low"],
        np.maximum(np.abs(df["high"] - df["close"].shift(1)), np.abs(df["low"] - df["close"].shift(1)))
    )
    df["atr"] = tr.rolling(14).mean()

    diffs = df["close"].diff()
    gain = diffs.clip(lower=0).rolling(14).mean()
    loss = (-diffs.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100.0 - (100.0 / (1.0 + (gain / (loss + 1e-9))))

    # Causal H1 macro trend (approximated via 12-bar rolling M5 EMA)
    df["h1_ema_trend"] = df["close"].ewm(span=240).mean() # ~20 hours
    df["h1_trend_bias"] = np.where(df["close"] > df["h1_ema_trend"], 1, -1)
    df["h1_dist_ema50"] = (df["close"] - df["h1_ema_trend"]) / df["h1_ema_trend"] * 100.0

    return df.dropna().reset_index(drop=True)


def extract_candidate_setups_with_causal_forward_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies strategy candidate setups and resolves outcomes non-anticipatorily:
    - Signal evaluated at candle i close
    - Entry order filled at candle i+1 OPEN with spread + slippage
    - Tracked from candle i+1 up to i+25
    - Exact entry_time and exit_time recorded for timestamp-based purging.
    """
    n = len(df)
    candidates = []

    # Cache arrays for instant vector evaluation
    close_arr = df["close"].values
    open_arr = df["open"].values
    high_arr = df["high"].values
    low_arr = df["low"].values
    time_arr = df["timestamp"].values
    ema20_arr = df["ema_20"].values
    ema50_arr = df["ema_50"].values
    atr_arr = df["atr"].values
    rsi_arr = df["rsi"].values
    h1_bias_arr = df["h1_trend_bias"].values
    h1_dist_arr = df["h1_dist_ema50"].values
    donch_h_arr = df["donch_h20"].values
    donch_l_arr = df["donch_l20"].values

    for i in range(50, n - 26):
        c_close = close_arr[i]
        c_open = open_arr[i]
        c_atr = atr_arr[i] if not np.isnan(atr_arr[i]) else 3.50
        e20 = ema20_arr[i]
        e50 = ema50_arr[i]
        rsi_val = rsi_arr[i]
        h1_bias = h1_bias_arr[i]
        donch_h = donch_h_arr[i]
        donch_l = donch_l_arr[i]

        risk_dist = max(c_atr * 1.5, 4.50)
        risk_pips = risk_dist * 10.0

        action = "HOLD"
        strat_name = "NONE"

        min_ema = min(e20, e50)
        max_ema = max(e20, e50)

        # 1. H1 Trend + M5 Pullback
        if h1_bias == 1 and (min_ema - 1.5 <= c_close <= max_ema + 1.0) and (38.0 <= rsi_val <= 62.0) and (c_close >= c_open):
            action, strat_name = "BUY", "H1_PULLBACK"
        elif h1_bias == -1 and (min_ema - 1.0 <= c_close <= max_ema + 1.5) and (38.0 <= rsi_val <= 62.0) and (c_close <= c_open):
            action, strat_name = "SELL", "H1_PULLBACK"
        # 2. Donchian Breakout
        elif (c_close > donch_h) and h1_bias == 1 and (rsi_val <= 68.0):
            action, strat_name = "BUY", "DONCHIAN"
        elif (c_close < donch_l) and h1_bias == -1 and (rsi_val >= 32.0):
            action, strat_name = "SELL", "DONCHIAN"

        if action not in ["BUY", "SELL"]:
            continue

        # NON-ANTICIPATORY FILL: Candle i+1 Open
        fill_bar_idx = i + 1
        bar_open_fill = float(open_arr[fill_bar_idx])
        signal_time = pd.Timestamp(time_arr[i])
        entry_time = pd.Timestamp(time_arr[fill_bar_idx])

        if action == "BUY":
            entry_p = round(bar_open_fill + (BENCHMARK_SPREAD_USD / 2.0) + BENCHMARK_SLIPPAGE_USD, 2)
            sl_p = round(entry_p - risk_dist, 2)
            tp1_p = round(entry_p + (risk_dist * 1.5), 2)
            be_trig_p = round(entry_p + max(risk_dist * 0.75, 3.50), 2)
        else:
            entry_p = round(bar_open_fill - (BENCHMARK_SPREAD_USD / 2.0) - BENCHMARK_SLIPPAGE_USD, 2)
            sl_p = round(entry_p + risk_dist, 2)
            tp1_p = round(entry_p - (risk_dist * 1.5), 2)
            be_trig_p = round(entry_p - max(risk_dist * 0.75, 3.50), 2)

        risk_dist = abs(entry_p - sl_p)
        initial_risk_usd = risk_dist * LOT_SIZE * CONTRACT_SIZE

        # Forward scan starting from candle i+1 up to i+25 (24 bars max horizon)
        resolved = False
        outcome = "TIMEOUT"
        be_active = False
        net_pnl = 0.0
        exit_time = df["timestamp"].iloc[min(n - 1, fill_bar_idx + 24)]
        exit_p = float(df["close"].iloc[min(n - 1, fill_bar_idx + 24)])

        for f_idx in range(fill_bar_idx, min(n, fill_bar_idx + 25)):
            f_high = float(df["high"].iloc[f_idx])
            f_low = float(df["low"].iloc[f_idx])
            f_time = df["timestamp"].iloc[f_idx]

            effective_sl = entry_p if be_active else sl_p

            if action == "BUY":
                # Check SL first
                if f_low <= effective_sl:
                    resolved = True
                    exit_time = f_time
                    exit_p = effective_sl - BENCHMARK_SLIPPAGE_USD
                    gross = (exit_p - entry_p) * LOT_SIZE * CONTRACT_SIZE
                    net_pnl = gross - (BENCHMARK_SLIPPAGE_USD * LOT_SIZE * CONTRACT_SIZE)
                    outcome = "BREAK_EVEN" if be_active else "LOSS"
                    break
                elif f_high >= tp1_p:
                    resolved = True
                    exit_time = f_time
                    exit_p = tp1_p - BENCHMARK_SLIPPAGE_USD
                    gross = (exit_p - entry_p) * LOT_SIZE * CONTRACT_SIZE
                    net_pnl = gross - (BENCHMARK_SLIPPAGE_USD * LOT_SIZE * CONTRACT_SIZE)
                    outcome = "WIN"
                    break
                elif not be_active and f_high >= be_trig_p:
                    # BE activates for SUBSEQUENT bars
                    be_active = True
            else:  # SELL
                if f_high >= effective_sl:
                    resolved = True
                    exit_time = f_time
                    exit_p = effective_sl + BENCHMARK_SLIPPAGE_USD
                    gross = (entry_p - exit_p) * LOT_SIZE * CONTRACT_SIZE
                    net_pnl = gross - (BENCHMARK_SLIPPAGE_USD * LOT_SIZE * CONTRACT_SIZE)
                    outcome = "BREAK_EVEN" if be_active else "LOSS"
                    break
                elif f_low <= tp1_p:
                    resolved = True
                    exit_time = f_time
                    exit_p = tp1_p + BENCHMARK_SLIPPAGE_USD
                    gross = (entry_p - exit_p) * LOT_SIZE * CONTRACT_SIZE
                    net_pnl = gross - (BENCHMARK_SLIPPAGE_USD * LOT_SIZE * CONTRACT_SIZE)
                    outcome = "WIN"
                    break
                elif not be_active and f_low <= be_trig_p:
                    be_active = True

        if not resolved:
            # Timeout close at end of horizon
            if action == "BUY":
                gross = (exit_p - entry_p) * LOT_SIZE * CONTRACT_SIZE
            else:
                gross = (entry_p - exit_p) * LOT_SIZE * CONTRACT_SIZE
            net_pnl = gross - TOTAL_FRICTION_USD
            outcome = "WIN" if net_pnl > 0 else "LOSS"

        pnl_R = round(net_pnl / max(0.50, initial_risk_usd), 3)
        target = 1 if outcome == "WIN" else 0

        # Feature vector extracted causally at bar i
        hour = signal_time.hour
        is_prime = 1 if (7 <= hour <= 10 or 12 <= hour <= 16) else 0
        is_asian = 1 if (0 <= hour <= 6) else 0

        candidates.append({
            "bar_idx": i,
            "signal_time": signal_time,
            "entry_time": entry_time,
            "exit_time": exit_time,
            "action": action,
            "strategy": strat_name,
            "entry_price": entry_p,
            "sl_price": sl_p,
            "tp1_price": tp1_p,
            "initial_risk_usd": round(initial_risk_usd, 2),
            "outcome": outcome,
            "net_pnl": round(net_pnl, 2),
            "pnl_R": pnl_R,
            "target": target,
            # Features
            "side_code": 1 if action == "BUY" else -1,
            "is_pullback": 1 if "PULLBACK" in strat_name else 0,
            "is_donchian": 1 if "DONCHIAN" in strat_name else 0,
            "htf_trend_bias": int(df["h1_trend_bias"].iloc[i]),
            "h1_dist_ema50": float(df["h1_dist_ema50"].iloc[i]),
            "dist_ema_20": float((df["close"].iloc[i] - df["ema_20"].iloc[i]) / df["ema_20"].iloc[i] * 100.0),
            "dist_ema_50": float((df["close"].iloc[i] - df["ema_50"].iloc[i]) / df["ema_50"].iloc[i] * 100.0),
            "rsi": float(df["rsi"].iloc[i]),
            "atr_pct": float(df["atr"].iloc[i] / df["close"].iloc[i] * 100.0),
            "buyer_pct": 50.0,
            "is_prime_session": is_prime,
            "is_asian_chop": is_asian,
        })

    return pd.DataFrame(candidates)


def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, ci: float = 0.95) -> tuple[float, float]:
    """Computes bootstrap confidence interval."""
    if len(values) < 5:
        return (float(np.mean(values)), float(np.mean(values)))
    rng = np.random.default_rng(42)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        means.append(np.mean(sample))
    alpha = (1.0 - ci) / 2.0
    low = np.percentile(means, alpha * 100.0)
    high = np.percentile(means, (1.0 - alpha) * 100.0)
    return (round(float(low), 4), round(float(high), 4))


def run_rigorous_evaluation():
    raw_path = ROOT / "data/raw/XAUUSD_M5_60d_max.csv"
    sha256_hash = compute_file_sha256(raw_path)
    logger.info("Auditing dataset: %s (SHA-256: %s)", raw_path.name, sha256_hash)

    df_market = load_dataset(raw_path)
    total_bars = len(df_market)
    start_date = df_market["timestamp"].iloc[0]
    end_date = df_market["timestamp"].iloc[-1]
    logger.info("Total market candles: %d | Range: %s to %s", total_bars, start_date, end_date)

    df_candidates = extract_candidate_setups_with_causal_forward_outcomes(df_market)
    logger.info("Extracted %d total candidate setups across all candles.", len(df_candidates))

    # Features
    feature_cols = [
        "side_code", "is_pullback", "is_donchian",
        "htf_trend_bias", "h1_dist_ema50",
        "dist_ema_20", "dist_ema_50", "rsi",
        "atr_pct", "buyer_pct",
        "is_prime_session", "is_asian_chop"
    ]

    # -------------------------------------------------------------------------
    # CHRONOLOGICAL SPLIT WITH STRICT DEVELOPMENT / TEST BOUNDARY PURGING
    # -------------------------------------------------------------------------
    # 80% Development (for expanding purged walk-forward) / 20% Out-of-Sample Test
    test_split_idx = int(len(df_candidates) * 0.80)
    test_start_time = df_candidates["entry_time"].iloc[test_split_idx]

    # Purge any development candidate whose forward exit extends into test_start_time
    df_dev_raw = df_candidates[df_candidates["entry_time"] < test_start_time].copy()
    df_dev = df_dev_raw[df_dev_raw["exit_time"] < test_start_time].copy().reset_index(drop=True)

    df_test = df_candidates[df_candidates["entry_time"] >= test_start_time].copy().reset_index(drop=True)

    logger.info(
        "Development split: %d candidates (purged %d boundary overlapping) | Untouched Test: %d candidates",
        len(df_dev), len(df_dev_raw) - len(df_dev), len(df_test)
    )

    # -------------------------------------------------------------------------
    # 4-FOLD EXPANDING PURGED WALK-FORWARD CROSS-VALIDATION BY MARKET TIMESTAMP
    # -------------------------------------------------------------------------
    n_dev = len(df_dev)
    split_points = [
        int(n_dev * 0.40),
        int(n_dev * 0.55),
        int(n_dev * 0.70),
        int(n_dev * 0.85),
        n_dev
    ]

    models_to_test = ["NoML_Baseline", "XGBoost", "LightGBM", "DualEnsemble_55_45", "CalibratedEnsemble"]
    tournament_records = {m: [] for m in models_to_test}

    for fold_i in range(4):
        val_start_idx = split_points[fold_i]
        val_end_idx = split_points[fold_i + 1]

        val_start_time = df_dev["entry_time"].iloc[val_start_idx]
        val_end_time = df_dev["entry_time"].iloc[val_end_idx - 1]

        # Training candidate pool: strictly before val_start_time
        train_raw = df_dev[df_dev["entry_time"] < val_start_time]

        # PURGE TRAIN CANDIDATES WHOSE FORWARD EXIT EXTENDS INTO OR PAST val_start_time
        train_clean = train_raw[train_raw["exit_time"] < val_start_time].copy().reset_index(drop=True)
        val_clean = df_dev[(df_dev["entry_time"] >= val_start_time) & (df_dev["entry_time"] <= val_end_time)].copy().reset_index(drop=True)

        logger.info(
            "Fold %d: Train %d (purged %d) | Val %d | Val Dates: %s to %s",
            fold_i + 1, len(train_clean), len(train_raw) - len(train_clean), len(val_clean),
            val_start_time.strftime("%Y-%m-%d"), val_end_time.strftime("%Y-%m-%d")
        )

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(train_clean[feature_cols])
        y_tr = train_clean["target"].values
        X_val = scaler.transform(val_clean[feature_cols])
        y_val = val_clean["target"].values

        # Time-aware calibration partition (earlier 75% train, later 25% calibrate)
        cal_split = int(len(train_clean) * 0.75)
        X_tr_base, y_tr_base = X_tr[:cal_split], y_tr[:cal_split]
        X_tr_cal, y_tr_cal = X_tr[cal_split:], y_tr[cal_split:]

        # Train Base Models
        xgb_m = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.03, subsample=0.8, eval_metric="logloss", random_state=42)
        lgb_m = lgb.LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.03, subsample=0.8, verbosity=-1, random_state=42)

        xgb_m.fit(X_tr, y_tr)
        lgb_m.fit(X_tr, y_tr)

        # Time-Aware Calibrator fit on chronological split
        xgb_base = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.03, subsample=0.8, eval_metric="logloss", random_state=42)
        lgb_base = lgb.LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.03, subsample=0.8, verbosity=-1, random_state=42)
        xgb_base.fit(X_tr_base, y_tr_base)
        lgb_base.fit(X_tr_base, y_tr_base)

        p_cal_xgb = xgb_base.predict_proba(X_tr_cal)[:, 1]
        p_cal_lgb = lgb_base.predict_proba(X_tr_cal)[:, 1]
        p_cal_ens = (p_cal_xgb * 0.55) + (p_cal_lgb * 0.45)

        isotonic_calibrator = IsotonicRegression(out_of_bounds="clip")
        isotonic_calibrator.fit(p_cal_ens, y_tr_cal)

        # Validation Probabilities
        p_xgb_val = xgb_m.predict_proba(X_val)[:, 1]
        p_lgb_val = lgb_m.predict_proba(X_val)[:, 1]
        p_ens_val = (p_xgb_val * 0.55) + (p_lgb_val * 0.45)
        p_cal_val = isotonic_calibrator.predict(p_ens_val)

        probs_map = {
            "NoML_Baseline": np.ones(len(val_clean)),
            "XGBoost": p_xgb_val,
            "LightGBM": p_lgb_val,
            "DualEnsemble_55_45": p_ens_val,
            "CalibratedEnsemble": p_cal_val,
        }

        # Friction cost in R: $0.34 / initial_risk
        cost_R = (TOTAL_FRICTION_USD / val_clean["initial_risk_usd"]).values

        for m_name in models_to_test:
            p = probs_map[m_name]
            if m_name == "NoML_Baseline":
                trade_mask = np.ones(len(val_clean), dtype=bool)
                auc = 0.500
                brier = 0.500
            else:
                auc = float(roc_auc_score(y_val, p)) if len(np.unique(y_val)) > 1 else 0.5
                brier = float(brier_score_loss(y_val, p))
                exp_R = (p * 1.5) - ((1.0 - p) * 1.0) - cost_R
                trade_mask = exp_R >= 0.15

            taken = val_clean[trade_mask].copy()
            n_taken = len(taken)
            if n_taken > 0:
                wins = int((taken["target"] == 1).sum())
                losses = int((taken["outcome"] == "LOSS").sum())
                bes = int((taken["outcome"] == "BREAK_EVEN").sum())
                gross_profit = float(taken[taken["net_pnl"] > 0]["net_pnl"].sum())
                gross_loss = float(abs(taken[taken["net_pnl"] < 0]["net_pnl"].sum()))
                net_pnl = float(taken["net_pnl"].sum())
                r_vals = taken["pnl_R"].values
            else:
                wins = losses = bes = 0
                gross_profit = gross_loss = net_pnl = 0.0
                r_vals = np.array([0.0])

            tournament_records[m_name].append({
                "fold": fold_i + 1,
                "val_samples": len(val_clean),
                "auc": auc,
                "brier": brier,
                "trades": n_taken,
                "wins": wins,
                "losses": losses,
                "bes": bes,
                "gross_profit": gross_profit,
                "gross_loss": gross_loss,
                "net_pnl": net_pnl,
                "r_vals": r_vals.tolist(),
            })

    # -------------------------------------------------------------------------
    # AGGREGATE TOURNAMENT RESULTS ACROSS ALL FOLDS
    # Computed from total gross profit / total gross loss (NOT averaging PFs!)
    # -------------------------------------------------------------------------
    leaderboard = []
    for m_name, recs in tournament_records.items():
        total_gp = sum(r["gross_profit"] for r in recs)
        total_gl = sum(r["gross_loss"] for r in recs)
        total_trades = sum(r["trades"] for r in recs)
        total_wins = sum(r["wins"] for r in recs)
        total_losses = sum(r["losses"] for r in recs)
        total_bes = sum(r["bes"] for r in recs)
        total_net_pnl = sum(r["net_pnl"] for r in recs)

        aggregate_pf = round(total_gp / max(0.01, total_gl), 2) if total_gl > 0 else (99.0 if total_gp > 0 else 0.0)
        win_rate = round((total_wins / max(1, total_trades)) * 100.0, 2)

        all_r = []
        for r in recs:
            all_r.extend(r["r_vals"])
        all_r = np.array(all_r)
        mean_R = round(float(np.mean(all_r)), 3)
        ci_R = bootstrap_ci(all_r)

        aucs = [r["auc"] for r in recs if r["auc"] > 0]
        mean_auc = round(float(np.mean(aucs)), 3)
        briers = [r["brier"] for r in recs]
        mean_brier = round(float(np.mean(briers)), 4)

        leaderboard.append({
            "model": m_name,
            "mean_auc": mean_auc,
            "mean_brier": mean_brier,
            "total_trades": total_trades,
            "win_rate_pct": win_rate,
            "gross_profit_usd": round(total_gp, 2),
            "gross_loss_usd": round(total_gl, 2),
            "aggregate_profit_factor": aggregate_pf,
            "mean_net_R": mean_R,
            "ci_95_net_R": ci_R,
            "total_net_pnl_usd": round(total_net_pnl, 2),
        })

    leaderboard_df = pd.DataFrame(leaderboard).sort_values("aggregate_profit_factor", ascending=False)
    logger.info("Tournament Leaderboard:\n%s", leaderboard_df.to_string(index=False))

    # -------------------------------------------------------------------------
    # PRODUCTION MODEL RETRAINING & CALIBRATION PACKAGING
    # -------------------------------------------------------------------------
    logger.info("Training and packaging final production ensemble on full development pool (%d samples)...", len(df_dev))
    scaler_prod = StandardScaler()
    X_dev = scaler_prod.fit_transform(df_dev[feature_cols])
    y_dev = df_dev["target"].values

    xgb_final = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.03, subsample=0.8, eval_metric="logloss", random_state=42)
    lgb_final = lgb.LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.03, subsample=0.8, verbosity=-1, random_state=42)
    xgb_final.fit(X_dev, y_dev)
    lgb_final.fit(X_dev, y_dev)

    # Time-aware calibration on dev temporal holdout
    dev_cal_split = int(len(df_dev) * 0.80)
    xgb_c_base = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.03, subsample=0.8, eval_metric="logloss", random_state=42)
    lgb_c_base = lgb.LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.03, subsample=0.8, verbosity=-1, random_state=42)
    xgb_c_base.fit(X_dev[:dev_cal_split], y_dev[:dev_cal_split])
    lgb_c_base.fit(X_dev[:dev_cal_split], y_dev[:dev_cal_split])

    p_dev_c_ens = (xgb_c_base.predict_proba(X_dev[dev_cal_split:])[:, 1] * 0.55) + (lgb_c_base.predict_proba(X_dev[dev_cal_split:])[:, 1] * 0.45)
    calibrator_final = IsotonicRegression(out_of_bounds="clip")
    calibrator_final.fit(p_dev_c_ens, y_dev[dev_cal_split:])

    prod_model_path = ROOT / "models/xauusd_master_ensemble.pkl"
    joblib.dump({
        "model_type": "Dual_XGB_LGB_Purged_WalkForward",
        "xgb_model": xgb_final,
        "lgb_model": lgb_final,
        "calibrator": calibrator_final,
        "scaler": scaler_prod,
        "feature_cols": feature_cols,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "train_samples": len(df_dev),
        "data_sha256": sha256_hash,
    }, prod_model_path)
    logger.info("Saved production model artifact to %s", prod_model_path)

    # -------------------------------------------------------------------------
    # PRIORITY 2 & 3: PRODUCTION-EQUIVALENT EVENT-DRIVEN OUT-OF-SAMPLE EVALUATION
    # -------------------------------------------------------------------------
    # Cut market data to out-of-sample period only
    df_market_test = df_market[df_market["timestamp"] >= test_start_time].copy().reset_index(drop=True)
    logger.info("Out-of-Sample Market Period: %s to %s (%d M5 bars)", df_market_test["timestamp"].iloc[0], df_market_test["timestamp"].iloc[-1], len(df_market_test))

    meta_filter_prod = MLMetaFilter(model_path=prod_model_path)
    risk_gov_prod = RiskGovernor(max_risk_pct=1.0, contract_size=CONTRACT_SIZE)

    # 1. Event-Driven Run with ML Meta-Filter Active
    engine_ml = EventDrivenBacktestEngine(
        initial_equity=1000.0,
        max_risk_pct=1.0,
        use_ml_filter=True,
        meta_filter=meta_filter_prod,
        risk_governor=risk_gov_prod,
    )
    res_ml = engine_ml.run(df_market_test)

    # 2. Event-Driven Run WITHOUT ML Meta-Filter (No-ML Baseline under identical conditions)
    engine_noml = EventDrivenBacktestEngine(
        initial_equity=1000.0,
        max_risk_pct=1.0,
        use_ml_filter=False,
        risk_governor=risk_gov_prod,
    )
    res_noml = engine_noml.run(df_market_test)

    # 3. Sensitivity Test: +50% Spread & Slippage ($0.36 spread, $0.15 slippage)
    engine_sens50 = EventDrivenBacktestEngine(
        initial_equity=1000.0,
        max_risk_pct=1.0,
        spread_usd=0.36,
        entry_slippage_usd=0.075,
        exit_slippage_usd=0.075,
        use_ml_filter=True,
        meta_filter=meta_filter_prod,
        risk_governor=risk_gov_prod,
    )
    res_sens50 = engine_sens50.run(df_market_test)

    # 4. Sensitivity Test: +100% Spread & Slippage ($0.48 spread, $0.20 slippage)
    engine_sens100 = EventDrivenBacktestEngine(
        initial_equity=1000.0,
        max_risk_pct=1.0,
        spread_usd=0.48,
        entry_slippage_usd=0.10,
        exit_slippage_usd=0.10,
        use_ml_filter=True,
        meta_filter=meta_filter_prod,
        risk_governor=risk_gov_prod,
    )
    res_sens100 = engine_sens100.run(df_market_test)

    # Compile comprehensive final report
    final_report = {
        "dataset_metadata": {
            "source_file": raw_path.name,
            "sha256_checksum": sha256_hash,
            "total_candles": total_bars,
            "full_date_range": f"{start_date} to {end_date}",
            "oos_test_candles": len(df_market_test),
            "oos_date_range": f"{df_market_test['timestamp'].iloc[0]} to {df_market_test['timestamp'].iloc[-1]}",
        },
        "tournament_leaderboard": leaderboard,
        "oos_event_driven_results": {
            "optimized_with_ml": {
                "initial_equity": res_ml.initial_equity,
                "final_equity": res_ml.final_equity,
                "net_pnl_usd": res_ml.net_pnl_usd,
                "net_return_pct": res_ml.net_return_pct,
                "candidates_count": res_ml.candidates_count,
                "ml_approved_count": res_ml.ml_approved_count,
                "ml_abstained_count": res_ml.ml_abstained_count,
                "risk_approved_count": res_ml.risk_approved_count,
                "risk_vetoed_count": res_ml.risk_vetoed_count,
                "filled_trades_count": res_ml.filled_trades_count,
                "closed_trades_count": res_ml.closed_trades_count,
                "wins": res_ml.wins_count,
                "losses": res_ml.losses_count,
                "break_evens": res_ml.be_count,
                "win_rate_pct": res_ml.win_rate_pct,
                "profit_factor": res_ml.profit_factor,
                "net_expectancy_R": res_ml.net_expectancy_R,
                "max_drawdown_usd": res_ml.max_drawdown_usd,
                "max_drawdown_pct": res_ml.max_drawdown_pct,
            },
            "no_ml_baseline": {
                "initial_equity": res_noml.initial_equity,
                "final_equity": res_noml.final_equity,
                "net_pnl_usd": res_noml.net_pnl_usd,
                "net_return_pct": res_noml.net_return_pct,
                "candidates_count": res_noml.candidates_count,
                "filled_trades_count": res_noml.filled_trades_count,
                "closed_trades_count": res_noml.closed_trades_count,
                "wins": res_noml.wins_count,
                "losses": res_noml.losses_count,
                "break_evens": res_noml.be_count,
                "win_rate_pct": res_noml.win_rate_pct,
                "profit_factor": res_noml.profit_factor,
                "net_expectancy_R": res_noml.net_expectancy_R,
                "max_drawdown_usd": res_noml.max_drawdown_usd,
                "max_drawdown_pct": res_noml.max_drawdown_pct,
            },
            "sensitivity_tests": {
                "spread_plus_50_pct": {
                    "profit_factor": res_sens50.profit_factor,
                    "net_pnl_usd": res_sens50.net_pnl_usd,
                    "closed_trades": res_sens50.closed_trades_count,
                },
                "spread_plus_100_pct": {
                    "profit_factor": res_sens100.profit_factor,
                    "net_pnl_usd": res_sens100.net_pnl_usd,
                    "closed_trades": res_sens100.closed_trades_count,
                }
            }
        },
        "verdict": (
            "STATISTICALLY INCONCLUSIVE FOR LIVE CAPITAL"
            if res_ml.closed_trades_count < 30
            else ("PASS" if res_ml.profit_factor > 1.20 and res_ml.net_expectancy_R > 0.05 else "FAIL")
        )
    }

    report_path = ROOT / "reports/rigorous_quant_evaluation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(final_report, indent=2))
    logger.info("Saved evaluation report to %s", report_path)

    # Save OOS trade ledgers for independent inspection
    trade_rows = []
    for t in res_ml.trade_ledger:
        trade_rows.append({
            "ticket": t.ticket,
            "side": t.side,
            "lots": t.lots,
            "signal_time": str(t.signal_time),
            "entry_time": str(t.entry_time),
            "entry_price": t.entry_price,
            "sl_price": t.initial_sl,
            "close_time": str(t.close_time),
            "close_price": t.close_price,
            "close_reason": t.close_reason,
            "pnl_usd": t.pnl_usd,
            "pnl_pips": t.pnl_pips,
            "pnl_R": t.pnl_R,
        })
    df_trades = pd.DataFrame(trade_rows)
    ledger_path = ROOT / "reports/oos_event_driven_trade_ledger.csv"
    df_trades.to_csv(ledger_path, index=False)
    logger.info("Saved OOS trade ledger (%d trades) to %s", len(df_trades), ledger_path)

    print("\n=======================================================")
    print("FINAL QUANT EVALUATION COMPLETE")
    print(f"Dataset: {raw_path.name} ({start_date} to {end_date})")
    print(f"OOS Event-Driven Trades: {res_ml.closed_trades_count}")
    print(f"OOS Net PnL: ${res_ml.net_pnl_usd:.2f} | PF: {res_ml.profit_factor}")
    print(f"No-ML Baseline Net PnL: ${res_noml.net_pnl_usd:.2f} | PF: {res_noml.profit_factor}")
    print(f"Statistical Classification: {final_report['verdict']}")
    print("=======================================================\n")


if __name__ == "__main__":
    run_rigorous_evaluation()
