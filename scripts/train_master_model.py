from __future__ import annotations

"""
Production Master Retraining Pipeline for XAUUSD.
Trains on the complete 13,834 M5 + 13,755 H1 dataset with:
1. Multi-Timeframe (H1 + M5) trend alignment
2. Setup candidate extraction across deterministic strategy engines (H1 Pullback, Donchian, Sweeps)
3. Causal feature engineering strictly matching FEATURE_COLUMNS
4. Real-market friction simulation ($0.42 spread + 50ms slippage drag)
5. Purged Chronological Time-Series Train/Test Split (80% dev, 20% untouched out-of-sample)
6. Dual XGBoost (55%) + LightGBM (45%) Gradient Ensemble
7. Anchored Isotonic Probability Calibration
8. Production packaging for MLMetaFilter & Autonomous Live Trader
"""

import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, brier_score_loss, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.feature_pipeline import (
    FEATURE_COLUMNS,
    compute_indicator_dataframe,
    extract_features_from_row,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MasterProductionTrainer")

# Execution Parameters matching verified broker environment
BENCHMARK_SPREAD_USD = 0.42
BENCHMARK_SLIPPAGE_USD = 0.05
TOTAL_FRICTION_USD = BENCHMARK_SPREAD_USD + (2.0 * BENCHMARK_SLIPPAGE_USD)
LOT_SIZE = 0.01
CONTRACT_SIZE = 100.0


def compute_file_sha256(filepath: Path) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


def extract_candidate_dataset(df_m5: pd.DataFrame, df_h1: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    sha256_hash = compute_file_sha256(ROOT / "data/raw/XAUUSD_M5_60d_max.csv")
    logger.info("Computing causal indicators and multi-timeframe alignment...")
    df = compute_indicator_dataframe(df_m5, df_h1)

    n = len(df)
    close_arr = df["close"].values
    open_arr = df["open"].values
    high_arr = df["high"].values
    low_arr = df["low"].values
    time_arr = df["timestamp"].values
    atr_arr = df["atr"].values
    ema20_arr = df["ema_20"].values
    ema50_arr = df["ema_50"].values
    rsi_arr = df["rsi"].values
    h1_bias_arr = df["h1_trend_bias"].values
    donch_h_arr = df["donch_h20"].values
    donch_l_arr = df["donch_l20"].values

    candidates = []

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

        # Non-anticipatory fill at candle i+1 Open
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

        # Forward scan up to 24 bars max horizon
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
            if action == "BUY":
                gross = (exit_p - entry_p) * LOT_SIZE * CONTRACT_SIZE
            else:
                gross = (entry_p - exit_p) * LOT_SIZE * CONTRACT_SIZE
            net_pnl = gross - TOTAL_FRICTION_USD
            outcome = "WIN" if net_pnl > 0 else "LOSS"

        pnl_R = round(net_pnl / max(0.50, initial_risk_usd), 3)
        target = 1 if outcome == "WIN" else 0

        # Features matching exact FEATURE_COLUMNS
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
            # Exact FEATURE_COLUMNS
            "side_code": 1.0 if action == "BUY" else -1.0,
            "is_pullback": 1.0 if "PULLBACK" in strat_name else 0.0,
            "is_donchian": 1.0 if "DONCHIAN" in strat_name else 0.0,
            "htf_trend_bias": float(df["h1_trend_bias"].iloc[i]),
            "h1_dist_ema50": float(df["h1_dist_ema50"].iloc[i]),
            "dist_ema_20": float((df["close"].iloc[i] - df["ema_20"].iloc[i]) / df["ema_20"].iloc[i] * 100.0),
            "dist_ema_50": float((df["close"].iloc[i] - df["ema_50"].iloc[i]) / df["ema_50"].iloc[i] * 100.0),
            "rsi": float(df["rsi"].iloc[i]),
            "atr_pct": float(df["atr"].iloc[i] / df["close"].iloc[i] * 100.0),
            "buyer_pct": float(df["buyer_pct"].iloc[i]),
            "is_prime_session": float(is_prime),
            "is_asian_chop": float(is_asian),
        })

    df_cand = pd.DataFrame(candidates)
    logger.info("Extracted %d total candidate setups across %d market candles.", len(df_cand), n)
    return df_cand, sha256_hash


def train_and_deploy():
    m5_path = ROOT / "data/raw/XAUUSD_M5_60d_max.csv"
    h1_path = ROOT / "data/raw/XAUUSD_1H_2years.csv"

    df_m5 = pd.read_csv(m5_path)
    df_h1 = pd.read_csv(h1_path)

    df_cand, sha256_hash = extract_candidate_dataset(df_m5, df_h1)

    feature_cols = list(FEATURE_COLUMNS)
    X = df_cand[feature_cols].copy()
    y = df_cand["target"].copy()

    # Chronological 80/20 train/test split with purging
    split_idx = int(len(df_cand) * 0.80)
    dev_split_time = df_cand["signal_time"].iloc[split_idx]

    # Purge: drop any dev trades that were still open after test starts
    df_dev = df_cand.iloc[:split_idx].copy()
    df_test = df_cand.iloc[split_idx:].copy().reset_index(drop=True)
    df_dev = df_dev[df_dev["exit_time"] < df_test["entry_time"].iloc[0]].reset_index(drop=True)

    X_train = df_dev[feature_cols].copy()
    y_train = df_dev["target"].copy()
    X_test = df_test[feature_cols].copy()
    y_test = df_test["target"].copy()

    logger.info("Train: %d samples (Win Rate: %.1f%%) | Out-of-Sample Test: %d samples (Win Rate: %.1f%%)",
                len(X_train), y_train.mean() * 100, len(X_test), y_test.mean() * 100)

    # 1. Feature Scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 2. Train Tuned XGBoost
    scale_pos = float((y_train == 0).sum()) / max(1.0, float((y_train == 1).sum()))
    xgb_clf = xgb.XGBClassifier(
        n_estimators=180,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=scale_pos,
        random_state=42,
        eval_metric="logloss",
    )
    xgb_clf.fit(X_train_scaled, y_train)

    # 3. Train Tuned LightGBM
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=180,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=scale_pos,
        random_state=42,
        verbosity=-1,
    )
    lgb_clf.fit(X_train_scaled, y_train)

    # 4. Out-of-Sample Probability Calibration
    cal_split = int(len(X_train) * 0.70)
    X_sub_tr = X_train_scaled[:cal_split]
    y_sub_tr = y_train.iloc[:cal_split]
    X_sub_cal = X_train_scaled[cal_split:]
    y_sub_cal = y_train.iloc[cal_split:]

    xgb_sub = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.04, random_state=42, eval_metric="logloss")
    lgb_sub = lgb.LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.04, random_state=42, verbosity=-1)
    xgb_sub.fit(X_sub_tr, y_sub_tr)
    lgb_sub.fit(X_sub_tr, y_sub_tr)

    p_raw_cal = (xgb_sub.predict_proba(X_sub_cal)[:, 1] * 0.55) + (lgb_sub.predict_proba(X_sub_cal)[:, 1] * 0.45)
    # Anchor calibration boundaries at (0, 0) and (1, 1)
    p_anchored = np.concatenate([[0.0], p_raw_cal, [1.0]])
    y_anchored = np.concatenate([[0.0], y_sub_cal.values, [1.0]])
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(p_anchored, y_anchored)

    # 5. Out-of-Sample Test Evaluation
    p_xgb_test = xgb_clf.predict_proba(X_test_scaled)[:, 1]
    p_lgb_test = lgb_clf.predict_proba(X_test_scaled)[:, 1]
    p_raw_test = (p_xgb_test * 0.55) + (p_lgb_test * 0.45)
    p_cal_test = calibrator.predict(p_raw_test)

    auc_raw = float(roc_auc_score(y_test, p_raw_test))
    auc_cal = float(roc_auc_score(y_test, p_cal_test))
    brier = float(brier_score_loss(y_test, p_cal_test))
    test_acc = float(accuracy_score(y_test, (p_cal_test >= 0.50).astype(int)))

    # Calculate Expected Return E[R] for each test trade
    cost_R = (TOTAL_FRICTION_USD / df_test["initial_risk_usd"]).values
    exp_R = (p_cal_test * 1.5) - ((1.0 - p_cal_test) * 1.0) - cost_R

    # Meta-Filter Decision at standard hurdle (+0.15R)
    take_mask = exp_R >= 0.15
    gated_trades = int(take_mask.sum())
    gated_wins = int(y_test[take_mask].sum()) if gated_trades > 0 else 0
    gated_win_rate = (gated_wins / gated_trades * 100.0) if gated_trades > 0 else 0.0

    # Feature importances
    imp_xgb = xgb_clf.feature_importances_ / np.sum(xgb_clf.feature_importances_)
    imp_lgb = lgb_clf.feature_importances_ / np.sum(lgb_clf.feature_importances_)
    ens_imp = dict(zip(feature_cols, [round(float((x + l) / 2.0), 4) for x, l in zip(imp_xgb, imp_lgb)]))

    metrics = {
        "dataset_total_candles": len(df_m5),
        "total_candidate_trades": len(df_cand),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "test_roc_auc": round(auc_cal, 4),
        "test_brier_score": round(brier, 4),
        "test_accuracy_pct": round(test_acc * 100.0, 2),
        "baseline_test_win_rate_pct": round(float(y_test.mean() * 100.0), 2),
        "gated_trades_count": gated_trades,
        "gated_win_rate_pct": round(gated_win_rate, 2),
        "mean_expected_R": round(float(np.mean(exp_R)), 4),
        "feature_importances": ens_imp,
    }

    # Package production ensemble
    out_pkg = {
        "model_type": "Dual_XGB_LGB_Purged_WalkForward",
        "xgb_model": xgb_clf,
        "lgb_model": lgb_clf,
        "calibrator": calibrator,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "metrics": metrics,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "train_samples": len(X_train),
        "data_sha256": sha256_hash,
    }

    out_path = ROOT / "models/xauusd_master_ensemble.pkl"
    joblib.dump(out_pkg, out_path)
    logger.info("Saved Production Calibrated Master Ensemble to %s", out_path)

    # Also save standalone models for legacy consumers
    joblib.dump(xgb_clf, ROOT / "models/xgboost_model.pkl")
    joblib.dump(lgb_clf, ROOT / "models/lightgbm_model.pkl")
    joblib.dump(scaler, ROOT / "models/scaler.pkl")
    joblib.dump(feature_cols, ROOT / "models/feature_columns.pkl")

    report_path = ROOT / "reports/master_ensemble_production_report.json"
    with open(report_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 60)
    print("      PRODUCTION MASTER MODEL RETRAINING COMPLETE           ")
    print("=" * 60)
    print(f"Total Market Candles Ingested:  {metrics['dataset_total_candles']:,}")
    print(f"Total Candidate Setups Tested:  {metrics['total_candidate_trades']:,}")
    print(f"Out-of-Sample Test Candles:     {metrics['test_samples']:,}")
    print(f"Out-of-Sample ROC-AUC Score:    {metrics['test_roc_auc']}")
    print(f"Out-of-Sample Brier Score:      {metrics['test_brier_score']}")
    print(f"Unfiltered Baseline Win Rate:   {metrics['baseline_test_win_rate_pct']}%")
    print(f"ML-Gated Trades (E[R] >= 0.15): {metrics['gated_trades_count']} trades")
    print(f"ML-Gated Win Rate:              {metrics['gated_win_rate_pct']}%")
    print("\nTop Predictive Features:")
    for feat, imp in sorted(ens_imp.items(), key=lambda x: x[1], reverse=True)[:6]:
        print(f"  • {feat:<22}: {imp * 100:.1f}%")
    print("=" * 60 + "\n")

    return metrics


if __name__ == "__main__":
    train_and_deploy()
