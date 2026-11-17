from __future__ import annotations

"""
Production Master Retraining Pipeline for XAUUSD.
Trains on the complete 13,521 M5 + 13,728 H1 dataset with all 6 upgrades:
1. Multi-Timeframe (H1 + M5) trend alignment
2. Post-Judas Liquidity Sweep confirmation
3. Session Killzones (London, NY, Asian Chop filter)
4. Dynamic ATR-normalized volatility bands
5. Order Flow Imbalance & CVD slope
6. Dual XGBoost + LightGBM Gradient Ensemble
"""

import json
import logging
import sys
from pathlib import Path
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MasterProductionTrainer")


def build_master_dataset() -> tuple[pd.DataFrame, pd.Series]:
    m5_path = ROOT / "data/raw/XAUUSD_M5_60d_max.csv"
    h1_path = ROOT / "data/raw/XAUUSD_1H_2years.csv"

    df_m5 = pd.read_csv(m5_path)
    df_h1 = pd.read_csv(h1_path)
    logger.info("Loaded %d M5 bars and %d H1 bars", len(df_m5), len(df_h1))

    # 1. Causal Higher-Timeframe (H1) Integration
    df_h1["timestamp"] = pd.to_datetime(df_h1["timestamp"])
    df_h1 = df_h1.sort_values("timestamp").reset_index(drop=True)
    df_h1["h1_ema_50"] = df_h1["close"].ewm(span=50).mean()
    df_h1["h1_ema_200"] = df_h1["close"].ewm(span=200).mean()
    
    # Strictly lag H1 by 1 closed bar: M5 bars only ever observe completed past H1 bars
    df_h1["htf_trend_bias"] = np.where(df_h1["close"] > df_h1["h1_ema_200"], 1, -1)
    df_h1["htf_trend_bias"] = df_h1["htf_trend_bias"].shift(1)
    df_h1["h1_dist_ema50"] = ((df_h1["close"] - df_h1["h1_ema_50"]) / df_h1["h1_ema_50"] * 100).shift(1)
    df_h1_sub = df_h1[["timestamp", "htf_trend_bias", "h1_dist_ema50"]].dropna()

    df_m5["timestamp"] = pd.to_datetime(df_m5["timestamp"])
    df_m5 = df_m5.sort_values("timestamp").reset_index(drop=True)
    # Causal backward asof merge
    df_m5 = pd.merge_asof(df_m5, df_h1_sub, on="timestamp", direction="backward")

    close = df_m5["close"].values
    high = df_m5["high"].values
    low = df_m5["low"].values
    vol = df_m5["volume"].values
    timestamps = df_m5["timestamp"]
    n = len(df_m5)

    # 2. M5 Indicators without Look-Ahead (.bfill strictly eliminated)
    df_m5["ema_9"] = pd.Series(close).ewm(span=9).mean()
    df_m5["ema_21"] = pd.Series(close).ewm(span=21).mean()
    df_m5["ema_50"] = pd.Series(close).ewm(span=50).mean()
    df_m5["dist_ema_50"] = (df_m5["close"] - df_m5["ema_50"]) / df_m5["ema_50"] * 100

    # Session Killzones
    hours = timestamps.dt.hour.values
    df_m5["is_london"] = np.isin(hours, [7, 8, 9, 10]).astype(int)
    df_m5["is_ny_overlap"] = np.isin(hours, [12, 13, 14, 15, 16]).astype(int)
    df_m5["is_prime_session"] = np.isin(hours, [7, 8, 9, 10, 12, 13, 14, 15, 16]).astype(int)
    df_m5["is_asian_chop"] = np.isin(hours, [0, 1, 2, 3, 4, 5, 6]).astype(int)

    # Volatility & ATR (True causal, no future fill)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    df_m5["atr"] = pd.Series(tr).rolling(14).mean()
    df_m5["atr_pct"] = df_m5["atr"] / df_m5["close"] * 100
    df_m5["volatility_expansion"] = df_m5["atr"] / pd.Series(df_m5["atr"]).rolling(50).mean()

    # Premium/Discount Depth & Sweeps (Strictly shifted by 1 past bar)
    roll_h20 = pd.Series(high).rolling(20).max().shift(1).values
    roll_l20 = pd.Series(low).rolling(20).min().shift(1).values
    span = np.maximum(roll_h20 - roll_l20, 0.1)
    df_m5["discount_depth"] = (close - roll_l20) / span
    df_m5["sweep_high_confirmed"] = ((high > roll_h20) & (close < roll_h20)).astype(int)
    df_m5["sweep_low_confirmed"] = ((low < roll_l20) & (close > roll_l20)).astype(int)

    # Order Flow Proxies (Causal signed tick volume, honest names)
    p_diff = np.diff(close, prepend=close[0])
    v_delta = np.where(p_diff >= 0, vol, -vol)
    df_m5["signed_tick_volume_proxy"] = v_delta
    df_m5["cvd"] = np.cumsum(v_delta)
    df_m5["cvd_slope"] = pd.Series(df_m5["cvd"]).diff(10)

    rolling_buy = pd.Series(np.where(p_diff >= 0, vol, 0.0)).rolling(14).sum()
    rolling_tot = pd.Series(vol).rolling(14).sum()
    df_m5["signed_buyer_proxy_pct"] = (rolling_buy / np.maximum(rolling_tot, 1.0)) * 100

    # RSI (No future fill)
    delta_s = pd.Series(close).diff()
    gain = delta_s.where(delta_s > 0, 0.0).rolling(14).mean()
    loss = (-delta_s.where(delta_s < 0, 0.0)).rolling(14).mean()
    rs = gain / np.maximum(loss, 1e-6)
    df_m5["rsi"] = 100 - (100 / (1 + rs))

    # Rigorous Target Labeling (BE is NOT a Win, SL before TP is 0)
    targets = []
    for i in range(n):
        if i >= n - 15:
            targets.append(0)
            continue

        entry = close[i]
        curr_atr = df_m5["atr"].iloc[i] if pd.notnull(df_m5["atr"].iloc[i]) else 3.0
        sl_dist = max(curr_atr * 1.25, 3.0) # Realistic SL buffer
        tp_dist = sl_dist * 1.5           # 1:1.5 Risk-Reward Target

        f_high = high[i+1 : min(n, i+16)]
        f_low = low[i+1 : min(n, i+16)]

        # Setup direction determination
        is_short = (df_m5["discount_depth"].iloc[i] > 0.65) or (df_m5["sweep_high_confirmed"].iloc[i] == 1) or (df_m5["htf_trend_bias"].iloc[i] == -1)

        hit_win = False
        hit_loss = False

        if is_short:
            current_sl = entry + sl_dist
            tp_target = entry - tp_dist

            for h_val, l_val in zip(f_high, f_low):
                # Conservative same-bar tie-breaker: if both hit in the same bar, assume SL hit first
                if h_val >= current_sl and l_val <= tp_target:
                    hit_loss = True
                    break
                if h_val >= current_sl:
                    hit_loss = True
                    break
                if l_val <= tp_target:
                    hit_win = True
                    break
        else:
            current_sl = entry - sl_dist
            tp_target = entry + tp_dist

            for h_val, l_val in zip(f_high, f_low):
                # Conservative same-bar tie-breaker
                if l_val <= current_sl and h_val >= tp_target:
                    hit_loss = True
                    break
                if l_val <= current_sl:
                    hit_loss = True
                    break
                if h_val >= tp_target:
                    hit_win = True
                    break

        # STRICT TARGET: 1 ONLY if TP hit without hitting SL. BE or timeouts are 0!
        targets.append(1 if hit_win else 0)

    df_m5["target"] = targets

    # Drop warm-up rows (eliminates all NaNs causally without lookahead)
    feature_cols = [
        "htf_trend_bias", "h1_dist_ema50", "dist_ema_50", "is_prime_session", "is_asian_chop",
        "atr_pct", "volatility_expansion", "discount_depth",
        "sweep_high_confirmed", "sweep_low_confirmed",
        "signed_tick_volume_proxy", "cvd_slope", "signed_buyer_proxy_pct", "rsi"
    ]
    df_clean = df_m5[feature_cols + ["target"]].dropna().reset_index(drop=True)
    logger.info("Clean dataset after discarding warm-up rows: %d samples", len(df_clean))
    
    X = df_clean[feature_cols].copy()
    y = df_clean["target"].copy()
    return X, y


def train_and_deploy():
    X, y = build_master_dataset()
    logger.info("Total Samples: %d | Overall Non-Loss Win Rate: %.1f%%", len(X), y.mean() * 100)

    # 80/20 TimeSeries Split (Train on past, test on unseen latest 2,700 candles)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    logger.info("Train: %d samples | Test (Unseen): %d samples", len(X_train), len(X_test))

    # Train XGBoost
    scale_w = float((y_train == 0).sum()) / max(1.0, float((y_train == 1).sum()))
    xgb_clf = xgb.XGBClassifier(
        n_estimators=220,
        max_depth=5,
        learning_rate=0.035,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=scale_w,
        random_state=42,
        eval_metric="logloss"
    )
    xgb_clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # Train LightGBM
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=220,
        max_depth=5,
        learning_rate=0.035,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=scale_w,
        random_state=42,
        verbosity=-1
    )
    lgb_clf.fit(X_train, y_train)

    # Ensemble Evaluation on Unseen Test Set
    p_xgb = xgb_clf.predict_proba(X_test)[:, 1]
    p_lgb = lgb_clf.predict_proba(X_test)[:, 1]
    p_ensemble = (p_xgb * 0.55) + (p_lgb * 0.45)

    test_preds = (p_ensemble >= 0.50).astype(int)
    acc = accuracy_score(y_test, test_preds)
    roc = roc_auc_score(y_test, p_ensemble)

    # High Confidence Win Rate (Threshold >= 0.70)
    high_conf = p_ensemble >= 0.65
    high_conf_win_rate = (y_test[high_conf].sum() / max(1, high_conf.sum())) * 100
    high_conf_count = int(high_conf.sum())

    # Feature Importances
    imp_xgb = xgb_clf.feature_importances_ / np.sum(xgb_clf.feature_importances_)
    imp_lgb = lgb_clf.feature_importances_ / np.sum(lgb_clf.feature_importances_)
    ensemble_imp = dict(zip(X.columns, [round(float((x + l) / 2.0), 4) for x, l in zip(imp_xgb, imp_lgb)]))

    results = {
        "dataset_candles": len(X),
        "test_candles": len(X_test),
        "overall_accuracy_pct": round(acc * 100, 2),
        "roc_auc_score": round(roc, 3),
        "high_confidence_win_rate_pct": round(high_conf_win_rate, 2),
        "high_confidence_trades": high_conf_count,
        "feature_importances": ensemble_imp,
    }

    # Save to Master Ensemble
    out_pkg = {
        "xgb_model": xgb_clf,
        "lgb_model": lgb_clf,
        "feature_cols": list(X.columns),
        "metrics": results,
    }
    out_path = ROOT / "models/xauusd_master_ensemble.pkl"
    joblib.dump(out_pkg, out_path)
    logger.info("Saved Retrained Master Ensemble to %s", out_path)

    report_path = ROOT / "reports/master_ensemble_production_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*55)
    print("      MASTER PRODUCTION RETRAINING COMPLETE         ")
    print("="*55)
    print(f"Total Live M5 Bars Analyzed:     {results['dataset_candles']:,}")
    print(f"Unseen Test Bars Evaluated:      {results['test_candles']:,}")
    print(f"Test Accuracy:                   {results['overall_accuracy_pct']}%")
    print(f"High-Confidence Trade Win Rate:  {results['high_confidence_win_rate_pct']}%")
    print(f"Ensemble ROC-AUC Score:          {results['roc_auc_score']}")
    print("\nDominant Feature Importances:")
    for feat, imp in sorted(ensemble_imp.items(), key=lambda x: x[1], reverse=True):
        print(f"  • {feat:<24}: {imp * 100:.1f}%")
    print("="*55 + "\n")

    return results


if __name__ == "__main__":
    train_and_deploy()
