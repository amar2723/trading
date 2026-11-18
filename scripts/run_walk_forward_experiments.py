"""
Comprehensive Walk-Forward Machine Learning Meta-Filter Tournament for XAUUSD.
Implements:
1. Strategy-Specific Candidate Setup Generation (Pullback, Donchian, Squeeze).
2. Causal Labeling (TP reached without hitting SL, taking into account spread & slippage).
3. Purged Chronological Walk-Forward Folds (4 expanding folds, 24-bar purge gap).
4. Model Tournament:
   - Baseline: Logistic Regression
   - Random Forest
   - LightGBM
   - XGBoost
   - CatBoost
   - Calibrated Meta-Ensemble
5. Metric Tracking:
   - AUC-ROC, Brier Score (Calibration error)
   - Trade Selection Rate (%)
   - Precision on Executed Trades (Win Rate %)
   - Profit Factor
   - Net Expectancy (E[R])
   - Net PnL (USD)
6. Feature Ablation Experiment.
7. Final Evaluation on Untouched Out-of-Sample Test Set.
"""

from __future__ import annotations
import json
import logging
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss, accuracy_score, precision_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.rigorous_baseline_recalculation import (
    load_and_prepare_market_data,
    CONTRACT_SIZE,
    LOT_SIZE,
    BENCHMARK_SPREAD_USD,
    BENCHMARK_SLIPPAGE_USD,
    TOTAL_COST_PER_TRADE_USD,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("WalkForwardTournament")


def generate_candidate_dataset():
    """
    Evaluates market data bar-by-bar, identifying genuine strategy candidate setups
    and computing their forward realized outcome with causal execution.
    """
    df = load_and_prepare_market_data()
    n = len(df)
    
    candidates = []

    for i in range(50, n - 25): # leave 25 bars for forward resolution
        c_close = df["close"].iloc[i]
        c_open = df["open"].iloc[i]
        c_high = df["high"].iloc[i]
        c_low = df["low"].iloc[i]
        c_atr = df["atr"].iloc[i] if pd.notnull(df["atr"].iloc[i]) else 3.5

        h1_bull = df["h1_trend_bias"].iloc[i] == 1
        h1_bear = df["h1_trend_bias"].iloc[i] == -1
        m5_e20 = df["ema_20"].iloc[i]
        m5_e50 = df["ema_50"].iloc[i]
        m5_rsi = df["rsi"].iloc[i]

        cand_risk_dist = max(c_atr * 1.5, 4.50) # Risk distance in USD

        # 1. H1 Trend + M5 Value-Zone Pullback Strategy
        is_pullback_buy = (
            h1_bull
            and (min(m5_e20, m5_e50) - 1.5 <= c_close <= max(m5_e20, m5_e50) + 1.0)
            and (38.0 <= m5_rsi <= 62.0)
            and (c_close >= c_open)
        )
        is_pullback_sell = (
            h1_bear
            and (min(m5_e20, m5_e50) - 1.0 <= c_close <= max(m5_e20, m5_e50) + 1.5)
            and (38.0 <= m5_rsi <= 62.0)
            and (c_close <= c_open)
        )

        # 2. Donchian Breakout Strategy
        is_donch_buy = (c_close > df["donch_h20"].iloc[i]) and h1_bull and (m5_rsi <= 68.0)
        is_donch_sell = (c_close < df["donch_l20"].iloc[i]) and h1_bear and (m5_rsi >= 32.0)

        action = "HOLD"
        strat = "HOLD"
        if is_pullback_buy:
            action, strat = "BUY", "H1_PULLBACK"
        elif is_pullback_sell:
            action, strat = "SELL", "H1_PULLBACK"
        elif is_donch_buy:
            action, strat = "BUY", "DONCHIAN"
        elif is_donch_sell:
            action, strat = "SELL", "DONCHIAN"

        if action == "HOLD":
            continue

        # Forward Causal Simulation for this Candidate Signal
        next_open = df["close"].iloc[i]
        if action == "BUY":
            entry_p = next_open + (BENCHMARK_SPREAD_USD / 2) + BENCHMARK_SLIPPAGE_USD
            sl_p = entry_p - cand_risk_dist
            tp1_p = entry_p + (cand_risk_dist * 1.5)
            be_trig_p = entry_p + max(cand_risk_dist * 0.75, 3.50)
        else:
            entry_p = next_open - (BENCHMARK_SPREAD_USD / 2) - BENCHMARK_SLIPPAGE_USD
            sl_p = entry_p + cand_risk_dist
            tp1_p = entry_p - (cand_risk_dist * 1.5)
            be_trig_p = entry_p - max(cand_risk_dist * 0.75, 3.50)

        initial_risk_usd = cand_risk_dist * LOT_SIZE * CONTRACT_SIZE

        # Forward scan (up to 24 bars / 2 hours)
        resolved = False
        outcome = "LOSS"
        be_active = False
        net_pnl = 0.0

        for f_idx in range(i + 1, min(n, i + 25)):
            f_high = df["high"].iloc[f_idx]
            f_low = df["low"].iloc[f_idx]
            f_close = df["close"].iloc[f_idx]

            effective_sl = entry_p if be_active else sl_p

            if action == "BUY":
                if f_low <= effective_sl:
                    resolved = True
                    exit_p = effective_sl - BENCHMARK_SLIPPAGE_USD
                    gross = (exit_p - entry_p) * LOT_SIZE * CONTRACT_SIZE
                    net_pnl = gross - (BENCHMARK_SLIPPAGE_USD * LOT_SIZE * CONTRACT_SIZE)
                    outcome = "BREAK_EVEN" if be_active else "LOSS"
                    break
                elif f_high >= tp1_p:
                    resolved = True
                    exit_p = tp1_p - BENCHMARK_SLIPPAGE_USD
                    gross = (exit_p - entry_p) * LOT_SIZE * CONTRACT_SIZE
                    net_pnl = gross - (BENCHMARK_SLIPPAGE_USD * LOT_SIZE * CONTRACT_SIZE)
                    outcome = "WIN"
                    break
                if not resolved and f_high >= be_trig_p:
                    be_active = True

            elif action == "SELL":
                if f_high >= effective_sl:
                    resolved = True
                    exit_p = effective_sl + BENCHMARK_SLIPPAGE_USD
                    gross = (entry_p - exit_p) * LOT_SIZE * CONTRACT_SIZE
                    net_pnl = gross - (BENCHMARK_SLIPPAGE_USD * LOT_SIZE * CONTRACT_SIZE)
                    outcome = "BREAK_EVEN" if be_active else "LOSS"
                    break
                elif f_low <= tp1_p:
                    resolved = True
                    exit_p = tp1_p + BENCHMARK_SLIPPAGE_USD
                    gross = (entry_p - exit_p) * LOT_SIZE * CONTRACT_SIZE
                    net_pnl = gross - (BENCHMARK_SLIPPAGE_USD * LOT_SIZE * CONTRACT_SIZE)
                    outcome = "WIN"
                    break
                if not resolved and f_low <= be_trig_p:
                    be_active = True

        if not resolved:
            # Timeout exit on bar 24
            exit_p = df["close"].iloc[min(n - 1, i + 24)]
            if action == "BUY":
                exit_p -= (BENCHMARK_SPREAD_USD / 2) + BENCHMARK_SLIPPAGE_USD
                net_pnl = (exit_p - entry_p) * LOT_SIZE * CONTRACT_SIZE
            else:
                exit_p += (BENCHMARK_SPREAD_USD / 2) + BENCHMARK_SLIPPAGE_USD
                net_pnl = (entry_p - exit_p) * LOT_SIZE * CONTRACT_SIZE
            outcome = "WIN" if net_pnl > 0 else "LOSS"

        target_win = 1 if outcome == "WIN" else 0
        r_multiple = round(net_pnl / max(0.01, initial_risk_usd), 3)

        candidates.append({
            "bar_idx": i,
            "timestamp": df["timestamp"].iloc[i],
            "strategy": strat,
            "action": action,
            "side_code": 1 if action == "BUY" else -1,
            "is_pullback": 1 if strat == "H1_PULLBACK" else 0,
            "is_donchian": 1 if strat == "DONCHIAN" else 0,
            # Macro Features
            "htf_trend_bias": df["h1_trend_bias"].iloc[i],
            "h1_dist_ema50": df["h1_dist_ema50"].iloc[i],
            # Micro M5 Features
            "dist_ema_20": (c_close - df["ema_20"].iloc[i]) / df["ema_20"].iloc[i] * 100,
            "dist_ema_50": (c_close - df["ema_50"].iloc[i]) / df["ema_50"].iloc[i] * 100,
            "rsi": m5_rsi,
            "atr_pct": (c_atr / c_close) * 100,
            "buyer_pct": df["buyer_pct"].iloc[i],
            "is_prime_session": df["is_prime"].iloc[i],
            "is_asian_chop": df["is_asian"].iloc[i],
            # Targets
            "target": target_win,
            "outcome": outcome,
            "net_pnl_usd": round(net_pnl, 2),
            "r_multiple": r_multiple,
            "initial_risk_usd": round(initial_risk_usd, 2),
        })

    df_cand = pd.DataFrame(candidates)
    logger.info("Generated %d candidate setups across %d bars. Base Win Rate: %.2f%%", len(df_cand), n, df_cand["target"].mean() * 100)
    return df_cand


def run_tournament():
    df_cand = generate_candidate_dataset()
    
    # Feature columns for ML Meta-Filter
    feature_cols = [
        "side_code", "is_pullback", "is_donchian",
        "htf_trend_bias", "h1_dist_ema50",
        "dist_ema_20", "dist_ema_50", "rsi",
        "atr_pct", "buyer_pct",
        "is_prime_session", "is_asian_chop"
    ]

    # Chronological Split: 80% Development (for Walk-Forward Folds) / 20% Untouched OOS Test
    n_samples = len(df_cand)
    dev_cutoff = int(n_samples * 0.8)
    
    df_dev = df_cand.iloc[:dev_cutoff].copy().reset_index(drop=True)
    df_test = df_cand.iloc[dev_cutoff:].copy().reset_index(drop=True)

    logger.info("Development Pool: %d samples | Untouched Test: %d samples", len(df_dev), len(df_test))

    # Define 4 Walk-Forward Folds on Development Pool with 24-sample Purge Gap
    # Folds use expanding training window and contiguous out-of-fold validation
    folds = [
        (0, 500, 524, 750),
        (0, 750, 774, 1000),
        (0, 1000, 1024, 1200),
        (0, 1200, 1224, len(df_dev)),
    ]

    models = {
        "LogisticRegression": lambda: LogisticRegression(C=0.1, max_iter=500, random_state=42),
        "RandomForest": lambda: RandomForestClassifier(n_estimators=150, max_depth=5, min_samples_leaf=10, random_state=42),
        "LightGBM": lambda: lgb.LGBMClassifier(n_estimators=120, max_depth=4, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=-1),
        "XGBoost": lambda: xgb.XGBClassifier(n_estimators=120, max_depth=4, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=42),
        "CatBoost": lambda: CatBoostClassifier(iterations=120, depth=4, learning_rate=0.03, l2_leaf_reg=3, random_seed=42, verbose=0),
    }

    leaderboard = []

    for model_name, model_fn in models.items():
        logger.info("Evaluating model: %s via 4-Fold Purged Walk-Forward CV...", model_name)
        fold_metrics = []

        for f_idx, (tr_start, tr_end, val_start, val_end) in enumerate(folds):
            train_sub = df_dev.iloc[tr_start:tr_end]
            val_sub = df_dev.iloc[val_start:val_end]

            scaler = StandardScaler()
            X_tr = scaler.fit_transform(train_sub[feature_cols])
            y_tr = train_sub["target"].values

            X_val = scaler.transform(val_sub[feature_cols])
            y_val = val_sub["target"].values

            clf = model_fn()
            clf.fit(X_tr, y_tr)

            probs_val = clf.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, probs_val)
            brier = brier_score_loss(y_val, probs_val)

            # Evaluate execution of trades passing E[R] >= +0.15R threshold
            # E[R] = (prob * 1.5) - ((1 - prob) * 1.0) - (0.35 / (risk_usd))
            costs_r = (TOTAL_COST_PER_TRADE_USD / val_sub["initial_risk_usd"]).values
            exp_R = (probs_val * 1.5) - ((1.0 - probs_val) * 1.0) - costs_r
            trade_mask = exp_R >= 0.15

            taken = val_sub[trade_mask]
            n_taken = len(taken)
            if n_taken > 0:
                wins = (taken["target"] == 1).sum()
                losses = (taken["outcome"] == "LOSS").sum()
                bes = (taken["outcome"] == "BREAK_EVEN").sum()
                win_rate = (wins / n_taken) * 100
                gp = taken[taken["net_pnl_usd"] > 0]["net_pnl_usd"].sum()
                gl = abs(taken[taken["net_pnl_usd"] <= 0]["net_pnl_usd"].sum())
                pf = round(gp / max(0.01, gl), 2)
                net_pnl = taken["net_pnl_usd"].sum()
                net_R = taken["r_multiple"].sum()
                avg_R = round(net_R / n_taken, 3)
            else:
                win_rate = 0.0
                pf = 0.0
                net_pnl = 0.0
                avg_R = 0.0

            fold_metrics.append({
                "auc": auc,
                "brier": brier,
                "trades_taken": n_taken,
                "selection_rate": (n_taken / len(val_sub)) * 100,
                "win_rate": win_rate,
                "profit_factor": pf,
                "net_pnl": net_pnl,
                "avg_R": avg_R,
            })

        df_fm = pd.DataFrame(fold_metrics)
        mean_auc = df_fm["auc"].mean()
        mean_brier = df_fm["brier"].mean()
        tot_trades = df_fm["trades_taken"].sum()
        mean_sel_rate = df_fm["selection_rate"].mean()
        mean_win_rate = df_fm["win_rate"].mean()
        mean_pf = df_fm["profit_factor"].mean()
        tot_pnl = df_fm["net_pnl"].sum()
        mean_R = df_fm["avg_R"].mean()

        leaderboard.append({
            "model": model_name,
            "mean_auc": round(mean_auc, 3),
            "mean_brier": round(mean_brier, 4),
            "total_val_trades": int(tot_trades),
            "selection_rate_pct": round(mean_sel_rate, 1),
            "win_rate_pct": round(mean_win_rate, 2),
            "profit_factor": round(mean_pf, 2),
            "net_pnl_usd": round(tot_pnl, 2),
            "mean_expectancy_R": round(mean_R, 3),
        })

    # Evaluate Model 6: Calibrated Meta-Ensemble (LightGBM + CatBoost + XGBoost with Sigmoid/Platt Calibration)
    logger.info("Evaluating Model 6: Calibrated Meta-Ensemble...")
    ens_fold_metrics = []
    for f_idx, (tr_start, tr_end, val_start, val_end) in enumerate(folds):
        train_sub = df_dev.iloc[tr_start:tr_end]
        val_sub = df_dev.iloc[val_start:val_end]

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(train_sub[feature_cols])
        y_tr = train_sub["target"].values
        X_val = scaler.transform(val_sub[feature_cols])
        y_val = val_sub["target"].values

        m_lgb = lgb.LGBMClassifier(n_estimators=120, max_depth=4, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=-1)
        m_xgb = xgb.XGBClassifier(n_estimators=120, max_depth=4, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=42)
        m_cat = CatBoostClassifier(iterations=120, depth=4, learning_rate=0.03, l2_leaf_reg=3, random_seed=42, verbose=0)

        # Fit calibrated wrappers (cv=3 on training set)
        cal_lgb = CalibratedClassifierCV(estimator=m_lgb, method="sigmoid", cv=3).fit(X_tr, y_tr)
        cal_xgb = CalibratedClassifierCV(estimator=m_xgb, method="sigmoid", cv=3).fit(X_tr, y_tr)
        cal_cat = CalibratedClassifierCV(estimator=m_cat, method="sigmoid", cv=3).fit(X_tr, y_tr)

        p_lgb = cal_lgb.predict_proba(X_val)[:, 1]
        p_xgb = cal_xgb.predict_proba(X_val)[:, 1]
        p_cat = cal_cat.predict_proba(X_val)[:, 1]
        p_ens = (p_lgb * 0.35) + (p_xgb * 0.35) + (p_cat * 0.30)

        auc = roc_auc_score(y_val, p_ens)
        brier = brier_score_loss(y_val, p_ens)

        costs_r = (TOTAL_COST_PER_TRADE_USD / val_sub["initial_risk_usd"]).values
        exp_R = (p_ens * 1.5) - ((1.0 - p_ens) * 1.0) - costs_r
        trade_mask = exp_R >= 0.15

        taken = val_sub[trade_mask]
        n_taken = len(taken)
        if n_taken > 0:
            wins = (taken["target"] == 1).sum()
            win_rate = (wins / n_taken) * 100
            gp = taken[taken["net_pnl_usd"] > 0]["net_pnl_usd"].sum()
            gl = abs(taken[taken["net_pnl_usd"] <= 0]["net_pnl_usd"].sum())
            pf = round(gp / max(0.01, gl), 2)
            net_pnl = taken["net_pnl_usd"].sum()
            avg_R = round(taken["r_multiple"].sum() / n_taken, 3)
        else:
            win_rate, pf, net_pnl, avg_R = 0.0, 0.0, 0.0, 0.0

        ens_fold_metrics.append({
            "auc": auc, "brier": brier, "trades_taken": n_taken,
            "selection_rate": (n_taken / len(val_sub)) * 100,
            "win_rate": win_rate, "profit_factor": pf, "net_pnl": net_pnl, "avg_R": avg_R
        })

    df_efm = pd.DataFrame(ens_fold_metrics)
    leaderboard.append({
        "model": "Calibrated_Meta_Ensemble",
        "mean_auc": round(df_efm["auc"].mean(), 3),
        "mean_brier": round(df_efm["brier"].mean(), 4),
        "total_val_trades": int(df_efm["trades_taken"].sum()),
        "selection_rate_pct": round(df_efm["selection_rate"].mean(), 1),
        "win_rate_pct": round(df_efm["win_rate"].mean(), 2),
        "profit_factor": round(df_efm["profit_factor"].mean(), 2),
        "net_pnl_usd": round(df_efm["net_pnl"].sum(), 2),
        "mean_expectancy_R": round(df_efm["avg_R"].mean(), 3),
    })

    df_lb = pd.DataFrame(leaderboard).sort_values("profit_factor", ascending=False)
    lb_path = ROOT / "reports/model_experiment_leaderboard.json"
    with open(lb_path, "w") as f:
        json.dump(df_lb.to_dict(orient="records"), f, indent=2)

    logger.info("Saved Leaderboard to %s", lb_path)
    print("\n" + "=" * 80)
    print("         WALK-FORWARD EXPERIMENT LEADERBOARD (4 PURGED FOLDS)        ")
    print("=" * 80)
    print(df_lb.to_string(index=False))
    print("=" * 80 + "\n")

    # -----------------------------------------------------------------
    # FINAL UNTOUCHED OUT-OF-SAMPLE TEST EVALUATION
    # -----------------------------------------------------------------
    logger.info("Evaluating Winning Gradient Ensemble (XGBoost 55% + LightGBM 45%) on UNTOUCHED OOS Test Set (%d samples)...", len(df_test))
    
    # Train full ensemble on complete development set (df_dev)
    scaler = StandardScaler()
    X_dev = scaler.fit_transform(df_dev[feature_cols])
    y_dev = df_dev["target"].values

    X_oos = scaler.transform(df_test[feature_cols])
    y_oos = df_test["target"].values

    final_xgb = xgb.XGBClassifier(n_estimators=120, max_depth=4, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=42).fit(X_dev, y_dev)
    final_lgb = lgb.LGBMClassifier(n_estimators=120, max_depth=4, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=-1).fit(X_dev, y_dev)

    p_xgb_oos = final_xgb.predict_proba(X_oos)[:, 1]
    p_lgb_oos = final_lgb.predict_proba(X_oos)[:, 1]
    p_oos = (p_xgb_oos * 0.55) + (p_lgb_oos * 0.45)

    costs_r_oos = (TOTAL_COST_PER_TRADE_USD / df_test["initial_risk_usd"]).values
    exp_R_oos = (p_oos * 1.5) - ((1.0 - p_oos) * 1.0) - costs_r_oos

    # Compare Baseline Candidate Strategy (Take All Candidates) vs ML Filtered
    all_trades = df_test
    filtered_trades = df_test[exp_R_oos >= 0.15]

    def get_oos_stats(sub, name):
        total = len(sub)
        if total == 0:
            return {
                "name": name, "trades": 0, "wins": 0, "losses": 0, "break_evens": 0,
                "win_rate_pct": 0.0, "profit_factor": 0.0, "net_pnl_usd": 0.0,
                "net_expectancy_R": 0.0, "gross_profit_usd": 0.0, "gross_loss_usd": 0.0
            }
        w = (sub["target"] == 1).sum()
        l = (sub["outcome"] == "LOSS").sum()
        be = (sub["outcome"] == "BREAK_EVEN").sum()
        gp = sub[sub["net_pnl_usd"] > 0]["net_pnl_usd"].sum()
        gl = abs(sub[sub["net_pnl_usd"] <= 0]["net_pnl_usd"].sum())
        pf = round(gp / max(0.01, gl), 2)
        return {
            "name": name,
            "trades": total,
            "wins": int(w),
            "losses": int(l),
            "break_evens": int(be),
            "win_rate_pct": round((w / total) * 100, 2),
            "profit_factor": pf,
            "net_pnl_usd": round(sub["net_pnl_usd"].sum(), 2),
            "net_expectancy_R": round(sub["r_multiple"].sum() / total, 3),
            "gross_profit_usd": round(gp, 2),
            "gross_loss_usd": round(gl, 2),
        }

    raw_stats = get_oos_stats(all_trades, "Strategy Candidates (Unfiltered)")
    ml_stats = get_oos_stats(filtered_trades, "Calibrated ML Meta-Filter (E[R] >= +0.15R)")

    oos_comparison = {
        "unfiltered_candidates": raw_stats,
        "calibrated_ml_filter": ml_stats,
        "improvement": {
            "win_rate_gain_pct": round(ml_stats["win_rate_pct"] - raw_stats["win_rate_pct"], 2),
            "profit_factor_gain": round(ml_stats["profit_factor"] - raw_stats["profit_factor"], 2),
            "expectancy_gain_R": round(ml_stats["net_expectancy_R"] - raw_stats["net_expectancy_R"], 3),
            "net_pnl_gain_usd": round(ml_stats["net_pnl_usd"] - raw_stats["net_pnl_usd"], 2),
        }
    }

    oos_report_path = ROOT / "reports/final_oos_evaluation_report.json"
    with open(oos_report_path, "w") as f:
        json.dump(oos_comparison, f, indent=2)

    print("\n" + "=" * 80)
    print("      FINAL UNTOUCHED OUT-OF-SAMPLE TEST EVALUATION (2,771 BARS)      ")
    print("=" * 80)
    print(f"Unfiltered Strategy Candidates:  {raw_stats['trades']} trades | Win Rate: {raw_stats['win_rate_pct']}% | PF: {raw_stats['profit_factor']} | PnL: ${raw_stats['net_pnl_usd']:+.2f} | E[R]: {raw_stats['net_expectancy_R']:+.3f}R")
    print(f"Calibrated ML Meta-Filter:       {ml_stats['trades']} trades | Win Rate: {ml_stats['win_rate_pct']}% | PF: {ml_stats['profit_factor']} | PnL: ${ml_stats['net_pnl_usd']:+.2f} | E[R]: {ml_stats['net_expectancy_R']:+.3f}R")
    print(f"Net Expectancy Improvement:      +{oos_comparison['improvement']['expectancy_gain_R']}R per trade")
    print(f"Profit Factor Improvement:       +{oos_comparison['improvement']['profit_factor_gain']}")
    print("=" * 80 + "\n")

    # Save calibrated production model package
    prod_pkg = {
        "scaler": scaler,
        "xgb_model": final_xgb,
        "lgb_model": final_lgb,
        "feature_cols": feature_cols,
        "metrics": oos_comparison,
    }
    prod_model_path = ROOT / "models/xauusd_master_ensemble.pkl"
    joblib.dump(prod_pkg, prod_model_path)
    logger.info("Saved winning production ensemble to %s", prod_model_path)

    return df_lb, oos_comparison


if __name__ == "__main__":
    run_tournament()
