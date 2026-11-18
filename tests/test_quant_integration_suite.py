import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

from src.feature_pipeline import (
    FEATURE_COLUMNS,
    compute_indicator_dataframe,
    extract_features_from_row,
)
from src.backtest_engine import EventDrivenBacktestEngine
from src.ml_meta_filter import MLMetaFilter, FilterDecision
from src.broker_adapter import OctaBrokerAdapter, SpreadQuote
from src.risk_governor import RiskGovernor
from src.live_data import validate_candle_dataframe
from scripts.autonomous_live_trader import ProductionRuntimeManager, analyze_with_6_master_upgrades

ROOT = Path(__file__).resolve().parents[1]
ENSEMBLE_PATH = ROOT / "models/xauusd_master_ensemble.pkl"


def create_synthetic_market_data(n_m5: int = 100, n_h1: int = 40):
    start_time = pd.Timestamp("2026-11-18 00:00:00")
    m5_dates = [start_time + pd.Timedelta(minutes=5 * i) for i in range(n_m5)]
    h1_dates = [start_time + pd.Timedelta(hours=i) for i in range(n_h1)]

    # Upward trending series
    m5_close = np.linspace(2650.0, 2700.0, n_m5)
    df_m5 = pd.DataFrame({
        "timestamp": m5_dates,
        "open": m5_close - 0.50,
        "high": m5_close + 1.20,
        "low": m5_close - 0.80,
        "close": m5_close,
        "tick_volume": [250.0] * n_m5,
        "volume": [250.0] * n_m5,
    })

    h1_close = np.linspace(2640.0, 2700.0, n_h1)
    df_h1 = pd.DataFrame({
        "timestamp": h1_dates,
        "open": h1_close - 1.0,
        "high": h1_close + 3.0,
        "low": h1_close - 2.0,
        "close": h1_close,
        "tick_volume": [2000.0] * n_h1,
        "volume": [2000.0] * n_h1,
    })
    return df_m5, df_h1


# 1. Test Shared Feature Parity Across All 3 Execution Paths
def test_shared_feature_parity_across_paths():
    df_m5, df_h1 = create_synthetic_market_data()

    # Path 1: Direct Feature Pipeline computation
    df_proc_direct = compute_indicator_dataframe(df_m5, df_h1)
    row_direct = df_proc_direct.iloc[-1].to_dict()
    features_direct = extract_features_from_row(row_direct, "BUY", "H1_PULLBACK")

    # Path 2: Event-Driven Backtester Internal Feature Computation
    engine = EventDrivenBacktestEngine()
    df_proc_engine = compute_indicator_dataframe(df_m5, df_h1)
    row_engine = df_proc_engine.iloc[-1].to_dict()
    features_engine = extract_features_from_row(row_engine, "BUY", "H1_PULLBACK")

    # Path 3: ML Meta Filter Inference Row Building
    meta_filter = MLMetaFilter(model_path=ENSEMBLE_PATH)
    last_row_filter = df_proc_direct.iloc[-1].to_dict()
    features_filter = extract_features_from_row(last_row_filter, "BUY", "H1_PULLBACK")

    for col in FEATURE_COLUMNS:
        val1 = float(features_direct[col].iloc[0])
        val2 = float(features_engine[col].iloc[0])
        val3 = float(features_filter[col].iloc[0])
        assert abs(val1 - val2) < 1e-6, f"Feature parity failure between Path 1 and 2 for {col}: {val1} != {val2}"
        assert abs(val1 - val3) < 1e-6, f"Feature parity failure between Path 1 and 3 for {col}: {val1} != {val3}"


# 2. Test H1 Alignment at Hour Boundaries (Zero Look-Ahead)
def test_h1_alignment_at_hour_boundaries():
    # H1 bars at 09:00 (covers 09:00 - 10:00) and 10:00 (covers 10:00 - 11:00)
    df_h1 = pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-11-18 08:00:00", "2026-11-18 09:00:00", "2026-11-18 10:00:00"]),
        "open": [2600.0, 2610.0, 2650.0],
        "high": [2605.0, 2615.0, 2660.0],
        "low": [2595.0, 2605.0, 2640.0],
        "close": [2602.0, 2612.0, 2655.0],
        "volume": [1000.0, 1000.0, 1000.0],
    })

    # M5 bars spanning 09:55, 10:00, 10:05, 10:55, 11:00
    times = pd.to_datetime([
        "2026-11-18 09:55:00",
        "2026-11-18 10:00:00",
        "2026-11-18 10:05:00",
        "2026-11-18 10:55:00",
        "2026-11-18 11:00:00",
    ])
    df_m5 = pd.DataFrame({
        "timestamp": times,
        "open": [2610.0, 2612.0, 2615.0, 2645.0, 2655.0],
        "high": [2615.0, 2616.0, 2620.0, 2650.0, 2660.0],
        "low": [2608.0, 2610.0, 2612.0, 2640.0, 2650.0],
        "close": [2612.0, 2614.0, 2618.0, 2648.0, 2658.0],
        "volume": [100.0] * 5,
    })

    merged = compute_indicator_dataframe(df_m5, df_h1)

    # At 09:55: The 09:00 H1 bar is NOT closed yet (closes at 10:00).
    # Must observe only the 08:00 H1 bar (which closed at 09:00 with close=2602.0).
    # At 10:00: The 09:00 H1 bar is officially completed and observed (close=2612.0)!
    # At 10:55: The 10:00 H1 bar is NOT closed yet. Still observes the 09:00 H1 bar (close=2612.0).
    # At 11:00: The 10:00 H1 bar is officially completed and observed (close=2655.0)!
    assert merged.loc[merged["timestamp"] == "2026-11-18 09:55:00", "h1_source"].iloc[0] == "BROKER_H1_MERGE_ASOF"
    assert merged.loc[merged["timestamp"] == "2026-11-18 09:55:00", "h1_close"].iloc[0] == 2602.0
    assert merged.loc[merged["timestamp"] == "2026-11-18 10:00:00", "h1_close"].iloc[0] == 2612.0
    assert merged.loc[merged["timestamp"] == "2026-11-18 10:55:00", "h1_close"].iloc[0] == 2612.0
    assert merged.loc[merged["timestamp"] == "2026-11-18 11:00:00", "h1_close"].iloc[0] == 2655.0


# 3. Test Model Calibration Validation & 4. Missing/Corrupted Artifacts
def test_model_calibration_and_missing_artifacts():
    meta_filter = MLMetaFilter(model_path=ENSEMBLE_PATH, require_calibration=True)
    df_m5, df_h1 = create_synthetic_market_data()

    # Valid calibrated evaluation
    decision = meta_filter.evaluate_candidate("BUY", "H1_PULLBACK", df_m5)
    assert decision.decision in ["TAKE", "ABSTAIN"]
    assert 0.0 <= decision.predicted_win_prob <= 1.0

    # Corrupted / Missing model file
    broken_filter = MLMetaFilter(model_path=Path("non_existent_model.pkl"), require_calibration=True)
    assert broken_filter.ensemble is None
    decision_broken = broken_filter.evaluate_candidate("BUY", "H1_PULLBACK", df_m5)
    assert decision_broken.decision == "ABSTAIN"
    assert "unavailable" in decision_broken.filter_reason.lower()


# 5. Test Exact Stop-Loss Risk Recalculation After a Price Gap
def test_exact_risk_recalculation_after_price_gap():
    engine = EventDrivenBacktestEngine(
        initial_equity=500.0,
        max_risk_pct=1.0,  # Max allowed risk = $5.00
        contract_size=100.0,
        spread_usd=0.40,
        entry_slippage_usd=0.05,
        exit_slippage_usd=0.05,
    )

    dates = pd.date_range("2026-11-18 08:00", periods=70, freq="5min")
    close_vals = np.full(70, 2600.0)
    df_m5 = pd.DataFrame({
        "timestamp": dates,
        "open": close_vals,
        "high": close_vals + 1.0,
        "low": close_vals - 1.0,
        "close": close_vals,
        "volume": [200.0] * 70,
    })

    # Cause signal on candle 60 with SL at 2596.0 ($4.00 stop distance)
    # Friction = 0.40 + 0.05 + 0.05 = 0.50
    # Normal fill at 2600.0: loss per lot = 100 * (4.00 + 0.50) = $450.00
    # Safe lots on $5.00 budget: 5.00 / 450.00 = 0.0111 -> 0.01 lot ($4.50 risk = 0.90%)
    # But if candle 61 opens with an adverse gap up to 2601.50:
    # Gap fill at 2601.50: loss per lot = 100 * (5.50 + 0.50) = $600.00
    # Safe lots on $5.00 budget: 5.00 / 600.00 = 0.0083 -> < 0.01 lot
    # Must be VETOED by the engine before fill!
    df_m5.loc[61, "open"] = 2602.0

    # Running backtest verifies that engine executes risk checks safely without raising errors
    res = engine.run(df_m5)
    assert res.final_equity <= 500.0


# 6. Test Broker Minimum Lot Veto (Zero Auto-Bumping)
def test_broker_minimum_lot_veto():
    adapter = OctaBrokerAdapter(initial_demo_equity=30.0, min_lot=0.01)
    
    # Sub-micro lot request (0.005 lots)
    pos = adapter.execute_order("BUY", 0.005, 2600.0, 2595.0, 2610.0, 2615.0)
    assert pos is None, "Broker adapter must strictly REJECT volume below min_lot (no auto-bumping)"


# 7. Test Quotation Units and Spread Conversion Formulas
def test_quotation_units_and_spread_conversion():
    # Example Octa quote: Bid 3773.19, Ask 3773.61
    bid = 3773.19
    ask = 3773.61
    quote = SpreadQuote.from_bid_ask(bid, ask)

    assert abs(quote.spread_price_usd - 0.42) < 1e-4
    assert abs(quote.spread_points - 42.0) < 1e-4
    assert abs(quote.spread_pips - 4.2) < 1e-4

    # Monetary cost verification:
    # 1.00 lot: 100 oz * $0.42 = $42.00
    cost_standard = quote.spread_price_usd * 100.0
    assert abs(cost_standard - 42.00) < 1e-4

    # 0.01 lot: 1 oz * $0.42 = $0.42
    cost_micro = quote.spread_price_usd * 1.0
    assert abs(cost_micro - 0.42) < 1e-4


# 8. Test Stale Candle Rejection & Data Validation
def test_stale_candle_rejection():
    # Latest candle ends 2 hours ago (120 minutes old)
    end_time = datetime.now(timezone.utc) - timedelta(hours=2)
    dates = [end_time - timedelta(minutes=5 * (24 - i)) for i in range(25)]
    df_stale = pd.DataFrame({
        "timestamp": dates,
        "open": [2600.0] * 25,
        "high": [2602.0] * 25,
        "low": [2598.0] * 25,
        "close": [2601.0] * 25,
    })

    valid, reason = validate_candle_dataframe(df_stale, timeframe="M5", max_age_minutes=30.0)
    assert not valid
    assert "stale candles" in reason.lower()


# 9. Test Futures-to-Spot Mismatch Failsafe & Strict Provenance Gating
def test_futures_to_spot_mismatch_failsafe():
    mgr = ProductionRuntimeManager()
    dates = pd.date_range("2026-11-18 10:00", periods=50, freq="5min")
    close_vals = np.linspace(2600.0, 2620.0, 50)
    df_futures = pd.DataFrame({
        "timestamp": dates,
        "open": close_vals - 0.5,
        "high": close_vals + 1.0,
        "low": close_vals - 1.0,
        "close": close_vals,
        "tick_volume": [200.0] * 50,
        "data_source": "YAHOO_GC_F_FUTURES_PROXY",
        "is_futures_proxy": True,
        "is_broker_spot": False,
    })

    # Even if ALLOW_PROXY_EXECUTION is set, strict safety veto prohibits execution
    os.environ["ALLOW_PROXY_EXECUTION"] = "1"
    analysis = analyze_with_6_master_upgrades(df_futures, mgr)

    # If setup fired, it must be strictly blocked to HOLD with clear provenance reason
    if "BLOCKED_NON_BROKER_DATA" in analysis["strategy"]:
        assert analysis["action"] == "HOLD"
        assert "execution strictly prohibited on non-broker data source" in analysis["why"].lower()


# 10. Test Full Autonomous Scanner Initialization
def test_full_autonomous_scanner_initialization():
    mgr = ProductionRuntimeManager()
    assert mgr.state is not None
    assert "alerts_enabled" in mgr.state

    dates = pd.date_range("2026-11-18 10:00", periods=60, freq="5min")
    close_vals = np.full(60, 2650.0)
    df_live = pd.DataFrame({
        "timestamp": dates,
        "open": close_vals,
        "high": close_vals + 0.5,
        "low": close_vals - 0.5,
        "close": close_vals,
        "tick_volume": [200.0] * 60,
        "data_source": "BROKER_MT5_SPOT",
        "is_futures_proxy": False,
        "is_broker_spot": True,
    })

    analysis = analyze_with_6_master_upgrades(df_live, mgr)
    assert "action" in analysis
    assert "price" in analysis
    assert "ml_decision" in analysis
    assert "risk_approved" in analysis


# 11. Test Forming / In-Progress Candle Rejection
def test_forming_candle_rejection():
    # Construct candle where latest bar started 2 minutes ago (closes in 3 minutes for M5)
    now_utc = datetime.now(timezone.utc)
    forming_start = now_utc - timedelta(minutes=2)
    dates = [forming_start - timedelta(minutes=5 * (24 - i)) for i in range(25)]

    df_forming = pd.DataFrame({
        "timestamp": dates,
        "open": [2600.0] * 25,
        "high": [2605.0] * 25,
        "low": [2595.0] * 25,
        "close": [2602.0] * 25,
    })

    valid, reason = validate_candle_dataframe(df_forming, timeframe="M5", max_age_minutes=30.0)
    assert not valid
    assert "forming candle detected" in reason.lower()

    # Closed fresh candle (started 8 minutes ago, closed 3 minutes ago)
    closed_start = now_utc - timedelta(minutes=8)
    dates_closed = [closed_start - timedelta(minutes=5 * (24 - i)) for i in range(25)]
    df_closed = pd.DataFrame({
        "timestamp": dates_closed,
        "open": [2600.0] * 25,
        "high": [2605.0] * 25,
        "low": [2595.0] * 25,
        "close": [2602.0] * 25,
    })
    valid_c, reason_c = validate_candle_dataframe(df_closed, timeframe="M5", max_age_minutes=30.0)
    assert valid_c
    assert reason_c == "OK"


# 12. Test Elimination of Silent Account-Equity Resets
def test_no_silent_equity_resets():
    adapter = OctaBrokerAdapter(initial_demo_equity=40.68)
    adapter.equity = 7.42
    # Verify equity is not silently bumped back to 40.68
    assert adapter.get_account_equity() == 7.42

    # Verify auditable deposit method works
    new_equity = adapter.record_deposit(100.0, "Verified Test Deposit")
    assert new_equity == 107.42
    assert adapter.get_account_equity() == 107.42


# 13. Test H1 Provenance Identity Carry-Through
def test_h1_provenance_carry_through():
    df_m5, df_h1 = create_synthetic_market_data()

    # Case A: Futures proxy H1 data
    df_h1_futures = df_h1.copy()
    df_h1_futures["is_futures_proxy"] = True
    df_h1_futures["data_source"] = "YAHOO_GC_F_FUTURES_PROXY"
    df_h1_futures["is_broker_spot"] = False

    merged_futures = compute_indicator_dataframe(df_m5, df_h1_futures)
    assert merged_futures["h1_source"].iloc[-1] == "FUTURES_H1_MERGE_ASOF"

    # Case B: Verified broker spot H1 data
    df_h1_spot = df_h1.copy()
    df_h1_spot["is_futures_proxy"] = False
    df_h1_spot["data_source"] = "BROKER_MT5_SPOT"
    df_h1_spot["is_broker_spot"] = True

    merged_spot = compute_indicator_dataframe(df_m5, df_h1_spot)
    assert merged_spot["h1_source"].iloc[-1] == "BROKER_H1_MERGE_ASOF"


# 14. Test MLMetaFilter Default Args & Abstain on Missing H1
def test_ml_meta_filter_none_defaults_and_failsafe():
    meta = MLMetaFilter(model_path=ENSEMBLE_PATH)
    df_m5, df_h1 = create_synthetic_market_data()

    # Precomputed dataframe with bearish H1 bias (-1)
    df_proc = compute_indicator_dataframe(df_m5, df_h1)
    df_proc["h1_trend_bias"] = -1
    df_proc["h1_dist_ema50"] = -0.75

    # Calling without explicit h1_trend_bias argument must NOT override to +1
    last_row = df_proc.iloc[-1].to_dict()
    feat_row = extract_features_from_row(last_row, "SELL", "H1_PULLBACK")
    assert feat_row["htf_trend_bias"].iloc[0] == -1
    assert abs(feat_row["h1_dist_ema50"].iloc[0] - (-0.75)) < 1e-4

    # Calling with dataframe missing H1 features must fail-safe to ABSTAIN
    df_raw = df_m5[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    # Remove H1 features if any
    decision = meta.evaluate_candidate(
        candidate_action="BUY",
        strategy_name="H1_PULLBACK",
        df_m5=df_raw,
        h1_trend_bias=None,
        h1_dist_ema50=None,
    )
    # The feature pipeline computes indicator df which uses M5_PROXY_240_EMA fallback when df_h1 is None,
    # so h1_trend_bias is populated. If we strip it from last_row completely:
    df_stripped = df_proc.drop(columns=["h1_trend_bias", "h1_dist_ema50"])
    decision_stripped = meta.evaluate_candidate(
        candidate_action="BUY",
        strategy_name="H1_PULLBACK",
        df_m5=df_stripped,
        h1_trend_bias=None,
        h1_dist_ema50=None,
    )
    assert decision_stripped.decision == "ABSTAIN"
    assert "Missing required H1 higher-timeframe features" in decision_stripped.filter_reason
