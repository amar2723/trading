from __future__ import annotations

import math
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
import pytest

from src.macro_compass import MacroCompass, MacroSnapshot
from src.realtime.economic_calendar import NewsCalendarFilter, NewsStatus
from src.liquidity_map import LiquidityMap, LiquidityReferenceLevel
from src.risk_governor import RiskGovernor, RiskApproval


def _generate_synthetic_h1_candles(num_days: int = 14) -> pd.DataFrame:
    """Generates synthetic H1 candles spanning multiple weeks for HTF testing."""
    periods = num_days * 24
    dates = pd.date_range("2026-11-09 00:00", periods=periods, freq="1h", tz="UTC")
    rows = []
    base_price = 2650.0

    for i, d in enumerate(dates):
        c = base_price + np.sin(i / 10.0) * 20.0 + (i % 24) * 0.5
        o = c - 1.0
        h = max(o, c) + 3.0
        l = min(o, c) - 3.0
        rows.append({
            "timestamp": d,
            "open": round(o, 2),
            "high": round(h, 2),
            "low": round(l, 2),
            "close": round(c, 2),
            "tick_volume": 1000.0,
            "is_broker_spot": True,
            "data_source": "BROKER_MT5_SPOT",
        })

    return pd.DataFrame(rows)


def _generate_synthetic_m5_candles(num_bars: int = 100) -> pd.DataFrame:
    """Generates synthetic M5 candles."""
    dates = pd.date_range("2026-11-23 00:00", periods=num_bars, freq="5min", tz="UTC")
    rows = []
    base_price = 2660.0

    for i, d in enumerate(dates):
        c = base_price + np.sin(i / 8.0) * 5.0
        o = c - 0.5
        h = max(o, c) + 1.5
        l = min(o, c) - 1.5
        rows.append({
            "timestamp": d,
            "open": round(o, 2),
            "high": round(h, 2),
            "low": round(l, 2),
            "close": round(c, 2),
            "tick_volume": 200.0,
            "is_broker_spot": True,
            "data_source": "BROKER_MT5_SPOT",
        })

    return pd.DataFrame(rows)


# =========================================================================
# 1. Macro Compass Tests
# =========================================================================

def test_macro_compass_synthetic_snapshot():
    """Validates snapshot fields, calculation logic, and regime classifications."""
    mc = MacroCompass(ttl_seconds=60)
    snap = mc.get_snapshot()

    assert isinstance(snap, MacroSnapshot)
    assert -1.0 <= snap.concordance_score <= 1.0
    assert snap.regime in ["BULLISH_TAILWIND", "BEARISH_HEADWIND", "NEUTRAL"]
    assert snap.dxy_price > 0.0
    assert snap.us10y_yield >= 0.0


def test_macro_compass_caching():
    """Validates that rapid consecutive calls return cached snapshot."""
    mc = MacroCompass(ttl_seconds=120)
    snap1 = mc.get_snapshot()
    snap2 = mc.get_snapshot()

    assert snap1.timestamp == snap2.timestamp
    assert snap1.concordance_score == snap2.concordance_score


def test_macro_compass_regime_thresholds():
    """Tests concordance regime assignment."""
    mc = MacroCompass()
    assert mc._classify_regime(0.50) == "BULLISH_TAILWIND"
    assert mc._classify_regime(0.25) == "BULLISH_TAILWIND"
    assert mc._classify_regime(-0.50) == "BEARISH_HEADWIND"
    assert mc._classify_regime(-0.25) == "BEARISH_HEADWIND"
    assert mc._classify_regime(0.0) == "NEUTRAL"
    assert mc._classify_regime(0.20) == "NEUTRAL"


# =========================================================================
# 2. Economic Calendar Tests
# =========================================================================

def test_news_calendar_clear_status():
    """Tests normal clear status when no high impact events are active."""
    nf = NewsCalendarFilter(lockout_minutes=30)
    status = nf.get_news_status()

    assert isinstance(status, NewsStatus)
    assert isinstance(status.is_locked, bool)
    assert isinstance(status.summary, str)


def test_news_calendar_lockout_synthetic_event():
    """Tests lockout when an event is artificially injected within the 30m window."""
    nf = NewsCalendarFilter(lockout_minutes=30)
    now_utc = datetime.now(timezone.utc)

    # Inject high impact USD event 10 minutes in the future
    event_time = now_utc + timedelta(minutes=10)
    nf.cached_events = [{
        "date": event_time.isoformat(),
        "title": "US Non-Farm Payrolls (NFP)",
        "country": "USD",
        "impact": "High",
    }]
    nf.last_fetch_time = now_utc

    status = nf.get_news_status(current_dt=now_utc)
    assert status.is_locked is True
    assert status.minutes_to_event is not None
    assert abs(status.minutes_to_event) <= 30
    assert "News Blackout" in status.summary
    assert "ACTIVE" in status.summary
    assert "NFP" in status.active_event


def test_news_calendar_cleared_after_window():
    """Tests that trading resumes after the 30-minute post-event window passes."""
    nf = NewsCalendarFilter(lockout_minutes=30)
    now_utc = datetime.now(timezone.utc)

    # Event occurred 45 minutes ago -> lockout should be False
    past_event_time = now_utc - timedelta(minutes=45)
    nf.cached_events = [{
        "date": past_event_time.isoformat(),
        "title": "US CPI YoY",
        "country": "USD",
        "impact": "High",
    }]
    nf.last_fetch_time = now_utc

    # Avoid heuristic window triggering by choosing 10:00 UTC
    eval_dt = now_utc.replace(hour=10, minute=0, second=0)
    status = nf.get_news_status(current_dt=eval_dt)
    assert status.is_locked is False
    assert "CLEAR" in status.summary


# =========================================================================
# 3. HTF Liquidity Levels Tests
# =========================================================================

def test_htf_liquidity_computation():
    """Tests PDH, PDL, PDC, PWH, PWL, PWC, and NYMO calculation from H1 data."""
    df_h1 = _generate_synthetic_h1_candles(num_days=14)
    lmap = LiquidityMap()
    htf_levels = lmap.compute_htf_levels(df_h1)

    assert len(htf_levels) >= 5
    level_types = {lvl.level_type for lvl in htf_levels}
    assert "PDH" in level_types, "Must detect Prior Day High"
    assert "PDL" in level_types, "Must detect Prior Day Low"
    assert "PDC" in level_types, "Must detect Prior Day Close"
    assert "PWH" in level_types, "Must detect Prior Week High"
    assert "PWL" in level_types, "Must detect Prior Week Low"
    assert "NYMO" in level_types, "Must detect New York Midnight Open"

    # All HTF levels must be flagged as TIER_1_INSTITUTIONAL
    for lvl in htf_levels:
        assert lvl.tier == "TIER_1_INSTITUTIONAL"


def test_watchlist_htf_incorporation():
    """Tests that watchlist incorporates HTF levels and tier labels."""
    df_m5 = _generate_synthetic_m5_candles(num_bars=100)
    df_h1 = _generate_synthetic_h1_candles(num_days=14)
    lmap = LiquidityMap()

    wl = lmap.get_pre_sweep_liquidity_watchlist(df_m5, df_h1=df_h1)

    assert "htf_levels" in wl
    htf = wl["htf_levels"]
    assert htf["pdh"] is not None
    assert htf["pdl"] is not None
    assert htf["pwh"] is not None
    assert htf["pwl"] is not None
    assert htf["nymo"] is not None

    # Verify pools have tier classification
    for pool in wl.get("all_bsl_pools", []):
        assert pool["tier"] in ["TIER_1_INSTITUTIONAL", "TIER_2_SESSION"]
    for pool in wl.get("all_ssl_pools", []):
        assert pool["tier"] in ["TIER_1_INSTITUTIONAL", "TIER_2_SESSION"]


# =========================================================================
# 4. Volatility-Targeted Risk Governor Tests
# =========================================================================

def test_volatility_targeting_k_vol():
    """Tests that position size scales inversely with market ATR."""
    gov = RiskGovernor(max_risk_pct=1.0, atr_benchmark=3.50)

    # Baseline ATR (3.50) -> k_vol = 1.0, effective risk = 1.0%
    app_base = gov.review_trade(
        equity=10000.0,
        stop_dist_usd=5.0,
        atr_current=3.50,
        session_hour=14,
    )
    assert app_base.approved is True
    assert math.isclose(app_base.k_vol, 1.0, rel_tol=1e-2)
    assert math.isclose(app_base.effective_risk_pct, 1.0, rel_tol=1e-2)

    # High Volatility (ATR 7.00) -> k_vol = 0.50, effective risk = 0.50%
    app_high_vol = gov.review_trade(
        equity=10000.0,
        stop_dist_usd=5.0,
        atr_current=7.00,
        session_hour=14,
    )
    assert app_high_vol.approved is True
    assert math.isclose(app_high_vol.k_vol, 0.50, rel_tol=1e-2)
    assert math.isclose(app_high_vol.effective_risk_pct, 0.50, rel_tol=1e-2)
    # Position size must be smaller during high volatility
    assert app_high_vol.lot_size < app_base.lot_size


def test_conviction_weighted_scaling():
    """Tests that low conviction (<0.55) scales risk down to 0.5x."""
    gov = RiskGovernor(max_risk_pct=1.0, atr_benchmark=3.50)

    app_low_conv = gov.review_trade(
        equity=10000.0,
        stop_dist_usd=5.0,
        atr_current=3.50,
        conviction_score=0.52,  # Low conviction
        session_hour=14,
    )
    assert app_low_conv.approved is True
    assert math.isclose(app_low_conv.conviction_factor, 0.50, rel_tol=1e-2)
    assert math.isclose(app_low_conv.effective_risk_pct, 0.50, rel_tol=1e-2)


def test_consecutive_loss_cooldown_veto():
    """Tests that 3 consecutive losses engage the cooldown governor."""
    gov = RiskGovernor(consecutive_loss_limit=3)

    gov.record_trade_result(-50.0)
    gov.record_trade_result(-40.0)
    assert gov.consecutive_losses == 2

    # Still approved at 2 losses
    app_2 = gov.review_trade(equity=1000.0, stop_dist_usd=5.0, session_hour=14)
    assert app_2.approved is True

    # Third consecutive loss trips the circuit breaker
    gov.record_trade_result(-30.0)
    assert gov.consecutive_losses == 3

    app_3 = gov.review_trade(equity=1000.0, stop_dist_usd=5.0, session_hour=14)
    assert app_3.approved is False
    assert "Consecutive loss cooldown" in app_3.veto_reason

    # Reset upon winning trade
    gov.record_trade_result(100.0)
    assert gov.consecutive_losses == 0
    app_reset = gov.review_trade(equity=1000.0, stop_dist_usd=5.0, session_hour=14)
    assert app_reset.approved is True


def test_news_lockout_veto():
    """Tests that active economic news lockout vetoes trade execution."""
    gov = RiskGovernor()

    locked_status = NewsStatus(
        is_locked=True,
        active_event="US FOMC Statement",
        minutes_to_event=5,
        reason="Within 30m of US FOMC Statement",
        summary="🔴 *News Blackout*: ACTIVE",
        upcoming_events_today=[],
    )

    app = gov.review_trade(
        equity=1000.0,
        stop_dist_usd=5.0,
        session_hour=14,
        news_status=locked_status,
    )
    assert app.approved is False
    assert "RED-FOLDER NEWS BLACKOUT" in app.veto_reason
