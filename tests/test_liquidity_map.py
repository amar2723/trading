from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.liquidity_map import (
    LiquidityMap,
    LiquidityReferenceLevel,
    ForcedLiquidationMonitor,
    SweepEvent,
)


def _generate_synthetic_multiday_candles() -> pd.DataFrame:
    """Generates two full days of 5-minute candles (288 * 2 = 576 bars)."""
    dates = pd.date_range("2026-11-16 00:00", periods=576, freq="5min", tz="UTC")
    rows = []
    base_price = 2650.0

    for i, d in enumerate(dates):
        # Day 1: higher prices (2660-2680); Day 2: lower prices (2630-2650)
        day_offset = 20.0 if d.day == 16 else 0.0
        h_offset = np.sin(i / 15.0) * 8.0
        c = base_price + day_offset + h_offset
        o = c - 0.5
        h = max(o, c) + 2.0
        l = min(o, c) - 2.0
        vol = 150.0 + (i % 20) * 10.0

        rows.append({
            "timestamp": d,
            "open": round(o, 2),
            "high": round(h, 2),
            "low": round(l, 2),
            "close": round(c, 2),
            "tick_volume": vol,
        })

    return pd.DataFrame(rows)


def test_session_levels_detection():
    df = _generate_synthetic_multiday_candles()
    lmap = LiquidityMap()
    levels = lmap.compute_session_levels(df)

    assert len(levels) > 0
    level_types = set(lvl.level_type for lvl in levels)
    assert "PDH" in level_types, "Must detect Prior Day High"
    assert "PDL" in level_types, "Must detect Prior Day Low"
    assert "PDC" in level_types, "Must detect Prior Day Close"
    assert "SESSION_HIGH" in level_types, "Must detect Session High"
    assert "SESSION_LOW" in level_types, "Must detect Session Low"

    # Validate PDH > PDL
    pdh_level = next(l for l in levels if l.level_type == "PDH")
    pdl_level = next(l for l in levels if l.level_type == "PDL")
    assert pdh_level.price > pdl_level.price


def test_anchored_vwap_and_bands():
    df = _generate_synthetic_multiday_candles()
    lmap = LiquidityMap()
    vwap_dict = lmap.compute_anchored_vwap(df)

    assert "vwap" in vwap_dict
    assert "vwap_upper_1" in vwap_dict
    assert "vwap_lower_1" in vwap_dict
    assert "vwap_upper_2" in vwap_dict
    assert "vwap_lower_2" in vwap_dict

    # Check dispersion order
    assert vwap_dict["vwap_upper_2"] > vwap_dict["vwap_upper_1"] > vwap_dict["vwap"]
    assert vwap_dict["vwap"] > vwap_dict["vwap_lower_1"] > vwap_dict["vwap_lower_2"]


def test_volume_profile_value_area():
    df = _generate_synthetic_multiday_candles()
    lmap = LiquidityMap()
    vp_dict = lmap.compute_volume_profile(df, bins=40)

    assert "vp_poc" in vp_dict
    assert "vp_vah" in vp_dict
    assert "vp_val" in vp_dict

    # Value area high must be >= Point of Control >= Value Area Low
    assert vp_dict["vp_vah"] >= vp_dict["vp_poc"]
    assert vp_dict["vp_poc"] >= vp_dict["vp_val"]


def test_no_trade_stop_run_gate():
    """Single bar wick through level must be classified as UNCONFIRMED_WICK."""
    df = _generate_synthetic_multiday_candles()
    lmap = LiquidityMap()

    # Artificially modify the latest bar to wick high above Asia High and close below
    levels = lmap.compute_session_levels(df)
    asia_high = next(l for l in levels if l.session == "ASIA" and l.level_type == "SESSION_HIGH")

    last_idx = len(df) - 1
    df.loc[last_idx, "high"] = asia_high.price + 3.0  # Wick breach
    df.loc[last_idx, "close"] = asia_high.price - 1.0  # Closed back below
    df.loc[last_idx, "open"] = asia_high.price - 2.0

    events, status = lmap.evaluate_sweeps_and_structure(df)
    wick_events = [e for e in events if e.status == "UNCONFIRMED_WICK"]
    assert len(wick_events) > 0, "Active wick on current candle must be flagged UNCONFIRMED_WICK"
    assert any("STOP-RUN NO-TRADE GATE ACTIVE" in e.rationale for e in wick_events)


def test_multi_bar_sweep_confirmation():
    """Sweep on bar t followed by confirmed bearish closure on bar t+1."""
    df = _generate_synthetic_multiday_candles()
    lmap = LiquidityMap()

    levels = lmap.compute_session_levels(df)
    asia_high = next(l for l in levels if l.session == "ASIA" and l.level_type == "SESSION_HIGH")

    last_idx = len(df) - 1
    # Bar t-1 wicks above Asia high and closes inside
    df.loc[last_idx - 1, "high"] = asia_high.price + 3.0
    df.loc[last_idx - 1, "close"] = asia_high.price - 1.0
    df.loc[last_idx - 1, "open"] = asia_high.price - 2.0

    # Bar t confirms with bearish close below previous close
    df.loc[last_idx, "open"] = asia_high.price - 1.0
    df.loc[last_idx, "close"] = asia_high.price - 3.5
    df.loc[last_idx, "high"] = asia_high.price - 0.5
    df.loc[last_idx, "low"] = asia_high.price - 4.0

    events, status = lmap.evaluate_sweeps_and_structure(df)
    confirmed = [e for e in events if e.status == "CONFIRMED_REJECTION"]
    assert len(confirmed) > 0, "Multi-bar bearish follow-through must produce CONFIRMED_REJECTION"
    assert "CONFIRMED_BEARISH_REJECTION" in status


def test_forced_liquidation_circuit_breaker():
    """Extreme 5m displacement (>2%) must trip the circuit breaker and pause trading."""
    df = _generate_synthetic_multiday_candles()
    monitor = ForcedLiquidationMonitor(velocity_threshold_pct=2.0, cooldown_bars=12)

    # Normal conditions
    tripped, msg = monitor.evaluate(df)
    assert not tripped
    assert msg == "NORMAL"

    # Extreme 2.8% sudden drop in single candle (Margin cascade)
    last_idx = len(df) - 1
    open_p = float(df.loc[last_idx - 1, "open"])
    df.loc[last_idx, "open"] = open_p
    df.loc[last_idx, "close"] = open_p * 0.972  # -2.8% collapse
    df.loc[last_idx, "low"] = open_p * 0.970
    df.loc[last_idx, "high"] = open_p

    tripped, msg = monitor.evaluate(df)
    assert tripped, "Displacement > 2% must engage circuit breaker"
    assert "FORCED_LIQUIDATION_ALERT" in msg

    # Next bar during cool-down with normal price movement
    new_bar = {
        "timestamp": df["timestamp"].iloc[-1] + pd.Timedelta(minutes=5),
        "open": open_p * 0.972,
        "high": open_p * 0.973,
        "low": open_p * 0.971,
        "close": open_p * 0.972,
        "tick_volume": 100.0,
    }
    df = pd.concat([df, pd.DataFrame([new_bar])], ignore_index=True)
    tripped_cooldown, msg_cooldown = monitor.evaluate(df)
    assert tripped_cooldown
    assert "CIRCUIT_BREAKER_COOLING" in msg_cooldown
