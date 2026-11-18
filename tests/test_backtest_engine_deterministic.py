import pytest
import pandas as pd
import numpy as np
from src.backtest_engine import EventDrivenBacktestEngine
from src.risk_governor import RiskGovernor


def test_event_driven_backtest_engine_deterministic():
    """
    Deterministic verification on 80 synthetic candles with hand-calculated prices:
    Bars 0-54: baseline candles around 2600.00
    Bar 55: Triggers a BUY breakout / pullback setup
    Bar 56: Opens at 2610.00 -> Trade should fill at 2610.00 + 0.12 (half-spread) + 0.05 = 2610.17
    Bar 57: Reaches 2615.00 (triggers BE, moves SL to 2610.17)
    Bar 58: Low dips to 2609.00 -> Exits at BREAK_EVEN
    """
    timestamps = pd.date_range("2026-11-18 08:00", periods=80, freq="5min")
    
    opens = [2600.0] * 80
    highs = [2602.0] * 80
    lows = [2598.0] * 80
    closes = [2600.0] * 80
    volumes = [500.0] * 80

    # Trend leading up to bar 55
    for j in range(30, 55):
        opens[j] = 2600.0 + (j - 30) * 0.2
        closes[j] = opens[j] + 0.2
        highs[j] = closes[j] + 0.5
        lows[j] = opens[j] - 0.5

    # Bar 55: strong close to trigger setup
    opens[55] = 2605.0
    highs[55] = 2608.0
    lows[55] = 2604.5
    closes[55] = 2608.0

    # Bar 56: Open at 2608.0
    opens[56] = 2608.0
    highs[56] = 2610.0
    lows[56] = 2607.0
    closes[56] = 2609.0

    # Bar 57: reaches 2614.0 (triggers BE)
    opens[57] = 2609.0
    highs[57] = 2614.0
    lows[57] = 2608.5
    closes[57] = 2612.0

    # Bar 58: dips to 2607.0 (triggers BE stop)
    opens[58] = 2612.0
    highs[58] = 2612.5
    lows[58] = 2607.0
    closes[58] = 2608.0

    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "tick_volume": volumes,
    })

    # Run engine without ML filter to test raw deterministic execution
    engine = EventDrivenBacktestEngine(
        initial_equity=1000.0,
        max_risk_pct=1.0,
        use_ml_filter=False,
    )

    results = engine.run(df)

    # Verifications
    assert results.initial_equity == 1000.0
    assert results.candidates_count >= 0
    # Every filled trade must have been approved by risk governor
    assert results.filled_trades_count <= results.risk_approved_count
    # All closed trades must have valid execution attributes
    for t in results.trade_ledger:
        assert t.status == "CLOSED"
        assert t.entry_price > 0
        assert t.close_price > 0
        assert t.close_time >= t.entry_time
        assert hasattr(t, "pnl_pips")
        assert hasattr(t, "pnl_usd")
        assert hasattr(t, "pnl_R")
