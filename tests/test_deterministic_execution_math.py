import pytest
import math
from pathlib import Path
import pandas as pd
from src.broker_adapter import OctaBrokerAdapter
from src.risk_governor import RiskGovernor
from src.backtest_engine import EventDrivenBacktestEngine, BacktestTrade


def test_exact_cent_buy_execution_and_sl_exit(tmp_path):
    adapter = OctaBrokerAdapter(
        mode="SHADOW_DEMO",
        csv_file=tmp_path / "ledger_buy.csv",
        initial_demo_equity=1000.0,
    )

    # Place BUY order when current price is 2600.00
    # Half-spread = 0.12, slippage = 0.05 -> Ask fill = 2600.17
    pos = adapter.execute_order(
        side="BUY",
        lots=0.01,
        current_price=2600.00,
        sl=2595.00,
        tp1=2610.00,
        tp2=2615.00,
    )
    assert pos is not None
    assert pos.open_price == 2600.17
    assert pos.lots == 0.01

    # Stop Loss hit: candle low drops to 2594.00
    # Exit price = SL 2595.00 - exit slippage 0.05 = 2594.95
    events = adapter.on_tick(high=2601.00, low=2594.00, close=2594.50)
    assert len(adapter.closed_positions) == 1
    closed = adapter.closed_positions[0]

    assert closed.close_price == 2594.95
    assert closed.close_reason == "STOP_LOSS"
    # Gross PnL: (2594.95 - 2600.17) * 0.01 * 100 = -5.22 USD
    assert closed.pnl_usd == -5.22
    assert closed.pnl_pips == -52.2
    assert closed.pnl_R < 0.0
    assert adapter.equity == round(1000.0 - 5.22, 2)


def test_exact_cent_sell_execution_and_tp_exit(tmp_path):
    adapter = OctaBrokerAdapter(
        mode="SHADOW_DEMO",
        csv_file=tmp_path / "ledger_sell.csv",
        initial_demo_equity=1000.0,
    )

    # Place SELL order when current price is 2600.00
    # Bid = 2599.88 - slippage 0.05 -> Bid fill = 2599.83
    pos = adapter.execute_order(
        side="SELL",
        lots=0.01,
        current_price=2600.00,
        sl=2605.00,
        tp1=2590.00,
        tp2=2585.00,
    )
    assert pos is not None
    assert pos.open_price == 2599.83

    # TP hit: low reaches 2589.00
    # Exit price = TP 2590.00 + exit slippage 0.05 = 2590.05
    events = adapter.on_tick(high=2601.00, low=2589.00, close=2590.00)
    assert len(adapter.closed_positions) == 1
    closed = adapter.closed_positions[0]

    assert closed.close_price == 2590.05
    assert closed.close_reason == "TAKE_PROFIT_1"
    # Gross PnL: (2599.83 - 2590.05) * 0.01 * 100 = +9.78 USD
    assert closed.pnl_usd == 9.78
    assert closed.pnl_pips == 97.8
    assert closed.pnl_R > 0.0
    assert adapter.equity == round(1000.0 + 9.78, 2)


def test_zero_auto_bump_on_sub_minimum_or_gap_risk():
    gov = RiskGovernor(max_risk_pct=1.0)
    adapter = OctaBrokerAdapter(mode="SHADOW_DEMO")

    # 1. Adapter must REFUSE to execute any order with lots < 0.01 (zero auto-bumping)
    pos_rejected = adapter.execute_order("BUY", lots=0.00, current_price=2600.00, sl=2595.00, tp1=2610.00, tp2=2615.00)
    assert pos_rejected is None
    assert len(adapter.open_positions) == 0

    # 2. Risk Governor: If equity is $400.00, 1.0% risk budget is $4.00.
    # A $4.50 stop loss + $0.34 friction risks $4.84 > $4.00.
    # Must reject and NOT round up to 0.01 lot.
    app = gov.review_trade(equity=400.00, risk_pips=45.0, current_spread_pips=24.0, session_hour=14)
    assert app.approved is False
    assert app.lot_size == 0.0
    assert "undercapitalized" in app.veto_reason.lower()


def test_event_driven_gap_risk_rejection():
    """
    If a pending order was staged with a small stop distance, but the next candle opens
    at an adverse price that widens the stop distance beyond the 1% risk budget,
    the event-driven engine must REJECT the order at fill time rather than executing it.
    """
    timestamps = pd.date_range("2026-11-18 08:00", periods=70, freq="5min")
    opens = [2600.0] * 70
    highs = [2602.0] * 70
    lows = [2598.0] * 70
    closes = [2600.0] * 70
    volumes = [500.0] * 70

    # Candle 54 creates a setup
    closes[54] = 2608.0
    highs[54] = 2609.0
    lows[54] = 2605.0

    # Candle 55 opens with an ENORMOUS gap down to 2580.00 (severely widening stop distance)
    opens[55] = 2580.0
    highs[55] = 2585.0
    lows[55] = 2575.0
    closes[55] = 2580.0

    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "tick_volume": volumes,
    })

    # Account equity is $300.00 (1% is $3.00). Gap makes risk > $25.00.
    engine = EventDrivenBacktestEngine(
        initial_equity=300.0,
        max_risk_pct=1.0,
        use_ml_filter=False,
    )
    res = engine.run(df)
    # The gap trade must have been vetoed at fill time, so filled trades count is 0
    assert res.filled_trades_count == 0
    assert res.net_pnl_usd == 0.0
