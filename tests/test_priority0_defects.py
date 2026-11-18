import pytest
from pathlib import Path
import pandas as pd
from src.broker_adapter import OctaBrokerAdapter, BrokerPosition
from src.risk_governor import RiskGovernor
from src.ml_meta_filter import MLMetaFilter, FilterDecision


def test_broker_adapter_pnl_pips_and_exits(tmp_path):
    csv_file = tmp_path / "test_trades.csv"
    adapter = OctaBrokerAdapter(mode="SHADOW_DEMO", csv_file=csv_file)

    # 1. Test BUY Stop Loss
    pos_buy = adapter.execute_order("BUY", 0.01, 2600.00, sl=2595.50, tp1=2609.00, tp2=2615.00)
    assert pos_buy is not None
    ticket_buy = pos_buy.ticket
    assert ticket_buy in adapter.open_positions

    # Trigger Stop Loss on tick
    events = adapter.on_tick(high=2602.00, low=2595.00, close=2596.00)
    assert ticket_buy not in adapter.open_positions
    assert len(adapter.closed_positions) == 1
    closed_buy = adapter.closed_positions[0]
    assert closed_buy.status == "CLOSED"
    assert closed_buy.close_reason in ["STOP_LOSS", "STOP_LOSS (Conservative)"]
    assert closed_buy.pnl_usd < 0
    assert hasattr(closed_buy, "pnl_pips")
    assert closed_buy.pnl_pips < 0

    # 2. Test SELL Take Profit
    pos_sell = adapter.execute_order("SELL", 0.01, 2600.00, sl=2604.50, tp1=2591.00, tp2=2585.00)
    assert pos_sell is not None
    ticket_sell = pos_sell.ticket
    events = adapter.on_tick(high=2601.00, low=2590.00, close=2591.00)
    assert ticket_sell not in adapter.open_positions
    closed_sell = [p for p in adapter.closed_positions if p.ticket == ticket_sell][0]
    assert closed_sell.status == "CLOSED"
    assert closed_sell.close_reason == "TAKE_PROFIT_1"
    assert closed_sell.pnl_usd > 0
    assert closed_sell.pnl_pips > 0


def test_broker_adapter_no_same_candle_be_exit(tmp_path):
    csv_file = tmp_path / "test_be.csv"
    adapter = OctaBrokerAdapter(mode="SHADOW_DEMO", csv_file=csv_file)

    # BUY at 2600.00, SL at 2595.50 (risk_dist = 4.50, be_dist = 3.50 -> 2603.50 trigger)
    pos = adapter.execute_order("BUY", 0.01, 2600.00, sl=2595.50, tp1=2620.00, tp2=2630.00)
    ticket = pos.ticket

    # Candle reaches high 2605.00 (triggers BE) but low is 2599.50 (below open price)
    # Under conservative, non-lookahead execution, BE activation CANNOT stop out on this same candle!
    events = adapter.on_tick(high=2605.00, low=2599.50, close=2601.00)

    # Position must STILL BE OPEN with SL now moved to pos.open_price for subsequent candles
    assert ticket in adapter.open_positions
    updated_pos = adapter.open_positions[ticket]
    assert updated_pos.be_active is True
    assert updated_pos.sl == updated_pos.open_price

    # On NEXT candle, if low hits 2599.50, it stops out at BREAK_EVEN
    events2 = adapter.on_tick(high=2601.00, low=2599.50, close=2600.00)
    assert ticket not in adapter.open_positions
    closed_pos = adapter.closed_positions[-1]
    assert closed_pos.close_reason == "BREAK_EVEN"


def test_ml_meta_filter_failsafe_abstain():
    # Test with non-existent model
    meta = MLMetaFilter(model_path=Path("non_existent_model.pkl"))
    assert meta.ensemble is None

    # Dummy candle dataframe
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-11-18 10:00", periods=30, freq="5min"),
        "open": [2600.0] * 30,
        "high": [2605.0] * 30,
        "low": [2595.0] * 30,
        "close": [2600.0] * 30,
        "tick_volume": [100.0] * 30,
    })

    # HOLD must return ABSTAIN
    dec_hold = meta.evaluate_candidate("HOLD", "PULLBACK", df)
    assert dec_hold.decision == "ABSTAIN"

    # Missing model must return ABSTAIN
    dec_buy = meta.evaluate_candidate("BUY", "PULLBACK", df)
    assert dec_buy.decision == "ABSTAIN"
    assert dec_buy.expected_R < 0.0

    # Test corrupted/mock ensemble with missing feature cols -> must return ABSTAIN, never raise
    meta.ensemble = {"xgb_model": None, "feature_cols": ["non_existent_column_123"]}
    dec_err = meta.evaluate_candidate("BUY", "PULLBACK", df)
    assert dec_err.decision == "ABSTAIN"
    assert "ABSTAIN" in dec_err.filter_reason


def test_risk_governor_1pct_ceiling_and_sizing():
    gov = RiskGovernor(max_risk_pct=1.0)

    # Case 1: $40.68 equity -> 1.0% = $0.41. Stop distance $4.50 -> risk $4.84 > $0.41
    # MUST BE VETOED, MUST NOT AUTO-BUMP TO 0.01 LOT
    # Spread is $0.24 (2.4 pips). With authoritative spread_price_usd=0.24:
    app_40 = gov.review_trade(equity=40.68, risk_pips=45.0, spread_price_usd=0.24, session_hour=14)
    assert app_40.approved is False
    assert app_40.lot_size == 0.0
    assert "undercapitalized" in app_40.veto_reason.lower()

    # Case 2: $100.00 equity -> 1.0% = $1.00. Risk $4.84 > $1.00
    # MUST BE VETOED
    app_100 = gov.review_trade(equity=100.00, risk_pips=45.0, spread_price_usd=0.24, session_hour=14)
    assert app_100.approved is False
    assert app_100.lot_size == 0.0

    # Case 3: $450.00 equity -> 1.0% = $4.50.
    # Total risk with $0.24 spread + $0.10 slippage = $4.84 > $4.50
    # MUST BE VETOED because friction pushes total risk to 1.08% > 1.0%
    app_450 = gov.review_trade(equity=450.00, risk_pips=45.0, spread_price_usd=0.24, session_hour=14)
    assert app_450.approved is False
    assert app_450.lot_size == 0.0

    # Case 4: $1,000.00 equity -> 1.0% = $10.00.
    # Raw volume = 10.00 / 484.00 = 0.0206 -> lot_size = 0.02 lot
    # Actual risk = 0.02 * 484.00 = $9.68 (0.97% of equity <= 1.0%)
    # MUST BE APPROVED!
    app_1000 = gov.review_trade(equity=1000.00, risk_pips=45.0, spread_price_usd=0.24, session_hour=14)
    assert app_1000.approved is True
    assert app_1000.lot_size == 0.02
    assert app_1000.risk_usd <= 10.00
    assert app_1000.risk_pct <= 1.00
