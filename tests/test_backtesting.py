from __future__ import annotations

import pandas as pd

from src.backtesting.backtester import Backtester, ExecutionConfig
from src.backtesting.performance import calculate_performance
from src.backtesting.risk_manager import RiskConfig, RiskManager


def signal_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=8, freq="5min"),
            "open": [100, 101, 102, 103, 104, 105, 106, 107],
            "high": [101, 105, 108, 110, 111, 112, 113, 114],
            "low": [99, 100, 101, 102, 103, 104, 105, 106],
            "close": [100, 104, 107, 109, 110, 111, 112, 113],
            "entry_type": ["BUY", "HOLD", "HOLD", "HOLD", "HOLD", "HOLD", "HOLD", "HOLD"],
            "sl_price": [98.0] + [None] * 7,
            "tp1": [104.0] + [None] * 7,
            "tp2": [108.0] + [None] * 7,
        }
    )


def test_position_sizing_percent_risk():
    manager = RiskManager(RiskConfig(initial_balance=10_000, percent_risk=1.0, contract_size=100))
    size = manager.position_size(10_000, 100, 98)
    assert size == 0.5


def test_risk_manager_blocks_after_consecutive_losses():
    manager = RiskManager(RiskConfig(max_consecutive_losses=2))
    manager.record_trade(-10)
    manager.record_trade(-10)
    assert not manager.can_trade(10_000, 10_000, 0, 0)


def test_trade_execution_and_pnl_calculation():
    backtester = Backtester(
        execution=ExecutionConfig(spread_points=0, slippage_points=0, commission_per_lot=0, execution_delay=0),
        risk=RiskConfig(initial_balance=10_000, sizing_mode="fixed_lot", fixed_lot=1.0, contract_size=1),
    )
    trades, equity = backtester.run(signal_frame())
    assert len(trades) == 1
    assert trades.loc[0, "result"] == "WIN"
    assert trades.loc[0, "pnl"] > 0
    assert equity["equity"].iloc[-1] >= 10_000


def test_drawdown_calculation():
    trades = pd.DataFrame({"pnl": [100.0, -50.0, 25.0]})
    equity = pd.DataFrame({"equity": [10_000, 10_100, 10_050, 10_075], "drawdown": [0, 0, -50, -25]})
    metrics = calculate_performance(trades, equity, 10_000)
    assert metrics["maximum_drawdown"] == -50.0
    assert metrics["net_profit"] == 75.0
