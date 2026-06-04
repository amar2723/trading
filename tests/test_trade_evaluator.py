from __future__ import annotations

import pandas as pd

from trade_evaluator import build_metrics, evaluate_trades, evaluate_buy_trades, simulate_buy_trade, simulate_trade


def test_simulate_buy_trade_tp1_before_sl():
    future = pd.DataFrame(
        [
            {"timestamp": "2024-01-01 00:10:00", "high": 11.6, "low": 10.5},
            {"timestamp": "2024-01-01 00:15:00", "high": 12.0, "low": 10.0},
        ]
    )
    outcome, exit_price, _, rr = simulate_buy_trade(future, sl=9.0, tp1=11.5, tp2=13.0)
    assert outcome == "TP1"
    assert exit_price == 11.5
    assert rr == 1.5


def test_simulate_sell_trade_tp2_before_tp1_and_sl():
    future = pd.DataFrame(
        [
            {"timestamp": "2024-01-01 00:10:00", "high": 10.5, "low": 7.0},
        ]
    )
    outcome, exit_price, _, rr = simulate_trade("SELL", future, sl=11.0, tp1=8.5, tp2=7.0)
    assert outcome == "TP2"
    assert exit_price == 7.0
    assert rr == 3.0


def test_build_metrics_profit_factor_and_drawdown():
    trades = pd.DataFrame({"rr": [1.5, -1.0, 3.0, -1.0]})
    metrics = build_metrics(trades)
    assert metrics["total_trades"] == 4
    assert metrics["win_rate"] == 50.0
    assert metrics["profit_factor"] == 2.25
    assert metrics["max_drawdown"] == -1.0


def test_evaluate_buy_trades_finds_trade():
    warmup = [
        {"timestamp": f"2024-01-01 00:{i:02d}:00", "open": 10, "high": 10.5, "low": 9.8, "close": 10.1, "volume": 100}
        for i in range(20)
    ]
    previous = {"timestamp": "2024-01-01 01:40:00", "open": 10, "high": 11, "low": 9, "close": 9.5, "volume": 100}
    signal = {"timestamp": "2024-01-01 01:45:00", "open": 9.4, "high": 12, "low": 8.8, "close": 11.5, "volume": 100}
    future = [{"timestamp": "2024-01-01 01:50:00", "open": 11.5, "high": 15.6, "low": 11.2, "close": 15, "volume": 100}]
    df = pd.DataFrame(warmup + [previous, signal] + future)
    trades, metrics = evaluate_buy_trades(df, lookahead=1)
    assert len(trades) == 1
    assert trades.loc[0, "outcome"] in {"TP1", "TP2"}
    assert metrics["total_trades"] == 1
