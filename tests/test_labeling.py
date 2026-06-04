from __future__ import annotations

import pandas as pd

from src.labeling.entry_generator import generate_entries
from src.labeling.label_generator import build_report, generate_labels
from src.labeling.sl_tp_generator import generate_sl_tp
from src.labeling.trade_simulator import simulate_trades


def pattern_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=8, freq="5min"),
            "open": [100, 101, 102, 103, 104, 105, 106, 107],
            "high": [101, 103, 106, 108, 109, 110, 111, 112],
            "low": [99, 100, 101, 102, 103, 104, 105, 106],
            "close": [100, 102, 105, 107, 108, 109, 110, 111],
            "atr": [2.0] * 8,
            "bullish_liquidity_sweep": [1, 0, 0, 0, 0, 0, 0, 0],
            "bullish_mss": [1, 0, 0, 0, 0, 0, 0, 0],
            "bullish_bos": [1, 0, 0, 0, 0, 0, 0, 0],
            "bearish_liquidity_sweep": [0] * 8,
            "bearish_mss": [0] * 8,
            "bearish_bos": [0] * 8,
            "sweep_low": [98.0] + [None] * 7,
            "sweep_high": [None] * 8,
            "resistance": [104.0] * 8,
            "support": [96.0] * 8,
            "next_liquidity_high": [108.0] * 8,
            "next_liquidity_low": [94.0] * 8,
        }
    )


def test_entry_logic_generates_buy():
    result = generate_entries(pattern_frame())
    assert result.loc[0, "entry_type"] == "BUY"
    assert result.loc[0, "target"] == 1
    assert result.loc[1, "entry_type"] == "HOLD"


def test_sl_logic_uses_sweep_low_for_buy():
    entries = generate_entries(pattern_frame())
    result = generate_sl_tp(entries)
    assert result.loc[0, "sl_price"] == 98.0
    assert result.loc[0, "risk_points"] == 2.0


def test_tp_logic_uses_liquidity_zones():
    entries = generate_entries(pattern_frame())
    result = generate_sl_tp(entries)
    assert result.loc[0, "tp1"] == 104.0
    assert result.loc[0, "tp2"] == 108.0
    assert result.loc[0, "risk_reward_ratio"] == 2.0


def test_trade_simulation_detects_win():
    planned = generate_sl_tp(generate_entries(pattern_frame()))
    result = simulate_trades(planned)
    assert result.loc[0, "trade_result"] == "WIN"
    assert result.loc[0, "max_favorable_excursion"] > 0


def test_label_generation_sets_profitable_trade_and_report():
    result = generate_labels(pattern_frame())
    assert result.loc[0, "profitable_trade"] == 1
    report = build_report(result)
    assert report["total_trades"] == 1
    assert report["win_rate"] == 100.0
