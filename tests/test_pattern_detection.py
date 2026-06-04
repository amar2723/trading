from __future__ import annotations

import pandas as pd

from src.pattern_detection.bos import detect_bos
from src.pattern_detection.fvg import detect_fvg
from src.pattern_detection.liquidity_sweep import detect_liquidity_sweeps
from src.pattern_detection.mss import detect_mss
from src.pattern_detection.order_blocks import detect_order_blocks
from src.pattern_detection.pattern_pipeline import detect_patterns


def base_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=8, freq="5min"),
            "open": [100, 101, 102, 101, 100, 103, 104, 105],
            "high": [101, 103, 104, 102, 101, 106, 107, 108],
            "low": [99, 100, 101, 98, 97, 102, 103, 104],
            "close": [100.5, 102, 101.5, 100, 100.5, 105, 106, 107],
            "tick_volume": [100] * 8,
            "spread": [20] * 8,
            "body_size": [0.5, 1, 0.5, 1, 0.5, 2, 2, 2],
            "swing_low": [False, False, True, False, False, False, False, False],
            "swing_high": [False, True, False, False, False, False, False, False],
            "swing_low_price": [None, None, 101, None, None, None, None, None],
            "swing_high_price": [None, 103, None, None, None, None, None, None],
            "lower_high": [False, False, False, False, True, False, False, False],
            "higher_low": [False, False, False, True, False, False, False, False],
        }
    )


def test_liquidity_sweep_detection():
    df = base_frame()
    df.loc[4, ["low", "close"]] = [99, 102]
    result = detect_liquidity_sweeps(df)
    assert result.loc[4, "bullish_liquidity_sweep"] == 1
    assert result.loc[4, "sweep_price"] == 101


def test_bos_detection():
    df = base_frame()
    df.loc[5, "close"] = 104
    result = detect_bos(df)
    assert result.loc[5, "bullish_bos"] == 1


def test_mss_detection_after_sweep():
    df = base_frame()
    df["bullish_liquidity_sweep"] = [0, 0, 0, 0, 1, 0, 0, 0]
    df["bearish_liquidity_sweep"] = 0
    df["previous_structure_high"] = 103
    df["previous_structure_low"] = 98
    df.loc[5, "close"] = 104
    result = detect_mss(df)
    assert result.loc[5, "bullish_mss"] == 1


def test_fvg_detection():
    df = base_frame()
    df.loc[0, "high"] = 100
    df.loc[2, "low"] = 102
    result = detect_fvg(df)
    assert result.loc[2, "bullish_fvg"] == 1
    assert result.loc[2, "fvg_bottom"] == 100
    assert result.loc[2, "fvg_top"] == 102


def test_order_block_detection():
    df = base_frame()
    df.loc[3, ["open", "close", "high", "low", "body_size"]] = [105, 100, 106, 99, 5]
    df.loc[4, ["open", "close", "high", "low", "body_size"]] = [100, 115, 116, 99, 15]
    result = detect_order_blocks(df)
    assert result.loc[4, "bullish_ob"] == 1
    assert result.loc[4, "ob_high"] == 106
    assert result.loc[4, "ob_low"] == 99


def test_pattern_pipeline_adds_core_columns():
    result = detect_patterns(base_frame())
    for column in ["bullish_liquidity_sweep", "bullish_mss", "bullish_bos", "bullish_fvg", "bullish_ob", "equilibrium"]:
        assert column in result.columns
