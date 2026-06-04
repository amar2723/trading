from __future__ import annotations

import pandas as pd

from src.outcome_tracker import record_signal_for_outcome, update_pending_outcomes


def test_record_and_update_buy_outcome(tmp_path):
    path = tmp_path / "signal_outcomes.csv"
    signal = {
        "time": "2024-01-01 00:00:00",
        "signal": "BUY",
        "entry": 10.0,
        "sl": 9.0,
        "tp1": 11.5,
        "confidence": 75,
        "reason": "Liquidity Sweep, Strong Body, Close Near High",
        "debug": {
            "bull_score": 3,
            "bear_score": 0,
            "bull_sweep": True,
            "bear_sweep": False,
            "close_above_high": True,
            "close_below_low": False,
            "strong_body": True,
            "close_near_high": True,
            "close_near_low": False,
            "average_body_last_20": 1.0,
            "previous": {"open": 10, "high": 10.5, "low": 9.5, "close": 9.8},
            "current": {"open": 9.7, "high": 10.2, "low": 9.0, "close": 10.0},
        },
    }
    record_signal_for_outcome(signal, path)
    candles = pd.DataFrame(
        [{"timestamp": f"2024-01-01 00:{i + 1:02d}:00", "high": 11.6, "low": 10.0, "open": 10, "close": 11} for i in range(20)]
    )
    outcomes = update_pending_outcomes(candles, path, lookahead=20)
    assert outcomes.loc[0, "outcome"] == "TP1"
    assert float(outcomes.loc[0, "rr_result"]) == 1.5
