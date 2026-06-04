from __future__ import annotations

import pandas as pd

from tune_pattern_from_outcomes import recommend_tuning


def test_recommend_tuning_waits_for_enough_data():
    report = recommend_tuning(pd.DataFrame(), min_completed=5)
    assert report["status"] == "not_enough_data"


def test_recommend_tuning_ready_with_completed_data():
    rows = []
    for i in range(10):
        rows.append(
            {
                "signal": "SELL" if i % 2 else "BUY",
                "rr_result": 1.5 if i < 6 else -1.0,
                "confidence": 75,
                "body_percentage": 0.8 if i < 6 else 0.4,
                "body_size": 5 if i < 6 else 2,
                "range_size": 6,
                "upper_wick": 1,
                "lower_wick": 1,
                "bull_score": 3,
                "bear_score": 3,
            }
        )
    report = recommend_tuning(pd.DataFrame(rows), min_completed=10)
    assert report["status"] == "ready"
    assert report["profit_factor"] > 1
    assert report["feature_comparison"]
