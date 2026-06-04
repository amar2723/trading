from __future__ import annotations

import json

import pandas as pd

from stability_tester import classify_performance_trend, load_best_strategy, rank_robustness


def test_load_best_strategy(tmp_path):
    path = tmp_path / "best_strategy.json"
    path.write_text(json.dumps({"best_parameters": {"minimum_body_size": 1.25}}), encoding="utf-8")
    assert load_best_strategy(path)["minimum_body_size"] == 1.25


def test_rank_robustness_weak_for_near_breakeven():
    results = pd.DataFrame(
        {
            "profit_factor": [0.98, 0.99, 1.0],
            "expectancy": [-0.01, 0.0, 0.01],
            "max_drawdown": [-10, -20, -30],
            "profitable": [False, False, False],
        }
    )
    assert rank_robustness(results) == "Weak"


def test_classify_performance_trend_stable():
    results = pd.DataFrame(
        {
            "candles": [1000, 5000, 10000],
            "profit_factor": [1.01, 1.03, 1.05],
            "expectancy": [0.01, 0.02, 0.03],
            "max_drawdown": [-10, -15, -18],
        }
    )
    assert classify_performance_trend(results) == "Stays Stable"
