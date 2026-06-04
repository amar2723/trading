from __future__ import annotations

import pandas as pd

from filter_optimizer import metrics_from_trades, optimize_filters


def test_metrics_from_trades():
    trades = pd.DataFrame({"rr": [3.0, -1.0, 1.5, -1.0]})
    total, win_rate, profit_factor, expectancy, drawdown = metrics_from_trades(trades)
    assert total == 4
    assert win_rate == 50.0
    assert profit_factor == 2.25
    assert expectancy == 0.625
    assert drawdown == -1.0


def test_optimize_filters_from_parameter_study(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    pd.DataFrame(
        [
            {
                "body_multiplier": 2.0,
                "require_sweep": True,
                "close_near_ratio": 0.35,
                "combined_trades": 50,
                "combined_win_rate": 40,
                "combined_profit_factor": 1.2,
                "combined_average_rr": 0.1,
                "combined_drawdown": -5,
            }
        ]
    ).to_csv(reports / "pattern_parameter_study.csv", index=False)
    pd.DataFrame({"rr": [1.5, -1], "confidence": [75, 75], "reason": ["Liquidity Sweep", ""]}).to_csv(
        reports / "trade_outcomes.csv", index=False
    )
    pd.DataFrame({"feature": ["body_size"], "winning_avg": [2], "losing_avg": [1], "difference": [1]}).to_csv(
        reports / "outcome_feature_analysis.csv", index=False
    )
    results, best = optimize_filters(reports)
    assert not results.empty
    assert "recommended_rule" in best
