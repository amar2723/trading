from __future__ import annotations

import json

import pandas as pd

from src.dashboard.charts import candlestick_chart, equity_chart, feature_importance_chart
from src.dashboard.data_loader import latest_signal, load_csv, load_json


def test_load_csv_and_json(tmp_path):
    csv_path = tmp_path / "data.csv"
    json_path = tmp_path / "metrics.json"
    pd.DataFrame({"timestamp": ["2024-01-01"], "close": [2000]}).to_csv(csv_path, index=False)
    json_path.write_text(json.dumps({"net_profit": 10}), encoding="utf-8")
    assert len(load_csv(csv_path)) == 1
    assert load_json(json_path)["net_profit"] == 10


def test_latest_signal(tmp_path):
    path = tmp_path / "signals.csv"
    pd.DataFrame(
        [
            {"timestamp": "2024-01-01", "signal": "HOLD", "confidence": 0},
            {"timestamp": "2024-01-02", "signal": "BUY", "confidence": 80},
        ]
    ).to_csv(path, index=False)
    assert latest_signal(path)["signal"] == "BUY"


def test_chart_builders_return_figures():
    candles = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="5min"),
            "open": [1, 2, 3],
            "high": [2, 3, 4],
            "low": [0, 1, 2],
            "close": [1.5, 2.5, 3.5],
        }
    )
    equity = pd.DataFrame({"timestamp": candles["timestamp"], "equity": [100, 110, 105], "balance": [100, 110, 105]})
    importance = pd.DataFrame({"feature": ["atr"], "importance": [0.5]})
    assert candlestick_chart(candles).data
    assert equity_chart(equity).data
    assert feature_importance_chart(importance).data
