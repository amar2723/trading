from __future__ import annotations

import pandas as pd

from src.prediction import PredictionPipeline


def test_prediction_pipeline_schema():
    rows = []
    for i in range(150):
        rows.append(
            {
                "timestamp": pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=5 * i),
                "open": 100 + i * 0.1,
                "high": 101 + i * 0.1,
                "low": 99 + i * 0.1,
                "close": 100.5 + i * 0.1,
                "volume": 100,
            }
        )
    prediction = PredictionPipeline().predict(pd.DataFrame(rows))
    for key in ["signal", "confidence", "entry", "stop_loss", "tp1", "tp2", "tp3", "rr_ratio", "reason"]:
        assert key in prediction
