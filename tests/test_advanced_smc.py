from __future__ import annotations

import pandas as pd

from src.advanced_smc import AdvancedSMCEngine


def test_advanced_engine_outputs_signal_object():
    rows = []
    for i in range(120):
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
    signal = AdvancedSMCEngine().predict(pd.DataFrame(rows))
    assert signal.signal in {"BUY", "SELL", "NO TRADE"}
    assert 0 <= signal.confidence <= 100
