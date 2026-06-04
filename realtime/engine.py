from __future__ import annotations

import time
from dataclasses import asdict

from src.config import DEFAULT_CONFIG
from src.data_ingestion.mt5_client import MT5Client
from src.feature_engineering.indicators import add_features
from src.pattern_detection.concepts import add_market_concepts
from src.signals import generate_trade_plan


class RealtimeEngine:
    def __init__(self, symbol: str = DEFAULT_CONFIG.symbol, timeframe: str = "M5", model_path: str | None = None):
        self.client = MT5Client(symbol)
        self.timeframe = timeframe
        self.model_path = model_path
        self.last_candle_time = None
        self.paper_trades: list[dict] = []

    def start(self) -> None:
        self.client.initialize()

    def stop(self) -> None:
        self.client.shutdown()

    def poll_once(self) -> dict:
        df = self.client.fetch_ohlcv(self.timeframe, DEFAULT_CONFIG.bars)
        df = add_market_concepts(add_features(df)).dropna()
        latest = df.iloc[-1]
        plan = generate_trade_plan(latest, self.model_path)
        payload = asdict(plan)
        payload["time"] = str(latest["time"])
        if plan.signal != "HOLD" and latest["time"] != self.last_candle_time:
            self.paper_trades.append(payload)
        self.last_candle_time = latest["time"]
        return payload

    def run(self, sleep_seconds: int = 5) -> None:
        self.start()
        try:
            while True:
                print(self.poll_once())
                time.sleep(sleep_seconds)
        finally:
            self.stop()


if __name__ == "__main__":
    RealtimeEngine().run()
