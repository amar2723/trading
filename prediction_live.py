from __future__ import annotations

import argparse
import json

from src.live_data import fetch_live_candles
from src.prediction import PredictionPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run prediction pipeline on live MT5 candles.")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--timeframe", default="M5")
    parser.add_argument("--bars", type=int, default=500)
    parser.add_argument("--min-confidence", type=float, default=70.0)
    args = parser.parse_args()
    candles = fetch_live_candles(args.symbol, args.timeframe, args.bars)
    prediction = PredictionPipeline(args.min_confidence).predict(candles)
    print(json.dumps(prediction, indent=2))
    print("Probabilities/confidence only. Never guarantee accuracy.")


if __name__ == "__main__":
    main()
