from __future__ import annotations

import argparse
import json

from src.live_data import fetch_live_candles
from src.multi_timeframe import analyze_h1_trend, confirm_mtf_entry, validate_m15_structure


def run_mtf_prediction(symbol: str = "XAUUSD", min_confidence: float = 75.0) -> dict:
    h1 = fetch_live_candles(symbol, "H1", 500)
    m15 = fetch_live_candles(symbol, "M15", 500)
    m5 = fetch_live_candles(symbol, "M5", 500)
    trend = analyze_h1_trend(h1)
    structure = validate_m15_structure(m15)
    signal = confirm_mtf_entry(trend, structure, m5, min_confidence)
    return signal.to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run H1/M15/M5 multi-timeframe prediction.")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--min-confidence", type=float, default=75.0)
    args = parser.parse_args()
    print(json.dumps(run_mtf_prediction(args.symbol, args.min_confidence), indent=2))
    print("Probability-style output only. Never guarantee accuracy.")


if __name__ == "__main__":
    main()
