from __future__ import annotations

import argparse
import json

from src.advanced_smc import AdvancedSMCEngine
from src.live_data import fetch_live_candles


def main() -> None:
    parser = argparse.ArgumentParser(description="Advanced liquidity-based prediction engine for XAUUSD M5.")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--timeframe", default="M5")
    parser.add_argument("--bars", type=int, default=500)
    parser.add_argument("--min-confidence", type=float, default=70.0)
    args = parser.parse_args()
    candles = fetch_live_candles(args.symbol, args.timeframe, args.bars)
    signal = AdvancedSMCEngine(args.symbol, args.min_confidence).predict(candles)
    print(json.dumps(signal.to_dict(), indent=2))
    print("\nNever guarantee accuracy. Treat this as a probability-based demo signal and manage risk.")


if __name__ == "__main__":
    main()
