from __future__ import annotations

import argparse

from src.live_data import fetch_live_candles
from src.pattern_detector import detect_pattern


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Phase 1 detector over historical MT5 candles.")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--timeframe", default="M5", choices=["M1", "M5", "M15"])
    parser.add_argument("--bars", type=int, default=1000)
    args = parser.parse_args()

    candles = fetch_live_candles(args.symbol, args.timeframe, args.bars)
    buy_signals = []
    sell_signals = []
    none_count = 0

    for end in range(25, len(candles) + 1):
        signal = detect_pattern(candles.iloc[:end], use_closed_candle=False)
        payload = signal.to_dict()
        if signal.signal == "BUY":
            buy_signals.append(payload)
        elif signal.signal == "SELL":
            sell_signals.append(payload)
        else:
            none_count += 1

    print(f"BUY Signals Found: {len(buy_signals)}")
    print(f"SELL Signals Found: {len(sell_signals)}")
    print(f"NONE Signals Found: {none_count}")

    print("\nSample BUY Signals:")
    for signal in buy_signals[:5]:
        print({key: signal.get(key) for key in ["time", "signal", "entry", "sl", "tp1", "confidence", "reason"]})

    print("\nSample SELL Signals:")
    for signal in sell_signals[:5]:
        print({key: signal.get(key) for key in ["time", "signal", "entry", "sl", "tp1", "confidence", "reason"]})


if __name__ == "__main__":
    main()
