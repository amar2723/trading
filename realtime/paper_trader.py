from __future__ import annotations

import argparse
import json

from realtime.engine import RealtimeEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper trading mode for XAUUSD signals")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--timeframe", default="M5", choices=["M1", "M5", "M15"])
    parser.add_argument("--model-path")
    parser.add_argument("--sleep", type=int, default=5)
    args = parser.parse_args()
    engine = RealtimeEngine(args.symbol, args.timeframe, args.model_path)
    engine.start()
    try:
        while True:
            print(json.dumps(engine.poll_once(), indent=2))
            import time

            time.sleep(args.sleep)
    finally:
        engine.stop()


if __name__ == "__main__":
    main()
