from __future__ import annotations

import argparse

from src.realtime.realtime_pipeline import RealtimePipeline, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the realtime XAUUSD trading engine.")
    parser.add_argument("--config", default="config/realtime_config.json")
    parser.add_argument("--once", action="store_true", help="Run one polling cycle and exit")
    args = parser.parse_args()
    pipeline = RealtimePipeline(load_config(args.config))
    if args.once:
        pipeline.start()
        try:
            print(pipeline.run_once())
        finally:
            pipeline.stop()
    else:
        pipeline.run_forever()


if __name__ == "__main__":
    main()
