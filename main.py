from __future__ import annotations

import argparse
import time

from src.live_data import fetch_live_candles
from src.market_filters import add_filter_features, passes_chop_filter, passes_session_filter, passes_trend_filter
from mtf_live import run_mtf_prediction
from src.outcome_tracker import record_signal_for_outcome, update_pending_outcomes
from src.pattern_detector import detect_pattern, print_debug
from src.signal_logger import log_debug_signal, log_signal
from src.telegram_alert import send_telegram_alert


SYMBOL = "XAUUSD"
TIMEFRAME = "M5"
BARS = 100
SLEEP_SECONDS = 5
DEFAULT_SESSION = "NEW_YORK"


def scan_once(
    symbol: str,
    timeframe: str,
    bars: int,
    last_signal_key: tuple[str, str] | None = None,
    last_candle_time: str | None = None,
    session: str = DEFAULT_SESSION,
    trend_alignment: bool = False,
    avoid_chop: bool = False,
    min_chop_ratio: float = 4.0,
) -> tuple[dict, tuple[str, str] | None, str | None]:
    candles = fetch_live_candles(symbol, timeframe, bars)
    candles = add_filter_features(candles)
    update_pending_outcomes(candles)
    detected = detect_pattern(candles)
    signal = detected.to_dict()
    candle_time = signal["time"]

    if candle_time == last_candle_time:
        return signal, last_signal_key, last_candle_time

    print_debug(detected)
    log_debug_signal(signal)

    if signal["signal"] in {"BUY", "SELL"}:
        signal_candle = candles[candles["timestamp"].astype(str).eq(signal["time"])]
        signal_row = signal_candle.iloc[-1] if not signal_candle.empty else candles.iloc[-2]
        if not passes_session_filter(signal["time"], session):
            print(f"Rejected by session filter: {session}")
            return signal, last_signal_key, candle_time
        if not passes_trend_filter(signal_row, signal["signal"], trend_alignment):
            print("Rejected by trend alignment filter")
            return signal, last_signal_key, candle_time
        if not passes_chop_filter(signal_row, avoid_chop, min_chop_ratio):
            print("Rejected by chop filter")
            return signal, last_signal_key, candle_time

        signal_key = (signal["time"], signal["signal"])
        if signal_key != last_signal_key:
            log_signal(signal)
            record_signal_for_outcome(signal)
            send_telegram_alert(symbol, signal)
            print(signal)
            return signal, signal_key, candle_time
    else:
        print(f"{signal['time']} NONE")

    return signal, last_signal_key, candle_time


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 live XAUUSD pattern scanner.")
    parser.add_argument("--symbol", default=SYMBOL)
    parser.add_argument("--timeframe", default=TIMEFRAME, choices=["M1", "M5", "M15"])
    parser.add_argument("--bars", type=int, default=BARS)
    parser.add_argument("--sleep", type=int, default=SLEEP_SECONDS)
    parser.add_argument("--session", default=DEFAULT_SESSION, choices=["ALL", "LONDON", "NEW_YORK", "OVERLAP"])
    parser.add_argument("--trend-alignment", action="store_true")
    parser.add_argument("--avoid-chop", action="store_true")
    parser.add_argument("--min-chop-ratio", type=float, default=4.0)
    parser.add_argument("--mtf", action="store_true", help="Use H1/M15/M5 multi-timeframe engine")
    parser.add_argument("--once", action="store_true", help="Run one scan and exit")
    args = parser.parse_args()

    last_signal_key: tuple[str, str] | None = None
    last_candle_time: str | None = None
    print("Phase 1 live scanner started. No trading. Collecting signals only.")

    if args.mtf:
        result = run_mtf_prediction(args.symbol)
        print(result)
        return

    if args.once:
        scan_once(
            args.symbol,
            args.timeframe,
            args.bars,
            session=args.session,
            trend_alignment=args.trend_alignment,
            avoid_chop=args.avoid_chop,
            min_chop_ratio=args.min_chop_ratio,
        )
        return

    while True:
        try:
            _, last_signal_key, last_candle_time = scan_once(
                args.symbol,
                args.timeframe,
                args.bars,
                last_signal_key,
                last_candle_time,
                args.session,
                args.trend_alignment,
                args.avoid_chop,
                args.min_chop_ratio,
            )
        except Exception as exc:
            print(f"Scanner error: {exc}")

        time.sleep(args.sleep)


if __name__ == "__main__":
    main()
