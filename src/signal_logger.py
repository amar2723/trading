from __future__ import annotations

import csv
from pathlib import Path


LOG_PATH = Path("logs/signals.csv")
DEBUG_LOG_PATH = Path("logs/debug_signals.csv")
FIELDS = ["time", "signal", "entry", "sl", "tp1", "tp2", "liquidity_target", "trailing_stop", "confidence", "reason"]
DEBUG_FIELDS = [
    "timestamp",
    "bull_score",
    "bear_score",
    "bull_sweep",
    "bear_sweep",
    "close_above_high",
    "close_below_low",
    "strong_body",
    "close_near_high",
    "close_near_low",
    "signal",
]


def log_signal(signal: dict, path: Path = LOG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: signal.get(field) for field in FIELDS})


def read_signal_history(path: Path = LOG_PATH):
    import pandas as pd

    if not path.exists():
        return pd.DataFrame(columns=FIELDS)
    return pd.read_csv(path)


def log_debug_signal(signal: dict, path: Path = DEBUG_LOG_PATH) -> None:
    debug = signal.get("debug") or {}
    if not debug:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    row = {
        "timestamp": debug.get("timestamp"),
        "bull_score": debug.get("bull_score"),
        "bear_score": debug.get("bear_score"),
        "bull_sweep": debug.get("bull_sweep"),
        "bear_sweep": debug.get("bear_sweep"),
        "close_above_high": debug.get("close_above_high"),
        "close_below_low": debug.get("close_below_low"),
        "strong_body": debug.get("strong_body"),
        "close_near_high": debug.get("close_near_high"),
        "close_near_low": debug.get("close_near_low"),
        "signal": signal.get("signal"),
    }
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DEBUG_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
