from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable


LOG_PATH = Path("logs/signals.csv")
DEBUG_LOG_PATH = Path("logs/debug_signals.csv")
IMAGE_PLAN_LOG_PATH = Path("logs/image_trade_plans.csv")
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
IMAGE_PLAN_FIELDS = [
    "time",
    "symbol",
    "signal",
    "confidence",
    "entry",
    "stop_loss",
    "tp1",
    "tp2",
    "tp3",
    "risk_reward",
    "bias",
    "trend",
    "recent_pressure",
    "decision",
    "reason",
]


def log_signal(signal: dict, path: Path = LOG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _normalize_signal_log(path)
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
    rows = _read_signal_rows(path)
    return pd.DataFrame(rows, columns=FIELDS)


def _normalize_signal_log(path: Path) -> None:
    if not path.exists():
        return
    rows = _read_signal_rows(path)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _read_signal_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        raw_rows = list(reader)
    if not raw_rows:
        return rows

    header = raw_rows[0]
    data_rows = raw_rows[1:] if "time" in header and "signal" in header else raw_rows
    for raw in data_rows:
        if not raw:
            continue
        rows.append(_row_to_signal_dict(raw))
    return rows


def _row_to_signal_dict(raw: Iterable[str]) -> dict:
    values = list(raw)
    row = {field: None for field in FIELDS}
    if len(values) >= len(FIELDS):
        for field, value in zip(FIELDS, values[: len(FIELDS)]):
            row[field] = value
        return row

    old_fields = ["time", "signal", "entry", "sl", "tp1", "confidence", "reason"]
    for field, value in zip(old_fields, values):
        row[field] = value
    return row


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


def log_image_trade_plan(plan: dict, symbol: str, decision: str, path: Path = IMAGE_PLAN_LOG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    row = {
        "time": plan.get("live_time") or plan.get("timestamp"),
        "symbol": symbol,
        "signal": plan.get("signal"),
        "confidence": plan.get("confidence"),
        "entry": plan.get("entry"),
        "stop_loss": plan.get("stop_loss"),
        "tp1": plan.get("tp1"),
        "tp2": plan.get("tp2"),
        "tp3": plan.get("tp3"),
        "risk_reward": plan.get("risk_reward"),
        "bias": plan.get("bias"),
        "trend": plan.get("trend"),
        "recent_pressure": plan.get("recent_pressure"),
        "decision": decision,
        "reason": ", ".join(plan.get("reason") or []),
    }
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=IMAGE_PLAN_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
