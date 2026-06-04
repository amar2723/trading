from __future__ import annotations

from pathlib import Path

import pandas as pd


OUTCOME_PATH = Path("logs/signal_outcomes.csv")
LOOKAHEAD_CANDLES = 20


OUTCOME_FIELDS = [
    "time",
    "signal",
    "entry",
    "sl",
    "tp1",
    "tp2",
    "risk",
    "confidence",
    "reason",
    "bull_score",
    "bear_score",
    "bull_sweep",
    "bear_sweep",
    "close_above_high",
    "close_below_low",
    "strong_body",
    "close_near_high",
    "close_near_low",
    "previous_open",
    "previous_high",
    "previous_low",
    "previous_close",
    "current_open",
    "current_high",
    "current_low",
    "current_close",
    "average_body_last_20",
    "body_size",
    "range_size",
    "body_percentage",
    "upper_wick",
    "lower_wick",
    "outcome",
    "exit_time",
    "exit_price",
    "rr_result",
]


def record_signal_for_outcome(signal: dict, path: Path = OUTCOME_PATH) -> None:
    if signal.get("signal") not in {"BUY", "SELL"}:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = read_outcomes(path)
    if not existing.empty:
        duplicate = existing["time"].astype(str).eq(str(signal.get("time"))) & existing["signal"].eq(signal.get("signal"))
        if duplicate.any():
            return

    debug = signal.get("debug") or {}
    previous = debug.get("previous") or {}
    current = debug.get("current") or {}
    entry = float(signal["entry"])
    sl = float(signal["sl"])
    risk = abs(entry - sl)
    tp2 = signal.get("tp2")
    if pd.isna(tp2) if tp2 is not None else True:
        tp2 = entry + 3 * risk if signal["signal"] == "BUY" else entry - 3 * risk
    body_size = abs(current.get("close", 0) - current.get("open", 0))
    range_size = max(current.get("high", 0) - current.get("low", 0), 0)

    row = {
        "time": signal.get("time"),
        "signal": signal.get("signal"),
        "entry": entry,
        "sl": sl,
        "tp1": signal.get("tp1"),
        "tp2": tp2,
        "risk": risk,
        "confidence": signal.get("confidence"),
        "reason": signal.get("reason"),
        "bull_score": debug.get("bull_score"),
        "bear_score": debug.get("bear_score"),
        "bull_sweep": debug.get("bull_sweep"),
        "bear_sweep": debug.get("bear_sweep"),
        "close_above_high": debug.get("close_above_high"),
        "close_below_low": debug.get("close_below_low"),
        "strong_body": debug.get("strong_body"),
        "close_near_high": debug.get("close_near_high"),
        "close_near_low": debug.get("close_near_low"),
        "previous_open": previous.get("open"),
        "previous_high": previous.get("high"),
        "previous_low": previous.get("low"),
        "previous_close": previous.get("close"),
        "current_open": current.get("open"),
        "current_high": current.get("high"),
        "current_low": current.get("low"),
        "current_close": current.get("close"),
        "average_body_last_20": debug.get("average_body_last_20"),
        "body_size": body_size,
        "range_size": range_size,
        "body_percentage": body_size / range_size if range_size else 0,
        "upper_wick": current.get("high", 0) - max(current.get("open", 0), current.get("close", 0)),
        "lower_wick": min(current.get("open", 0), current.get("close", 0)) - current.get("low", 0),
        "outcome": "PENDING",
        "exit_time": "",
        "exit_price": "",
        "rr_result": "",
    }
    pd.DataFrame([row], columns=OUTCOME_FIELDS).to_csv(path, mode="a", index=False, header=not path.exists())


def update_pending_outcomes(candles: pd.DataFrame, path: Path = OUTCOME_PATH, lookahead: int = LOOKAHEAD_CANDLES) -> pd.DataFrame:
    outcomes = read_outcomes(path)
    if outcomes.empty:
        return outcomes

    candles = candles.copy()
    candles["timestamp"] = pd.to_datetime(candles["timestamp"], errors="coerce")
    outcomes["time"] = pd.to_datetime(outcomes["time"], errors="coerce")
    outcomes["exit_time"] = outcomes["exit_time"].astype(str)

    for idx, trade in outcomes[outcomes["outcome"].eq("PENDING")].iterrows():
        signal_time = trade["time"]
        future = candles[candles["timestamp"] > signal_time].head(lookahead)
        if len(future) < lookahead:
            continue
        outcome, exit_price, exit_time, rr_result = simulate_outcome(trade, future)
        outcomes.loc[idx, "outcome"] = outcome
        outcomes.loc[idx, "exit_price"] = exit_price if exit_price is not None else ""
        outcomes.loc[idx, "exit_time"] = exit_time or ""
        outcomes.loc[idx, "rr_result"] = rr_result

    outcomes.to_csv(path, index=False)
    return outcomes


def simulate_outcome(trade: pd.Series, future: pd.DataFrame) -> tuple[str, float | None, str | None, float]:
    signal = trade["signal"]
    sl = float(trade["sl"])
    tp1 = float(trade["tp1"])
    tp2 = float(trade["tp2"])

    for _, candle in future.iterrows():
        high = float(candle["high"])
        low = float(candle["low"])
        timestamp = str(candle["timestamp"])
        if signal == "BUY":
            if low <= sl:
                return "SL", sl, timestamp, -1.0
            if high >= tp2:
                return "TP2", tp2, timestamp, 3.0
            if high >= tp1:
                return "TP1", tp1, timestamp, 1.5
        else:
            if high >= sl:
                return "SL", sl, timestamp, -1.0
            if low <= tp2:
                return "TP2", tp2, timestamp, 3.0
            if low <= tp1:
                return "TP1", tp1, timestamp, 1.5
    return "NO_HIT", None, None, 0.0


def read_outcomes(path: Path = OUTCOME_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=OUTCOME_FIELDS)
    return pd.read_csv(path)
