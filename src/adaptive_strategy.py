from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.market_filters import passes_chop_filter, passes_session_filter, passes_trend_filter


DEFAULT_STRATEGY_PATH = Path("reports/outcome_strategy.json")


def load_adaptive_strategy(path: str | Path = DEFAULT_STRATEGY_PATH) -> dict[str, Any]:
    strategy_path = Path(path)
    if not strategy_path.exists():
        return {}
    return json.loads(strategy_path.read_text(encoding="utf-8"))


def strategy_detector_kwargs(strategy: dict[str, Any]) -> dict[str, Any]:
    pattern = strategy.get("pattern", {})
    kwargs: dict[str, Any] = {}
    for source_key, target_key in [
        ("min_score", "min_score"),
        ("body_multiplier", "body_multiplier"),
        ("close_near_ratio", "close_near_ratio"),
        ("require_sweep", "require_sweep"),
    ]:
        if source_key in pattern and pattern[source_key] is not None:
            kwargs[target_key] = pattern[source_key]
    return kwargs


def adaptive_rejection_reason(signal: dict, signal_row: pd.Series, strategy: dict[str, Any]) -> str | None:
    if not strategy:
        return None
    if signal.get("signal") not in {"BUY", "SELL"}:
        return None

    allowed_signals = strategy.get("allowed_signals") or ["BUY", "SELL"]
    if signal["signal"] not in allowed_signals:
        return f"adaptive direction filter allows only {', '.join(allowed_signals)}"

    min_confidence = float(strategy.get("min_confidence", 0.0) or 0.0)
    if float(signal.get("confidence") or 0.0) < min_confidence:
        return f"confidence below adaptive minimum {min_confidence:.0f}%"

    allowed_hours = strategy.get("allowed_hours")
    if allowed_hours:
        ts = pd.to_datetime(signal.get("time"), errors="coerce")
        if pd.isna(ts) or int(ts.hour) not in {int(hour) for hour in allowed_hours}:
            return f"outside adaptive hours {allowed_hours}"

    session = strategy.get("session", "ALL")
    if not passes_session_filter(signal.get("time"), session):
        return f"outside adaptive session {session}"

    if not passes_trend_filter(signal_row, signal["signal"], bool(strategy.get("trend_alignment", False))):
        return "adaptive trend alignment failed"

    if not passes_chop_filter(
        signal_row,
        bool(strategy.get("avoid_chop", False)),
        float(strategy.get("min_chop_ratio", 4.0) or 4.0),
    ):
        return "adaptive chop filter failed"

    min_rr = strategy.get("min_tp1_rr")
    if min_rr is not None:
        entry = signal.get("entry")
        sl = signal.get("sl")
        tp1 = signal.get("tp1")
        try:
            risk = abs(float(entry) - float(sl))
            reward = abs(float(tp1) - float(entry))
            rr = reward / risk if risk > 0 else 0.0
        except Exception:
            rr = 0.0
        if rr < float(min_rr):
            return f"TP1 RR below adaptive minimum {float(min_rr):.2f}"

    return None
