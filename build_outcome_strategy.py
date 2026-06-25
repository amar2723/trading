from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


REPORTS_DIR = Path("reports")
LOGS_DIR = Path("logs")


def load_all_outcomes() -> pd.DataFrame:
    frames = [
        _load_live_outcomes(LOGS_DIR / "signal_outcomes.csv"),
        _load_trade_report(REPORTS_DIR / "trade_outcomes.csv", "historical_evaluator"),
        _load_trade_report(REPORTS_DIR / "strategy_lab_trades.csv", "strategy_lab"),
    ]
    data = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True)
    if data.empty:
        return data
    data["time"] = pd.to_datetime(data["time"], errors="coerce")
    data = data.dropna(subset=["time", "signal", "rr"])
    data = data[data["signal"].isin(["BUY", "SELL"])]
    data["hour"] = data["time"].dt.hour
    data["session"] = data["hour"].map(_session_name)
    data["win"] = data["rr"] > 0
    data["loss"] = data["rr"] < 0
    return data.sort_values("time").reset_index(drop=True)


def _load_live_outcomes(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 2:
        return pd.DataFrame()
    data = pd.read_csv(path)
    if data.empty:
        return data
    data["rr"] = pd.to_numeric(data.get("rr_result"), errors="coerce")
    data["source"] = "live_outcomes"
    data["confidence"] = pd.to_numeric(data.get("confidence"), errors="coerce")
    data["chop_ratio"] = pd.NA
    data["trend_direction"] = pd.NA
    return _standard_columns(data)


def _load_trade_report(path: Path, source: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 2:
        return pd.DataFrame()
    data = pd.read_csv(path)
    if data.empty:
        return data
    if "rr" not in data.columns and "rr_result" in data.columns:
        data["rr"] = pd.to_numeric(data["rr_result"], errors="coerce")
    data["rr"] = pd.to_numeric(data.get("rr"), errors="coerce")
    data["source"] = source
    if "confidence" in data.columns:
        data["confidence"] = pd.to_numeric(data["confidence"], errors="coerce")
    else:
        data["confidence"] = pd.NA
    if "chop_ratio" not in data.columns:
        data["chop_ratio"] = pd.NA
    if "trend_direction" not in data.columns:
        data["trend_direction"] = pd.NA
    return _standard_columns(data)


def _standard_columns(data: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "time",
        "signal",
        "entry",
        "sl",
        "tp1",
        "tp2",
        "risk",
        "outcome",
        "rr",
        "confidence",
        "reason",
        "trend_direction",
        "chop_ratio",
        "source",
    ]
    for column in columns:
        if column not in data.columns:
            data[column] = pd.NA
    return data[columns].copy()


def _session_name(hour: int) -> str:
    if 7 <= hour < 12:
        return "LONDON"
    if 13 <= hour < 16:
        return "OVERLAP"
    if 16 <= hour < 20:
        return "NEW_YORK"
    return "OTHER"


def metrics(data: pd.DataFrame) -> dict[str, Any]:
    if data.empty:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "profit_factor": None,
            "expectancy": 0.0,
            "max_drawdown": 0.0,
        }
    wins = data[data["rr"] > 0]
    losses = data[data["rr"] < 0]
    gross_profit = float(wins["rr"].sum())
    gross_loss = abs(float(losses["rr"].sum()))
    equity = data["rr"].cumsum()
    drawdown = equity - equity.cummax()
    return {
        "trades": int(len(data)),
        "win_rate": float(len(wins) / len(data) * 100),
        "profit_factor": gross_profit / gross_loss if gross_loss else None,
        "expectancy": float(data["rr"].mean()),
        "max_drawdown": float(drawdown.min()) if not drawdown.empty else 0.0,
    }


def segment_table(data: pd.DataFrame, column: str, min_trades: int = 5) -> pd.DataFrame:
    rows = []
    for value, group in data.groupby(column, dropna=False):
        result = metrics(group)
        if result["trades"] >= min_trades:
            rows.append({column: value, **result, "score": score_metrics(result)})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["score", "profit_factor", "expectancy"], ascending=[False, False, False])


def score_metrics(result: dict[str, Any]) -> float:
    trades = int(result.get("trades") or 0)
    if trades < 5:
        return -999.0
    pf = result.get("profit_factor")
    pf_value = float(pf) if pf is not None and math.isfinite(float(pf)) else 5.0
    expectancy = float(result.get("expectancy") or 0.0)
    drawdown = abs(float(result.get("max_drawdown") or 0.0))
    win_rate = float(result.get("win_rate") or 0.0) / 100.0
    sample_bonus = min(trades, 100) / 100.0
    return pf_value * 2.0 + expectancy * 3.0 + win_rate + sample_bonus - drawdown * 0.10


def build_strategy(data: pd.DataFrame) -> dict[str, Any]:
    best_lab = _read_best_lab_strategy()
    by_signal = segment_table(data, "signal", min_trades=8)
    by_hour = segment_table(data, "hour", min_trades=4)
    by_session = segment_table(data, "session", min_trades=8)
    by_trend = segment_table(data.dropna(subset=["trend_direction"]), "trend_direction", min_trades=8)

    allowed_signals = _allowed_signals(by_signal)
    if not allowed_signals:
        allowed_signals = ["BUY", "SELL"]

    allowed_hours = _allowed_hours(by_hour)
    best_session = _best_session(by_session)
    trend_alignment = _should_use_trend(by_trend, allowed_signals)

    strategy = {
        "mode": "paper_research_only",
        "warning": "No accuracy guarantee. Use for demo filtering until validated on more completed live outcomes.",
        "allowed_signals": allowed_signals,
        "allowed_hours": allowed_hours,
        "session": best_session,
        "trend_alignment": trend_alignment,
        "avoid_chop": True,
        "min_chop_ratio": float(best_lab.get("min_chop_ratio", 4.0) or 4.0),
        "min_confidence": 90 if _needs_strict_confidence(data) else 75,
        "min_tp1_rr": 1.5,
        "pattern": {
            "min_score": int(best_lab.get("min_score", 4)),
            "body_multiplier": float(best_lab.get("body_multiplier", 1.5)),
            "close_near_ratio": float(best_lab.get("close_near_ratio", 0.35)),
            "require_sweep": bool(best_lab.get("require_sweep", True)),
        },
        "evidence": {
            "all": metrics(data),
            "by_signal": by_signal.to_dict("records"),
            "by_hour": by_hour.head(10).to_dict("records"),
            "by_session": by_session.to_dict("records"),
            "by_trend": by_trend.to_dict("records") if not by_trend.empty else [],
        },
    }
    return strategy


def _read_best_lab_strategy() -> dict[str, Any]:
    path = REPORTS_DIR / "strategy_lab_best.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload.get("best_strategy", {})
    except Exception:
        return {}


def _allowed_signals(by_signal: pd.DataFrame) -> list[str]:
    if by_signal.empty:
        return []
    allowed = []
    for _, row in by_signal.iterrows():
        pf = row["profit_factor"]
        expectancy = float(row["expectancy"])
        if pf is not None and float(pf) >= 1.05 and expectancy > 0:
            allowed.append(str(row["signal"]))
    return allowed


def _allowed_hours(by_hour: pd.DataFrame) -> list[int]:
    if by_hour.empty:
        return []
    hours = []
    for _, row in by_hour.head(6).iterrows():
        pf = row["profit_factor"]
        if pf is not None and float(pf) >= 1.20 and float(row["expectancy"]) > 0:
            hours.append(int(row["hour"]))
    return sorted(hours)


def _best_session(by_session: pd.DataFrame) -> str:
    if by_session.empty:
        return "ALL"
    best = by_session.iloc[0]
    pf = best["profit_factor"]
    if pf is not None and float(pf) >= 1.10 and float(best["expectancy"]) > 0:
        session = str(best["session"])
        return "ALL" if session == "OTHER" else session
    return "ALL"


def _should_use_trend(by_trend: pd.DataFrame, allowed_signals: list[str]) -> bool:
    if by_trend.empty:
        return False
    if len(allowed_signals) == 1:
        return False
    best = by_trend.iloc[0]
    pf = best["profit_factor"]
    return bool(pf is not None and float(pf) >= 1.20 and float(best["expectancy"]) > 0)


def _needs_strict_confidence(data: pd.DataFrame) -> bool:
    if "confidence" not in data.columns or data["confidence"].dropna().empty:
        return False
    table = segment_table(data.assign(confidence_bucket=data["confidence"].fillna(0).astype(float)), "confidence_bucket", min_trades=8)
    if table.empty:
        return False
    best_conf = float(table.iloc[0]["confidence_bucket"])
    return best_conf >= 90


def save_strategy(data: pd.DataFrame, strategy: dict[str, Any], output_dir: str | Path = REPORTS_DIR) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "outcome_strategy.json"
    csv_path = output / "outcome_strategy_dataset.csv"
    html_path = output / "outcome_strategy_report.html"

    json_path.write_text(json.dumps(strategy, indent=2, default=str), encoding="utf-8")
    data.to_csv(csv_path, index=False)
    html_path.write_text(
        "<html><head><title>Outcome Strategy</title></head><body>"
        "<h1>Outcome-Built Strategy</h1>"
        "<p><strong>Mode:</strong> paper research only. This is probability filtering, not guaranteed accuracy.</p>"
        "<h2>Recommended Strategy</h2>"
        f"<pre>{json.dumps(strategy, indent=2, default=str)}</pre>"
        "<h2>All Outcome Data</h2>"
        f"{data.tail(200).to_html(index=False)}"
        "</body></html>",
        encoding="utf-8",
    )
    return {"json": json_path, "csv": csv_path, "html": html_path}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an adaptive rule strategy from all available trade outcomes.")
    parser.add_argument("--output-dir", default="reports")
    args = parser.parse_args()

    data = load_all_outcomes()
    if data.empty:
        raise SystemExit("No outcome data found.")

    strategy = build_strategy(data)
    paths = save_strategy(data, strategy, args.output_dir)
    print("Outcome rows:", len(data))
    print("Recommended strategy:")
    print(json.dumps(strategy, indent=2, default=str))
    print("Report:", paths["html"])
    print("Strategy JSON:", paths["json"])


if __name__ == "__main__":
    main()
