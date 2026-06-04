from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.outcome_tracker import OUTCOME_PATH, read_outcomes


MIN_COMPLETED = 30


def load_completed(path: Path = OUTCOME_PATH) -> pd.DataFrame:
    data = read_outcomes(path)
    if data.empty:
        return data
    completed = data[~data["outcome"].isin(["PENDING", ""])]
    completed = completed.copy()
    completed["rr_result"] = pd.to_numeric(completed["rr_result"], errors="coerce").fillna(0)
    for column in ["body_percentage", "body_size", "range_size", "upper_wick", "lower_wick", "confidence", "bull_score", "bear_score"]:
        if column in completed.columns:
            completed[column] = pd.to_numeric(completed[column], errors="coerce")
    return completed


def recommend_tuning(completed: pd.DataFrame, min_completed: int = MIN_COMPLETED) -> dict:
    if len(completed) < min_completed:
        return {
            "status": "not_enough_data",
            "completed_trades": int(len(completed)),
            "minimum_required": min_completed,
            "message": "Keep the live scanner running until more outcomes are complete.",
            "recommendations": {},
        }

    winners = completed[completed["rr_result"] > 0]
    losers = completed[completed["rr_result"] < 0]
    if winners.empty or losers.empty:
        return {
            "status": "not_enough_class_balance",
            "completed_trades": int(len(completed)),
            "wins": int(len(winners)),
            "losses": int(len(losers)),
            "recommendations": {},
        }

    recommendations = {}
    feature_comparison = compare_features(winners, losers)

    if _mean(winners, "body_percentage") > _mean(losers, "body_percentage"):
        recommendations["CLOSE_NEAR_RATIO"] = "Consider lowering only after enough samples; winners have stronger closes/body position."

    if _mean(winners, "body_size") > _mean(losers, "body_size"):
        recommendations["BODY_MULTIPLIER"] = "Consider increasing BODY_MULTIPLIER slightly if trade count remains high."
    elif _mean(winners, "body_size") < _mean(losers, "body_size"):
        recommendations["BODY_MULTIPLIER"] = "Do not increase BODY_MULTIPLIER yet; winners are not larger-bodied than losers."

    buy_pf = profit_factor(completed[completed["signal"].eq("BUY")])
    sell_pf = profit_factor(completed[completed["signal"].eq("SELL")])
    if buy_pf is not None and sell_pf is not None:
        if buy_pf < 1 and sell_pf > 1:
            recommendations["BUY_FILTER"] = "BUY is weaker than SELL. Consider collecting SELL-only live data or raising BUY min score to 4."
        if sell_pf < 1 and buy_pf > 1:
            recommendations["SELL_FILTER"] = "SELL is weaker than BUY. Consider raising SELL min score to 4."

    return {
        "status": "ready",
        "completed_trades": int(len(completed)),
        "win_rate": float((completed["rr_result"] > 0).mean() * 100),
        "profit_factor": profit_factor(completed),
        "buy_profit_factor": buy_pf,
        "sell_profit_factor": sell_pf,
        "feature_comparison": feature_comparison,
        "recommendations": recommendations,
    }


def compare_features(winners: pd.DataFrame, losers: pd.DataFrame) -> list[dict]:
    rows = []
    for column in ["confidence", "body_percentage", "body_size", "range_size", "upper_wick", "lower_wick", "bull_score", "bear_score"]:
        if column not in winners.columns:
            continue
        win_avg = _mean(winners, column)
        lose_avg = _mean(losers, column)
        rows.append({"feature": column, "winner_avg": win_avg, "loser_avg": lose_avg, "difference": win_avg - lose_avg})
    return sorted(rows, key=lambda item: abs(item["difference"]), reverse=True)


def profit_factor(data: pd.DataFrame) -> float | None:
    if data.empty:
        return None
    wins = data[data["rr_result"] > 0]["rr_result"].sum()
    losses = abs(data[data["rr_result"] < 0]["rr_result"].sum())
    return float(wins / losses) if losses else None


def save_recommendations(report: dict, output_dir: str | Path = "reports") -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "pattern_tuning_recommendations.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    feature_rows = report.get("feature_comparison") or []
    feature_html = pd.DataFrame(feature_rows).to_html(index=False) if feature_rows else "<p>No feature comparison available yet.</p>"
    rec_rows = "".join(f"<li><strong>{key}</strong>: {value}</li>" for key, value in (report.get("recommendations") or {}).items())
    (output / "pattern_tuning_recommendations.html").write_text(
        f"<html><body><h1>Pattern Tuning Recommendations</h1><pre>{json.dumps({k: v for k, v in report.items() if k != 'feature_comparison'}, indent=2)}</pre><h2>Feature Comparison</h2>{feature_html}<h2>Recommendations</h2><ul>{rec_rows}</ul></body></html>",
        encoding="utf-8",
    )


def _mean(data: pd.DataFrame, column: str) -> float:
    return float(pd.to_numeric(data[column], errors="coerce").mean())


def main() -> None:
    parser = argparse.ArgumentParser(description="Recommend Phase 1 pattern tuning from completed live outcomes.")
    parser.add_argument("--input", default=str(OUTCOME_PATH))
    parser.add_argument("--min-completed", type=int, default=MIN_COMPLETED)
    parser.add_argument("--output-dir", default="reports")
    args = parser.parse_args()
    completed = load_completed(Path(args.input))
    report = recommend_tuning(completed, args.min_completed)
    save_recommendations(report, args.output_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
