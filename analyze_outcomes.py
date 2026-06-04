from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.outcome_tracker import OUTCOME_PATH, read_outcomes


def analyze(path: Path = OUTCOME_PATH) -> tuple[dict, pd.DataFrame]:
    data = read_outcomes(path)
    completed = data[~data["outcome"].isin(["PENDING", ""])] if not data.empty else data
    if completed.empty:
        return {"total_completed": 0}, pd.DataFrame()

    completed["rr_result"] = pd.to_numeric(completed["rr_result"], errors="coerce").fillna(0)
    wins = completed[completed["rr_result"] > 0]
    losses = completed[completed["rr_result"] < 0]
    gross_profit = wins["rr_result"].sum()
    gross_loss = abs(losses["rr_result"].sum())
    metrics = {
        "total_completed": int(len(completed)),
        "win_rate": float(len(wins) / len(completed) * 100),
        "profit_factor": float(gross_profit / gross_loss) if gross_loss else None,
        "average_rr": float(completed["rr_result"].mean()),
        "buy_trades": int(completed["signal"].eq("BUY").sum()),
        "sell_trades": int(completed["signal"].eq("SELL").sum()),
    }

    feature_rows = []
    for column in [
        "bull_score",
        "bear_score",
        "confidence",
        "average_body_last_20",
        "body_size",
        "range_size",
        "body_percentage",
        "upper_wick",
        "lower_wick",
    ]:
        if column not in completed.columns:
            continue
        completed[column] = pd.to_numeric(completed[column], errors="coerce")
        feature_rows.append(
            {
                "feature": column,
                "winning_avg": completed.loc[completed["rr_result"] > 0, column].mean(),
                "losing_avg": completed.loc[completed["rr_result"] < 0, column].mean(),
                "difference": completed.loc[completed["rr_result"] > 0, column].mean()
                - completed.loc[completed["rr_result"] < 0, column].mean(),
            }
        )
    return metrics, pd.DataFrame(feature_rows).sort_values("difference", ascending=False)


def save_analysis(output_dir: str | Path = "reports") -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics, features = analyze()
    features.to_csv(output / "outcome_feature_analysis.csv", index=False)
    rows = "".join(f"<tr><th>{key}</th><td>{value}</td></tr>" for key, value in metrics.items())
    feature_html = features.to_html(index=False) if not features.empty else "<p>No completed outcomes yet.</p>"
    (output / "outcome_analysis.html").write_text(
        f"<html><body><h1>Live Outcome Analysis</h1><h2>Metrics</h2><table>{rows}</table><h2>Feature Comparison</h2>{feature_html}</body></html>",
        encoding="utf-8",
    )
    print(metrics)
    if not features.empty:
        print(features.to_string(index=False))


if __name__ == "__main__":
    save_analysis()
