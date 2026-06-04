from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import pandas as pd


CONFIDENCE_LEVELS = [50, 60, 70, 80, 90]
BODY_MULTIPLIERS = [1.0, 1.25, 1.5, 2.0]
BOOLEAN_OPTIONS = [False, True]


def load_reports(report_dir: str | Path = "reports") -> dict[str, pd.DataFrame]:
    base = Path(report_dir)
    return {
        "outcome_features": _read_csv(base / "outcome_feature_analysis.csv"),
        "trade_outcomes": _read_csv(base / "trade_outcomes.csv"),
        "parameter_study": _read_csv(base / "pattern_parameter_study.csv"),
    }


def optimize_filters(report_dir: str | Path = "reports") -> tuple[pd.DataFrame, dict]:
    reports = load_reports(report_dir)
    parameter_study = reports["parameter_study"]
    trade_outcomes = reports["trade_outcomes"]
    outcome_features = reports["outcome_features"]

    rows = []
    for min_confidence, min_body, trend_alignment, require_sweep, close_near_extreme in product(
        CONFIDENCE_LEVELS,
        BODY_MULTIPLIERS,
        BOOLEAN_OPTIONS,
        BOOLEAN_OPTIONS,
        BOOLEAN_OPTIONS,
    ):
        metrics = evaluate_combination(
            parameter_study,
            trade_outcomes,
            min_confidence,
            min_body,
            trend_alignment,
            require_sweep,
            close_near_extreme,
        )
        rows.append(metrics)

    results = pd.DataFrame(rows)
    results = results.sort_values(["profit_factor", "expectancy", "trades"], ascending=[False, False, False]).reset_index(drop=True)
    best = build_recommendation(results.iloc[0].to_dict() if not results.empty else {}, outcome_features)
    return results, best


def evaluate_combination(
    parameter_study: pd.DataFrame,
    trade_outcomes: pd.DataFrame,
    min_confidence: int,
    min_body: float,
    trend_alignment: bool,
    require_sweep: bool,
    close_near_extreme: bool,
) -> dict:
    matched = pd.DataFrame()
    if not parameter_study.empty:
        matched = parameter_study[
            parameter_study["body_multiplier"].astype(float).eq(min_body)
            & parameter_study["require_sweep"].map(_to_bool).eq(require_sweep)
        ].copy()
        if close_near_extreme:
            matched = matched[matched["close_near_ratio"].astype(float).le(0.25)]
        else:
            matched = matched[matched["close_near_ratio"].astype(float).ge(0.25)]
        if trend_alignment:
            # Trend was not part of the original parameter study. Apply a conservative penalty until live trend data exists.
            matched["combined_trades"] = (matched["combined_trades"] * 0.75).round()
            matched["combined_drawdown"] = matched["combined_drawdown"] * 0.8

    if not matched.empty:
        chosen = matched.sort_values(["combined_profit_factor", "combined_average_rr"], ascending=False).iloc[0]
        trades = int(chosen.get("combined_trades", 0))
        win_rate = float(chosen.get("combined_win_rate", 0.0))
        profit_factor = _safe_float(chosen.get("combined_profit_factor"))
        expectancy = _safe_float(chosen.get("combined_average_rr"))
        drawdown = _safe_float(chosen.get("combined_drawdown"))
    else:
        filtered = filter_trade_outcomes(trade_outcomes, min_confidence, require_sweep, close_near_extreme)
        trades, win_rate, profit_factor, expectancy, drawdown = metrics_from_trades(filtered)

    if min_confidence > 75:
        # Existing reports mostly use 75/90 confidence. Penalize stricter thresholds when no direct support exists.
        trades = int(trades * 0.8)
    if min_confidence > 85:
        trades = int(trades * 0.7)

    return {
        "minimum_confidence": min_confidence,
        "minimum_body_size": min_body,
        "trend_alignment": trend_alignment,
        "liquidity_sweep_required": require_sweep,
        "close_near_extreme": close_near_extreme,
        "trades": trades,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "drawdown": drawdown,
    }


def filter_trade_outcomes(trades: pd.DataFrame, min_confidence: int, require_sweep: bool, close_near_extreme: bool) -> pd.DataFrame:
    if trades.empty:
        return trades
    filtered = trades.copy()
    if "confidence" in filtered.columns:
        filtered = filtered[pd.to_numeric(filtered["confidence"], errors="coerce").fillna(0) >= min_confidence]
    if require_sweep and "reason" in filtered.columns:
        filtered = filtered[filtered["reason"].fillna("").str.contains("Liquidity Sweep", case=False)]
    if close_near_extreme and "reason" in filtered.columns:
        filtered = filtered[
            filtered["reason"].fillna("").str.contains("Close Near High|Close Near Low", case=False, regex=True)
        ]
    return filtered


def metrics_from_trades(trades: pd.DataFrame) -> tuple[int, float, float | None, float, float]:
    if trades.empty or "rr" not in trades.columns:
        return 0, 0.0, None, 0.0, 0.0
    rr = pd.to_numeric(trades["rr"], errors="coerce").fillna(0)
    wins = rr[rr > 0]
    losses = rr[rr < 0]
    gross_profit = wins.sum()
    gross_loss = abs(losses.sum())
    equity = rr.cumsum()
    drawdown = equity - equity.cummax()
    return (
        int(len(trades)),
        float(len(wins) / len(trades) * 100) if len(trades) else 0.0,
        float(gross_profit / gross_loss) if gross_loss else None,
        float(rr.mean()) if len(rr) else 0.0,
        float(drawdown.min()) if len(drawdown) else 0.0,
    )


def build_recommendation(best: dict, outcome_features: pd.DataFrame) -> dict:
    recommendation = {
        "best_parameters": best,
        "recommended_rule": (
            f"Use confidence >= {best.get('minimum_confidence')}, "
            f"body >= {best.get('minimum_body_size')}x average body, "
            f"trend alignment {'ON' if best.get('trend_alignment') else 'OFF'}, "
            f"liquidity sweep {'REQUIRED' if best.get('liquidity_sweep_required') else 'OPTIONAL'}, "
            f"close near extreme {'ON' if best.get('close_near_extreme') else 'OFF'}."
        ),
        "notes": [
            "Ranked primarily by Profit Factor, then Expectancy, then trade count.",
            "Trend alignment is marked as experimental unless trend columns are added to live outcomes.",
            "Use at least 30 completed live outcomes before trusting live-based recommendations.",
        ],
    }
    if not outcome_features.empty:
        recommendation["feature_evidence"] = outcome_features.head(10).to_dict(orient="records")
    return recommendation


def save_outputs(results: pd.DataFrame, best: dict, report_dir: str | Path = "reports") -> dict[str, Path]:
    base = Path(report_dir)
    base.mkdir(parents=True, exist_ok=True)
    top20 = results.head(20)
    html_path = base / "filter_optimization.html"
    json_path = base / "best_strategy.json"
    csv_path = base / "filter_optimization_top20.csv"
    top20.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(best, indent=2, default=str), encoding="utf-8")
    html_path.write_text(
        f"""
<html>
  <head><title>Filter Optimization</title></head>
  <body>
    <h1>Filter Optimization</h1>
    <h2>Recommended Strategy</h2>
    <pre>{json.dumps(best, indent=2, default=str)}</pre>
    <h2>Top 20 Parameter Combinations</h2>
    {top20.to_html(index=False)}
  </body>
</html>
""",
        encoding="utf-8",
    )
    return {"html": html_path, "json": json_path, "csv": csv_path}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "on"}


def _safe_float(value) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize rule-based filters from generated reports.")
    parser.add_argument("--reports", default="reports")
    args = parser.parse_args()
    results, best = optimize_filters(args.reports)
    paths = save_outputs(results, best, args.reports)
    print("Top 20 parameter combinations ranked by Profit Factor:")
    print(results.head(20).to_string(index=False))
    print("Recommended Strategy:")
    print(best["recommended_rule"])
    print("HTML:", paths["html"])
    print("JSON:", paths["json"])


if __name__ == "__main__":
    main()
