from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.live_data import fetch_live_candles
from stability_tester import classify_performance_trend, rank_robustness
from trade_evaluator import evaluate_trades


PERIODS = [1000, 5000, 10000]


def load_strategy(path: str | Path = "reports/strict_best_strategy.json") -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_strict_stability(symbol: str, timeframe: str, strategy_path: str | Path, output_dir: str | Path = "reports") -> pd.DataFrame:
    strategy = load_strategy(strategy_path)
    rows = []
    for bars in PERIODS:
        candles = fetch_live_candles(symbol, timeframe, bars)
        _, _, _, metrics = evaluate_trades(
            candles,
            session=strategy.get("session", "ALL"),
            trend_alignment=bool(strategy.get("trend_alignment", False)),
            avoid_chop=bool(strategy.get("avoid_chop", False)),
            min_chop_ratio=float(strategy.get("min_chop_ratio", 4.0)),
        )
        rows.append(
            {
                "candles": bars,
                "trade_count": metrics["total_trades"],
                "win_rate": metrics["win_rate"],
                "profit_factor": metrics["profit_factor"],
                "expectancy": metrics["average_rr"],
                "max_drawdown": metrics["max_drawdown"],
                "profitable": (metrics["profit_factor"] or 0) > 1 and metrics["average_rr"] > 0,
            }
        )
    results = pd.DataFrame(rows)
    save_report(results, strategy, output_dir)
    return results


def save_report(results: pd.DataFrame, strategy: dict, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    trend = classify_performance_trend(results)
    rank = rank_robustness(results)
    enriched = results.copy()
    enriched["performance_trend"] = trend
    enriched["robustness_rank"] = rank
    enriched.to_csv(output / "strict_stability_summary.csv", index=False)
    (output / "strict_stability_report.html").write_text(
        f"<html><body><h1>Strict Strategy Stability Report</h1><h2>Strategy</h2><pre>{json.dumps(strategy, indent=2)}</pre><h2>Trend</h2><p>{trend}</p><h2>Robustness</h2><p>{rank}</p>{enriched.to_html(index=False)}</body></html>",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Test strict best strategy stability.")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--timeframe", default="M5", choices=["M1", "M5", "M15"])
    parser.add_argument("--strategy", default="reports/strict_best_strategy.json")
    parser.add_argument("--output-dir", default="reports")
    args = parser.parse_args()
    results = run_strict_stability(args.symbol, args.timeframe, args.strategy, args.output_dir)
    print(results.to_string(index=False))
    print("Performance Trend:", classify_performance_trend(results))
    print("Robustness Rank:", rank_robustness(results))


if __name__ == "__main__":
    main()
