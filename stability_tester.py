from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.live_data import fetch_live_candles
from trade_evaluator import evaluate_trades


PERIODS = [1000, 5000, 10000]


def load_best_strategy(path: str | Path = "reports/best_strategy.json") -> dict:
    strategy_path = Path(path)
    if not strategy_path.exists():
        raise FileNotFoundError(f"Best strategy file not found: {strategy_path}")
    payload = json.loads(strategy_path.read_text(encoding="utf-8"))
    return payload.get("best_parameters", payload)


def run_stability_test(symbol: str, timeframe: str, strategy_path: str | Path, output_dir: str | Path = "reports") -> pd.DataFrame:
    strategy = load_best_strategy(strategy_path)
    rows = []
    for bars in PERIODS:
        candles = fetch_live_candles(symbol, timeframe, bars)
        _, _, _, metrics = evaluate_trades(
            candles,
            lookahead=20,
            min_score=3,
            body_multiplier=float(strategy.get("minimum_body_size", 1.5)),
            close_near_ratio=0.25 if strategy.get("close_near_extreme") else 0.35,
            require_sweep=bool(strategy.get("liquidity_sweep_required")),
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


def save_report(results: pd.DataFrame, strategy: dict, output_dir: str | Path) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    html_path = output / "stability_report.html"
    csv_path = output / "stability_summary.csv"
    performance_trend = classify_performance_trend(results)
    robustness_rank = rank_robustness(results)
    results = results.copy()
    results["performance_trend"] = performance_trend
    results["robustness_rank"] = robustness_rank
    results.to_csv(csv_path, index=False)
    stable = bool(results["profitable"].all()) if not results.empty else False
    verdict = "Strategy remains profitable across all tested periods." if stable else "Strategy is NOT stable across all tested periods."
    html_path.write_text(
        f"""
<html>
  <head><title>Strategy Stability Report</title></head>
  <body>
    <h1>Strategy Stability Report</h1>
    <h2>Best Strategy Tested</h2>
    <pre>{json.dumps(strategy, indent=2)}</pre>
    <h2>Verdict</h2>
    <p><strong>{verdict}</strong></p>
    <h2>Performance Trend</h2>
    <p><strong>{performance_trend}</strong></p>
    <h2>Robustness Rank</h2>
    <p><strong>{robustness_rank}</strong></p>
    <h2>1000 vs 5000 vs 10000 Candles</h2>
    {results.to_html(index=False)}
  </body>
</html>
""",
        encoding="utf-8",
    )
    return html_path


def classify_performance_trend(results: pd.DataFrame) -> str:
    if results.empty or len(results) < 2:
        return "Unknown"
    ordered = results.sort_values("candles")
    pf = ordered["profit_factor"].fillna(0).tolist()
    expectancy = ordered["expectancy"].fillna(0).tolist()
    drawdown = ordered["max_drawdown"].abs().fillna(0).tolist()

    pf_improves = pf[-1] > pf[0]
    expectancy_improves = expectancy[-1] > expectancy[0]
    drawdown_worsens = drawdown[-1] > drawdown[0]

    if pf_improves and expectancy_improves and not drawdown_worsens:
        return "Improves"
    if abs(pf[-1] - pf[0]) <= 0.1 and abs(expectancy[-1] - expectancy[0]) <= 0.05:
        return "Stays Stable"
    return "Deteriorates"


def rank_robustness(results: pd.DataFrame) -> str:
    if results.empty:
        return "Poor"
    pf_min = results["profit_factor"].fillna(0).min()
    expectancy_min = results["expectancy"].fillna(0).min()
    profitable_all = bool(results["profitable"].all())
    max_dd = results["max_drawdown"].abs().fillna(0).max()

    if profitable_all and pf_min >= 1.3 and expectancy_min > 0.1 and max_dd <= 20:
        return "Strong"
    if profitable_all and pf_min >= 1.1 and expectancy_min > 0 and max_dd <= 35:
        return "Moderate"
    if pf_min >= 0.95 and max_dd <= 75:
        return "Weak"
    return "Poor"


def main() -> None:
    parser = argparse.ArgumentParser(description="Test best strategy stability over multiple candle periods.")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--timeframe", default="M5", choices=["M1", "M5", "M15"])
    parser.add_argument("--strategy", default="reports/best_strategy.json")
    parser.add_argument("--output-dir", default="reports")
    args = parser.parse_args()
    results = run_stability_test(args.symbol, args.timeframe, args.strategy, args.output_dir)
    print(results.to_string(index=False))
    print("Performance Trend:", classify_performance_trend(results))
    print("Robustness Rank:", rank_robustness(results))
    print("Stable across all periods:", bool(results["profitable"].all()))
    print("HTML Report:", Path(args.output_dir) / "stability_report.html")


if __name__ == "__main__":
    main()
