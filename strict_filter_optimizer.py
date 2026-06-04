from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import pandas as pd

from src.live_data import fetch_live_candles
from trade_evaluator import evaluate_trades


SESSIONS = ["ALL", "LONDON", "NEW_YORK", "OVERLAP"]
TREND_OPTIONS = [False, True]
CHOP_OPTIONS = [False, True]
CHOP_RATIOS = [3.0, 4.0, 5.0]


def optimize_strict_filters(candles: pd.DataFrame, lookahead: int = 20) -> pd.DataFrame:
    rows = []
    for session, trend_alignment, avoid_chop, min_chop_ratio in product(SESSIONS, TREND_OPTIONS, CHOP_OPTIONS, CHOP_RATIOS):
        _, buy, sell, combined = evaluate_trades(
            candles,
            lookahead=lookahead,
            session=session,
            trend_alignment=trend_alignment,
            avoid_chop=avoid_chop,
            min_chop_ratio=min_chop_ratio,
        )
        rows.append(
            {
                "session": session,
                "trend_alignment": trend_alignment,
                "avoid_chop": avoid_chop,
                "min_chop_ratio": min_chop_ratio,
                "trades": combined["total_trades"],
                "win_rate": combined["win_rate"],
                "profit_factor": combined["profit_factor"],
                "expectancy": combined["average_rr"],
                "drawdown": combined["max_drawdown"],
                "buy_trades": buy["total_trades"],
                "buy_profit_factor": buy["profit_factor"],
                "sell_trades": sell["total_trades"],
                "sell_profit_factor": sell["profit_factor"],
                "score": score(combined),
            }
        )
    return pd.DataFrame(rows).sort_values(["profit_factor", "expectancy", "trades"], ascending=[False, False, False])


def score(metrics: dict) -> float:
    trades = metrics.get("total_trades") or 0
    if trades < 20:
        return -999
    pf = metrics.get("profit_factor") or 0
    expectancy = metrics.get("average_rr") or 0
    drawdown = abs(metrics.get("max_drawdown") or 0)
    return pf * 2 + expectancy - drawdown / 50


def save_report(results: pd.DataFrame, output_dir: str | Path = "reports") -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "strict_filter_optimization.csv"
    html_path = output / "strict_filter_optimization.html"
    json_path = output / "strict_best_strategy.json"
    top = results.head(20)
    best = top.iloc[0].to_dict() if not top.empty else {}
    results.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(best, indent=2, default=str), encoding="utf-8")
    html_path.write_text(
        f"<html><body><h1>Strict Filter Optimization</h1><h2>Best Strategy</h2><pre>{json.dumps(best, indent=2, default=str)}</pre><h2>Top 20</h2>{top.to_html(index=False)}</body></html>",
        encoding="utf-8",
    )
    return {"csv": csv_path, "html": html_path, "json": json_path}


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize session, trend, and chop filters.")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--timeframe", default="M5", choices=["M1", "M5", "M15"])
    parser.add_argument("--bars", type=int, default=5000)
    parser.add_argument("--lookahead", type=int, default=20)
    parser.add_argument("--output-dir", default="reports")
    args = parser.parse_args()
    candles = fetch_live_candles(args.symbol, args.timeframe, args.bars)
    results = optimize_strict_filters(candles, args.lookahead)
    paths = save_report(results, args.output_dir)
    print(results.head(20).to_string(index=False))
    print("HTML:", paths["html"])
    print("Best JSON:", paths["json"])


if __name__ == "__main__":
    main()
