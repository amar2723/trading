from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.live_data import fetch_live_candles
from trade_evaluator import evaluate_trades


def run_parameter_study(candles: pd.DataFrame, lookahead: int = 20) -> pd.DataFrame:
    rows = []
    for min_score in [3, 4]:
        for body_multiplier in [1.0, 1.25, 1.5, 1.75, 2.0]:
            for close_near_ratio in [0.15, 0.25, 0.35]:
                for require_sweep in [False, True]:
                    trades, buy, sell, combined = evaluate_trades(
                        candles,
                        lookahead=lookahead,
                        min_score=min_score,
                        body_multiplier=body_multiplier,
                        close_near_ratio=close_near_ratio,
                        require_sweep=require_sweep,
                    )
                    rows.append(
                        {
                            "min_score": min_score,
                            "body_multiplier": body_multiplier,
                            "close_near_ratio": close_near_ratio,
                            "require_sweep": require_sweep,
                            "buy_trades": buy["total_trades"],
                            "buy_win_rate": buy["win_rate"],
                            "buy_profit_factor": buy["profit_factor"],
                            "sell_trades": sell["total_trades"],
                            "sell_win_rate": sell["win_rate"],
                            "sell_profit_factor": sell["profit_factor"],
                            "combined_trades": combined["total_trades"],
                            "combined_win_rate": combined["win_rate"],
                            "combined_profit_factor": combined["profit_factor"],
                            "combined_drawdown": combined["max_drawdown"],
                            "combined_average_rr": combined["average_rr"],
                            "score": score(combined),
                        }
                    )
    return pd.DataFrame(rows).sort_values("score", ascending=False)


def score(metrics: dict) -> float:
    pf = metrics.get("profit_factor") or 0
    avg_rr = metrics.get("average_rr") or 0
    dd = abs(metrics.get("max_drawdown") or 0)
    trades = metrics.get("total_trades") or 0
    if trades < 20:
        return -999
    return pf * 2 + avg_rr - dd / 50


def main() -> None:
    parser = argparse.ArgumentParser(description="Find better Phase 1 pattern filters.")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--timeframe", default="M5", choices=["M1", "M5", "M15"])
    parser.add_argument("--bars", type=int, default=1000)
    parser.add_argument("--lookahead", type=int, default=20)
    parser.add_argument("--output", default="reports/pattern_parameter_study.csv")
    args = parser.parse_args()

    candles = fetch_live_candles(args.symbol, args.timeframe, args.bars)
    results = run_parameter_study(candles, args.lookahead)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)

    print("Top 10 Settings:")
    print(results.head(10).to_string(index=False))
    print("Saved:", output)


if __name__ == "__main__":
    main()
