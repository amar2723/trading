from __future__ import annotations

import argparse
import json
import math
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd

from src.live_data import fetch_live_candles
from trade_evaluator import evaluate_trades


SESSIONS = ["ALL", "LONDON", "NEW_YORK", "OVERLAP"]
MIN_SCORES = [3]
BODY_MULTIPLIERS = [1.5, 2.0]
CLOSE_NEAR_RATIOS = [0.25, 0.35]
REQUIRE_SWEEP = [True]
TREND_ALIGNMENT = [False, True]
AVOID_CHOP = [False, True]
CHOP_RATIOS = [4.0]


def build_strategy_grid() -> list[dict[str, Any]]:
    strategies: list[dict[str, Any]] = []
    for min_score, body, close_near, require_sweep, session, trend, avoid_chop, chop_ratio in product(
        MIN_SCORES,
        BODY_MULTIPLIERS,
        CLOSE_NEAR_RATIOS,
        REQUIRE_SWEEP,
        SESSIONS,
        TREND_ALIGNMENT,
        AVOID_CHOP,
        CHOP_RATIOS,
    ):
        strategies.append(
            {
                "min_score": min_score,
                "body_multiplier": body,
                "close_near_ratio": close_near,
                "require_sweep": require_sweep,
                "session": session,
                "trend_alignment": trend,
                "avoid_chop": avoid_chop,
                "min_chop_ratio": chop_ratio,
            }
        )
    return strategies


def evaluate_strategy(candles: pd.DataFrame, strategy: dict[str, Any], lookahead: int) -> tuple[pd.DataFrame, dict, dict, dict]:
    return evaluate_trades(
        candles,
        lookahead=lookahead,
        min_score=strategy["min_score"],
        body_multiplier=strategy["body_multiplier"],
        close_near_ratio=strategy["close_near_ratio"],
        require_sweep=strategy["require_sweep"],
        session=strategy["session"],
        trend_alignment=strategy["trend_alignment"],
        avoid_chop=strategy["avoid_chop"],
        min_chop_ratio=strategy["min_chop_ratio"],
    )


def score_strategy(metrics: dict[str, Any]) -> float:
    trades = int(metrics.get("total_trades") or 0)
    if trades < 15:
        return -9999.0

    pf = metrics.get("profit_factor")
    pf_value = float(pf) if pf is not None and math.isfinite(float(pf)) else 10.0
    expectancy = float(metrics.get("average_rr") or 0.0)
    drawdown = abs(float(metrics.get("max_drawdown") or 0.0))
    win_rate = float(metrics.get("win_rate") or 0.0) / 100.0

    trade_penalty = 0.0 if trades >= 30 else (30 - trades) * 0.08
    return pf_value * 2.0 + expectancy * 3.0 + win_rate - drawdown * 0.15 - trade_penalty


def summarize_trades(trades: pd.DataFrame) -> dict[str, Any]:
    if trades.empty:
        return {
            "best_time_window": "No trades",
            "buy_trades": 0,
            "sell_trades": 0,
            "buy_win_rate": 0.0,
            "sell_win_rate": 0.0,
            "sharpe_ratio": 0.0,
        }

    data = trades.copy()
    data["time"] = pd.to_datetime(data["time"], errors="coerce")
    data["hour"] = data["time"].dt.hour
    hourly = (
        data.groupby("hour")
        .agg(trades=("rr", "count"), expectancy=("rr", "mean"), wins=("rr", lambda s: int((s > 0).sum())))
        .reset_index()
    )
    hourly["win_rate"] = hourly["wins"] / hourly["trades"] * 100
    hourly = hourly[hourly["trades"] >= max(2, min(5, len(data) // 10))]
    if hourly.empty:
        best_time = "Not enough hourly samples"
    else:
        best = hourly.sort_values(["expectancy", "win_rate", "trades"], ascending=[False, False, False]).iloc[0]
        best_time = f"{int(best['hour']):02d}:00-{int(best['hour']) + 1:02d}:00"

    buy = data[data["signal"].eq("BUY")]
    sell = data[data["signal"].eq("SELL")]
    rr_std = float(data["rr"].std(ddof=0) or 0.0)
    sharpe = float(data["rr"].mean() / rr_std * math.sqrt(len(data))) if rr_std > 0 else 0.0

    return {
        "best_time_window": best_time,
        "buy_trades": int(len(buy)),
        "sell_trades": int(len(sell)),
        "buy_win_rate": float((buy["rr"] > 0).mean() * 100) if not buy.empty else 0.0,
        "sell_win_rate": float((sell["rr"] > 0).mean() * 100) if not sell.empty else 0.0,
        "sharpe_ratio": sharpe,
    }


def run_strategy_lab(candles: pd.DataFrame, lookahead: int = 20, train_ratio: float = 0.7) -> dict[str, Any]:
    split = int(len(candles) * train_ratio)
    train = candles.iloc[:split].reset_index(drop=True)
    test = candles.iloc[split:].reset_index(drop=True)
    rows: list[dict[str, Any]] = []

    for strategy in build_strategy_grid():
        train_trades, train_buy, train_sell, train_combined = evaluate_strategy(train, strategy, lookahead)
        test_trades, test_buy, test_sell, test_combined = evaluate_strategy(test, strategy, lookahead)
        test_extra = summarize_trades(test_trades)

        rows.append(
            {
                **strategy,
                "train_trades": train_combined["total_trades"],
                "train_win_rate": train_combined["win_rate"],
                "train_profit_factor": train_combined["profit_factor"],
                "train_expectancy": train_combined["average_rr"],
                "train_drawdown": train_combined["max_drawdown"],
                "test_trades": test_combined["total_trades"],
                "test_win_rate": test_combined["win_rate"],
                "test_profit_factor": test_combined["profit_factor"],
                "test_expectancy": test_combined["average_rr"],
                "test_drawdown": test_combined["max_drawdown"],
                "test_sharpe": test_extra["sharpe_ratio"],
                "test_buy_trades": test_extra["buy_trades"],
                "test_buy_win_rate": test_extra["buy_win_rate"],
                "test_sell_trades": test_extra["sell_trades"],
                "test_sell_win_rate": test_extra["sell_win_rate"],
                "best_time_window": test_extra["best_time_window"],
                "train_score": score_strategy(train_combined),
                "test_score": score_strategy(test_combined),
            }
        )

    results = pd.DataFrame(rows)
    results = results.sort_values(["test_score", "test_profit_factor", "test_expectancy", "test_trades"], ascending=[False, False, False, False])
    best = results.iloc[0].to_dict() if not results.empty else {}
    best_strategy = {key: best[key] for key in build_strategy_grid()[0].keys()} if best else {}
    best_trades, best_buy, best_sell, best_combined = evaluate_strategy(candles, best_strategy, lookahead) if best_strategy else (pd.DataFrame(), {}, {}, {})
    return {
        "results": results,
        "best_strategy": best_strategy,
        "best_row": best,
        "best_trades": best_trades,
        "best_buy": best_buy,
        "best_sell": best_sell,
        "best_combined": best_combined,
        "best_summary": summarize_trades(best_trades),
    }


def save_lab_outputs(lab: dict[str, Any], output_dir: str | Path = "reports") -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    results_path = output / "strategy_lab_results.csv"
    trades_path = output / "strategy_lab_trades.csv"
    best_path = output / "strategy_lab_best.json"
    report_path = output / "strategy_lab_report.html"

    results: pd.DataFrame = lab["results"]
    trades: pd.DataFrame = lab["best_trades"]
    results.to_csv(results_path, index=False)
    trades.to_csv(trades_path, index=False)

    best_payload = {
        "important": "Research output only. Do not treat as guaranteed accuracy or live trading advice.",
        "best_strategy": lab["best_strategy"],
        "best_row": lab["best_row"],
        "combined_metrics_all_candles": lab["best_combined"],
        "buy_metrics_all_candles": lab["best_buy"],
        "sell_metrics_all_candles": lab["best_sell"],
        "trade_summary_all_candles": lab["best_summary"],
    }
    best_path.write_text(json.dumps(best_payload, indent=2, default=str), encoding="utf-8")

    top20 = results.head(20)
    report_path.write_text(
        "<html><head><title>Strategy Lab Report</title></head><body>"
        "<h1>XAUUSD M5 Strategy Lab</h1>"
        "<p><strong>Mode:</strong> research/backtest only. No result guarantees future accuracy.</p>"
        "<h2>Recommended Rule Set</h2>"
        f"<pre>{json.dumps(best_payload, indent=2, default=str)}</pre>"
        "<h2>Top 20 Strategies</h2>"
        f"{top20.to_html(index=False)}"
        "<h2>Best Strategy Trades</h2>"
        f"{trades.tail(100).to_html(index=False) if not trades.empty else '<p>No trades found.</p>'}"
        "</body></html>",
        encoding="utf-8",
    )
    return {"results": results_path, "trades": trades_path, "best": best_path, "report": report_path}


def main() -> None:
    parser = argparse.ArgumentParser(description="Research multiple XAUUSD M5 rule strategies and report the best current fit.")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--timeframe", default="M5", choices=["M1", "M5", "M15"])
    parser.add_argument("--bars", type=int, default=5000)
    parser.add_argument("--lookahead", type=int, default=20)
    parser.add_argument("--output-dir", default="reports")
    args = parser.parse_args()

    candles = fetch_live_candles(args.symbol, args.timeframe, args.bars)
    lab = run_strategy_lab(candles, lookahead=args.lookahead)
    paths = save_lab_outputs(lab, args.output_dir)

    print("Best Strategy:")
    print(json.dumps(lab["best_strategy"], indent=2, default=str))
    print("Combined Metrics:")
    print(json.dumps(lab["best_combined"], indent=2, default=str))
    print("Trade Summary:")
    print(json.dumps(lab["best_summary"], indent=2, default=str))
    print("Report:", paths["report"])
    print("Best JSON:", paths["best"])


if __name__ == "__main__":
    main()
