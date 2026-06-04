from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def calculate_performance(trades: pd.DataFrame, equity_curve: pd.DataFrame, initial_balance: float) -> dict:
    if trades.empty:
        return _empty_metrics()
    pnl = trades["pnl"].astype(float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    returns = equity_curve["equity"].pct_change().dropna() if not equity_curve.empty else pd.Series(dtype=float)
    drawdown = equity_curve["drawdown"] if "drawdown" in equity_curve.columns else pd.Series(dtype=float)
    net_profit = float(pnl.sum())
    max_dd = float(drawdown.min()) if not drawdown.empty else 0.0
    gross_profit = float(wins.sum())
    gross_loss = float(abs(losses.sum()))
    return {
        "total_trades": int(len(trades)),
        "winning_trades": int(len(wins)),
        "losing_trades": int(len(losses)),
        "win_rate": float(len(wins) / len(trades) * 100),
        "average_win": float(wins.mean()) if len(wins) else 0.0,
        "average_loss": float(losses.mean()) if len(losses) else 0.0,
        "profit_factor": gross_profit / gross_loss if gross_loss else None,
        "expectancy": float(pnl.mean()),
        "recovery_factor": net_profit / abs(max_dd) if max_dd else None,
        "maximum_drawdown": max_dd,
        "net_profit": net_profit,
        "sharpe_ratio": _sharpe(returns),
        "sortino_ratio": _sortino(returns),
        "calmar_ratio": net_profit / abs(max_dd) if max_dd else None,
        "best_trade": float(pnl.max()),
        "worst_trade": float(pnl.min()),
        "average_trade": float(pnl.mean()),
        "longest_winning_streak": _streak(pnl > 0),
        "longest_losing_streak": _streak(pnl < 0),
    }


def save_metrics(metrics: dict, output_dir: str | Path) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    metrics_path = path / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics_path


def monte_carlo(trades: pd.DataFrame, simulations: int = 1000, initial_balance: float = 10_000.0, seed: int = 42) -> dict:
    if trades.empty:
        return {"expected_returns": 0.0, "worst_case_drawdown": 0.0, "risk_of_ruin": 0.0}
    rng = np.random.default_rng(seed)
    pnl = trades["pnl"].to_numpy(dtype=float)
    finals = []
    drawdowns = []
    ruin_count = 0
    for _ in range(simulations):
        sample = rng.choice(pnl, size=len(pnl), replace=True)
        equity = initial_balance + sample.cumsum()
        peak = np.maximum.accumulate(equity)
        dd = equity - peak
        finals.append(equity[-1] - initial_balance)
        drawdowns.append(dd.min())
        ruin_count += int(equity.min() <= initial_balance * 0.5)
    return {
        "expected_returns": float(np.mean(finals)),
        "worst_case_drawdown": float(np.percentile(drawdowns, 5)),
        "risk_of_ruin": float(ruin_count / simulations * 100),
    }


def walk_forward_splits(df: pd.DataFrame, periods: int = 3) -> list[dict]:
    n = len(df)
    if periods <= 0 or n == 0:
        return []
    chunk = n // (periods + 2)
    splits = []
    for i in range(periods):
        train_end = chunk * (i + 2)
        validate_end = train_end + chunk
        test_end = validate_end + chunk
        splits.append({"train": df.iloc[:train_end], "validate": df.iloc[train_end:validate_end], "test": df.iloc[validate_end:test_end]})
    return splits


def _empty_metrics() -> dict:
    return {k: 0.0 for k in ["total_trades", "winning_trades", "losing_trades", "win_rate", "average_win", "average_loss", "expectancy", "maximum_drawdown", "net_profit", "best_trade", "worst_trade", "average_trade", "longest_winning_streak", "longest_losing_streak"]} | {"profit_factor": None, "recovery_factor": None, "sharpe_ratio": 0.0, "sortino_ratio": 0.0, "calmar_ratio": None}


def _sharpe(returns: pd.Series) -> float:
    return float((returns.mean() / returns.std()) * np.sqrt(252)) if len(returns) and returns.std() else 0.0


def _sortino(returns: pd.Series) -> float:
    downside = returns[returns < 0]
    return float((returns.mean() / downside.std()) * np.sqrt(252)) if len(downside) and downside.std() else 0.0


def _streak(mask: pd.Series) -> int:
    best = current = 0
    for value in mask:
        current = current + 1 if value else 0
        best = max(best, current)
    return int(best)
