from __future__ import annotations

import logging
from pathlib import Path

import joblib
import pandas as pd

from src.backtesting.backtester import Backtester, ExecutionConfig
from src.backtesting.equity_curve import add_return_tables, save_visual_reports
from src.backtesting.performance import calculate_performance, monte_carlo, save_metrics, walk_forward_splits
from src.backtesting.risk_manager import RiskConfig
from src.training.train_pipeline import predict_probabilities


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def run_backtest_pipeline(
    data_path: str | Path,
    model_path: str | Path | None = None,
    output_dir: str | Path = "reports/backtest",
    execution: ExecutionConfig | None = None,
    risk: RiskConfig | None = None,
) -> dict:
    data = pd.read_csv(data_path)
    if model_path:
        bundle = joblib.load(model_path)
        probabilities = predict_probabilities(bundle, data)
        data = pd.concat([data.reset_index(drop=True), probabilities], axis=1)
        data["predicted_signal"] = probabilities[["buy_probability", "sell_probability", "hold_probability"]].idxmax(axis=1).str.replace("_probability", "").str.upper()

    backtester = Backtester(execution=execution, risk=risk)
    trades, equity = backtester.run(data)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    trades.to_csv(output / "trade_log.csv", index=False)
    equity.to_csv(output / "equity_curve.csv", index=False)

    metrics = calculate_performance(trades, equity, backtester.risk_config.initial_balance)
    metrics["monte_carlo"] = monte_carlo(trades, initial_balance=backtester.risk_config.initial_balance)
    metrics["walk_forward_periods"] = len(walk_forward_splits(data))
    save_metrics(metrics, output)

    monthly, yearly = add_return_tables(equity)
    monthly.to_csv(output / "monthly_returns.csv", index=False)
    yearly.to_csv(output / "yearly_returns.csv", index=False)
    save_visual_reports(trades, equity, metrics, output)
    (output / "performance_report.html").write_text(_performance_html(metrics), encoding="utf-8")
    return {"output_dir": str(output), "metrics": metrics, "trades": len(trades)}


def _performance_html(metrics: dict) -> str:
    rows = "".join(f"<tr><th>{key}</th><td>{value}</td></tr>" for key, value in metrics.items())
    return f"<html><body><h1>Backtest Performance Report</h1><table>{rows}</table></body></html>"
