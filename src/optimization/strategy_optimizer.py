from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from src.backtesting.backtest_pipeline import run_backtest_pipeline
from src.backtesting.backtester import ExecutionConfig
from src.backtesting.risk_manager import RiskConfig


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OptimizationWeights:
    profit_factor: float = 1.5
    expectancy: float = 1.0
    sharpe_ratio: float = 1.2
    maximum_drawdown: float = 1.4
    risk_of_ruin: float = 2.0


def objective_score(metrics: dict, weights: OptimizationWeights | None = None) -> float:
    """Higher is better: rewards PF/expectancy/Sharpe and penalizes DD/ruin."""
    w = weights or OptimizationWeights()
    profit_factor = metrics.get("profit_factor") or 0.0
    expectancy = metrics.get("expectancy") or 0.0
    sharpe = metrics.get("sharpe_ratio") or 0.0
    max_drawdown = abs(metrics.get("maximum_drawdown") or 0.0)
    risk_of_ruin = (metrics.get("monte_carlo") or {}).get("risk_of_ruin", 0.0) or 0.0

    return (
        w.profit_factor * min(float(profit_factor), 10.0)
        + w.expectancy * float(expectancy)
        + w.sharpe_ratio * float(sharpe)
        - w.maximum_drawdown * (max_drawdown / 1000.0)
        - w.risk_of_ruin * (float(risk_of_ruin) / 10.0)
    )


class StrategyOptimizer:
    def __init__(
        self,
        data_path: str | Path,
        model_path: str | Path | None = None,
        output_dir: str | Path = "reports/optimization",
        weights: OptimizationWeights | None = None,
    ):
        self.data_path = Path(data_path)
        self.model_path = Path(model_path) if model_path else None
        self.output_dir = Path(output_dir)
        self.weights = weights or OptimizationWeights()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def optimize(self, trials: int = 50) -> dict:
        try:
            import optuna
        except ImportError as exc:
            raise RuntimeError("Install optuna to run strategy optimization.") from exc

        def objective(trial) -> float:
            params = self._suggest(trial)
            result = self._run_trial(params, trial.number)
            score = objective_score(result["metrics"], self.weights)
            trial.set_user_attr("metrics", result["metrics"])
            trial.set_user_attr("params", params)
            return score

        study = optuna.create_study(direction="maximize", study_name="xauusd-risk-adjusted-optimization")
        study.optimize(objective, n_trials=trials, show_progress_bar=False)
        best = {
            "best_score": study.best_value,
            "best_params": study.best_trial.user_attrs["params"],
            "best_metrics": study.best_trial.user_attrs["metrics"],
            "weights": asdict(self.weights),
        }
        (self.output_dir / "best_optimization.json").write_text(json.dumps(best, indent=2, default=str), encoding="utf-8")
        pd.DataFrame([self._trial_row(trial) for trial in study.trials]).to_csv(self.output_dir / "optimization_trials.csv", index=False)
        return best

    def _suggest(self, trial) -> dict:
        return {
            "spread_points": trial.suggest_float("spread_points", 10.0, 45.0),
            "slippage_points": trial.suggest_float("slippage_points", 1.0, 15.0),
            "commission_per_lot": trial.suggest_float("commission_per_lot", 3.0, 12.0),
            "execution_delay": trial.suggest_int("execution_delay", 0, 2),
            "tp1_close_fraction": trial.suggest_float("tp1_close_fraction", 0.25, 0.75),
            "percent_risk": trial.suggest_float("percent_risk", 0.25, 2.0),
            "max_daily_loss": trial.suggest_float("max_daily_loss", 100.0, 600.0),
            "max_consecutive_losses": trial.suggest_int("max_consecutive_losses", 2, 6),
            "max_drawdown_percent": trial.suggest_float("max_drawdown_percent", 5.0, 25.0),
        }

    def _run_trial(self, params: dict, trial_number: int) -> dict:
        trial_dir = self.output_dir / f"trial_{trial_number:04d}"
        execution = ExecutionConfig(
            spread_points=params["spread_points"],
            commission_per_lot=params["commission_per_lot"],
            slippage_points=params["slippage_points"],
            execution_delay=params["execution_delay"],
            tp1_close_fraction=params["tp1_close_fraction"],
        )
        risk = RiskConfig(
            sizing_mode="percent_risk",
            percent_risk=params["percent_risk"],
            max_daily_loss=params["max_daily_loss"],
            max_consecutive_losses=params["max_consecutive_losses"],
            max_drawdown_percent=params["max_drawdown_percent"],
        )
        return run_backtest_pipeline(self.data_path, self.model_path, trial_dir, execution, risk)

    @staticmethod
    def _trial_row(trial) -> dict:
        metrics = trial.user_attrs.get("metrics", {})
        mc = metrics.get("monte_carlo") or {}
        return {
            "number": trial.number,
            "score": trial.value,
            "profit_factor": metrics.get("profit_factor"),
            "expectancy": metrics.get("expectancy"),
            "sharpe_ratio": metrics.get("sharpe_ratio"),
            "maximum_drawdown": metrics.get("maximum_drawdown"),
            "risk_of_ruin": mc.get("risk_of_ruin"),
            **trial.user_attrs.get("params", {}),
        }
