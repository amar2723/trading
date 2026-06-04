from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


def classification_metrics(y_true: pd.Series, probabilities: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (probabilities >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)) if pd.Series(y_true).nunique() > 1 else 0.5,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
    }


def trading_metrics(df: pd.DataFrame) -> dict:
    trades = df[df.get("entry_type", "HOLD").isin(["BUY", "SELL"])] if "entry_type" in df.columns else df
    if trades.empty:
        return {"win_rate": 0.0, "profit_factor": None, "expectancy": 0.0, "average_rr": 0.0, "max_drawdown": 0.0, "net_profit": 0.0}

    pnl = _profit_points(trades)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    equity = pnl.cumsum()
    drawdown = equity - equity.cummax()
    gross_profit = float(wins.sum())
    gross_loss = float(abs(losses.sum()))
    return {
        "win_rate": float(len(wins) / len(trades) * 100),
        "profit_factor": gross_profit / gross_loss if gross_loss else None,
        "expectancy": float(pnl.mean()),
        "average_rr": float(trades["risk_reward_ratio"].mean()) if "risk_reward_ratio" in trades.columns else 0.0,
        "max_drawdown": float(drawdown.min()) if not drawdown.empty else 0.0,
        "net_profit": float(pnl.sum()),
    }


def save_metrics(metrics: dict, report_dir: str | Path) -> Path:
    path = Path(report_dir)
    path.mkdir(parents=True, exist_ok=True)
    metrics_path = path / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics_path


def write_html_report(title: str, payload: dict, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, indent=2)
    path.write_text(f"<html><body><h1>{title}</h1><pre>{body}</pre></body></html>", encoding="utf-8")
    return path


def _profit_points(trades: pd.DataFrame) -> pd.Series:
    if "exit_price" not in trades.columns or "entry_price" not in trades.columns:
        return trades.get("profitable_trade", pd.Series(0, index=trades.index)).astype(float)
    buy = trades.get("entry_type", "").eq("BUY")
    sell = trades.get("entry_type", "").eq("SELL")
    exit_price = trades["exit_price"].fillna(trades["entry_price"])
    return pd.Series(np.select([buy, sell], [exit_price - trades["entry_price"], trades["entry_price"] - exit_price], default=0.0), index=trades.index)
