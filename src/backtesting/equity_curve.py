from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


def add_return_tables(equity: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if equity.empty or "timestamp" not in equity.columns:
        return pd.DataFrame(), pd.DataFrame()
    data = equity.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    data = data.dropna(subset=["timestamp"]).set_index("timestamp")
    returns = data["equity"].pct_change().fillna(0)
    monthly = returns.resample("ME").sum().rename("monthly_return").reset_index()
    yearly = returns.resample("YE").sum().rename("yearly_return").reset_index()
    return monthly, yearly


def save_visual_reports(trades: pd.DataFrame, equity: pd.DataFrame, metrics: dict, output_dir: str | Path) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    _equity_fig(equity).write_html(path / "equity_curve.html")
    _drawdown_fig(equity).write_html(path / "drawdown_report.html")
    _trade_fig(trades).write_html(path / "trade_analysis.html")
    dashboard = go.Figure()
    dashboard.add_trace(go.Scatter(x=equity.get("timestamp", []), y=equity.get("equity", []), name="Equity"))
    dashboard.add_trace(go.Scatter(x=equity.get("timestamp", []), y=equity.get("balance", []), name="Balance"))
    dashboard.update_layout(title=f"Performance Dashboard | Net Profit: {metrics.get('net_profit', 0):.2f}")
    dashboard.write_html(path / "performance_dashboard.html")


def _equity_fig(equity: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity.get("timestamp", []), y=equity.get("equity", []), name="Equity"))
    fig.add_trace(go.Scatter(x=equity.get("timestamp", []), y=equity.get("balance", []), name="Balance"))
    fig.update_layout(title="Equity and Balance Curve")
    return fig


def _drawdown_fig(equity: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity.get("timestamp", []), y=equity.get("drawdown", []), name="Drawdown", fill="tozeroy"))
    fig.update_layout(title="Drawdown Report")
    return fig


def _trade_fig(trades: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=trades.get("entry_time", []), y=trades.get("pnl", []), name="Trade PnL"))
    fig.update_layout(title="Trade Analysis")
    return fig
