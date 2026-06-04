from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def candlestick_chart(df: pd.DataFrame) -> go.Figure:
    time_col = "timestamp" if "timestamp" in df.columns else "time"
    fig = go.Figure()
    if df.empty or time_col not in df.columns:
        fig.update_layout(title="Candlestick Chart")
        return fig
    fig.add_trace(
        go.Candlestick(
            x=df[time_col],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="XAUUSD",
        )
    )
    _markers(fig, df, time_col, "bullish_liquidity_sweep", "low", "Bullish Sweep")
    _markers(fig, df, time_col, "bearish_liquidity_sweep", "high", "Bearish Sweep")
    _markers(fig, df, time_col, "bullish_mss", "close", "Bullish MSS")
    _markers(fig, df, time_col, "bearish_mss", "close", "Bearish MSS")
    _markers(fig, df, time_col, "bullish_bos", "close", "Bullish BOS")
    _markers(fig, df, time_col, "bearish_bos", "close", "Bearish BOS")
    _markers(fig, df, time_col, "bullish_ob", "ob_low", "Bullish OB")
    _markers(fig, df, time_col, "bearish_ob", "ob_high", "Bearish OB")
    fig.update_layout(height=650, xaxis_rangeslider_visible=False, title="XAUUSD Candles and Smart Money Concepts")
    return fig


def equity_chart(equity: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if equity.empty:
        fig.update_layout(title="Equity Curve")
        return fig
    fig.add_trace(go.Scatter(x=equity.get("timestamp"), y=equity.get("equity"), name="Equity"))
    fig.add_trace(go.Scatter(x=equity.get("timestamp"), y=equity.get("balance"), name="Balance"))
    fig.update_layout(height=420, title="Equity and Balance")
    return fig


def drawdown_chart(equity: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if equity.empty:
        fig.update_layout(title="Drawdown")
        return fig
    fig.add_trace(go.Scatter(x=equity.get("timestamp"), y=equity.get("drawdown"), fill="tozeroy", name="Drawdown"))
    fig.update_layout(height=320, title="Drawdown")
    return fig


def feature_importance_chart(importance: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if importance.empty or not {"feature", "importance"}.issubset(importance.columns):
        fig.update_layout(title="Feature Importance")
        return fig
    top = importance.head(25).iloc[::-1]
    fig.add_trace(go.Bar(x=top["importance"], y=top["feature"], orientation="h", name="Importance"))
    fig.update_layout(height=600, title="Top Feature Importance")
    return fig


def trade_pnl_chart(trades: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if trades.empty:
        fig.update_layout(title="Trade PnL")
        return fig
    fig.add_trace(go.Bar(x=trades.get("entry_time"), y=trades.get("pnl"), name="PnL"))
    fig.update_layout(height=360, title="Trade PnL")
    return fig


def _markers(fig: go.Figure, df: pd.DataFrame, time_col: str, flag_col: str, price_col: str, name: str) -> None:
    if flag_col not in df.columns or price_col not in df.columns:
        return
    points = df[df[flag_col].fillna(0).astype(bool)]
    if points.empty:
        return
    fig.add_trace(go.Scatter(x=points[time_col], y=points[price_col], mode="markers", name=name))
