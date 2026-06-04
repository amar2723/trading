from __future__ import annotations

import pandas as pd
import streamlit as st

from src.dashboard.charts import candlestick_chart, drawdown_chart, equity_chart, feature_importance_chart, trade_pnl_chart


def signal_cards(signal: dict) -> None:
    cols = st.columns(6)
    cols[0].metric("Signal", signal.get("signal", "HOLD"))
    cols[1].metric("Confidence", _fmt(signal.get("confidence"), "%"))
    cols[2].metric("Entry", _fmt(signal.get("entry")))
    cols[3].metric("SL", _fmt(signal.get("sl")))
    cols[4].metric("TP1", _fmt(signal.get("tp1")))
    cols[5].metric("TP2", _fmt(signal.get("tp2")))


def backtest_summary(metrics: dict) -> None:
    cols = st.columns(5)
    cols[0].metric("Net Profit", _fmt(metrics.get("net_profit")))
    cols[1].metric("Win Rate", _fmt(metrics.get("win_rate"), "%"))
    cols[2].metric("Profit Factor", _fmt(metrics.get("profit_factor")))
    cols[3].metric("Max Drawdown", _fmt(metrics.get("maximum_drawdown")))
    cols[4].metric("Trades", metrics.get("total_trades", 0))


def render_market(df: pd.DataFrame) -> None:
    st.plotly_chart(candlestick_chart(df), use_container_width=True)
    if not df.empty:
        st.dataframe(df.tail(100), use_container_width=True, height=320)


def render_backtest(trades: pd.DataFrame, equity: pd.DataFrame, metrics: dict) -> None:
    backtest_summary(metrics)
    st.plotly_chart(equity_chart(equity), use_container_width=True)
    st.plotly_chart(drawdown_chart(equity), use_container_width=True)
    st.plotly_chart(trade_pnl_chart(trades), use_container_width=True)
    if not trades.empty:
        st.dataframe(trades, use_container_width=True, height=320)


def render_model_report(metrics: dict, importance: pd.DataFrame) -> None:
    st.json(metrics)
    st.plotly_chart(feature_importance_chart(importance), use_container_width=True)
    if not importance.empty:
        st.dataframe(importance.head(50), use_container_width=True, height=320)


def _fmt(value, suffix: str = "") -> str:
    if value is None or value == "":
        return "-"
    try:
        return f"{float(value):.2f}{suffix}"
    except Exception:
        return str(value)
