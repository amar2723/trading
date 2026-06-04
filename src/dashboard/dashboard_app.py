from __future__ import annotations

import streamlit as st

from src.dashboard.components import render_backtest, render_market, render_model_report, signal_cards
from src.dashboard.data_loader import available_files, latest_signal, load_csv, load_json


def run_dashboard() -> None:
    st.set_page_config(page_title="XAUUSD Trading AI", layout="wide")
    st.title("XAUUSD Trading AI")

    with st.sidebar:
        market_file = st.selectbox("Market / Pattern Data", _choices("data/patterns", "data/processed", "data/raw"))
        signal_log = st.text_input("Signal Log", "logs/signals.csv")
        backtest_dir = st.text_input("Backtest Reports", "reports/backtest")
        reports_dir = st.text_input("Model Reports", "reports")
        refresh = st.toggle("Auto Refresh", value=False)

    if refresh:
        st.rerun()

    signal = latest_signal(signal_log)
    signal_cards(signal)

    tabs = st.tabs(["Market", "Live Signals", "Backtest", "Model Reports"])

    with tabs[0]:
        render_market(load_csv(market_file) if market_file else load_csv("data/raw/XAUUSD_M5.csv"))

    with tabs[1]:
        signals = load_csv(signal_log)
        if signals.empty:
            st.info("No live signals logged yet.")
        else:
            st.dataframe(signals.tail(250), use_container_width=True, height=520)

    with tabs[2]:
        trades = load_csv(f"{backtest_dir}/trade_log.csv")
        equity = load_csv(f"{backtest_dir}/equity_curve.csv")
        metrics = load_json(f"{backtest_dir}/metrics.json")
        render_backtest(trades, equity, metrics)

    with tabs[3]:
        metrics = load_json(f"{reports_dir}/metrics.json")
        importance = load_csv(f"{reports_dir}/feature_importance.csv")
        render_model_report(metrics, importance)


def _choices(*dirs: str) -> list[str]:
    files: list[str] = []
    for directory in dirs:
        files.extend(available_files(directory))
    return files or [""]


if __name__ == "__main__":
    run_dashboard()
