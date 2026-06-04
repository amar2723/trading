from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from live_data import fetch_live_candles
from pattern_detector import detect_pattern
from signal_logger import read_signal_history


st.set_page_config(page_title="XAUUSD Phase 1 Scanner", layout="wide")
st.title("XAUUSD Phase 1 Live Scanner")

symbol = st.sidebar.text_input("Symbol", "XAUUSD")
timeframe = st.sidebar.selectbox("Timeframe", ["M5", "M1", "M15"], index=0)
bars = st.sidebar.number_input("Candles", min_value=20, max_value=500, value=100, step=10)

try:
    candles = fetch_live_candles(symbol, timeframe, int(bars))
    signal = detect_pattern(candles)
    current_price = candles.iloc[-1]["close"] if not candles.empty else None

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Current Price", "-" if current_price is None else f"{current_price:.2f}")
    c2.metric("Current Signal", signal.signal)
    c3.metric("Entry", "-" if signal.entry is None else f"{signal.entry:.2f}")
    c4.metric("Stop Loss", "-" if signal.sl is None else f"{signal.sl:.2f}")
    c5.metric("TP1", "-" if signal.tp1 is None else f"{signal.tp1:.2f}")
    c6.metric("Confidence", f"{signal.confidence:.0f}%")

    c7, c8, c9 = st.columns(3)
    c7.metric("TP2 / Liquidity", "-" if signal.tp2 is None else f"{signal.tp2:.2f}")
    c8.metric("Liquidity Target", "-" if signal.liquidity_target is None else f"{signal.liquidity_target:.2f}")
    c9.metric("Trailing Stop Ref", "-" if signal.trailing_stop is None else f"{signal.trailing_stop:.2f}")

    fig = go.Figure(
        data=[
            go.Candlestick(
                x=candles["timestamp"],
                open=candles["open"],
                high=candles["high"],
                low=candles["low"],
                close=candles["close"],
                name=symbol,
            )
        ]
    )
    fig.update_layout(height=600, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Last 100 Candles")
    st.dataframe(candles.tail(100), use_container_width=True, height=320)

    st.subheader("Signal History")
    history = read_signal_history()
    st.dataframe(history.tail(250), use_container_width=True, height=320)
except Exception as exc:
    st.error(str(exc))
