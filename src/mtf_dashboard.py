from __future__ import annotations

import streamlit as st

from live_data import fetch_live_candles
from multi_timeframe import analyze_h1_trend, confirm_mtf_entry, validate_m15_structure


st.set_page_config(page_title="XAUUSD MTF Engine", layout="wide")
st.title("XAUUSD Multi-Timeframe Prediction Engine")

symbol = st.sidebar.text_input("Symbol", "XAUUSD")
min_confidence = st.sidebar.slider("Minimum Confidence", 0, 100, 75)

try:
    h1 = fetch_live_candles(symbol, "H1", 500)
    m15 = fetch_live_candles(symbol, "M15", 500)
    m5 = fetch_live_candles(symbol, "M5", 500)
    trend = analyze_h1_trend(h1)
    structure = validate_m15_structure(m15)
    signal = confirm_mtf_entry(trend, structure, m5, float(min_confidence))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("H1 Trend", trend.trend_direction)
    c2.metric("Trend Strength", f"{trend.trend_strength:.2f}")
    c3.metric("M15 Bullish", str(structure.bullish_structure_valid))
    c4.metric("M15 Bearish", str(structure.bearish_structure_valid))

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Current Signal", signal.signal)
    c6.metric("Confidence", f"{signal.confidence:.0f}%")
    c7.metric("Entry", "-" if signal.entry is None else f"{signal.entry:.2f}")
    c8.metric("SL", "-" if signal.stop_loss is None else f"{signal.stop_loss:.2f}")

    c9, c10, c11 = st.columns(3)
    c9.metric("TP1", "-" if signal.tp1 is None else f"{signal.tp1:.2f}")
    c10.metric("TP2", "-" if signal.tp2 is None else f"{signal.tp2:.2f}")
    c11.metric("TP3", "-" if signal.tp3 is None else f"{signal.tp3:.2f}")

    st.subheader("Reasons")
    st.write(signal.reasons)
    st.subheader("Raw Signal")
    st.json(signal.to_dict())
except Exception as exc:
    st.error(str(exc))
