from __future__ import annotations

import streamlit as st

from live_data import fetch_live_candles
from prediction import PredictionPipeline


st.set_page_config(page_title="XAUUSD Prediction Engine", layout="wide")
st.title("XAUUSD Liquidity Prediction Engine")

symbol = st.sidebar.text_input("Symbol", "XAUUSD")
timeframe = st.sidebar.selectbox("Timeframe", ["M5", "M1", "M15"], index=0)
bars = st.sidebar.number_input("Candles", min_value=200, max_value=2000, value=500, step=100)
min_confidence = st.sidebar.slider("Minimum Confidence", 0, 100, 70)

try:
    candles = fetch_live_candles(symbol, timeframe, int(bars))
    prediction = PredictionPipeline(float(min_confidence)).predict(candles)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Prediction", prediction["signal"])
    c2.metric("Confidence", f"{prediction['confidence']:.0f}%")
    c3.metric("Trend", prediction.get("trend_direction"))
    c4.metric("RR", "-" if prediction.get("rr_ratio") is None else f"{prediction['rr_ratio']:.2f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Entry", "-" if prediction.get("entry") is None else f"{prediction['entry']:.2f}")
    c6.metric("SL", "-" if prediction.get("stop_loss") is None else f"{prediction['stop_loss']:.2f}")
    c7.metric("TP1", "-" if prediction.get("tp1") is None else f"{prediction['tp1']:.2f}")
    c8.metric("TP2", "-" if prediction.get("tp2") is None else f"{prediction['tp2']:.2f}")

    c9, c10 = st.columns(2)
    c9.metric("TP3", "-" if prediction.get("tp3") is None else f"{prediction['tp3']:.2f}")
    c10.metric("Nearest Liquidity", f"Buy: {prediction.get('nearest_buy_liquidity')} | Sell: {prediction.get('nearest_sell_liquidity')}")

    st.subheader("Reasoning")
    st.write(prediction.get("reason"))
    st.subheader("Raw Prediction")
    st.json(prediction)
except Exception as exc:
    st.error(str(exc))
