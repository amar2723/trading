from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from advanced_smc.engine import AdvancedSMCEngine
from image_trade_analyzer import analyze_chart_image
from live_data import fetch_live_candles, fetch_live_tick
from pattern_detector import detect_pattern
from signal_logger import log_image_trade_plan, read_signal_history


def _price(value) -> str:
    try:
        if value is None or pd.isna(value):
            return "-"
        return f"{float(value):.2f}"
    except Exception:
        return "-"


def _risk_reward(action: str, entry, stop_loss, tp1) -> float | None:
    try:
        entry_f = float(entry)
        sl_f = float(stop_loss)
        tp_f = float(tp1)
    except Exception:
        return None
    risk = abs(entry_f - sl_f)
    reward = abs(tp_f - entry_f)
    if risk <= 0:
        return None
    return reward / risk


def _load_strategy() -> dict:
    path = Path("reports/outcome_strategy.json")
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _investment_decision(action: str, confidence: float, strategy: dict) -> tuple[str, str]:
    if action not in {"BUY", "SELL"}:
        return "WAIT / NO TRADE", "No high-probability setup is active right now."
    if confidence < 70:
        return "WAIT / NO TRADE", "Confidence is below 70%, so the setup is filtered out."
    allowed = strategy.get("allowed_signals") if strategy else None
    if allowed and action not in allowed:
        return "WAIT / NO TRADE", f"Outcome strategy currently allows {allowed}, but engine found {action}."
    return f"PAPER {action} IDEA", "Demo only: use tiny risk, wait for candle close, and never treat this as guaranteed."


def _image_decision(action: str, confidence: float, strategy: dict, min_confidence: float) -> tuple[str, str]:
    if action not in {"BUY", "SELL"}:
        return "WAIT / NO TRADE", "Image and live data do not show a clear trade plan."
    if confidence < min_confidence:
        return "WAIT / NO TRADE", f"Image confidence is below {min_confidence:.0f}%."
    allowed = strategy.get("allowed_signals") if strategy else None
    if allowed and action not in allowed:
        return "WAIT / NO TRADE", f"Outcome data currently favors {allowed}, but the image suggests {action}."
    return f"PAPER {action} SETUP", "Live price levels are shown below. Demo only; confirm on MT5 before any action."


st.set_page_config(page_title="XAUUSD Phase 1 Scanner", layout="wide")
st.title("XAUUSD Phase 1 Live Scanner")

symbol = st.sidebar.text_input("Symbol", "XAUUSD")
timeframe = st.sidebar.selectbox("Timeframe", ["M5", "M1", "M15"], index=0)
bars = st.sidebar.number_input("Candles", min_value=20, max_value=500, value=100, step=10)
show_trade_idea = st.sidebar.checkbox("Show Trade Idea", value=True)
idea_engine = st.sidebar.selectbox("Idea Engine", ["Advanced Liquidity", "Basic Sweep"], index=0)
min_confidence = st.sidebar.slider("Minimum Confidence", 0, 100, 70)
show_image_analysis = st.sidebar.checkbox("Upload Chart Image", value=False)

try:
    candles = fetch_live_candles(symbol, timeframe, int(bars))
    try:
        live_tick = fetch_live_tick(symbol)
    except Exception:
        live_tick = {}
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
    if live_tick:
        t1, t2, t3 = st.columns(3)
        t1.metric("Live Bid", _price(live_tick.get("bid")))
        t2.metric("Live Ask", _price(live_tick.get("ask")))
        t3.metric("Tick Time", str(live_tick.get("time", "-")))

    if show_trade_idea:
        st.subheader("Trade Idea / Where To Place Trade")
        st.caption("Paper-trading research only. This is not guaranteed and not financial advice.")
        if idea_engine == "Advanced Liquidity":
            idea = AdvancedSMCEngine(symbol, float(min_confidence)).predict(candles).to_dict()
            reasons = idea.get("reason") or []
            action = idea.get("signal")
            entry = idea.get("entry")
            stop_loss = idea.get("stop_loss")
            tp1 = idea.get("tp1")
            tp2 = idea.get("tp2")
            tp3 = idea.get("tp3")
            rr = idea.get("risk_reward")
            confidence = float(idea.get("confidence") or 0)
        else:
            idea = signal.to_dict()
            action = "NO TRADE" if idea.get("signal") == "NONE" else idea.get("signal")
            entry = idea.get("entry")
            stop_loss = idea.get("sl")
            tp1 = idea.get("tp1")
            tp2 = idea.get("tp2")
            tp3 = None
            rr = _risk_reward(action, entry, stop_loss, tp1)
            confidence = float(idea.get("confidence") or 0)
            reasons = [item.strip() for item in str(idea.get("reason") or "Confidence below threshold").split(",") if item.strip()]

        strategy = _load_strategy()
        decision, decision_note = _investment_decision(action, confidence, strategy)
        box = st.container(border=True)
        with box:
            st.markdown(f"**Suggestion:** `{decision}`")
            st.write(decision_note)
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("Signal", action)
            m2.metric("Confidence", f"{confidence:.0f}%")
            m3.metric("Entry", _price(entry))
            m4.metric("Stop Loss", _price(stop_loss))
            m5.metric("TP1", _price(tp1))
            m6.metric("RR to TP1", "-" if rr is None else f"{rr:.2f}")
            m7, m8 = st.columns(2)
            m7.metric("TP2", _price(tp2))
            m8.metric("TP3", _price(tp3))
            st.write("Reason:", ", ".join(reasons) if reasons else "No strong setup.")
            if strategy:
                evidence = strategy.get("evidence", {})
                st.caption(
                    "Outcome filter: "
                    f"allowed={strategy.get('allowed_signals')} | "
                    f"session={strategy.get('session')} | "
                    f"hours={strategy.get('allowed_hours')} | "
                    f"PF={evidence.get('all', {}).get('profit_factor')}"
                )

    if show_image_analysis:
        st.subheader("Image Trade Analysis")
        st.caption("Upload a chart screenshot. The app will estimate visual bias and combine it with live MT5 price levels.")
        uploaded = st.file_uploader("Upload chart image", type=["png", "jpg", "jpeg", "webp"])
        if uploaded is not None:
            st.image(uploaded, caption="Uploaded chart", use_container_width=True)
            uploaded.seek(0)
            image_analysis = analyze_chart_image(uploaded, candles, live_tick).to_dict()
            image_analysis["live_time"] = str(live_tick.get("time", candles.iloc[-1].get("timestamp", "")) if live_tick else candles.iloc[-1].get("timestamp", ""))
            strategy = _load_strategy()
            decision, note = _image_decision(image_analysis["signal"], float(image_analysis["confidence"]), strategy, float(min_confidence))
            log_image_trade_plan(image_analysis, symbol, decision)
            st.markdown(f"**Live Image Suggestion:** `{decision}`")
            st.write(note)
            i1, i2, i3, i4, i5, i6 = st.columns(6)
            i1.metric("Image Signal", image_analysis["signal"])
            i2.metric("Confidence", f"{float(image_analysis['confidence']):.0f}%")
            i3.metric("Bias", image_analysis["bias"])
            i4.metric("Trend", image_analysis["trend"])
            i5.metric("Recent Pressure", image_analysis["recent_pressure"])
            i6.metric("RR", "-" if image_analysis["risk_reward"] is None else f"{float(image_analysis['risk_reward']):.2f}")
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Entry", _price(image_analysis["entry"]))
            p2.metric("Stop Loss", _price(image_analysis["stop_loss"]))
            p3.metric("TP1", _price(image_analysis["tp1"]))
            p4.metric("TP2", _price(image_analysis["tp2"]))
            st.metric("TP3", _price(image_analysis["tp3"]))
            st.caption(f"Entry source: {image_analysis.get('live_price_source')} | Live time: {image_analysis.get('live_time')}")
            st.write("Reason:", ", ".join(image_analysis["reason"]))
            st.warning(image_analysis["warning"])
            with st.expander("Raw image analysis"):
                st.json(image_analysis)

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
    st.info(
        "If this is running on Streamlit Community Cloud, live MT5 data will not work there because "
        "MetaTrader 5 requires a Windows terminal session. Use the local mobile URL or a Windows VPS "
        "for live trading data."
    )
    st.subheader("Deployment Mode")
    st.write("Cloud app is loaded, but live MT5 connection is unavailable in this environment.")
    st.write("For real-time data: keep MT5 open on your Windows PC/VPS and run the dashboard there.")
