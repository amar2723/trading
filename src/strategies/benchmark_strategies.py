from __future__ import annotations

"""
Deterministic Strategy Modules for XAUUSD (Gold).
Contains the top-ranked systematic strategy families from empirical research:
1. H1 Trend + M5 Pullback (Trend-Following Momentum)
2. Donchian Breakout (Multi-bar Volatility Channel)
3. Volatility Squeeze (Bollinger Band contraction inside Keltner Channel)
4. Liquidity Sweep & Reclaim (Session Key-Level Manipulation)
"""

from dataclasses import dataclass
from typing import Literal
import numpy as np
import pandas as pd


@dataclass
class CandidateSignal:
    strategy_name: str
    action: Literal["BUY", "SELL", "HOLD"]
    entry: float
    sl: float
    tp1: float
    tp2: float
    be_trigger: float
    risk_reward: str
    risk_pips: float
    confidence: float
    rationale: str
    bar_time: str


class H1TrendM5PullbackStrategy:
    """Strategy A: H1 Macro Trend + M5 Mean-Reversion Pullback."""
    name = "H1_TREND_M5_PULLBACK"

    @classmethod
    def evaluate(cls, df_m5: pd.DataFrame, df_h1: pd.DataFrame | None = None) -> CandidateSignal:
        close = df_m5["close"].values
        high = df_m5["high"].values
        low = df_m5["low"].values
        current_price = float(close[-1])
        bar_time = str(df_m5["timestamp"].iloc[-1])

        # Causal H1 Trend
        h1_trend_bull = True
        h1_trend_bear = False
        if df_h1 is not None and len(df_h1) >= 20:
            h1_closed = df_h1.iloc[:-1] # Completed H1 bars only
            h1_c = h1_closed["close"].values
            h1_e50 = float(pd.Series(h1_c).ewm(span=50).mean().iloc[-1])
            h1_e200 = float(pd.Series(h1_c).ewm(span=200).mean().iloc[-1])
            h1_trend_bull = h1_c[-1] > h1_e50 > h1_e200
            h1_trend_bear = h1_c[-1] < h1_e50 < h1_e200

        # M5 Pullback indicators
        m5_e20 = float(pd.Series(close).ewm(span=20).mean().iloc[-1])
        m5_e50 = float(pd.Series(close).ewm(span=50).mean().iloc[-1])

        diffs = pd.Series(close).diff()
        gain = diffs.clip(lower=0).rolling(14).mean()
        loss = (-diffs.clip(upper=0)).rolling(14).mean()
        rsi = float((100 - (100 / (1 + (gain / (loss + 1e-9))))).iloc[-1])

        # ATR
        tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
        atr = float(pd.Series(tr).rolling(14).mean().iloc[-1])
        risk_dist = round(max(atr * 1.25, 3.20), 2)

        # Bullish Pullback: H1 is bull, M5 pulls back near EMA 50, RSI is not overbought (42 - 62)
        if h1_trend_bull and current_price >= m5_e50 - (risk_dist * 0.5) and current_price <= m5_e20 + 2.0 and rsi < 65.0:
            entry = round(current_price, 2)
            sl = round(entry - risk_dist, 2)
            tp1 = round(entry + (risk_dist * 1.5), 2)
            tp2 = round(entry + (risk_dist * 2.8), 2)
            be_trig = round(entry + 2.00, 2)
            return CandidateSignal(
                strategy_name=cls.name,
                action="BUY",
                entry=entry,
                sl=sl,
                tp1=tp1,
                tp2=tp2,
                be_trigger=be_trig,
                risk_reward="1 : 1.5",
                risk_pips=risk_dist * 10,
                confidence=85.0,
                rationale=f"H1 Bull Trend confirmed. M5 Pullback to EMA50 support with RSI {rsi:.1f} Safe.",
                bar_time=bar_time,
            )

        # Bearish Pullback: H1 is bear, M5 pulls back up near EMA 50, RSI is not oversold (38 - 58)
        if h1_trend_bear and current_price <= m5_e50 + (risk_dist * 0.5) and current_price >= m5_e20 - 2.0 and rsi > 35.0:
            entry = round(current_price, 2)
            sl = round(entry + risk_dist, 2)
            tp1 = round(entry - (risk_dist * 1.5), 2)
            tp2 = round(entry - (risk_dist * 2.8), 2)
            be_trig = round(entry - 2.00, 2)
            return CandidateSignal(
                strategy_name=cls.name,
                action="SELL",
                entry=entry,
                sl=sl,
                tp1=tp1,
                tp2=tp2,
                be_trigger=be_trig,
                risk_reward="1 : 1.5",
                risk_pips=risk_dist * 10,
                confidence=85.0,
                rationale=f"H1 Bear Trend confirmed. M5 Pullback to EMA50 resistance with RSI {rsi:.1f} Safe.",
                bar_time=bar_time,
            )

        return CandidateSignal(
            strategy_name=cls.name, action="HOLD", entry=current_price, sl=current_price,
            tp1=current_price, tp2=current_price, be_trigger=current_price,
            risk_reward="1:1", risk_pips=0.0, confidence=50.0,
            rationale="No clean H1 pullback alignment.", bar_time=bar_time
        )


class DonchianBreakoutStrategy:
    """Strategy B: 20-period Donchian Breakout with ATR filter."""
    name = "DONCHIAN_BREAKOUT"

    @classmethod
    def evaluate(cls, df_m5: pd.DataFrame, df_h1: pd.DataFrame | None = None) -> CandidateSignal:
        close = df_m5["close"].values
        high = df_m5["high"].values
        low = df_m5["low"].values
        current_price = float(close[-1])
        bar_time = str(df_m5["timestamp"].iloc[-1])

        # 20-period Donchian Channels (shifted by 1 to prevent lookahead)
        donch_high = float(pd.Series(high).rolling(20).max().shift(1).iloc[-1])
        donch_low = float(pd.Series(low).rolling(20).min().shift(1).iloc[-1])

        tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
        atr = float(pd.Series(tr).rolling(14).mean().iloc[-1])
        risk_dist = round(max(atr * 1.25, 3.20), 2)

        if current_price > donch_high:
            entry = round(current_price, 2)
            sl = round(entry - risk_dist, 2)
            tp1 = round(entry + (risk_dist * 1.5), 2)
            tp2 = round(entry + (risk_dist * 2.8), 2)
            return CandidateSignal(
                strategy_name=cls.name, action="BUY", entry=entry, sl=sl, tp1=tp1, tp2=tp2,
                be_trigger=round(entry + 2.0, 2), risk_reward="1 : 1.5", risk_pips=risk_dist * 10,
                confidence=84.0, rationale=f"Breakout above 20-bar Donchian High (${donch_high:.2f}).",
                bar_time=bar_time
            )
        elif current_price < donch_low:
            entry = round(current_price, 2)
            sl = round(entry + risk_dist, 2)
            tp1 = round(entry - (risk_dist * 1.5), 2)
            tp2 = round(entry - (risk_dist * 2.8), 2)
            return CandidateSignal(
                strategy_name=cls.name, action="SELL", entry=entry, sl=sl, tp1=tp1, tp2=tp2,
                be_trigger=round(entry - 2.0, 2), risk_reward="1 : 1.5", risk_pips=risk_dist * 10,
                confidence=84.0, rationale=f"Breakout below 20-bar Donchian Low (${donch_low:.2f}).",
                bar_time=bar_time
            )

        return CandidateSignal(
            strategy_name=cls.name, action="HOLD", entry=current_price, sl=current_price,
            tp1=current_price, tp2=current_price, be_trigger=current_price,
            risk_reward="1:1", risk_pips=0.0, confidence=50.0,
            rationale="Price within 20-bar Donchian channel.", bar_time=bar_time
        )


class VolatilitySqueezeStrategy:
    """Strategy C: Bollinger Band contraction inside Keltner Channels."""
    name = "VOLATILITY_SQUEEZE"

    @classmethod
    def evaluate(cls, df_m5: pd.DataFrame, df_h1: pd.DataFrame | None = None) -> CandidateSignal:
        close = df_m5["close"].values
        high = df_m5["high"].values
        low = df_m5["low"].values
        current_price = float(close[-1])
        bar_time = str(df_m5["timestamp"].iloc[-1])

        # Bollinger Bands (20, 2.0)
        s_close = pd.Series(close)
        sma20 = s_close.rolling(20).mean()
        std20 = s_close.rolling(20).std()
        bb_upper = float((sma20 + (2.0 * std20)).iloc[-1])
        bb_lower = float((sma20 - (2.0 * std20)).iloc[-1])

        # ATR & Keltner Channels (20, 1.5)
        tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
        atr = float(pd.Series(tr).rolling(14).mean().iloc[-1])
        kc_upper = float((sma20 + (1.5 * pd.Series(tr).rolling(20).mean())).iloc[-1])
        kc_lower = float((sma20 - (1.5 * pd.Series(tr).rolling(20).mean())).iloc[-1])

        risk_dist = round(max(atr * 1.25, 3.20), 2)
        momentum = current_price - float(sma20.iloc[-1])

        # Expansion fire
        is_firing_long = bb_upper > kc_upper and momentum > 0
        is_firing_short = bb_lower < kc_lower and momentum < 0

        if is_firing_long:
            entry = round(current_price, 2)
            sl = round(entry - risk_dist, 2)
            tp1 = round(entry + (risk_dist * 1.5), 2)
            tp2 = round(entry + (risk_dist * 2.8), 2)
            return CandidateSignal(
                strategy_name=cls.name, action="BUY", entry=entry, sl=sl, tp1=tp1, tp2=tp2,
                be_trigger=round(entry + 2.0, 2), risk_reward="1 : 1.5", risk_pips=risk_dist * 10,
                confidence=86.0, rationale="Volatility Squeeze Expansion fired Bullish.",
                bar_time=bar_time
            )
        elif is_firing_short:
            entry = round(current_price, 2)
            sl = round(entry + risk_dist, 2)
            tp1 = round(entry - (risk_dist * 1.5), 2)
            tp2 = round(entry - (risk_dist * 2.8), 2)
            return CandidateSignal(
                strategy_name=cls.name, action="SELL", entry=entry, sl=sl, tp1=tp1, tp2=tp2,
                be_trigger=round(entry - 2.0, 2), risk_reward="1 : 1.5", risk_pips=risk_dist * 10,
                confidence=86.0, rationale="Volatility Squeeze Expansion fired Bearish.",
                bar_time=bar_time
            )

        return CandidateSignal(
            strategy_name=cls.name, action="HOLD", entry=current_price, sl=current_price,
            tp1=current_price, tp2=current_price, be_trigger=current_price,
            risk_reward="1:1", risk_pips=0.0, confidence=50.0,
            rationale="Volatility within normal compression band.", bar_time=bar_time
        )
