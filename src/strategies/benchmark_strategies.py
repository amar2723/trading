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
    """Strategy A: H1 Macro Trend + M5 Value-Zone Pullback."""
    name = "H1_TREND_M5_PULLBACK"

    @classmethod
    def evaluate(cls, df_m5: pd.DataFrame, df_h1: pd.DataFrame | None = None) -> CandidateSignal:
        close = df_m5["close"].values
        high = df_m5["high"].values
        low = df_m5["low"].values
        current_price = float(close[-1])
        bar_time = str(df_m5["timestamp"].iloc[-1])

        # Causal H1 Trend
        h1_trend_bull = False
        h1_trend_bear = False
        if "h1_trend_bias" in df_m5.columns:
            bias_val = int(df_m5["h1_trend_bias"].iloc[-1])
            h1_trend_bull = (bias_val == 1)
            h1_trend_bear = (bias_val == -1)
        elif df_h1 is not None and len(df_h1) >= 20:
            h1_closed = df_h1.iloc[:-1]  # Completed H1 bars only
            h1_c = h1_closed["close"].values
            h1_e50 = float(pd.Series(h1_c).ewm(span=50).mean().iloc[-1])
            h1_trend_bull = float(h1_c[-1]) > h1_e50
            h1_trend_bear = float(h1_c[-1]) < h1_e50

        # M5 Value Zone EMAs
        m5_e20 = float(pd.Series(close).ewm(span=20).mean().iloc[-1])
        m5_e50 = float(pd.Series(close).ewm(span=50).mean().iloc[-1])

        diffs = pd.Series(close).diff()
        gain = diffs.clip(lower=0).rolling(14).mean()
        loss = (-diffs.clip(upper=0)).rolling(14).mean()
        rsi = float((100 - (100 / (1 + (gain / (loss + 1e-9))))).iloc[-1])

        # ATR & Risk distance (minimum 4.50 pts / 45 pips)
        tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
        atr = float(pd.Series(tr).rolling(14).mean().iloc[-1])
        risk_dist = round(max(atr * 1.5, 4.50), 2)

        # Bullish Pullback Setup:
        # 1. H1 Bull Trend confirmed
        # 2. Price is in the M5 Value Zone between EMA 20 and EMA 50
        # 3. RSI is in healthy reload zone (40 <= RSI <= 60), NOT exhausted
        # 4. Current bar closes bullish (close >= open)
        is_bull_pullback = (
            h1_trend_bull
            and (min(m5_e20, m5_e50) - 1.5 <= current_price <= max(m5_e20, m5_e50) + 1.0)
            and (38.0 <= rsi <= 62.0)
            and (close[-1] >= df_m5["open"].iloc[-1])
        )

        if is_bull_pullback:
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
                rationale=f"H1 Bull Trend confirmed. Clean M5 Pullback into EMA 20/50 value zone with RSI {rsi:.1f} (Reloaded).",
                bar_time=bar_time,
            )

        # Bearish Pullback Setup:
        # 1. H1 Bear Trend confirmed
        # 2. Price is in the M5 Value Zone between EMA 20 and EMA 50
        # 3. RSI is in healthy reload zone (40 <= RSI <= 62), NOT oversold (< 35)
        # 4. Current bar closes bearish (close <= open)
        is_bear_pullback = (
            h1_trend_bear
            and (min(m5_e20, m5_e50) - 1.0 <= current_price <= max(m5_e20, m5_e50) + 1.5)
            and (38.0 <= rsi <= 62.0)
            and (close[-1] <= df_m5["open"].iloc[-1])
        )

        if is_bear_pullback:
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
                rationale=f"H1 Bear Trend confirmed. Clean M5 Pullback up into EMA 20/50 resistance with RSI {rsi:.1f} (Healthy).",
                bar_time=bar_time,
            )

        return CandidateSignal(
            strategy_name=cls.name, action="HOLD", entry=current_price, sl=current_price,
            tp1=current_price, tp2=current_price, be_trigger=current_price,
            risk_reward="1:1", risk_pips=0.0, confidence=50.0,
            rationale="No clean H1 pullback alignment.", bar_time=bar_time
        )


class DonchianBreakoutStrategy:
    """Strategy B: 20-period Donchian Breakout with H1 Trend & ATR filter."""
    name = "DONCHIAN_BREAKOUT"

    @classmethod
    def evaluate(cls, df_m5: pd.DataFrame, df_h1: pd.DataFrame | None = None) -> CandidateSignal:
        close = df_m5["close"].values
        high = df_m5["high"].values
        low = df_m5["low"].values
        current_price = float(close[-1])
        bar_time = str(df_m5["timestamp"].iloc[-1])

        # Causal H1 Trend
        h1_trend_bull = False
        h1_trend_bear = False
        if "h1_trend_bias" in df_m5.columns:
            bias_val = int(df_m5["h1_trend_bias"].iloc[-1])
            h1_trend_bull = (bias_val == 1)
            h1_trend_bear = (bias_val == -1)
        elif df_h1 is not None and len(df_h1) >= 20:
            h1_c = df_h1.iloc[:-1]["close"].values
            h1_e50 = float(pd.Series(h1_c).ewm(span=50).mean().iloc[-1])
            h1_trend_bull = float(h1_c[-1]) > h1_e50
            h1_trend_bear = float(h1_c[-1]) < h1_e50

        # 20-period Donchian Channels (shifted by 1 to prevent lookahead)
        donch_high = float(pd.Series(high).rolling(20).max().shift(1).iloc[-1])
        donch_low = float(pd.Series(low).rolling(20).min().shift(1).iloc[-1])

        diffs = pd.Series(close).diff()
        gain = diffs.clip(lower=0).rolling(14).mean()
        loss = (-diffs.clip(upper=0)).rolling(14).mean()
        rsi = float((100 - (100 / (1 + (gain / (loss + 1e-9))))).iloc[-1])

        tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
        atr = float(pd.Series(tr).rolling(14).mean().iloc[-1])
        risk_dist = round(max(atr * 1.5, 4.50), 2)

        # Only take Bullish Breakout if H1 is Bullish AND RSI is not overbought (> 68)
        if current_price > donch_high and h1_trend_bull and rsi <= 68.0:
            entry = round(current_price, 2)
            sl = round(entry - risk_dist, 2)
            tp1 = round(entry + (risk_dist * 1.5), 2)
            tp2 = round(entry + (risk_dist * 2.8), 2)
            return CandidateSignal(
                strategy_name=cls.name, action="BUY", entry=entry, sl=sl, tp1=tp1, tp2=tp2,
                be_trigger=round(entry + 2.0, 2), risk_reward="1 : 1.5", risk_pips=risk_dist * 10,
                confidence=84.0, rationale=f"H1 Bull Breakout above 20-bar Donchian High (${donch_high:.2f}) with RSI {rsi:.1f}.",
                bar_time=bar_time
            )
        # Only take Bearish Breakout if H1 is Bearish AND RSI is not oversold (< 32)
        elif current_price < donch_low and h1_trend_bear and rsi >= 32.0:
            entry = round(current_price, 2)
            sl = round(entry + risk_dist, 2)
            tp1 = round(entry - (risk_dist * 1.5), 2)
            tp2 = round(entry - (risk_dist * 2.8), 2)
            return CandidateSignal(
                strategy_name=cls.name, action="SELL", entry=entry, sl=sl, tp1=tp1, tp2=tp2,
                be_trigger=round(entry - 2.0, 2), risk_reward="1 : 1.5", risk_pips=risk_dist * 10,
                confidence=84.0, rationale=f"H1 Bear Breakout below 20-bar Donchian Low (${donch_low:.2f}) with RSI {rsi:.1f}.",
                bar_time=bar_time
            )

        return CandidateSignal(
            strategy_name=cls.name, action="HOLD", entry=current_price, sl=current_price,
            tp1=current_price, tp2=current_price, be_trigger=current_price,
            risk_reward="1:1", risk_pips=0.0, confidence=50.0,
            rationale="Price within 20-bar Donchian channel or counter-trend.", bar_time=bar_time
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

        # Causal H1 Trend
        h1_trend_bull = False
        h1_trend_bear = False
        if "h1_trend_bias" in df_m5.columns:
            bias_val = int(df_m5["h1_trend_bias"].iloc[-1])
            h1_trend_bull = (bias_val == 1)
            h1_trend_bear = (bias_val == -1)
        elif df_h1 is not None and len(df_h1) >= 20:
            h1_c = df_h1.iloc[:-1]["close"].values
            h1_e50 = float(pd.Series(h1_c).ewm(span=50).mean().iloc[-1])
            h1_trend_bull = float(h1_c[-1]) > h1_e50
            h1_trend_bear = float(h1_c[-1]) < h1_e50

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

        diffs = pd.Series(close).diff()
        gain = diffs.clip(lower=0).rolling(14).mean()
        loss = (-diffs.clip(upper=0)).rolling(14).mean()
        rsi = float((100 - (100 / (1 + (gain / (loss + 1e-9))))).iloc[-1])

        risk_dist = round(max(atr * 1.5, 4.50), 2)
        momentum = current_price - float(sma20.iloc[-1])

        # Expansion fire WITH H1 Trend Alignment & RSI reload zone (NOT exhausted)
        is_firing_long = bb_upper > kc_upper and momentum > 0 and h1_trend_bull and (45.0 <= rsi <= 65.0)
        is_firing_short = bb_lower < kc_lower and momentum < 0 and h1_trend_bear and (35.0 <= rsi <= 55.0)

        if is_firing_long:
            entry = round(current_price, 2)
            sl = round(entry - risk_dist, 2)
            tp1 = round(entry + (risk_dist * 1.5), 2)
            tp2 = round(entry + (risk_dist * 2.8), 2)
            return CandidateSignal(
                strategy_name=cls.name, action="BUY", entry=entry, sl=sl, tp1=tp1, tp2=tp2,
                be_trigger=round(entry + 2.0, 2), risk_reward="1 : 1.5", risk_pips=risk_dist * 10,
                confidence=86.0, rationale=f"Bullish Squeeze Expansion confirmed by H1 trend (RSI {rsi:.1f}).",
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
                confidence=86.0, rationale=f"Bearish Squeeze Expansion confirmed by H1 trend (RSI {rsi:.1f}).",
                bar_time=bar_time
            )

        return CandidateSignal(
            strategy_name=cls.name, action="HOLD", entry=current_price, sl=current_price,
            tp1=current_price, tp2=current_price, be_trigger=current_price,
            risk_reward="1:1", risk_pips=0.0, confidence=50.0,
            rationale="Volatility within normal compression band or exhausted.", bar_time=bar_time
        )
