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

from src.liquidity_map import LiquidityMap


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
    _lmap = LiquidityMap()

    @classmethod
    def evaluate(cls, df_m5: pd.DataFrame, df_h1: pd.DataFrame | None = None) -> CandidateSignal:
        close = df_m5["close"].values
        high = df_m5["high"].values
        low = df_m5["low"].values
        current_price = float(close[-1])
        bar_time = str(df_m5["timestamp"].iloc[-1])

        # Liquidity & Circuit Breaker Pre-Flight Check
        events, status_note = cls._lmap.evaluate_sweeps_and_structure(df_m5)
        if "CIRCUIT_BREAKER" in status_note or "FORCED_LIQUIDATION" in status_note:
            return CandidateSignal(
                strategy_name=cls.name, action="HOLD", entry=current_price, sl=current_price,
                tp1=current_price, tp2=current_price, be_trigger=current_price,
                risk_reward="1:1", risk_pips=0.0, confidence=0.0,
                rationale=f"VETO: Circuit breaker engaged ({status_note}). Standing down to preserve capital.",
                bar_time=bar_time,
            )

        active_wicks = [e for e in events if e.status == "UNCONFIRMED_WICK"]
        confirmed_rejections = [e for e in events if e.status == "CONFIRMED_REJECTION"]

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

        # ATR & Risk distance (minimum 6.50 pts / 65 pips for noise resilience)
        tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
        atr = float(pd.Series(tr).rolling(14).mean().iloc[-1])
        risk_dist = round(max(atr * 1.8, 6.50), 2)

        # Bullish Pullback Setup:
        # 1. H1 Bull Trend confirmed
        # 2. Price is in the M5 Value Zone between EMA 20 and EMA 50
        # 3. RSI is in healthy reload zone (40 <= RSI <= 60), NOT exhausted
        # 4. Current bar closes bullish (close >= open)
        # 5. Liquidity Gate: No active unconfirmed resistance wick or confirmed bearish rejection
        has_bearish_rejection = any(e.sweep_type == "BEARISH_SWEEP" and e.status == "CONFIRMED_REJECTION" for e in confirmed_rejections)
        has_bullish_rejection = any(e.sweep_type == "BULLISH_SWEEP" and e.status == "CONFIRMED_REJECTION" for e in confirmed_rejections)
        has_bearish_wick = any(e.sweep_type == "BEARISH_SWEEP" and e.status == "UNCONFIRMED_WICK" for e in active_wicks)
        has_bullish_wick = any(e.sweep_type == "BULLISH_SWEEP" and e.status == "UNCONFIRMED_WICK" for e in active_wicks)

        is_bull_pullback = (
            h1_trend_bull
            and (min(m5_e20, m5_e50) - 1.5 <= current_price <= max(m5_e20, m5_e50) + 1.0)
            and (38.0 <= rsi <= 62.0)
            and (close[-1] >= df_m5["open"].iloc[-1])
            and not has_bearish_rejection
            and not has_bearish_wick
        )

        if is_bull_pullback:
            entry = round(current_price, 2)
            sl = round(entry - risk_dist, 2)
            tp1 = round(entry + (risk_dist * 1.5), 2)
            tp2 = round(entry + (risk_dist * 2.8), 2)
            be_trig = round(entry + max(risk_dist * 0.85, 5.00), 2)
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
                confidence=78.0,
                rationale=f"H1 Bull Trend confirmed. Clean M5 Pullback into EMA 20/50 value zone with RSI {rsi:.1f} (Reloaded).",
                bar_time=bar_time,
            )

        # Bearish Pullback Setup:
        # 1. H1 Bear Trend confirmed
        # 2. Price is in the M5 Value Zone between EMA 20 and EMA 50
        # 3. RSI is in healthy reload zone (40 <= RSI <= 62), NOT oversold (< 35)
        # 4. Current bar closes bearish (close <= open)
        # 5. Liquidity Gate: No active unconfirmed support wick or confirmed bullish bounce
        is_bear_pullback = (
            h1_trend_bear
            and (min(m5_e20, m5_e50) - 1.0 <= current_price <= max(m5_e20, m5_e50) + 1.5)
            and (38.0 <= rsi <= 62.0)
            and (close[-1] <= df_m5["open"].iloc[-1])
            and not has_bullish_rejection
            and not has_bullish_wick
        )

        if is_bear_pullback:
            entry = round(current_price, 2)
            sl = round(entry + risk_dist, 2)
            tp1 = round(entry - (risk_dist * 1.5), 2)
            tp2 = round(entry - (risk_dist * 2.8), 2)
            be_trig = round(entry - max(risk_dist * 0.85, 5.00), 2)
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
                confidence=78.0,
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
    """Strategy B: 20-period Donchian Breakout with H1 Trend, ATR, & Liquidity filter."""
    name = "DONCHIAN_BREAKOUT"
    _lmap = LiquidityMap()

    @classmethod
    def evaluate(cls, df_m5: pd.DataFrame, df_h1: pd.DataFrame | None = None) -> CandidateSignal:
        close = df_m5["close"].values
        high = df_m5["high"].values
        low = df_m5["low"].values
        current_price = float(close[-1])
        bar_time = str(df_m5["timestamp"].iloc[-1])

        # Liquidity & Circuit Breaker Pre-Flight Check
        events, status_note = cls._lmap.evaluate_sweeps_and_structure(df_m5)
        if "CIRCUIT_BREAKER" in status_note or "FORCED_LIQUIDATION" in status_note:
            return CandidateSignal(
                strategy_name=cls.name, action="HOLD", entry=current_price, sl=current_price,
                tp1=current_price, tp2=current_price, be_trigger=current_price,
                risk_reward="1:1", risk_pips=0.0, confidence=0.0,
                rationale=f"VETO: Circuit breaker engaged ({status_note}). Standing down to preserve capital.",
                bar_time=bar_time,
            )

        active_wicks = [e for e in events if e.status == "UNCONFIRMED_WICK"]
        confirmed_rejections = [e for e in events if e.status == "CONFIRMED_REJECTION"]
        has_bearish_rejection = any(e.sweep_type == "BEARISH_SWEEP" and e.status == "CONFIRMED_REJECTION" for e in confirmed_rejections)
        has_bullish_rejection = any(e.sweep_type == "BULLISH_SWEEP" and e.status == "CONFIRMED_REJECTION" for e in confirmed_rejections)
        has_bearish_wick = any(e.sweep_type == "BEARISH_SWEEP" and e.status == "UNCONFIRMED_WICK" for e in active_wicks)
        has_bullish_wick = any(e.sweep_type == "BULLISH_SWEEP" and e.status == "UNCONFIRMED_WICK" for e in active_wicks)

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
        risk_dist = round(max(atr * 1.8, 6.50), 2)

        # Only take Bullish Breakout if H1 is Bullish AND RSI is in momentum range (not overbought > 62) AND no resistance rejection
        if current_price > donch_high and h1_trend_bull and (45.0 <= rsi <= 62.0) and not has_bearish_rejection and not has_bearish_wick:
            entry = round(current_price, 2)
            sl = round(entry - risk_dist, 2)
            tp1 = round(entry + (risk_dist * 1.5), 2)
            tp2 = round(entry + (risk_dist * 2.8), 2)
            be_trig = round(entry + max(risk_dist * 0.85, 5.00), 2)
            return CandidateSignal(
                strategy_name=cls.name, action="BUY", entry=entry, sl=sl, tp1=tp1, tp2=tp2,
                be_trigger=be_trig, risk_reward="1 : 1.5", risk_pips=risk_dist * 10,
                confidence=78.0, rationale=f"H1 Bull Breakout above 20-bar Donchian High (${donch_high:.2f}) with RSI {rsi:.1f}.",
                bar_time=bar_time
            )
        # Only take Bearish Breakout if H1 is Bearish AND RSI is in momentum range (not oversold < 38) AND no support rejection
        elif current_price < donch_low and h1_trend_bear and (38.0 <= rsi <= 55.0) and not has_bullish_rejection and not has_bullish_wick:
            entry = round(current_price, 2)
            sl = round(entry + risk_dist, 2)
            tp1 = round(entry - (risk_dist * 1.5), 2)
            tp2 = round(entry - (risk_dist * 2.8), 2)
            be_trig = round(entry - max(risk_dist * 0.85, 5.00), 2)
            return CandidateSignal(
                strategy_name=cls.name, action="SELL", entry=entry, sl=sl, tp1=tp1, tp2=tp2,
                be_trigger=be_trig, risk_reward="1 : 1.5", risk_pips=risk_dist * 10,
                confidence=78.0, rationale=f"H1 Bear Breakout below 20-bar Donchian Low (${donch_low:.2f}) with RSI {rsi:.1f}.",
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
    _lmap = LiquidityMap()

    @classmethod
    def evaluate(cls, df_m5: pd.DataFrame, df_h1: pd.DataFrame | None = None) -> CandidateSignal:
        close = df_m5["close"].values
        high = df_m5["high"].values
        low = df_m5["low"].values
        current_price = float(close[-1])
        bar_time = str(df_m5["timestamp"].iloc[-1])

        # Liquidity & Circuit Breaker Pre-Flight Check
        events, status_note = cls._lmap.evaluate_sweeps_and_structure(df_m5)
        if "CIRCUIT_BREAKER" in status_note or "FORCED_LIQUIDATION" in status_note:
            return CandidateSignal(
                strategy_name=cls.name, action="HOLD", entry=current_price, sl=current_price,
                tp1=current_price, tp2=current_price, be_trigger=current_price,
                risk_reward="1:1", risk_pips=0.0, confidence=0.0,
                rationale=f"VETO: Circuit breaker engaged ({status_note}). Standing down to preserve capital.",
                bar_time=bar_time,
            )

        active_wicks = [e for e in events if e.status == "UNCONFIRMED_WICK"]
        confirmed_rejections = [e for e in events if e.status == "CONFIRMED_REJECTION"]
        has_bearish_rejection = any(e.sweep_type == "BEARISH_SWEEP" and e.status == "CONFIRMED_REJECTION" for e in confirmed_rejections)
        has_bullish_rejection = any(e.sweep_type == "BULLISH_SWEEP" and e.status == "CONFIRMED_REJECTION" for e in confirmed_rejections)
        has_bearish_wick = any(e.sweep_type == "BEARISH_SWEEP" and e.status == "UNCONFIRMED_WICK" for e in active_wicks)
        has_bullish_wick = any(e.sweep_type == "BULLISH_SWEEP" and e.status == "UNCONFIRMED_WICK" for e in active_wicks)

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

        risk_dist = round(max(atr * 1.8, 6.50), 2)
        momentum = current_price - float(sma20.iloc[-1])

        # Expansion fire WITH H1 Trend Alignment, RSI reload zone, and Liquidity Clearance
        is_firing_long = bb_upper > kc_upper and momentum > 0 and h1_trend_bull and (45.0 <= rsi <= 62.0) and not has_bearish_rejection and not has_bearish_wick
        is_firing_short = bb_lower < kc_lower and momentum < 0 and h1_trend_bear and (38.0 <= rsi <= 55.0) and not has_bullish_rejection and not has_bullish_wick

        if is_firing_long:
            entry = round(current_price, 2)
            sl = round(entry - risk_dist, 2)
            tp1 = round(entry + (risk_dist * 1.5), 2)
            tp2 = round(entry + (risk_dist * 2.8), 2)
            be_trig = round(entry + max(risk_dist * 0.85, 5.00), 2)
            return CandidateSignal(
                strategy_name=cls.name, action="BUY", entry=entry, sl=sl, tp1=tp1, tp2=tp2,
                be_trigger=be_trig, risk_reward="1 : 1.5", risk_pips=risk_dist * 10,
                confidence=78.0, rationale=f"Bullish Squeeze Expansion confirmed by H1 trend (RSI {rsi:.1f}).",
                bar_time=bar_time
            )
        elif is_firing_short:
            entry = round(current_price, 2)
            sl = round(entry + risk_dist, 2)
            tp1 = round(entry - (risk_dist * 1.5), 2)
            tp2 = round(entry - (risk_dist * 2.8), 2)
            be_trig = round(entry - max(risk_dist * 0.85, 5.00), 2)
            return CandidateSignal(
                strategy_name=cls.name, action="SELL", entry=entry, sl=sl, tp1=tp1, tp2=tp2,
                be_trigger=be_trig, risk_reward="1 : 1.5", risk_pips=risk_dist * 10,
                confidence=78.0, rationale=f"Bearish Squeeze Expansion confirmed by H1 trend (RSI {rsi:.1f}).",
                bar_time=bar_time
            )

        return CandidateSignal(
            strategy_name=cls.name, action="HOLD", entry=current_price, sl=current_price,
            tp1=current_price, tp2=current_price, be_trigger=current_price,
            risk_reward="1:1", risk_pips=0.0, confidence=50.0,
            rationale="Volatility within normal compression band or exhausted.", bar_time=bar_time
        )


class LiquiditySweepReversalStrategy:
    """
    Premier Institutional Liquidity Strategy: Post-Sweep Judas Reversal (Turtle Soup).
    Triggers ONLY when resting liquidity (BSL or SSL) is swept and confirmed rejected.
    Filters out 100% of trend-pullback and consolidation chop trades.
    """
    name = "LIQUIDITY_SWEEP_REVERSAL"
    _lmap = LiquidityMap()

    @classmethod
    def evaluate(cls, df_m5: pd.DataFrame, df_h1: pd.DataFrame | None = None) -> CandidateSignal:
        close = df_m5["close"].values
        current_price = float(close[-1])
        bar_time = str(df_m5["timestamp"].iloc[-1])

        # 1. Pre-flight Circuit Breaker & Sweep Detection
        events, status_note = cls._lmap.evaluate_sweeps_and_structure(df_m5)
        if "CIRCUIT_BREAKER" in status_note or "FORCED_LIQUIDATION" in status_note:
            return CandidateSignal(
                strategy_name=cls.name, action="HOLD", entry=current_price, sl=current_price,
                tp1=current_price, tp2=current_price, be_trigger=current_price,
                risk_reward="1:1", risk_pips=0.0, confidence=0.0,
                rationale=f"VETO: Circuit breaker engaged ({status_note}). Capital strictly preserved.",
                bar_time=bar_time,
            )

        watchlist = cls._lmap.get_pre_sweep_liquidity_watchlist(df_m5)
        nearest_bsl = watchlist.get("nearest_bsl")
        nearest_ssl = watchlist.get("nearest_ssl")
        bsl_info = f"${nearest_bsl['price']:.2f} ({nearest_bsl['distance_pips']}p away)" if nearest_bsl else "None"
        ssl_info = f"${nearest_ssl['price']:.2f} ({nearest_ssl['distance_pips']}p away)" if nearest_ssl else "None"

        confirmed_rejections = [e for e in events if e.status == "CONFIRMED_REJECTION"]
        if not confirmed_rejections:
            return CandidateSignal(
                strategy_name=cls.name, action="HOLD", entry=current_price, sl=current_price,
                tp1=current_price, tp2=current_price, be_trigger=current_price,
                risk_reward="1:1", risk_pips=0.0, confidence=50.0,
                rationale=f"AWAITING_SWEEP: No verified liquidity sweep completed. Nearest BSL: {bsl_info} | Nearest SSL: {ssl_info}.",
                bar_time=bar_time,
            )

        # Look for the strongest confirmed sweep event
        latest_rejection = confirmed_rejections[-1]
        fair_value = watchlist.get("fair_value_targets", {})
        session_vwap = fair_value.get("session_vwap") or current_price

        # Case A: Bearish Sweep (Buy-Side Liquidity Swept and Rejected -> Enter SHORT)
        if latest_rejection.sweep_type == "BEARISH_SWEEP":
            entry = round(current_price, 2)
            sweep_high = latest_rejection.breach_price
            raw_risk = max(sweep_high - entry + 0.50, 4.00)
            risk_dist = round(min(raw_risk, 10.00), 2)
            sl = round(entry + risk_dist, 2)

            tp1_target = session_vwap if (session_vwap < entry - (risk_dist * 1.2)) else (entry - (risk_dist * 1.5))
            tp1 = round(tp1_target, 2)

            tp2_target = nearest_ssl["price"] if (nearest_ssl and nearest_ssl["price"] < tp1) else (entry - (risk_dist * 2.8))
            tp2 = round(tp2_target, 2)

            be_trig = round(entry - max(risk_dist * 0.85, 3.50), 2)
            risk_pips = round(risk_dist * 10.0, 1)

            return CandidateSignal(
                strategy_name=cls.name, action="SELL", entry=entry, sl=sl, tp1=tp1, tp2=tp2,
                be_trigger=be_trig, risk_reward="1 : 2.0", risk_pips=risk_pips,
                confidence=86.0,
                rationale=f"POST-SWEEP SHORT: Swept {latest_rejection.level.level_type} (${latest_rejection.level.price:.2f}) wick ${sweep_high:.2f}. Confirmed bearish bar closure inside range.",
                bar_time=bar_time,
            )

        # Case B: Bullish Sweep (Sell-Side Liquidity Swept and Rejected -> Enter LONG)
        elif latest_rejection.sweep_type == "BULLISH_SWEEP":
            entry = round(current_price, 2)
            sweep_low = latest_rejection.breach_price
            raw_risk = max(entry - sweep_low + 0.50, 4.00)
            risk_dist = round(min(raw_risk, 10.00), 2)
            sl = round(entry - risk_dist, 2)

            tp1_target = session_vwap if (session_vwap > entry + (risk_dist * 1.2)) else (entry + (risk_dist * 1.5))
            tp1 = round(tp1_target, 2)

            tp2_target = nearest_bsl["price"] if (nearest_bsl and nearest_bsl["price"] > tp1) else (entry + (risk_dist * 2.8))
            tp2 = round(tp2_target, 2)

            be_trig = round(entry + max(risk_dist * 0.85, 3.50), 2)
            risk_pips = round(risk_dist * 10.0, 1)

            return CandidateSignal(
                strategy_name=cls.name, action="BUY", entry=entry, sl=sl, tp1=tp1, tp2=tp2,
                be_trigger=be_trig, risk_reward="1 : 2.0", risk_pips=risk_pips,
                confidence=86.0,
                rationale=f"POST-SWEEP LONG: Swept {latest_rejection.level.level_type} (${latest_rejection.level.price:.2f}) wick ${sweep_low:.2f}. Confirmed bullish bar closure inside range.",
                bar_time=bar_time,
            )

        return CandidateSignal(
            strategy_name=cls.name, action="HOLD", entry=current_price, sl=current_price,
            tp1=current_price, tp2=current_price, be_trigger=current_price,
            risk_reward="1:1", risk_pips=0.0, confidence=50.0,
            rationale="No actionable post-sweep structure detected.", bar_time=bar_time
        )

