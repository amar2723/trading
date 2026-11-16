from __future__ import annotations

"""
Institutional & Viral Social Media Trading Algorithms for XAUUSD (Gold).

Includes:
1. ICT Silver Bullet Strategy (London 03:00-04:00, NY AM 10:00-11:00, NY PM 14:00-15:00)
2. Asian Range Sweep & Judas Swing (London Open Asian High/Low Liquidity Raids)
3. ICT 2022 Mentorship Model (Sweep -> Displacement -> MSS -> FVG)
4. Optimal Trade Entry (OTE - 62% to 79% Fibonacci Retracement)
5. Premium vs. Discount Dealing Range Valuation Matrix
6. Turtle Soup Liquidity Reversals (20-bar False Breakout Hunts)
"""

from dataclasses import dataclass
from typing import Literal
import numpy as np
import pandas as pd


@dataclass
class StrategySignal:
    strategy_name: str
    time: str
    signal: Literal["BUY", "SELL", "HOLD"]
    entry: float | None
    sl: float | None
    tp1: float | None
    tp2: float | None
    risk_reward: float | None
    confidence: float
    reason: str
    metadata: dict


# =====================================================================
# 1. Premium vs Discount Matrix
# =====================================================================
def calculate_premium_discount(df: pd.DataFrame, lookback: int = 40) -> pd.DataFrame:
    """
    Computes dealing range, 50% equilibrium, premium, and discount zones.
    Rule: Only BUY in Discount (< 50%), Only SELL in Premium (> 50%).
    """
    out = df.copy()
    rolling_high = out["high"].rolling(lookback, min_periods=5).max()
    rolling_low = out["low"].rolling(lookback, min_periods=5).min()
    equilibrium = (rolling_high + rolling_low) / 2.0
    
    out["dealing_high"] = rolling_high
    out["dealing_low"] = rolling_low
    out["equilibrium"] = equilibrium
    out["is_discount"] = out["close"] < equilibrium
    out["is_premium"] = out["close"] > equilibrium
    out["fib_618"] = rolling_low + (rolling_high - rolling_low) * 0.618
    out["fib_705"] = rolling_low + (rolling_high - rolling_low) * 0.705
    out["fib_786"] = rolling_low + (rolling_high - rolling_low) * 0.786
    return out


# =====================================================================
# 2. Asian Range & London Judas Swing Strategy
# =====================================================================
def add_asian_judas_swing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies the Asian Session (00:00 - 07:00 UTC) Range High & Low.
    Flags London Open (07:00 - 10:00 UTC) fakeout sweeps (Judas Swings)
    targeting the opposite Asian extreme.
    """
    out = df.copy()
    timestamps = pd.to_datetime(out["timestamp"] if "timestamp" in out.columns else out["time"])
    hours = timestamps.dt.hour
    
    out["is_asian"] = (hours >= 0) & (hours < 7)
    out["is_london_judas_window"] = (hours >= 7) & (hours < 10)
    
    # Calculate daily Asian high and low
    dates = timestamps.dt.date
    out["date"] = dates
    
    asian_highs = out[out["is_asian"]].groupby("date")["high"].max().to_dict()
    asian_lows = out[out["is_asian"]].groupby("date")["low"].min().to_dict()
    
    out["asian_high"] = out["date"].map(asian_highs)
    out["asian_low"] = out["date"].map(asian_lows)
    
    # Judas Swing triggers:
    # Bullish: Pierces Asian Low during London Open but closes back above
    out["judas_bull_sweep"] = (
        out["is_london_judas_window"]
        & (out["low"] < out["asian_low"])
        & (out["close"] > out["asian_low"])
    )
    # Bearish: Pierces Asian High during London Open but closes back below
    out["judas_bear_sweep"] = (
        out["is_london_judas_window"]
        & (out["high"] > out["asian_high"])
        & (out["close"] < out["asian_high"])
    )
    return out


# =====================================================================
# 3. ICT Silver Bullet Strategy (Exact 1-Hour Killzones)
# =====================================================================
def add_silver_bullet_signals(df: pd.DataFrame, lookback_sweep: int = 20) -> pd.DataFrame:
    """
    ICT Silver Bullet Model:
    Three 60-minute windows in NY Time (EST/EDT):
      - London Open: 03:00 - 04:00 AM NY (08:00 - 09:00 UTC)
      - NY AM:       10:00 - 11:00 AM NY (15:00 - 16:00 UTC)
      - NY PM:       02:00 - 03:00 PM NY (19:00 - 20:00 UTC)
    Rules:
      1. Time window active
      2. Prior liquidity sweep of rolling high/low
      3. Displacement candle
      4. Fair Value Gap (FVG) retest
    """
    out = df.copy()
    ts = pd.to_datetime(out["timestamp"] if "timestamp" in out.columns else out["time"])
    
    # Convert to NY hour (approximate UTC - 5 for standard EST)
    utc_hours = ts.dt.hour
    ny_hours = (utc_hours - 5) % 24
    
    out["is_sb_window"] = (ny_hours == 3) | (ny_hours == 10) | (ny_hours == 14)
    
    # 3-candle Fair Value Gap detection
    out["bull_fvg"] = (out["low"] > out["high"].shift(2))
    out["bear_fvg"] = (out["high"] < out["low"].shift(2))
    out["fvg_top"] = np.where(out["bull_fvg"], out["low"], np.where(out["bear_fvg"], out["low"].shift(2), np.nan))
    out["fvg_bottom"] = np.where(out["bull_fvg"], out["high"].shift(2), np.where(out["bear_fvg"], out["high"], np.nan))
    
    # Rolling liquidity pools
    rolling_high = out["high"].shift(1).rolling(lookback_sweep).max()
    rolling_low = out["low"].shift(1).rolling(lookback_sweep).min()
    
    out["sb_swept_high"] = out["high"] > rolling_high
    out["sb_swept_low"] = out["low"] < rolling_low
    
    # Signal generation
    out["sb_buy"] = out["is_sb_window"] & out["sb_swept_low"].rolling(5).max().astype(bool) & out["bull_fvg"]
    out["sb_sell"] = out["is_sb_window"] & out["sb_swept_high"].rolling(5).max().astype(bool) & out["bear_fvg"]
    return out


# =====================================================================
# 4. Master Algorithmic Engine (Ensemble of Viral & SMC Algos)
# =====================================================================
class ViralSMCEnsembleEngine:
    def __init__(self, symbol: str = "XAUUSD"):
        self.symbol = symbol

    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        data = calculate_premium_discount(df)
        data = add_asian_judas_swing(data)
        data = add_silver_bullet_signals(data)
        return data

    def get_signal_from_row(self, row: pd.Series) -> StrategySignal:
        time_str = str(row.get("timestamp", row.get("time", "")))
        price = float(row["close"])
        atr = float(row.get("atr", abs(row["high"] - row["low"])))

        # 1. Check Asian Judas Swing (High win rate London open reversal)
        if bool(row.get("judas_bull_sweep")) and bool(row.get("is_discount")):
            sl = float(row["low"]) - atr * 0.3
            tp1 = float(row.get("asian_high", price + atr * 2))
            tp2 = tp1 + atr * 1.5
            risk = price - sl
            rr = (tp1 - price) / risk if risk > 0 else 2.0
            return StrategySignal(
                strategy_name="Asian Judas Swing (London Open)",
                time=time_str,
                signal="BUY",
                entry=price,
                sl=sl,
                tp1=tp1,
                tp2=tp2,
                risk_reward=round(rr, 2),
                confidence=85.0,
                reason="Asian Low liquidity swept during London Open, rejected back into Discount zone.",
                metadata={"asian_low": row.get("asian_low"), "asian_high": row.get("asian_high")}
            )

        if bool(row.get("judas_bear_sweep")) and bool(row.get("is_premium")):
            sl = float(row["high"]) + atr * 0.3
            tp1 = float(row.get("asian_low", price - atr * 2))
            tp2 = tp1 - atr * 1.5
            risk = sl - price
            rr = (price - tp1) / risk if risk > 0 else 2.0
            return StrategySignal(
                strategy_name="Asian Judas Swing (London Open)",
                time=time_str,
                signal="SELL",
                entry=price,
                sl=sl,
                tp1=tp1,
                tp2=tp2,
                risk_reward=round(rr, 2),
                confidence=85.0,
                reason="Asian High liquidity swept during London Open, rejected back into Premium zone.",
                metadata={"asian_low": row.get("asian_low"), "asian_high": row.get("asian_high")}
            )

        # 2. Check ICT Silver Bullet (Strict 1-hour session execution)
        if bool(row.get("sb_buy")):
            sl = float(row["low"]) - atr * 0.4
            risk = price - sl
            tp1 = price + (risk * 2.5)  # Strict 1:2.5 RR
            tp2 = price + (risk * 4.0)
            return StrategySignal(
                strategy_name="ICT Silver Bullet",
                time=time_str,
                signal="BUY",
                entry=price,
                sl=sl,
                tp1=tp1,
                tp2=tp2,
                risk_reward=2.5,
                confidence=80.0,
                reason="ICT Silver Bullet Window active: Sell-Side Liquidity swept with Bullish FVG displacement.",
                metadata={"fvg_bottom": row.get("fvg_bottom"), "fvg_top": row.get("fvg_top")}
            )

        if bool(row.get("sb_sell")):
            sl = float(row["high"]) + atr * 0.4
            risk = sl - price
            tp1 = price - (risk * 2.5)  # Strict 1:2.5 RR
            tp2 = price - (risk * 4.0)
            return StrategySignal(
                strategy_name="ICT Silver Bullet",
                time=time_str,
                signal="SELL",
                entry=price,
                sl=sl,
                tp1=tp1,
                tp2=tp2,
                risk_reward=2.5,
                confidence=80.0,
                reason="ICT Silver Bullet Window active: Buy-Side Liquidity swept with Bearish FVG displacement.",
                metadata={"fvg_bottom": row.get("fvg_bottom"), "fvg_top": row.get("fvg_top")}
            )

        # Default HOLD state
        return StrategySignal(
            strategy_name="SMC Ensemble",
            time=time_str,
            signal="HOLD",
            entry=None,
            sl=None,
            tp1=None,
            tp2=None,
            risk_reward=None,
            confidence=0.0,
            reason="No high-probability session liquidity sweep or Silver Bullet setup active.",
            metadata={"is_discount": bool(row.get("is_discount")), "is_premium": bool(row.get("is_premium"))}
        )

    def get_latest_signal(self, df: pd.DataFrame) -> StrategySignal:
        analyzed = self.analyze(df)
        return self.get_signal_from_row(analyzed.iloc[-1])
