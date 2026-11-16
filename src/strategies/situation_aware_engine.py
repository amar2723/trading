from __future__ import annotations

"""
Situation-Aware & Order Flow Trading Engine for XAUUSD.

Combines:
1. Dynamic Market Regime Detection (Trending vs Ranging vs News Expansion)
2. Order Flow Buyer/Seller Imbalance & CVD Absorption
3. Smart Money Concepts (Sweeps, FVG, MSS, Order Blocks)
4. Premium vs Discount Matrix
"""

from dataclasses import dataclass
from typing import Literal
import pandas as pd

from src.market_regime.regime_classifier import classify_market_regime
from src.order_flow.absorption_detector import detect_absorption
from src.strategies.viral_algos import calculate_premium_discount, add_asian_judas_swing


@dataclass
class SituationSignal:
    time: str
    signal: Literal["BUY", "SELL", "HOLD"]
    regime: str
    buyer_dominance: float
    seller_dominance: float
    absorption_detected: bool
    entry: float | None
    sl: float | None
    tp1: float | None
    tp2: float | None
    risk_reward: float | None
    confidence: float
    situation_reason: str


class SituationAwareEngine:
    def __init__(self, symbol: str = "XAUUSD", min_confidence: float = 75.0):
        self.symbol = symbol
        self.min_confidence = min_confidence

    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enriches the candle feed with Order Flow, Market Regimes, and Structure."""
        data = classify_market_regime(df)
        data = detect_absorption(data)
        data = calculate_premium_discount(data)
        data = add_asian_judas_swing(data)
        return data

    def evaluate_bar(self, row: pd.Series) -> SituationSignal:
        time_str = str(row.get("timestamp", row.get("time", "")))
        price = float(row["close"])
        atr = float(row.get("atr", abs(row["high"] - row["low"])))
        regime = str(row.get("market_regime", "RANGING_CONSOLIDATION"))
        buyer_dom = float(row.get("buyer_dominance_pct", 50.0))
        seller_dom = float(row.get("seller_dominance_pct", 50.0))
        bull_abs = bool(row.get("bullish_absorption", False))
        bear_abs = bool(row.get("bearish_absorption", False))
        is_discount = bool(row.get("is_discount", False))
        is_premium = bool(row.get("is_premium", False))

        # -------------------------------------------------------------
        # SITUATION 1: VOLATILITY EXPANSION (News / High Risk Spike)
        # -------------------------------------------------------------
        if regime == "VOLATILITY_EXPANSION":
            return SituationSignal(
                time=time_str,
                signal="HOLD",
                regime=regime,
                buyer_dominance=buyer_dom,
                seller_dominance=seller_dom,
                absorption_detected=False,
                entry=None,
                sl=None,
                tp1=None,
                tp2=None,
                risk_reward=None,
                confidence=0.0,
                situation_reason="Market in extreme Volatility Expansion (News/Chaos). Standing aside to preserve capital."
            )

        # -------------------------------------------------------------
        # SITUATION 2: TRENDING BULLISH REGIME
        # -------------------------------------------------------------
        if regime == "TRENDING_BULLISH":
            # In a bull trend: ONLY BUY pullbacks in Discount or when buyers dominate
            if (is_discount or bull_abs) and buyer_dom >= 52.0:
                sl = float(row["low"]) - atr * 0.4
                risk = price - sl
                if risk > 0:
                    tp1 = price + (risk * 2.0)
                    tp2 = price + (risk * 3.5)
                    conf = 85.0 if bull_abs else 78.0
                    return SituationSignal(
                        time=time_str,
                        signal="BUY",
                        regime=regime,
                        buyer_dominance=buyer_dom,
                        seller_dominance=seller_dom,
                        absorption_detected=bull_abs,
                        entry=price,
                        sl=sl,
                        tp1=tp1,
                        tp2=tp2,
                        risk_reward=2.0,
                        confidence=conf,
                        situation_reason=f"Bullish Trend continuation pullback with {buyer_dom:.1f}% buyer pressure." + (" Institutional absorption confirmed." if bull_abs else "")
                    )

        # -------------------------------------------------------------
        # SITUATION 3: TRENDING BEARISH REGIME
        # -------------------------------------------------------------
        if regime == "TRENDING_BEARISH":
            # In a bear trend: ONLY SELL rallies in Premium or when sellers dominate
            if (is_premium or bear_abs) and seller_dom >= 52.0:
                sl = float(row["high"]) + atr * 0.4
                risk = sl - price
                if risk > 0:
                    tp1 = price - (risk * 2.0)
                    tp2 = price - (risk * 3.5)
                    conf = 85.0 if bear_abs else 78.0
                    return SituationSignal(
                        time=time_str,
                        signal="SELL",
                        regime=regime,
                        buyer_dominance=buyer_dom,
                        seller_dominance=seller_dom,
                        absorption_detected=bear_abs,
                        entry=price,
                        sl=sl,
                        tp1=tp1,
                        tp2=tp2,
                        risk_reward=2.0,
                        confidence=conf,
                        situation_reason=f"Bearish Trend continuation rally with {seller_dom:.1f}% seller pressure." + (" Institutional absorption confirmed." if bear_abs else "")
                    )

        # -------------------------------------------------------------
        # SITUATION 4: RANGING / CONSOLIDATION REGIME (Liquidity Raids)
        # -------------------------------------------------------------
        if regime == "RANGING_CONSOLIDATION":
            # Buy side: Liquidity sweep + Institutional absorption of sellers
            if bull_abs and is_discount:
                sl = float(row["low"]) - atr * 0.35
                risk = price - sl
                if risk > 0:
                    tp1 = price + (risk * 2.2)
                    tp2 = price + (risk * 3.8)
                    return SituationSignal(
                        time=time_str,
                        signal="BUY",
                        regime=regime,
                        buyer_dominance=buyer_dom,
                        seller_dominance=seller_dom,
                        absorption_detected=True,
                        entry=price,
                        sl=sl,
                        tp1=tp1,
                        tp2=tp2,
                        risk_reward=2.2,
                        confidence=88.0,
                        situation_reason="Ranging market low swept with Smart Money absorption (CVD Divergence)."
                    )

            # Sell side: Liquidity sweep + Institutional absorption of buyers
            if bear_abs and is_premium:
                sl = float(row["high"]) + atr * 0.35
                risk = sl - price
                if risk > 0:
                    tp1 = price - (risk * 2.2)
                    tp2 = price - (risk * 3.8)
                    return SituationSignal(
                        time=time_str,
                        signal="SELL",
                        regime=regime,
                        buyer_dominance=buyer_dom,
                        seller_dominance=seller_dom,
                        absorption_detected=True,
                        entry=price,
                        sl=sl,
                        tp1=tp1,
                        tp2=tp2,
                        risk_reward=2.2,
                        confidence=88.0,
                        situation_reason="Ranging market high swept with Smart Money absorption (CVD Divergence)."
                    )

        # Default HOLD
        return SituationSignal(
            time=time_str,
            signal="HOLD",
            regime=regime,
            buyer_dominance=buyer_dom,
            seller_dominance=seller_dom,
            absorption_detected=False,
            entry=None,
            sl=None,
            tp1=None,
            tp2=None,
            risk_reward=None,
            confidence=0.0,
            situation_reason=f"No high-edge confluence in current {regime} regime."
        )
