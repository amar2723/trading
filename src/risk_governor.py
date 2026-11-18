from __future__ import annotations

"""
Independent Risk Governor for XAUUSD Trading.
Holds absolute veto authority over all strategies and models.
Enforces:
1. Capital protection sizing (RiskUSD = Equity * r)
2. Maximum spread constraint
3. Single concurrent position rule
4. Daily drawdown circuit breaker (Max 2R loss / day)
5. Minimum broker lot and volume step verification
"""

import logging
import math
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger("RiskGovernor")

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None


@dataclass
class RiskApproval:
    approved: bool
    lot_size: float
    risk_usd: float
    risk_pct: float
    veto_reason: str | None = None


class RiskGovernor:
    """
    Independent Risk Governor for XAUUSD Trading.
    Holds absolute veto authority over all strategies and models.
    Enforces:
    1. Strict <= 1.0% Capital Protection Ceiling (RiskUSD <= Equity * max_risk_pct)
    2. Exact Gold Contract Specifications (100 oz/lot, 0.01 lot step, 0.01 min lot)
    3. Mandatory inclusion of spread and slippage execution friction in risk budget
    4. Prohibition of automatic volume bump-ups: if calculated lot < min_lot, REJECT trade
    5. Maximum spread constraint
    6. Single concurrent position rule
    7. Daily drawdown circuit breaker (Max 2R loss / day)
    8. Asian session chop shield (00:00 - 06:00 UTC)

    REAL-MARKET RISK DISCLOSURE:
    While modeled initial stop-loss risk is strictly capped at <= 1.0%, real market conditions
    (e.g., weekend price gaps, black-swan illiquidity, and fast-moving slippage) can cause
    realized exit losses that exceed the modeled stop-loss budget during live market execution.
    """

    def __init__(
        self,
        max_risk_pct: float = 1.0,         # Maximum 1.0% risk per trade
        max_spread_pips: float = 35.0,     # Veto if spread exceeds 3.5 points ($0.35/oz)
        max_open_positions: int = 1,       # Strictly 1 active trade at a time
        max_daily_losses_R: float = 2.0,   # Hard circuit breaker at 2R loss
        contract_size: float = 100.0,      # 100 oz per 1.00 lot for XAUUSD
        min_volume: float = 0.01,          # Minimum broker volume
        volume_step: float = 0.01,         # Broker volume step
        round_trip_slippage_usd: float = 0.10,  # $0.05 entry + $0.05 exit slippage per oz
    ):
        self.max_risk_pct = max_risk_pct
        self.max_spread_pips = max_spread_pips
        self.max_open_positions = max_open_positions
        self.max_daily_losses_R = max_daily_losses_R
        self.contract_size = contract_size
        self.min_volume = min_volume
        self.volume_step = volume_step
        self.round_trip_slippage_usd = round_trip_slippage_usd

    def review_trade(
        self,
        equity: float,
        risk_pips: float | None = None,
        stop_dist_usd: float | None = None,
        entry_price: float | None = None,
        sl_price: float | None = None,
        spread_price_usd: float | None = None,
        current_spread_pips: float | None = None,
        open_positions_count: int = 0,
        daily_loss_R: float = 0.0,
        is_reconciled: bool = True,
        session_hour: int | None = None,
        spread_included_in_prices: bool = False,
    ) -> RiskApproval:
        # Check 0: Asian session chop shield (00:00 - 06:00 UTC)
        if session_hour is not None and 0 <= session_hour <= 6:
            return RiskApproval(
                approved=False, lot_size=0.0, risk_usd=0.0, risk_pct=0.0,
                veto_reason=f"VETO: Asian session chop shield active (Hour {session_hour:02d}:00 UTC). Empirical win rate < 20% in low-liquidity hours."
            )
        # Check 1: Reconciliation integrity
        if not is_reconciled:
            return RiskApproval(
                approved=False, lot_size=0.0, risk_usd=0.0, risk_pct=0.0,
                veto_reason="VETO: Broker reconciliation is out of sync."
            )

        # Check 2: Daily loss circuit breaker
        if daily_loss_R >= self.max_daily_losses_R:
            return RiskApproval(
                approved=False, lot_size=0.0, risk_usd=0.0, risk_pct=0.0,
                veto_reason=f"VETO: Daily loss limit ({daily_loss_R:.1f}R >= {self.max_daily_losses_R:.1f}R) reached. Circuit breaker engaged."
            )

        # Check 3: Concurrent position limit
        if open_positions_count >= self.max_open_positions:
            return RiskApproval(
                approved=False, lot_size=0.0, risk_usd=0.0, risk_pct=0.0,
                veto_reason=f"VETO: Active position limit ({open_positions_count}/{self.max_open_positions}) reached. No concurrent exposure."
            )

        # Check 4: Authoritative Monetary Spread Handling
        # Standardized quotation: 1 point = $0.01; 1 pip = 10 points = $0.10 per ounce.
        if spread_price_usd is not None and spread_price_usd >= 0:
            spread_usd_per_oz = float(spread_price_usd)
            spread_pips_val = round(spread_usd_per_oz / 0.10, 1)
        elif current_spread_pips is not None and current_spread_pips >= 0:
            spread_pips_val = float(current_spread_pips)
            # Unified conversion: 1 pip = $0.10 price move per ounce
            spread_usd_per_oz = round(spread_pips_val * 0.10, 4)
        else:
            # Default to verified Octa typical spread: 4.2 pips = $0.42/oz
            spread_usd_per_oz = 0.42
            spread_pips_val = 4.2

        if spread_pips_val > self.max_spread_pips:
            return RiskApproval(
                approved=False, lot_size=0.0, risk_usd=0.0, risk_pct=0.0,
                veto_reason=f"VETO: Current spread ({spread_pips_val:.1f} pips / ${spread_usd_per_oz:.2f}) exceeds maximum allowed ({self.max_spread_pips:.1f} pips)."
            )

        # Check 5: Precise Stop Distance Determination
        if entry_price is not None and sl_price is not None:
            dist_usd = abs(entry_price - sl_price)
        elif stop_dist_usd is not None and stop_dist_usd > 0:
            dist_usd = stop_dist_usd
        elif risk_pips is not None and risk_pips > 0:
            dist_usd = risk_pips * 0.10  # 1 pip = $0.10 per ounce on XAUUSD
        else:
            return RiskApproval(
                approved=False, lot_size=0.0, risk_usd=0.0, risk_pct=0.0,
                veto_reason="VETO: Invalid stop loss distance (dist_usd <= 0)."
            )

        if dist_usd <= 0.05:
            return RiskApproval(
                approved=False, lot_size=0.0, risk_usd=0.0, risk_pct=0.0,
                veto_reason="VETO: Stop loss distance too small (dist_usd <= $0.05)."
            )

        # Execution friction modeling per ounce (Double-Counting Prevention)
        # If entry and SL are already priced with Ask/Bid spread, friction only includes slippage
        if spread_included_in_prices:
            friction_usd_per_oz = self.round_trip_slippage_usd
        else:
            friction_usd_per_oz = spread_usd_per_oz + self.round_trip_slippage_usd

        # Total monetary loss per 1.00 standard lot (100 oz) if stop loss is hit
        loss_per_lot_usd = self.contract_size * (dist_usd + friction_usd_per_oz)

        # Optional MT5 broker verification when connected
        if mt5 is not None:
            try:
                calc_pnl = mt5.order_calc_profit(0, "XAUUSD", 1.0, 2600.0, 2600.0 - dist_usd)
                if calc_pnl is not None and calc_pnl < 0:
                    broker_loss = abs(calc_pnl)
                    loss_per_lot_usd = broker_loss + (self.contract_size * friction_usd_per_oz)
            except Exception:
                pass

        # Maximum allowable risk budget in USD (Strict 1.0% limit)
        max_allowed_risk_usd = round(equity * (self.max_risk_pct / 100.0), 4)

        # Raw lot size that strictly respects the risk budget
        raw_volume = max_allowed_risk_usd / loss_per_lot_usd

        # Quantize down to the nearest broker volume_step
        calculated_lot = round(math.floor(raw_volume / self.volume_step) * self.volume_step, 2)

        # Check if the calculated lot is below the broker's minimum volume
        if calculated_lot < self.min_volume:
            # Check the monetary risk of trading the broker minimum (0.01 lot)
            min_lot_risk_usd = round(self.min_volume * loss_per_lot_usd, 2)
            min_lot_risk_pct = round((min_lot_risk_usd / max(0.01, equity)) * 100.0, 2)

            # CRITICAL SAFETY RULE: Never automatically bump volume to min_lot if it exceeds risk!
            return RiskApproval(
                approved=False,
                lot_size=0.0,
                risk_usd=min_lot_risk_usd,
                risk_pct=min_lot_risk_pct,
                veto_reason=(
                    f"VETO: Account undercapitalized for minimum broker lot ({self.min_volume:.2f} lot). "
                    f"Minimum risk is ${min_lot_risk_usd:.2f} ({min_lot_risk_pct:.2f}% of equity), "
                    f"which exceeds the {self.max_risk_pct:.1f}% risk ceiling (${max_allowed_risk_usd:.2f})."
                )
            )

        # If calculated_lot >= min_volume, compute exact exposure
        actual_risk_usd = round(calculated_lot * loss_per_lot_usd, 2)
        actual_risk_pct = round((actual_risk_usd / max(0.01, equity)) * 100.0, 2)

        # Final sanity check: ensure rounding never breached the 1.0% ceiling
        if actual_risk_pct > self.max_risk_pct:
            return RiskApproval(
                approved=False,
                lot_size=0.0,
                risk_usd=actual_risk_usd,
                risk_pct=actual_risk_pct,
                veto_reason=f"VETO: Actual risk ({actual_risk_pct:.2f}%) exceeds maximum allowed {self.max_risk_pct:.1f}%."
            )

        return RiskApproval(
            approved=True,
            lot_size=calculated_lot,
            risk_usd=actual_risk_usd,
            risk_pct=actual_risk_pct,
            veto_reason=None
        )
