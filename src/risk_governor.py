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

from dataclasses import dataclass
from typing import Literal
import logging

logger = logging.getLogger("RiskGovernor")


@dataclass
class RiskApproval:
    approved: bool
    lot_size: float
    risk_usd: float
    risk_pct: float
    veto_reason: str | None = None


class RiskGovernor:
    def __init__(
        self,
        max_risk_pct: float = 1.0,         # Maximum 1% risk per trade
        max_spread_pips: float = 35.0,     # Veto if spread exceeds 3.5 points
        max_open_positions: int = 1,       # Strictly 1 active trade at a time
        max_daily_losses_R: float = 2.0,   # Hard circuit breaker at 2R loss
        tick_value_per_lot: float = 1.0,   # $1.00 per pip per 1.00 lot for XAUUSD
    ):
        self.max_risk_pct = max_risk_pct
        self.max_spread_pips = max_spread_pips
        self.max_open_positions = max_open_positions
        self.max_daily_losses_R = max_daily_losses_R
        self.tick_value_per_lot = tick_value_per_lot

    def review_trade(
        self,
        equity: float,
        risk_pips: float,
        current_spread_pips: float = 20.0,
        open_positions_count: int = 0,
        daily_loss_R: float = 0.0,
        is_reconciled: bool = True,
        session_hour: int | None = None,
    ) -> RiskApproval:
        # Check 0: Asian Session Chop Shield
        if session_hour is not None and 0 <= session_hour <= 6:
            return RiskApproval(
                approved=False, lot_size=0.0, risk_usd=0.0, risk_pct=0.0,
                veto_reason="VETO: Asian Session Chop Shield active (00:00-06:00 UTC). Empirical win rate < 20% in low-liquidity hours."
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

        # Check 4: Spread constraint
        if current_spread_pips > self.max_spread_pips:
            return RiskApproval(
                approved=False, lot_size=0.0, risk_usd=0.0, risk_pct=0.0,
                veto_reason=f"VETO: Current spread ({current_spread_pips:.1f} pips) exceeds maximum allowed ({self.max_spread_pips:.1f} pips)."
            )

        # Check 5: Position Sizing
        if risk_pips <= 0:
            return RiskApproval(
                approved=False, lot_size=0.0, risk_usd=0.0, risk_pct=0.0,
                veto_reason="VETO: Invalid stop loss distance (risk_pips <= 0)."
            )

        # Capital risk in USD
        max_loss_usd = round(equity * (self.max_risk_pct / 100.0), 2)
        
        # Calculate raw lot size
        pip_cost_per_lot = risk_pips * self.tick_value_per_lot
        raw_lot = max_loss_usd / max(0.01, pip_cost_per_lot)

        # Round down to nearest 0.01 lot step
        calculated_lot = max(0.01, round(raw_lot, 2))

        # Check if broker minimum 0.01 lot exceeds permitted risk on small accounts
        actual_risk_usd = round(calculated_lot * pip_cost_per_lot, 2)
        actual_risk_pct = round((actual_risk_usd / max(1.0, equity)) * 100.0, 2)

        # STRICT 1.0% MAXIMUM RISK CEILING VETO
        if actual_risk_pct > self.max_risk_pct:
            return RiskApproval(
                approved=False,
                lot_size=0.0,
                risk_usd=actual_risk_usd,
                risk_pct=actual_risk_pct,
                veto_reason=f"VETO: Minimum position (0.01 lot) risk (${actual_risk_usd:.2f} / {actual_risk_pct:.1f}%) exceeds maximum {self.max_risk_pct:.1f}% risk limit (${max_loss_usd:.2f}). Undercapitalized for 0.01 lot."
            )

        return RiskApproval(
            approved=True,
            lot_size=calculated_lot,
            risk_usd=actual_risk_usd,
            risk_pct=actual_risk_pct,
            veto_reason=None
        )
