from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RiskConfig:
    initial_balance: float = 10_000.0
    fixed_lot: float = 0.1
    sizing_mode: str = "percent_risk"
    fixed_dollar_risk: float = 100.0
    percent_risk: float = 1.0
    contract_size: float = 100.0
    max_daily_loss: float = 300.0
    max_consecutive_losses: int = 4
    max_open_positions: int = 1
    max_drawdown_percent: float = 20.0


class RiskManager:
    def __init__(self, config: RiskConfig):
        self.config = config
        self.consecutive_losses = 0
        self.paused = False

    def position_size(self, balance: float, entry: float, stop: float) -> float:
        risk_points = abs(entry - stop)
        if risk_points <= 0:
            return 0.0
        if self.config.sizing_mode == "fixed_lot":
            return self.config.fixed_lot
        if self.config.sizing_mode == "fixed_dollar_risk":
            dollars = self.config.fixed_dollar_risk
        else:
            dollars = balance * (self.config.percent_risk / 100)
        return max(dollars / (risk_points * self.config.contract_size), 0.0)

    def can_trade(self, balance: float, equity_peak: float, daily_pnl: float, open_positions: int) -> bool:
        if self.paused:
            return False
        if open_positions >= self.config.max_open_positions:
            return False
        if daily_pnl <= -abs(self.config.max_daily_loss):
            return False
        if self.consecutive_losses >= self.config.max_consecutive_losses:
            return False
        if equity_peak > 0:
            drawdown_pct = (equity_peak - balance) / equity_peak * 100
            if drawdown_pct >= self.config.max_drawdown_percent:
                self.paused = True
                return False
        return True

    def record_trade(self, pnl: float) -> None:
        if pnl < 0:
            self.consecutive_losses += 1
        elif pnl > 0:
            self.consecutive_losses = 0
