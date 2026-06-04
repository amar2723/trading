from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class Trade:
    entry_time: str
    exit_time: str | None
    signal: str
    entry_price: float
    exit_price: float | None
    sl: float
    tp1: float
    tp2: float
    size: float
    duration: int
    rr_ratio: float
    result: str
    pnl: float
    costs: float


class Portfolio:
    def __init__(self, initial_balance: float):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.equity = initial_balance
        self.equity_peak = initial_balance
        self.closed_pnl = 0.0
        self.trades: list[Trade] = []
        self.curve: list[dict] = []

    def close_trade(self, trade: Trade) -> None:
        self.trades.append(trade)
        self.closed_pnl += trade.pnl
        self.balance += trade.pnl
        self.equity = self.balance
        self.equity_peak = max(self.equity_peak, self.equity)

    def snapshot(self, timestamp: str, floating_pnl: float = 0.0) -> None:
        equity = self.balance + floating_pnl
        self.equity = equity
        self.equity_peak = max(self.equity_peak, equity)
        self.curve.append(
            {
                "timestamp": timestamp,
                "balance": self.balance,
                "equity": equity,
                "floating_pnl": floating_pnl,
                "closed_pnl": self.closed_pnl,
                "drawdown": equity - self.equity_peak,
            }
        )

    def trade_log(self) -> list[dict]:
        return [asdict(trade) for trade in self.trades]

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        wins = [trade for trade in self.trades if trade.pnl > 0]
        return len(wins) / len(self.trades) * 100

    @property
    def loss_rate(self) -> float:
        if not self.trades:
            return 0.0
        losses = [trade for trade in self.trades if trade.pnl < 0]
        return len(losses) / len(self.trades) * 100
