from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from src.backtesting.portfolio import Portfolio, Trade
from src.backtesting.risk_manager import RiskConfig, RiskManager


logger = logging.getLogger(__name__)


@dataclass
class ExecutionConfig:
    spread_points: float = 25.0
    commission_per_lot: float = 7.0
    slippage_points: float = 5.0
    execution_delay: int = 1
    point: float = 0.01
    tp1_close_fraction: float = 0.5
    move_sl_to_breakeven: bool = True
    lookahead: int = 100


class Backtester:
    def __init__(self, execution: ExecutionConfig | None = None, risk: RiskConfig | None = None):
        self.execution = execution or ExecutionConfig()
        self.risk_config = risk or RiskConfig()
        self.risk = RiskManager(self.risk_config)
        self.portfolio = Portfolio(self.risk_config.initial_balance)

    def run(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        data = df.copy().reset_index(drop=True)
        time_col = "timestamp" if "timestamp" in data.columns else "time" if "time" in data.columns else None
        daily_pnl: dict[str, float] = {}

        for i, row in data.iterrows():
            timestamp = str(row[time_col]) if time_col else str(i)
            self.portfolio.snapshot(timestamp)
            signal = self._signal(row)
            if signal == "HOLD":
                continue

            day = timestamp[:10]
            if not self.risk.can_trade(self.portfolio.balance, self.portfolio.equity_peak, daily_pnl.get(day, 0.0), 0):
                continue

            entry_idx = min(i + self.execution.execution_delay, len(data) - 1)
            entry_row = data.iloc[entry_idx]
            plan = self._trade_plan(signal, row, entry_row)
            if plan is None:
                continue

            size = self.risk.position_size(self.portfolio.balance, plan["entry"], plan["sl"])
            if size <= 0:
                continue

            future = data.iloc[entry_idx + 1 : entry_idx + 1 + self.execution.lookahead]
            trade = self._simulate_trade(signal, plan, size, future, timestamp)
            self.portfolio.close_trade(trade)
            self.risk.record_trade(trade.pnl)
            daily_pnl[day] = daily_pnl.get(day, 0.0) + trade.pnl

        return pd.DataFrame(self.portfolio.trade_log()), pd.DataFrame(self.portfolio.curve)

    def _signal(self, row: pd.Series) -> str:
        if "predicted_signal" in row and row["predicted_signal"] in {"BUY", "SELL", "HOLD"}:
            return row["predicted_signal"]
        if "entry_type" in row and row["entry_type"] in {"BUY", "SELL"}:
            return row["entry_type"]
        if row.get("buy_probability", 0) > max(row.get("sell_probability", 0), row.get("hold_probability", 0)):
            return "BUY"
        if row.get("sell_probability", 0) > max(row.get("buy_probability", 0), row.get("hold_probability", 0)):
            return "SELL"
        return "HOLD"

    def _trade_plan(self, signal: str, signal_row: pd.Series, entry_row: pd.Series) -> dict | None:
        entry = float(entry_row["open"])
        entry += self._price_adjustment(signal)
        sl = signal_row.get("sl_price", signal_row.get("sl"))
        tp1 = signal_row.get("tp1")
        tp2 = signal_row.get("tp2")
        if pd.isna(sl) or pd.isna(tp1):
            return None
        if pd.isna(tp2):
            tp2 = tp1
        return {"entry": entry, "sl": float(sl), "tp1": float(tp1), "tp2": float(tp2)}

    def _simulate_trade(self, signal: str, plan: dict, size: float, future: pd.DataFrame, entry_time: str) -> Trade:
        entry = plan["entry"]
        sl = plan["sl"]
        tp1 = plan["tp1"]
        tp2 = plan["tp2"]
        costs = self._costs(size)
        remaining_size = size
        realized = -costs
        tp1_hit = False
        exit_price = entry
        exit_time = None
        result = "NO_EXIT"
        duration = 0
        active_sl = sl

        for duration, (_, candle) in enumerate(future.iterrows(), start=1):
            exit_time = str(candle.get("timestamp", candle.get("time", duration)))
            if signal == "BUY":
                sl_hit = candle["low"] <= active_sl
                tp1_now = candle["high"] >= tp1
                tp2_now = candle["high"] >= tp2
            else:
                sl_hit = candle["high"] >= active_sl
                tp1_now = candle["low"] <= tp1
                tp2_now = candle["low"] <= tp2

            if sl_hit and not tp1_hit:
                exit_price = active_sl
                realized += self._pnl(signal, entry, exit_price, remaining_size)
                result = "LOSS"
                break
            if tp1_now and not tp1_hit:
                closed_size = remaining_size * self.execution.tp1_close_fraction
                realized += self._pnl(signal, entry, tp1, closed_size)
                remaining_size -= closed_size
                tp1_hit = True
                if self.execution.move_sl_to_breakeven:
                    active_sl = entry
            if tp2_now:
                exit_price = tp2
                realized += self._pnl(signal, entry, exit_price, remaining_size)
                result = "WIN"
                break
            if sl_hit and tp1_hit:
                exit_price = active_sl
                realized += self._pnl(signal, entry, exit_price, remaining_size)
                result = "PARTIAL"
                break

        if result == "NO_EXIT" and tp1_hit:
            result = "PARTIAL"
            exit_price = tp1
        elif result == "NO_EXIT":
            exit_price = entry

        risk = abs(entry - sl)
        reward = abs(tp1 - entry)
        rr = reward / risk if risk > 0 else 0.0
        return Trade(entry_time, exit_time, signal, entry, exit_price, sl, tp1, tp2, size, duration, rr, result, realized, costs)

    def _price_adjustment(self, signal: str) -> float:
        raw = (self.execution.spread_points + self.execution.slippage_points) * self.execution.point
        return raw if signal == "BUY" else -raw

    def _costs(self, size: float) -> float:
        return self.execution.commission_per_lot * size

    def _pnl(self, signal: str, entry: float, exit_price: float, size: float) -> float:
        points = exit_price - entry if signal == "BUY" else entry - exit_price
        return points * size * self.risk_config.contract_size
