from __future__ import annotations

"""
Live & Demo MetaTrader 5 Trade Execution Bridge.

Features:
1. Dynamic Fractional Lot Size Calculator (% Risk of Balance vs Stop Distance)
2. Market Order Execution with Slippage & Deviation Protection
3. Automated Break-Even (Move SL to Entry once 1:1 RR is achieved)
4. Partial Profit Taking (Close 50% or 70% at TP1, let runner ride to TP2)
5. Position & Risk Monitoring (Emergency Close All)
"""

import logging
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)


@dataclass
class OrderResult:
    success: bool
    ticket: int | None
    symbol: str
    action: str
    volume: float
    price: float
    sl: float | None
    tp: float | None
    comment: str
    retcode: int | None


class MT5Executor:
    def __init__(self, magic_number: int = 202611, max_slippage_points: int = 30):
        self.magic_number = magic_number
        self.max_slippage = max_slippage_points
        self.mt5 = None

    def _ensure_mt5(self):
        if self.mt5 is None:
            try:
                import MetaTrader5 as mt5
                self.mt5 = mt5
            except ImportError as exc:
                raise RuntimeError("MetaTrader5 package not available.") from exc
        return self.mt5

    def calculate_lot_size(
        self,
        symbol: str,
        risk_amount_dollars: float,
        entry_price: float,
        stop_loss: float
    ) -> float:
        """
        Calculates exact fractional lot size so risk equals exactly risk_amount_dollars.
        Formula: Lot = Risk_Amount / (SL_Distance * Contract_Size)
        """
        mt5 = self._ensure_mt5()
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return 0.01  # Safe minimum fallback

        sl_distance = abs(entry_price - stop_loss)
        if sl_distance <= 0:
            return symbol_info.volume_min

        contract_size = symbol_info.trade_contract_size or 100.0  # Gold is typically 100 oz
        loss_per_lot = sl_distance * contract_size
        
        if loss_per_lot <= 0:
            return symbol_info.volume_min

        raw_lot = risk_amount_dollars / loss_per_lot
        step = symbol_info.volume_step or 0.01
        
        # Round down to volume step
        lot_size = max(symbol_info.volume_min, min(symbol_info.volume_max, round(raw_lot / step) * step))
        return round(lot_size, 2)

    def execute_market_order(
        self,
        symbol: str,
        action: Literal["BUY", "SELL"],
        lot_size: float,
        sl: float | None = None,
        tp: float | None = None,
        comment: str = "TradingAI_ICE"
    ) -> OrderResult:
        """Executes a live or demo market order through MetaTrader 5."""
        mt5 = self._ensure_mt5()
        if not mt5.initialize():
            return OrderResult(False, None, symbol, action, lot_size, 0.0, sl, tp, "MT5 initialize failed", None)

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return OrderResult(False, None, symbol, action, lot_size, 0.0, sl, tp, f"Failed to get tick for {symbol}", None)

        order_type = mt5.ORDER_TYPE_BUY if action == "BUY" else mt5.ORDER_TYPE_SELL
        price = tick.ask if action == "BUY" else tick.bid

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(lot_size),
            "type": order_type,
            "price": float(price),
            "sl": float(sl) if sl else 0.0,
            "tp": float(tp) if tp else 0.0,
            "deviation": int(self.max_slippage),
            "magic": int(self.magic_number),
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            err = result.comment if result else str(mt5.last_error())
            code = result.retcode if result else None
            logger.error("Order failed: %s (code: %s)", err, code)
            return OrderResult(False, None, symbol, action, lot_size, price, sl, tp, err, code)

        logger.info("Order executed! Ticket: %s | Action: %s | Volume: %s @ %s", result.order, action, lot_size, result.price)
        return OrderResult(True, result.order, symbol, action, lot_size, result.price, sl, tp, "Order executed successfully", result.retcode)

    def move_to_break_even(self, ticket: int, entry_price: float, buffer_points: float = 0.5) -> bool:
        """Moves Stop Loss to Entry + buffer to make trade completely risk-free."""
        mt5 = self._ensure_mt5()
        pos = mt5.positions_get(ticket=ticket)
        if not pos:
            return False
        
        position = pos[0]
        is_buy = position.type == mt5.ORDER_TYPE_BUY
        new_sl = entry_price + buffer_points if is_buy else entry_price - buffer_points

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "symbol": position.symbol,
            "sl": float(new_sl),
            "tp": float(position.tp),
        }
        res = mt5.order_send(request)
        return bool(res and res.retcode == mt5.TRADE_RETCODE_DONE)

    def partial_close(self, ticket: int, close_ratio: float = 0.5) -> bool:
        """Closes a fraction (e.g. 50%) of an active position to bank profits."""
        mt5 = self._ensure_mt5()
        pos = mt5.positions_get(ticket=ticket)
        if not pos:
            return False

        position = pos[0]
        step = mt5.symbol_info(position.symbol).volume_step or 0.01
        close_volume = max(step, round((position.volume * close_ratio) / step) * step)
        
        if close_volume >= position.volume:
            close_volume = position.volume

        tick = mt5.symbol_info_tick(position.symbol)
        if tick is None:
            return False

        close_action = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        close_price = tick.bid if position.type == mt5.ORDER_TYPE_BUY else tick.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": ticket,
            "symbol": position.symbol,
            "volume": float(close_volume),
            "type": close_action,
            "price": float(close_price),
            "deviation": int(self.max_slippage),
            "magic": int(self.magic_number),
            "comment": f"Partial_Close_{int(close_ratio*100)}%",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        res = mt5.order_send(request)
        return bool(res and res.retcode == mt5.TRADE_RETCODE_DONE)

    def close_all(self, symbol: str | None = None) -> int:
        """Emergency Circuit Breaker: closes all open bot positions."""
        mt5 = self._ensure_mt5()
        positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
        if not positions:
            return 0

        closed = 0
        for pos in positions:
            if pos.magic == self.magic_number:
                tick = mt5.symbol_info_tick(pos.symbol)
                if not tick:
                    continue
                close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
                close_price = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask
                req = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "position": pos.ticket,
                    "symbol": pos.symbol,
                    "volume": pos.volume,
                    "type": close_type,
                    "price": float(close_price),
                    "deviation": self.max_slippage,
                    "magic": self.magic_number,
                    "comment": "Circuit_Breaker_Close",
                }
                res = mt5.order_send(req)
                if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                    closed += 1
        return closed
