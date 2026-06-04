from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class TradeSignal:
    symbol: str
    signal: str
    entry: float | None
    stop_loss: float | None
    tp1: float | None
    tp2: float | None
    tp3: float | None
    risk_reward: float | None
    confidence: float
    reason: list[str]
    timestamp: str

    def to_dict(self) -> dict:
        return asdict(self)
