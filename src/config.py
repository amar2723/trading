from dataclasses import dataclass


MIN_SCORE = 3
BODY_MULTIPLIER = 2.0
LOOKBACK = 20
CLOSE_NEAR_RATIO = 0.35
RISK_REWARD = 1.5
REQUIRE_SWEEP = True
LIQUIDITY_LOOKBACK = 50
TP2_R_MULTIPLE = 3.0
TRAILING_STOP_R = 1.0


@dataclass(frozen=True)
class TradingConfig:
    symbol: str = "XAUUSD"
    timeframes: tuple[str, ...] = ("M1", "M5", "M15")
    bars: int = 3000
    atr_period: int = 14
    rsi_period: int = 14
    swing_lookback: int = 3
    sweep_lookback: int = 20
    displacement_atr_mult: float = 1.2
    fvg_min_atr_mult: float = 0.15
    risk_percent: float = 1.0
    min_rr: float = 1.2
    spread_points: float = 25.0
    slippage_points: float = 5.0
    commission_per_lot: float = 7.0
    point: float = 0.01
    contract_size: float = 100.0


DEFAULT_CONFIG = TradingConfig()
