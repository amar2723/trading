from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from config import BODY_MULTIPLIER, CLOSE_NEAR_RATIO, LIQUIDITY_LOOKBACK, LOOKBACK, MIN_SCORE, REQUIRE_SWEEP, RISK_REWARD, TP2_R_MULTIPLE, TRAILING_STOP_R


@dataclass
class PatternSignal:
    time: str
    signal: str
    entry: float | None
    sl: float | None
    tp1: float | None
    tp2: float | None
    liquidity_target: float | None
    trailing_stop: float | None
    confidence: float
    reason: str
    debug: dict

    def to_dict(self) -> dict:
        return asdict(self)


def detect_pattern(
    df: pd.DataFrame,
    rr: float = RISK_REWARD,
    use_closed_candle: bool = True,
    min_score: int = MIN_SCORE,
    body_multiplier: float = BODY_MULTIPLIER,
    close_near_ratio: float = CLOSE_NEAR_RATIO,
    require_sweep: bool = REQUIRE_SWEEP,
) -> PatternSignal:
    """
    Detect the relaxed two-candle sweep pattern with score-based rules.

    By default it uses the last two fully closed candles. MT5 includes the
    currently forming candle as the last row, so live scanning should compare
    df[-3] and df[-2], not df[-2] and df[-1].
    """
    required_rows = LOOKBACK + 3 if use_closed_candle else LOOKBACK + 2
    if len(df) < required_rows:
        return _none_signal()

    previous_idx = -3 if use_closed_candle else -2
    current_idx = -2 if use_closed_candle else -1

    previous = df.iloc[previous_idx]
    current = df.iloc[current_idx]
    recent = df.iloc[current_idx - LOOKBACK : current_idx]
    timestamp = str(current.get("timestamp", current.get("time", "")))

    current_body = abs(float(current["close"] - current["open"]))
    average_body = float((recent["close"] - recent["open"]).abs().mean())
    candle_range = max(float(current["high"] - current["low"]), 0.00001)

    bull_sweep = bool(current["low"] < previous["low"])
    bear_sweep = bool(current["high"] > previous["high"])
    close_above_high = bool(current["close"] > previous["high"])
    close_below_low = bool(current["close"] < previous["low"])
    strong_body = bool(current_body > body_multiplier * average_body) if average_body > 0 else False
    close_near_high = bool((current["high"] - current["close"]) <= candle_range * close_near_ratio)
    close_near_low = bool((current["close"] - current["low"]) <= candle_range * close_near_ratio)

    bull_conditions = {
        "Liquidity Sweep": bull_sweep,
        "Close Above Previous High": close_above_high,
        "Strong Body": strong_body,
        "Close Near High": close_near_high,
    }
    bear_conditions = {
        "Liquidity Sweep": bear_sweep,
        "Close Below Previous Low": close_below_low,
        "Strong Body": strong_body,
        "Close Near Low": close_near_low,
    }
    bull_score = sum(bull_conditions.values())
    bear_score = sum(bear_conditions.values())

    debug = {
        "timestamp": timestamp,
        "previous": _candle_dict(previous),
        "current": _candle_dict(current),
        "average_body_last_20": average_body,
        "bull_score": bull_score,
        "bear_score": bear_score,
        "bull_sweep": bull_sweep,
        "bear_sweep": bear_sweep,
        "close_above_high": close_above_high,
        "close_below_low": close_below_low,
        "strong_body": strong_body,
        "close_near_high": close_near_high,
        "close_near_low": close_near_low,
        "bull_conditions": bull_conditions,
        "bear_conditions": bear_conditions,
    }

    bull_allowed = bull_score >= min_score and bull_score >= bear_score and (bull_sweep or not require_sweep)
    bear_allowed = bear_score >= min_score and (bear_sweep or not require_sweep)

    if bull_allowed:
        entry = float(current["close"])
        sl = float(current["low"])
        risk = entry - sl
        tp1 = entry + risk * rr if risk > 0 else None
        liquidity = _upside_liquidity(df, current_idx, entry)
        tp2 = _tp2("BUY", entry, risk, liquidity)
        trailing_stop = entry - risk * TRAILING_STOP_R if risk > 0 else None
        return PatternSignal(timestamp, "BUY", entry, sl, tp1, tp2, liquidity, trailing_stop, _score_confidence(bull_score), _reason(bull_conditions), debug)

    if bear_allowed:
        entry = float(current["close"])
        sl = float(current["high"])
        risk = sl - entry
        tp1 = entry - risk * rr if risk > 0 else None
        liquidity = _downside_liquidity(df, current_idx, entry)
        tp2 = _tp2("SELL", entry, risk, liquidity)
        trailing_stop = entry + risk * TRAILING_STOP_R if risk > 0 else None
        return PatternSignal(timestamp, "SELL", entry, sl, tp1, tp2, liquidity, trailing_stop, _score_confidence(bear_score), _reason(bear_conditions), debug)

    return PatternSignal(timestamp, "NONE", None, None, None, None, None, None, 0.0, "", debug)


def print_debug(signal: PatternSignal) -> None:
    debug = signal.debug
    if not debug:
        return
    previous = debug["previous"]
    current = debug["current"]
    print("\nPrevious Candle:")
    print(f"open={previous['open']} high={previous['high']} low={previous['low']} close={previous['close']}")
    print("Current Candle:")
    print(f"open={current['open']} high={current['high']} low={current['low']} close={current['close']}")
    print(f"Average Body Size Last 20 Candles: {debug['average_body_last_20']:.5f}")
    print("\nBullish Conditions:")
    print(f"Bullish Sweep: {debug['bull_sweep']}")
    print(f"Close Above High: {debug['close_above_high']}")
    print(f"Strong Body: {debug['strong_body']}")
    print(f"Close Near High: {debug['close_near_high']}")
    print(f"Bullish Score = {debug['bull_score']}/4")
    print("\nBearish Conditions:")
    print(f"Bearish Sweep: {debug['bear_sweep']}")
    print(f"Close Below Low: {debug['close_below_low']}")
    print(f"Strong Body: {debug['strong_body']}")
    print(f"Close Near Low: {debug['close_near_low']}")
    print(f"Bearish Score = {debug['bear_score']}/4")
    print(f"\nSignal: {signal.signal}")
    if signal.reason:
        print(f"Reason: {signal.reason}")
    print(f"Confidence: {signal.confidence:.0f}%\n")


def _score_confidence(score: int) -> float:
    return {3: 75.0, 4: 90.0}.get(score, 0.0)


def _reason(conditions: dict[str, bool]) -> str:
    return ", ".join(name for name, passed in conditions.items() if passed)


def _candle_dict(candle: pd.Series) -> dict:
    return {
        "open": float(candle["open"]),
        "high": float(candle["high"]),
        "low": float(candle["low"]),
        "close": float(candle["close"]),
    }


def _upside_liquidity(df: pd.DataFrame, current_idx: int, entry: float) -> float | None:
    recent = df.iloc[max(0, current_idx - LIQUIDITY_LOOKBACK) : current_idx]
    highs = recent.loc[recent["high"] > entry, "high"].sort_values()
    if highs.empty:
        return None
    return float(highs.iloc[0])


def _downside_liquidity(df: pd.DataFrame, current_idx: int, entry: float) -> float | None:
    recent = df.iloc[max(0, current_idx - LIQUIDITY_LOOKBACK) : current_idx]
    lows = recent.loc[recent["low"] < entry, "low"].sort_values(ascending=False)
    if lows.empty:
        return None
    return float(lows.iloc[0])


def _tp2(signal: str, entry: float, risk: float, liquidity: float | None) -> float | None:
    if risk <= 0:
        return None
    fallback = entry + TP2_R_MULTIPLE * risk if signal == "BUY" else entry - TP2_R_MULTIPLE * risk
    if liquidity is None:
        return fallback
    if signal == "BUY" and liquidity > entry:
        return max(liquidity, fallback)
    if signal == "SELL" and liquidity < entry:
        return min(liquidity, fallback)
    return fallback


def _none_signal() -> PatternSignal:
    return PatternSignal("", "NONE", None, None, None, None, None, None, 0.0, "", {})
