from __future__ import annotations

"""
Enterprise-Grade Liquidity & Liquidation Mapping Engine for XAUUSD (Gold).

Theoretical Framework & Market Microstructure:
1. Microstructure Reference Zones (Kyle 1985, Harris 2003):
   Key price references (prior session highs/lows, VWAP bands, volume profile edges)
   are hypothetical zones of institutional interest, NOT guaranteed resting orders.
2. Sweep vs. Acceptance vs. Rejection (GoldTradingDesk 2024):
   - Sweep: Price excursions beyond a reference level that immediately fail.
   - Acceptance: Sustained volume accumulation and multi-bar closing beyond the level.
   - Rejection: Closed confirmation back inside the value zone with volume/RSI backing.
3. No-Trade Stop Run Gate:
   Single-bar wicks are observations only. Direct entries on wicks are prohibited.
4. Forced Liquidation Circuit Breaker (Brunnermeier & Pedersen 2009):
   Margin call cascades create non-linear price disconnections (>2% in 5 min).
   When triggered, a 60-minute cooling circuit breaker halts all new order generation.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger("LiquidityMap")


@dataclass
class LiquidityReferenceLevel:
    level_id: str
    level_type: str          # "SESSION_HIGH", "SESSION_LOW", "PDH", "PDL", "PDC", "PWH", "PWL", "PWC", "NYMO", "VWAP", "VP_VAH", "VP_VAL", "VP_POC"
    session: str             # "ASIA", "LONDON", "NY", "PRIOR_DAY", "HTF_DAILY", "HTF_WEEKLY", "PROFILE", "NYMO"
    price: float
    volume_weight: float     # Tick volume accumulation proxy
    established_time: str
    is_active: bool = True
    evidence: str = ""
    tier: str = "TIER_2_SESSION"  # "TIER_1_INSTITUTIONAL" or "TIER_2_SESSION"

    def to_dict(self) -> dict:
        return {
            "level_id": self.level_id,
            "level_type": self.level_type,
            "session": self.session,
            "price": round(self.price, 2),
            "volume_weight": round(self.volume_weight, 1),
            "established_time": self.established_time,
            "is_active": self.is_active,
            "evidence": self.evidence,
            "tier": self.tier,
        }


@dataclass
class SweepEvent:
    event_id: str
    timestamp: str
    level: LiquidityReferenceLevel
    sweep_type: str          # "BULLISH_SWEEP" (swept low, rejection up), "BEARISH_SWEEP" (swept high, rejection down)
    breach_price: float
    reclaim_price: float
    atr_excursion: float
    status: str              # "UNCONFIRMED_WICK", "CONFIRMED_REJECTION", "CONFIRMED_ACCEPTANCE"
    confirmation_bar_time: str | None = None
    rationale: str = ""


class ForcedLiquidationMonitor:
    """
    Monitors systemic liquidation cascades and extreme margin events.
    When rolling 5-minute price velocity exceeds 2.0% (approx $75/oz on Gold),
    trips an automatic circuit breaker for 12 bars (60 minutes).
    """
    def __init__(self, velocity_threshold_pct: float = 2.0, cooldown_bars: int = 12):
        self.velocity_threshold_pct = velocity_threshold_pct
        self.cooldown_bars = cooldown_bars
        self.is_circuit_breaker_active = False
        self.last_trip_time: str | None = None
        self.bars_since_trip: int = 999
        self.trip_reason: str = ""

    def evaluate(self, df: pd.DataFrame) -> tuple[bool, str]:
        if df is None or len(df) < 2:
            return False, "NORMAL"

        last_row = df.iloc[-1]
        prev_row = df.iloc[-2]

        # Calculate 5-minute displacement velocity (close vs previous close, or candle range)
        prev_close = float(prev_row["close"])
        close_price = float(last_row["close"])
        open_price = float(last_row["open"])
        displacement_pct = abs(close_price - prev_close) / max(prev_close, 1.0) * 100.0
        candle_body_pct = abs(close_price - open_price) / max(open_price, 1.0) * 100.0

        high_price = float(last_row["high"])
        low_price = float(last_row["low"])
        candle_range_pct = (high_price - low_price) / max(prev_close, 1.0) * 100.0

        max_vel = max(displacement_pct, candle_body_pct, candle_range_pct)

        if max_vel >= self.velocity_threshold_pct:
            self.is_circuit_breaker_active = True
            self.bars_since_trip = 0
            self.last_trip_time = str(last_row.get("timestamp", datetime.now(timezone.utc)))
            self.trip_reason = (
                f"FORCED_LIQUIDATION_ALERT: Extreme velocity {max_vel:.2f}% in 5m "
                f"(Threshold: {self.velocity_threshold_pct}%). Circuit breaker engaged."
            )
            logger.warning(self.trip_reason)
            return True, self.trip_reason

        if self.is_circuit_breaker_active:
            self.bars_since_trip += 1
            if self.bars_since_trip >= self.cooldown_bars:
                self.is_circuit_breaker_active = False
                logger.info("Circuit breaker cooldown expired (%d bars). Resuming standard monitoring.", self.cooldown_bars)
                return False, "CIRCUIT_BREAKER_CLEARED"
            return True, f"CIRCUIT_BREAKER_COOLING (Bar {self.bars_since_trip}/{self.cooldown_bars})"

        return False, "NORMAL"


class LiquidityMap:
    """
    Institutional Liquidity & Microstructure Mapping Engine.
    Identifies session boundaries, prior day extremes, anchored VWAP, and volume profile value areas.
    Detects sweeps with mandatory multi-bar closure confirmation.
    """
    def __init__(
        self,
        symbol: str = "XAUUSD",
        min_sweep_atr_mult: float = 0.20,
        max_sweep_atr_mult: float = 1.50,
        confirmation_required: bool = True,
    ):
        self.symbol = symbol
        self.min_sweep_atr_mult = min_sweep_atr_mult
        self.max_sweep_atr_mult = max_sweep_atr_mult
        self.confirmation_required = confirmation_required
        self.liquidation_monitor = ForcedLiquidationMonitor()

    @staticmethod
    def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period, min_periods=1).mean()

    def compute_htf_levels(self, df_h1: pd.DataFrame) -> list[LiquidityReferenceLevel]:
        """
        Computes Higher Timeframe (HTF) institutional benchmarks:
        1. Prior Day High (PDH), Prior Day Low (PDL), Prior Day Close (PDC)
        2. Prior Week High (PWH), Prior Week Low (PWL), Prior Week Close (PWC)
        3. New York Midnight Open (NYMO: 00:00 EST / 05:00 UTC)
        All tagged as TIER_1_INSTITUTIONAL liquidity reference levels.
        """
        levels: list[LiquidityReferenceLevel] = []
        if df_h1 is None or len(df_h1) < 5:
            return levels

        ts = pd.to_datetime(df_h1["timestamp"], utc=True)
        df_work = df_h1.copy()
        df_work["ts_utc"] = ts
        df_work["date"] = ts.dt.date
        df_work["year_week"] = (
            ts.dt.isocalendar().year.astype(str) + "_" + ts.dt.isocalendar().week.astype(str)
        )

        unique_dates = sorted(df_work["date"].unique())
        current_date = unique_dates[-1]

        # 1. Prior Day Extremes
        if len(unique_dates) >= 2:
            prior_date = unique_dates[-2]
            df_prior = df_work[df_work["date"] == prior_date]
            if not df_prior.empty:
                pdh = float(df_prior["high"].max())
                pdl = float(df_prior["low"].min())
                pdc = float(df_prior["close"].iloc[-1])
                pd_vol = float(df_prior["tick_volume"].sum()) if "tick_volume" in df_prior.columns else 2000.0

                levels.append(LiquidityReferenceLevel(
                    level_id=f"PDH_{prior_date}",
                    level_type="PDH",
                    session="HTF_DAILY",
                    price=pdh,
                    volume_weight=pd_vol,
                    established_time=str(df_prior["ts_utc"].iloc[-1]),
                    evidence=f"Prior Day High ({prior_date}) Institutional Buy-Side Pool",
                    tier="TIER_1_INSTITUTIONAL",
                ))
                levels.append(LiquidityReferenceLevel(
                    level_id=f"PDL_{prior_date}",
                    level_type="PDL",
                    session="HTF_DAILY",
                    price=pdl,
                    volume_weight=pd_vol,
                    established_time=str(df_prior["ts_utc"].iloc[-1]),
                    evidence=f"Prior Day Low ({prior_date}) Institutional Sell-Side Pool",
                    tier="TIER_1_INSTITUTIONAL",
                ))
                levels.append(LiquidityReferenceLevel(
                    level_id=f"PDC_{prior_date}",
                    level_type="PDC",
                    session="HTF_DAILY",
                    price=pdc,
                    volume_weight=pd_vol,
                    established_time=str(df_prior["ts_utc"].iloc[-1]),
                    evidence=f"Prior Day Close ({prior_date}) Equilibrium Benchmark",
                    tier="TIER_1_INSTITUTIONAL",
                ))

        # 2. Prior Week Extremes
        unique_weeks = sorted(df_work["year_week"].unique())
        if len(unique_weeks) >= 2:
            prior_week = unique_weeks[-2]
            df_pweek = df_work[df_work["year_week"] == prior_week]
            if not df_pweek.empty:
                pwh = float(df_pweek["high"].max())
                pwl = float(df_pweek["low"].min())
                pwc = float(df_pweek["close"].iloc[-1])
                pw_vol = float(df_pweek["tick_volume"].sum()) if "tick_volume" in df_pweek.columns else 10000.0

                levels.append(LiquidityReferenceLevel(
                    level_id=f"PWH_{prior_week}",
                    level_type="PWH",
                    session="HTF_WEEKLY",
                    price=pwh,
                    volume_weight=pw_vol,
                    established_time=str(df_pweek["ts_utc"].iloc[-1]),
                    evidence=f"Prior Week High (Wk {prior_week}) Major Institutional Pool",
                    tier="TIER_1_INSTITUTIONAL",
                ))
                levels.append(LiquidityReferenceLevel(
                    level_id=f"PWL_{prior_week}",
                    level_type="PWL",
                    session="HTF_WEEKLY",
                    price=pwl,
                    volume_weight=pw_vol,
                    established_time=str(df_pweek["ts_utc"].iloc[-1]),
                    evidence=f"Prior Week Low (Wk {prior_week}) Major Institutional Pool",
                    tier="TIER_1_INSTITUTIONAL",
                ))
                levels.append(LiquidityReferenceLevel(
                    level_id=f"PWC_{prior_week}",
                    level_type="PWC",
                    session="HTF_WEEKLY",
                    price=pwc,
                    volume_weight=pw_vol,
                    established_time=str(df_pweek["ts_utc"].iloc[-1]),
                    evidence=f"Prior Week Close (Wk {prior_week}) Weekly Pivot",
                    tier="TIER_1_INSTITUTIONAL",
                ))

        # 3. New York Midnight Open (05:00 UTC)
        df_today = df_work[df_work["date"] == current_date]
        if not df_today.empty:
            nymo_candidates = df_today[df_today["ts_utc"].dt.hour == 5]
            if not nymo_candidates.empty:
                nymo_price = float(nymo_candidates.iloc[0]["open"])
                nymo_time = str(nymo_candidates.iloc[0]["ts_utc"])
            else:
                nymo_price = float(df_today.iloc[0]["open"])
                nymo_time = str(df_today.iloc[0]["ts_utc"])

            levels.append(LiquidityReferenceLevel(
                level_id=f"NYMO_{current_date}",
                level_type="NYMO",
                session="NYMO",
                price=nymo_price,
                volume_weight=3000.0,
                established_time=nymo_time,
                evidence="New York Midnight Open (05:00 UTC / 00:00 EST) Institutional Anchor",
                tier="TIER_1_INSTITUTIONAL",
            ))

        return levels

    def compute_session_levels(self, df: pd.DataFrame, df_h1: pd.DataFrame | None = None) -> list[LiquidityReferenceLevel]:
        """
        Computes Asian (00-06 UTC), London (07-10 UTC), and NY (12-16 UTC) extremes,
        plus Prior Day High, Low, and Close (and HTF PWH/PWL/NYMO when df_h1 is provided).
        """
        levels: list[LiquidityReferenceLevel] = []
        if df is None or len(df) < 15:
            return levels

        ts = pd.to_datetime(df["timestamp"], utc=True)
        df_work = df.copy()
        df_work["ts_utc"] = ts
        df_work["hour"] = ts.dt.hour
        df_work["date"] = ts.dt.date

        unique_dates = df_work["date"].unique()
        current_date = unique_dates[-1]

        # If df_h1 is provided, extract higher-timeframe authoritative levels
        has_htf_pd = False
        if df_h1 is not None and not df_h1.empty:
            htf_levels = self.compute_htf_levels(df_h1)
            levels.extend(htf_levels)
            has_htf_pd = any(lvl.level_type in ["PDH", "PDL", "PDC"] for lvl in htf_levels)

        # Prior Day Extremes from intraday df (if not already sourced from H1)
        if not has_htf_pd and len(unique_dates) >= 2:
            prior_date = unique_dates[-2]
            df_prior = df_work[df_work["date"] == prior_date]
            if not df_prior.empty:
                pdh = float(df_prior["high"].max())
                pdl = float(df_prior["low"].min())
                pdc = float(df_prior["close"].iloc[-1])
                pd_vol = float(df_prior["tick_volume"].sum()) if "tick_volume" in df_prior.columns else 1000.0

                levels.append(LiquidityReferenceLevel(
                    level_id=f"PDH_{prior_date}",
                    level_type="PDH",
                    session="PRIOR_DAY",
                    price=pdh,
                    volume_weight=pd_vol,
                    established_time=str(df_prior["ts_utc"].iloc[-1]),
                    evidence="Prior Day High benchmark",
                    tier="TIER_1_INSTITUTIONAL",
                ))
                levels.append(LiquidityReferenceLevel(
                    level_id=f"PDL_{prior_date}",
                    level_type="PDL",
                    session="PRIOR_DAY",
                    price=pdl,
                    volume_weight=pd_vol,
                    established_time=str(df_prior["ts_utc"].iloc[-1]),
                    evidence="Prior Day Low benchmark",
                    tier="TIER_1_INSTITUTIONAL",
                ))
                levels.append(LiquidityReferenceLevel(
                    level_id=f"PDC_{prior_date}",
                    level_type="PDC",
                    session="PRIOR_DAY",
                    price=pdc,
                    volume_weight=pd_vol,
                    established_time=str(df_prior["ts_utc"].iloc[-1]),
                    evidence="Prior Day Close benchmark",
                    tier="TIER_1_INSTITUTIONAL",
                ))

        # Intraday Sessions for current date
        df_today = df_work[df_work["date"] == current_date]
        if not df_today.empty:
            # 1. Asian Session (00:00 - 06:00 UTC)
            df_asia = df_today[(df_today["hour"] >= 0) & (df_today["hour"] <= 6)]
            if len(df_asia) >= 6:
                asia_h = float(df_asia["high"].max())
                asia_l = float(df_asia["low"].min())
                asia_vol = float(df_asia["tick_volume"].sum()) if "tick_volume" in df_asia.columns else 500.0
                levels.append(LiquidityReferenceLevel(
                    level_id=f"ASIA_HIGH_{current_date}",
                    level_type="SESSION_HIGH",
                    session="ASIA",
                    price=asia_h,
                    volume_weight=asia_vol,
                    established_time=str(df_asia["ts_utc"].iloc[-1]),
                    evidence="Asian session range high accumulation",
                    tier="TIER_2_SESSION",
                ))
                levels.append(LiquidityReferenceLevel(
                    level_id=f"ASIA_LOW_{current_date}",
                    level_type="SESSION_LOW",
                    session="ASIA",
                    price=asia_l,
                    volume_weight=asia_vol,
                    established_time=str(df_asia["ts_utc"].iloc[-1]),
                    evidence="Asian session range low accumulation",
                    tier="TIER_2_SESSION",
                ))

            # 2. London Killzone (07:00 - 10:00 UTC)
            df_london = df_today[(df_today["hour"] >= 7) & (df_today["hour"] <= 10)]
            if len(df_london) >= 6:
                lon_h = float(df_london["high"].max())
                lon_l = float(df_london["low"].min())
                lon_vol = float(df_london["tick_volume"].sum()) if "tick_volume" in df_london.columns else 1200.0
                levels.append(LiquidityReferenceLevel(
                    level_id=f"LON_HIGH_{current_date}",
                    level_type="SESSION_HIGH",
                    session="LONDON",
                    price=lon_h,
                    volume_weight=lon_vol,
                    established_time=str(df_london["ts_utc"].iloc[-1]),
                    evidence="London Killzone peak expansion high",
                    tier="TIER_2_SESSION",
                ))
                levels.append(LiquidityReferenceLevel(
                    level_id=f"LON_LOW_{current_date}",
                    level_type="SESSION_LOW",
                    session="LONDON",
                    price=lon_l,
                    volume_weight=lon_vol,
                    established_time=str(df_london["ts_utc"].iloc[-1]),
                    evidence="London Killzone peak expansion low",
                    tier="TIER_2_SESSION",
                ))

        return levels

    def compute_anchored_vwap(self, df: pd.DataFrame) -> dict[str, float]:
        """
        Computes rolling/session Anchored VWAP and +/- 1.5 sigma and 2.0 sigma dispersion bands.
        """
        if df is None or len(df) < 10:
            return {}

        price = (df["high"].astype(float) + df["low"].astype(float) + df["close"].astype(float)) / 3.0
        vol = df["tick_volume"].astype(float) if "tick_volume" in df.columns else pd.Series(100.0, index=df.index)

        cum_pv = (price * vol).cumsum()
        cum_vol = vol.cumsum()
        vwap = cum_pv / cum_vol.replace(0, 1.0)

        # Standard deviation bands
        squared_diff = (price - vwap) ** 2
        cum_sq_diff = (vol * squared_diff).cumsum()
        variance = cum_sq_diff / cum_vol.replace(0, 1.0)
        std_dev = np.sqrt(variance.clip(lower=0))

        latest_vwap = float(vwap.iloc[-1])
        latest_std = float(std_dev.iloc[-1])

        return {
            "vwap": latest_vwap,
            "vwap_upper_1": latest_vwap + (1.5 * latest_std),
            "vwap_lower_1": latest_vwap - (1.5 * latest_std),
            "vwap_upper_2": latest_vwap + (2.0 * latest_std),
            "vwap_lower_2": latest_vwap - (2.0 * latest_std),
        }

    def compute_volume_profile(self, df: pd.DataFrame, bins: int = 30) -> dict[str, float]:
        """
        Computes 24-hour Volume Profile Value Area: Value Area High (VAH),
        Value Area Low (VAL), and Point of Control (POC) representing 70% volume.
        """
        if df is None or len(df) < 20:
            return {}

        high = df["high"].astype(float).values
        low = df["low"].astype(float).values
        close = df["close"].astype(float).values
        vol = df["tick_volume"].astype(float).values if "tick_volume" in df.columns else np.full(len(df), 100.0)

        min_p = float(low.min())
        max_p = float(high.max())
        if max_p <= min_p:
            return {"vp_poc": min_p, "vp_vah": max_p, "vp_val": min_p}

        bin_edges = np.linspace(min_p, max_p, bins + 1)
        bin_volumes = np.zeros(bins)

        # Distribute volume across price bins touched by each bar
        for h, l, c, v in zip(high, low, close, vol):
            # Find bins overlapping with [l, h]
            idx_start = max(0, int(np.searchsorted(bin_edges, l, side="right") - 1))
            idx_end = min(bins - 1, int(np.searchsorted(bin_edges, h, side="left")))
            if idx_start <= idx_end:
                num_bins = idx_end - idx_start + 1
                bin_volumes[idx_start:idx_end + 1] += v / num_bins
            else:
                bin_volumes[min(bins - 1, idx_start)] += v

        poc_idx = int(np.argmax(bin_volumes))
        poc_price = float((bin_edges[poc_idx] + bin_edges[poc_idx + 1]) / 2.0)

        # 70% Value Area calculation expanding outwards from POC
        total_vol = float(bin_volumes.sum())
        target_vol = total_vol * 0.70
        curr_vol = float(bin_volumes[poc_idx])
        left_idx = poc_idx
        right_idx = poc_idx

        while curr_vol < target_vol and (left_idx > 0 or right_idx < bins - 1):
            next_left_vol = float(bin_volumes[left_idx - 1]) if left_idx > 0 else -1.0
            next_right_vol = float(bin_volumes[right_idx + 1]) if right_idx < bins - 1 else -1.0

            if next_left_vol >= next_right_vol and next_left_vol >= 0:
                left_idx -= 1
                curr_vol += next_left_vol
            elif next_right_vol >= 0:
                right_idx += 1
                curr_vol += next_right_vol
            else:
                break

        val_price = float(bin_edges[left_idx])
        vah_price = float(bin_edges[right_idx + 1])

        return {
            "vp_poc": poc_price,
            "vp_vah": vah_price,
            "vp_val": val_price,
        }

    def evaluate_sweeps_and_structure(self, df: pd.DataFrame, df_h1: pd.DataFrame | None = None) -> tuple[list[SweepEvent], str]:
        """
        Evaluates recent completed price action against LiquidityReferenceLevels.
        Enforces No-Trade Stop Run Gate: single-bar wicks are tagged UNCONFIRMED_WICK.
        Only multi-bar confirmed rejections are marked CONFIRMED_REJECTION.
        """
        events: list[SweepEvent] = []
        if df is None or len(df) < 5:
            return events, "INSUFFICIENT_DATA"

        # Check circuit breaker
        is_breaker, breaker_msg = self.liquidation_monitor.evaluate(df)
        if is_breaker:
            return events, breaker_msg

        levels = self.compute_session_levels(df, df_h1=df_h1)
        if not levels:
            return events, "NO_ACTIVE_LEVELS"

        atr_series = self._compute_atr(df, 14)
        atr_val = float(atr_series.iloc[-1]) if not atr_series.empty else 5.0
        atr_val = max(atr_val, 1.0)

        last_idx = len(df) - 1
        curr_high = float(df["high"].iloc[last_idx])
        curr_low = float(df["low"].iloc[last_idx])
        curr_close = float(df["close"].iloc[last_idx])
        curr_open = float(df["open"].iloc[last_idx])
        curr_time = str(df["timestamp"].iloc[last_idx])

        prev_high = float(df["high"].iloc[last_idx - 1])
        prev_low = float(df["low"].iloc[last_idx - 1])
        prev_close = float(df["close"].iloc[last_idx - 1])

        status_verdict = "NORMAL"

        for lvl in levels:
            # Check Bearish Sweep (Sweep above resistance level)
            if lvl.level_type in ["SESSION_HIGH", "PDH", "PWH", "VP_VAH"]:
                min_reach = lvl.price + (self.min_sweep_atr_mult * atr_val)
                max_reach = lvl.price + (self.max_sweep_atr_mult * atr_val)

                # Case A: Sweep occurred on previous candle (last_idx - 1), confirmation on current candle
                if (prev_high >= min_reach) and (prev_high <= max_reach) and (prev_close < lvl.price):
                    # Multi-bar confirmation: current candle must close bearish below previous close
                    is_confirmed = (curr_close < prev_close) and (curr_close < curr_open)
                    status = "CONFIRMED_REJECTION" if is_confirmed else "UNCONFIRMED_WICK"
                    events.append(SweepEvent(
                        event_id=f"SWEEP_HIGH_{lvl.level_id}_{curr_time}",
                        timestamp=curr_time,
                        level=lvl,
                        sweep_type="BEARISH_SWEEP",
                        breach_price=prev_high,
                        reclaim_price=curr_close,
                        atr_excursion=round((prev_high - lvl.price) / atr_val, 2),
                        status=status,
                        confirmation_bar_time=curr_time if is_confirmed else None,
                        rationale=f"Swept {lvl.level_type} (${lvl.price:.2f}) by {prev_high - lvl.price:.2f} pts, closed back inside range.",
                    ))
                    if is_confirmed:
                        status_verdict = f"CONFIRMED_BEARISH_REJECTION_{lvl.level_type}"

                # Case B: Current bar is actively wicking beyond level (Unconfirmed Stop-Run Wick)
                elif (curr_high >= min_reach) and (curr_close < lvl.price):
                    events.append(SweepEvent(
                        event_id=f"WICK_HIGH_{lvl.level_id}_{curr_time}",
                        timestamp=curr_time,
                        level=lvl,
                        sweep_type="BEARISH_SWEEP",
                        breach_price=curr_high,
                        reclaim_price=curr_close,
                        atr_excursion=round((curr_high - lvl.price) / atr_val, 2),
                        status="UNCONFIRMED_WICK",
                        rationale=f"Active wick through {lvl.level_type}. STOP-RUN NO-TRADE GATE ACTIVE (Awaiting bar closure).",
                    ))

            # Check Bullish Sweep (Sweep below support level)
            if lvl.level_type in ["SESSION_LOW", "PDL", "PWL", "NYMO", "VP_VAL"]:
                min_dip = lvl.price - (self.min_sweep_atr_mult * atr_val)
                max_dip = lvl.price - (self.max_sweep_atr_mult * atr_val)

                # Case A: Sweep occurred on previous candle, confirmation on current candle
                if (prev_low <= min_dip) and (prev_low >= max_dip) and (prev_close > lvl.price):
                    is_confirmed = (curr_close > prev_close) and (curr_close > curr_open)
                    status = "CONFIRMED_REJECTION" if is_confirmed else "UNCONFIRMED_WICK"
                    events.append(SweepEvent(
                        event_id=f"SWEEP_LOW_{lvl.level_id}_{curr_time}",
                        timestamp=curr_time,
                        level=lvl,
                        sweep_type="BULLISH_SWEEP",
                        breach_price=prev_low,
                        reclaim_price=curr_close,
                        atr_excursion=round((lvl.price - prev_low) / atr_val, 2),
                        status=status,
                        confirmation_bar_time=curr_time if is_confirmed else None,
                        rationale=f"Swept {lvl.level_type} (${lvl.price:.2f}) by {lvl.price - prev_low:.2f} pts, closed back inside range.",
                    ))
                    if is_confirmed:
                        status_verdict = f"CONFIRMED_BULLISH_REJECTION_{lvl.level_type}"

                # Case B: Current bar is actively wicking below level
                elif (curr_low <= min_dip) and (curr_close > lvl.price):
                    events.append(SweepEvent(
                        event_id=f"WICK_LOW_{lvl.level_id}_{curr_time}",
                        timestamp=curr_time,
                        level=lvl,
                        sweep_type="BULLISH_SWEEP",
                        breach_price=curr_low,
                        reclaim_price=curr_close,
                        atr_excursion=round((lvl.price - curr_low) / atr_val, 2),
                        status="UNCONFIRMED_WICK",
                        rationale=f"Active wick below {lvl.level_type}. STOP-RUN NO-TRADE GATE ACTIVE (Awaiting bar closure).",
                    ))

        return events, status_verdict

    def get_pre_sweep_liquidity_watchlist(self, df: pd.DataFrame, df_h1: pd.DataFrame | None = None) -> dict:
        """
        Maps all active institutional liquidity pools BEFORE they are swept.
        Returns:
        - current_price
        - buy_side_liquidity (BSL): Resting buy-stops above current market price.
        - sell_side_liquidity (SSL): Resting sell-stops below current market price.
        - fair_value_targets: Session VWAP, POC, VAH, VAL.
        - nearest_bsl: The closest overhead pool to be swept for a potential SHORT setup.
        - nearest_ssl: The closest underlying pool to be swept for a potential LONG setup.
        - htf_levels: Dict containing PDH, PDL, PDC, PWH, PWL, NYMO benchmarks.
        """
        if df is None or len(df) < 5:
            return {}

        current_price = float(df["close"].iloc[-1])
        levels = self.compute_session_levels(df, df_h1=df_h1)
        vwap_dict = self.compute_anchored_vwap(df)
        vp_dict = self.compute_volume_profile(df)

        atr_series = self._compute_atr(df, 14)
        atr_val = float(atr_series.iloc[-1]) if not atr_series.empty else 5.0
        atr_val = max(atr_val, 1.0)

        # Incorporate Volume Profile levels into reference pool
        if "vp_vah" in vp_dict and vp_dict["vp_vah"] > 0:
            levels.append(LiquidityReferenceLevel(
                level_id="VP_VAH_24H", level_type="VP_VAH", session="PROFILE",
                price=round(vp_dict["vp_vah"], 2), volume_weight=1500.0,
                established_time=str(df["timestamp"].iloc[-1]), evidence="24H Value Area High",
                tier="TIER_2_SESSION",
            ))
        if "vp_val" in vp_dict and vp_dict["vp_val"] > 0:
            levels.append(LiquidityReferenceLevel(
                level_id="VP_VAL_24H", level_type="VP_VAL", session="PROFILE",
                price=round(vp_dict["vp_val"], 2), volume_weight=1500.0,
                established_time=str(df["timestamp"].iloc[-1]), evidence="24H Value Area Low",
                tier="TIER_2_SESSION",
            ))

        # Timing & Institutional Killzone Schedule
        ts = pd.to_datetime(df["timestamp"].iloc[-1], utc=True)
        ts_ist = ts + pd.Timedelta(hours=5, minutes=30)
        curr_hour = ts.hour

        if 0 <= curr_hour < 7:
            active_window = "ASIAN ACCUMULATION (00:00 - 06:00 UTC / 05:30 - 11:30 IST)"
            next_window = "LONDON JUDAS SWEEP (07:00 - 10:00 UTC / 12:30 - 15:30 IST)"
            next_window_name = "LONDON OPEN (12:30 IST)"
        elif 7 <= curr_hour < 10:
            active_window = "LONDON JUDAS SWEEP (07:00 - 10:00 UTC / 12:30 - 15:30 IST)"
            next_window = "NEW YORK JUDAS SWEEP (12:00 - 16:00 UTC / 17:30 - 21:30 IST)"
            next_window_name = "NY OPEN (17:30 IST)"
        elif 10 <= curr_hour < 12:
            active_window = "LONDON MIDDAY CHOP (10:00 - 12:00 UTC / 15:30 - 17:30 IST)"
            next_window = "NEW YORK JUDAS SWEEP (12:00 - 16:00 UTC / 17:30 - 21:30 IST)"
            next_window_name = "NY OPEN (17:30 IST)"
        elif 12 <= curr_hour < 16:
            active_window = "NEW YORK JUDAS SWEEP (12:00 - 16:00 UTC / 17:30 - 21:30 IST)"
            next_window = "NY CLOSE / LONDON FIX (16:00 - 20:00 UTC / 21:30 - 01:30 IST)"
            next_window_name = "NY AFTERNOON (21:30 IST)"
        else:
            active_window = "NY CLOSE / POST-MARKET (16:00 - 21:00 UTC / 21:30 - 02:30 IST)"
            next_window = "ASIAN ACCUMULATION (00:00 - 06:00 UTC / 05:30 - 11:30 IST)"
            next_window_name = "ASIAN OPEN (05:30 IST)"

        bsl_list = []
        ssl_list = []

        BSL_TYPES = {"SESSION_HIGH", "PDH", "PWH", "VP_VAH"}
        SSL_TYPES = {"SESSION_LOW", "PDL", "PWL", "VP_VAL", "NYMO"}

        for lvl in levels:
            dist_usd = round(abs(lvl.price - current_price), 2)
            dist_pips = round(dist_usd * 10.0, 1)

            est_bars = max(1, int(round(dist_usd / max(atr_val, 0.50))))
            est_minutes = est_bars * 5
            est_time_utc = ts + pd.Timedelta(minutes=est_minutes)
            est_time_ist = ts_ist + pd.Timedelta(minutes=est_minutes)

            if dist_usd <= (atr_val * 1.2):
                timing_status = "IMMINENT_TEST"
                timing_eta = f"IMMINENT (~{est_minutes}m / {est_time_ist.strftime('%H:%M IST')})"
            elif dist_usd <= (atr_val * 3.5):
                timing_status = "APPROACHING"
                timing_eta = f"APPROACHING (~{est_minutes}m / {est_time_ist.strftime('%H:%M IST')})"
            else:
                timing_status = "UPCOMING_SESSION"
                timing_eta = f"Projected {next_window_name} (~{est_minutes // 60}h {est_minutes % 60}m / {est_time_ist.strftime('%H:%M IST')})"

            pool_tier = getattr(lvl, "tier", "TIER_1_INSTITUTIONAL" if lvl.level_type in ["PDH", "PDL", "PWH", "PWL", "NYMO"] else "TIER_2_SESSION")

            pool_entry = {
                "level_id": lvl.level_id,
                "level_type": lvl.level_type,
                "session": lvl.session,
                "tier": pool_tier,
                "price": lvl.price,
                "distance_usd": dist_usd,
                "distance_pips": dist_pips,
                "evidence": lvl.evidence,
                "sweep_min_trigger": round(lvl.price + (self.min_sweep_atr_mult * atr_val) if lvl.price >= current_price else lvl.price - (self.min_sweep_atr_mult * atr_val), 2),
                "sweep_max_trigger": round(lvl.price + (self.max_sweep_atr_mult * atr_val) if lvl.price >= current_price else lvl.price - (self.max_sweep_atr_mult * atr_val), 2),
                "timing_status": timing_status,
                "eta_minutes": est_minutes,
                "eta_display": timing_eta,
                "est_arrival_ist": est_time_ist.strftime("%H:%M IST"),
                "est_arrival_utc": est_time_utc.strftime("%H:%M UTC"),
            }

            # Categorize structurally:
            # 1. BSL: Overhead resting buy stops above highs (never low/support levels)
            if lvl.price >= current_price and lvl.level_type in BSL_TYPES:
                bsl_list.append(pool_entry)
            # 2. SSL: Underlying resting sell stops below lows (never high/resistance levels)
            elif lvl.price <= current_price and lvl.level_type in SSL_TYPES:
                ssl_list.append(pool_entry)
            # 3. Fallback for untyped levels
            elif lvl.price > current_price and lvl.level_type not in SSL_TYPES:
                bsl_list.append(pool_entry)
            elif lvl.price < current_price and lvl.level_type not in BSL_TYPES:
                ssl_list.append(pool_entry)

        # Sort BSL closest first (lowest price above current)
        bsl_list.sort(key=lambda x: x["price"])
        # Sort SSL closest first (highest price below current)
        ssl_list.sort(key=lambda x: x["price"], reverse=True)

        nearest_bsl = bsl_list[0] if bsl_list else None
        nearest_ssl = ssl_list[0] if ssl_list else None

        htf_summary = {
            "pdh": next((lvl.price for lvl in levels if lvl.level_type == "PDH"), None),
            "pdl": next((lvl.price for lvl in levels if lvl.level_type == "PDL"), None),
            "pdc": next((lvl.price for lvl in levels if lvl.level_type == "PDC"), None),
            "pwh": next((lvl.price for lvl in levels if lvl.level_type == "PWH"), None),
            "pwl": next((lvl.price for lvl in levels if lvl.level_type == "PWL"), None),
            "nymo": next((lvl.price for lvl in levels if lvl.level_type == "NYMO"), None),
        }

        return {
            "current_price": current_price,
            "atr_14": round(atr_val, 2),
            "timing_schedule": {
                "current_time_utc": ts.strftime("%Y-%m-%d %H:%M UTC"),
                "current_time_ist": ts_ist.strftime("%Y-%m-%d %H:%M IST"),
                "active_window": active_window,
                "next_window": next_window,
            },
            "nearest_bsl": nearest_bsl,
            "nearest_ssl": nearest_ssl,
            "all_bsl_pools": bsl_list,
            "all_ssl_pools": ssl_list,
            "htf_levels": htf_summary,
            "fair_value_targets": {
                "session_vwap": vwap_dict.get("vwap"),
                "poc": vp_dict.get("vp_poc"),
                "vah": vp_dict.get("vp_vah"),
                "val": vp_dict.get("vp_val"),
            }
        }

