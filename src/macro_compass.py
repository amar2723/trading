from __future__ import annotations

"""
Institutional Macro Compass & Cross-Asset Concordance Engine for XAUUSD.
Tracks real-time macroeconomic drivers of gold:
1. US Dollar Index (DXY Proxy: DX-Y.NYB)
2. US 10-Year Treasury Yield (^TNX Proxy)
3. Macro Concordance Scoring (-1.0 to +1.0) for Gold Directional Bias
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import requests

logger = logging.getLogger("MacroCompass")

_MACRO_SESSION: requests.Session | None = None


def get_macro_session() -> requests.Session:
    global _MACRO_SESSION
    if _MACRO_SESSION is None:
        _MACRO_SESSION = requests.Session()
        _MACRO_SESSION.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json",
        })
    return _MACRO_SESSION


@dataclass
class MacroSnapshot:
    dxy_price: float
    dxy_change_pct: float
    us10y_yield: float
    us10y_change_bps: float
    concordance_score: float  # -1.0 (Strong Bearish Headwind) to +1.0 (Strong Bullish Tailwind)
    regime: str               # "BULLISH_TAILWIND", "BEARISH_HEADWIND", "NEUTRAL"
    summary: str
    timestamp: str
    is_live: bool


class MacroCompass:
    """
    Evaluates institutional macro tailwinds/headwinds for Gold trading.
    Senior desk rule: Never take aggressive Long Gold trades when DXY and 10Y Yields
    are surging simultaneously.
    """

    def __init__(self, cache_ttl_seconds: int = 120, ttl_seconds: int | None = None):
        self.cache_ttl = ttl_seconds if ttl_seconds is not None else cache_ttl_seconds
        self._cached_snapshot: MacroSnapshot | None = None
        self._last_fetch_ts: float = 0.0

    @staticmethod
    def _classify_regime(concordance: float) -> str:
        if concordance >= 0.25:
            return "BULLISH_TAILWIND"
        elif concordance <= -0.25:
            return "BEARISH_HEADWIND"
        return "NEUTRAL"

    def get_snapshot(self, force_refresh: bool = False) -> MacroSnapshot:
        now_ts = datetime.now(timezone.utc).timestamp()
        if not force_refresh and self._cached_snapshot is not None:
            if (now_ts - self._last_fetch_ts) < self.cache_ttl:
                return self._cached_snapshot

        session = get_macro_session()
        dxy_price = 104.50
        dxy_chg = 0.0
        us10y_val = 4.40
        us10y_bps = 0.0
        is_live = False

        # 1. Fetch DXY (US Dollar Index)
        try:
            r_dxy = session.get("https://query1.finance.yahoo.com/v8/finance/chart/DX-Y.NYB?interval=5m&range=1d", timeout=4)
            if r_dxy.status_code == 200:
                meta = r_dxy.json()["chart"]["result"][0]["meta"]
                cur = float(meta.get("regularMarketPrice", 104.50))
                prev = float(meta.get("chartPreviousClose", cur))
                dxy_price = round(cur, 2)
                dxy_chg = round(((cur - prev) / prev) * 100.0, 2) if prev > 0 else 0.0
                is_live = True
        except Exception as e:
            logger.debug("DXY quote fetch fallback: %s", e)

        # 2. Fetch US 10-Year Treasury Yield (^TNX)
        try:
            r_tnx = session.get("https://query1.finance.yahoo.com/v8/finance/chart/%5ETNX?interval=5m&range=1d", timeout=4)
            if r_tnx.status_code == 200:
                meta = r_tnx.json()["chart"]["result"][0]["meta"]
                cur_y = float(meta.get("regularMarketPrice", 4.40))
                prev_y = float(meta.get("chartPreviousClose", cur_y))
                us10y_val = round(cur_y, 3)
                us10y_bps = round((cur_y - prev_y) * 100.0, 1)  # Basis points
                is_live = True
        except Exception as e:
            logger.debug("US10Y quote fetch fallback: %s", e)

        # 3. Macro Concordance for Gold (-1.0 to +1.0)
        # Gold is inversely related to Dollar strength and 10Y real yields.
        # When DXY drops, score increases. When Yields drop, score increases.
        dxy_score = -np_clip(dxy_chg / 0.75, -1.0, 1.0)
        us10y_score = -np_clip(us10y_bps / 8.0, -1.0, 1.0)

        # Weighted composite: 60% DXY + 40% US10Y
        concordance = round(float((dxy_score * 0.60) + (us10y_score * 0.40)), 2)

        if concordance >= 0.25:
            regime = "BULLISH_TAILWIND"
            icon = "🟢"
        elif concordance <= -0.25:
            regime = "BEARISH_HEADWIND"
            icon = "🔴"
        else:
            regime = "NEUTRAL"
            icon = "⚖️"

        dxy_sign = "+" if dxy_chg >= 0 else ""
        us10y_sign = "+" if us10y_bps >= 0 else ""
        summary = (
            f"{icon} *Macro Regime*: `{regime}` (Concordance: `{concordance:+0.2f}`)\n"
            f"  ↳ 💵 *USD Index (DXY)*: `{dxy_price:.2f}` ({dxy_sign}{dxy_chg:.2f}%)\n"
            f"  ↳ 📈 *US 10Y Yield*: `{us10y_val:.3f}%` ({us10y_sign}{us10y_bps:.1f} bps)"
        )

        snapshot = MacroSnapshot(
            dxy_price=dxy_price,
            dxy_change_pct=dxy_chg,
            us10y_yield=us10y_val,
            us10y_change_bps=us10y_bps,
            concordance_score=concordance,
            regime=regime,
            summary=summary,
            timestamp=datetime.now(timezone.utc).isoformat(),
            is_live=is_live,
        )

        self._cached_snapshot = snapshot
        self._last_fetch_ts = now_ts
        return snapshot


def np_clip(val: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(max_val, val))
