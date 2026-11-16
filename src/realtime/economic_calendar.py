from __future__ import annotations

"""
High-Impact Economic News Filter & Red Folder Lockout.

Protects against catastrophic slippage and spread spikes during:
- Non-Farm Payrolls (NFP)
- Consumer Price Index (CPI)
- Federal Reserve Interest Rate Decisions (FOMC)
- Core PPI, Retail Sales, and GDP
"""

from datetime import datetime, timezone
import requests
import logging

logger = logging.getLogger(__name__)


class NewsCalendarFilter:
    def __init__(self, lockout_before_min: int = 30, lockout_after_min: int = 30):
        self.lockout_before = lockout_before_min
        self.lockout_after = lockout_after_min
        self.cached_events = []
        self.last_fetch_time = None

    def fetch_high_impact_news(self) -> list[dict]:
        """Fetches upcoming high-impact USD economic events."""
        try:
            # ForexFactory or public calendar feed
            url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
            resp = requests.get(url, timeout=8)
            if resp.status_code == 200:
                events = resp.json()
                high_impact_usd = [
                    e for e in events 
                    if e.get("country") == "USD" and e.get("impact") in ["High", "Holiday"]
                ]
                self.cached_events = high_impact_usd
                self.last_fetch_time = datetime.now(timezone.utc)
                return high_impact_usd
        except Exception as exc:
            logger.warning("Could not fetch remote economic calendar: %s. Using static news rules.", exc)
        return self.cached_events

    def is_news_lockout(self, current_dt: datetime | None = None) -> tuple[bool, str]:
        """
        Evaluates whether current time falls within the high-impact news blackout window.
        Returns (is_locked, reason).
        """
        now = current_dt or datetime.now(timezone.utc)
        
        # 1. Check live fetched calendar events
        if not self.cached_events or (self.last_fetch_time and (now - self.last_fetch_time).total_seconds() > 3600):
            self.fetch_high_impact_news()

        for event in self.cached_events:
            date_str = event.get("date")
            if not date_str:
                continue
            try:
                event_dt = datetime.fromisoformat(date_str)
                diff_min = (event_dt - now).total_seconds() / 60.0
                
                # Check if we are inside the [-lockout_after, +lockout_before] window
                if -self.lockout_after <= diff_min <= self.lockout_before:
                    title = event.get("title", "High-Impact USD News")
                    return True, f"NEWS LOCKOUT: {title} at {event_dt.strftime('%H:%M UTC')} (In {diff_min:.0f}m)"
            except Exception:
                continue

        # 2. Heuristic Rule: Common High-Impact Times (08:30 AM EST and 02:00 PM EST FOMC)
        # 08:30 AM EST = 13:30 UTC | 02:00 PM EST = 19:00 UTC
        utc_hour = now.hour
        utc_min = now.minute
        
        # 13:15 to 14:00 UTC (08:15 to 09:00 AM EST)
        if utc_hour == 13 and (15 <= utc_min <= 45):
            return True, "HEURISTIC NEWS WINDOW: 08:30 AM EST US Major Data Release Window (CPI/NFP/PPI)."

        # 18:45 to 19:30 UTC (01:45 to 02:30 PM EST FOMC)
        if (utc_hour == 18 and utc_min >= 45) or (utc_hour == 19 and utc_min <= 30):
            if now.weekday() == 2:  # Wednesday FOMC
                return True, "HEURISTIC NEWS WINDOW: 02:00 PM EST FOMC Rate Decision Window."

        return False, "Clear of high-impact news events."
