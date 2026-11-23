from __future__ import annotations

"""
High-Impact Economic News Filter & Red Folder Lockout.

Protects against catastrophic slippage and spread spikes during:
- Non-Farm Payrolls (NFP)
- Consumer Price Index (CPI)
- Federal Reserve Interest Rate Decisions (FOMC)
- Core PPI, Retail Sales, and GDP
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import requests

logger = logging.getLogger(__name__)


@dataclass
class NewsStatus:
    is_locked: bool
    active_event: str | None
    minutes_to_event: float | None
    reason: str
    summary: str
    upcoming_events_today: list[dict]


class NewsCalendarFilter:
    def __init__(
        self,
        lockout_before_min: int = 30,
        lockout_after_min: int = 30,
        lockout_minutes: int | None = None,
    ):
        if lockout_minutes is not None:
            self.lockout_before = lockout_minutes
            self.lockout_after = lockout_minutes
        else:
            self.lockout_before = lockout_before_min
            self.lockout_after = lockout_after_min
        self.cached_events = []
        self.last_fetch_time = None

    def fetch_high_impact_news(self) -> list[dict]:
        """Fetches upcoming high-impact USD economic events."""
        try:
            url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
            resp = requests.get(url, timeout=5)
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
            logger.debug("Could not fetch remote economic calendar: %s. Using cached/heuristic rules.", exc)
        return self.cached_events

    def get_news_status(self, current_dt: datetime | None = None) -> NewsStatus:
        """
        Evaluates full economic calendar status and returns structured NewsStatus.
        """
        now = current_dt or datetime.now(timezone.utc)

        # Refresh cache hourly
        if not self.cached_events or (self.last_fetch_time and (now - self.last_fetch_time).total_seconds() > 3600):
            self.fetch_high_impact_news()

        upcoming_today = []
        now_date_str = now.strftime("%Y-%m-%d")

        is_locked = False
        active_event = None
        min_diff = None
        lockout_reason = ""

        for event in self.cached_events:
            date_str = event.get("date")
            if not date_str:
                continue
            try:
                event_dt = datetime.fromisoformat(date_str)
                if event_dt.tzinfo is None:
                    event_dt = event_dt.replace(tzinfo=timezone.utc)

                if event_dt.strftime("%Y-%m-%d") == now_date_str:
                    upcoming_today.append({
                        "title": event.get("title", "High-Impact Event"),
                        "time_utc": event_dt.strftime("%H:%M UTC"),
                        "impact": event.get("impact", "High"),
                    })

                diff_min = (event_dt - now).total_seconds() / 60.0

                # Check if we are inside the [-lockout_after, +lockout_before] window
                if -self.lockout_after <= diff_min <= self.lockout_before:
                    is_locked = True
                    active_event = event.get("title", "High-Impact USD News")
                    min_diff = round(diff_min, 1)
                    lockout_reason = f"NEWS LOCKOUT: {active_event} at {event_dt.strftime('%H:%M UTC')} (In {diff_min:.0f}m)"
                    break
            except Exception:
                continue

        # Heuristic Rule: Common High-Impact Times (08:30 AM EST = 13:30 UTC | 02:00 PM EST = 19:00 UTC)
        if not is_locked:
            utc_hour = now.hour
            utc_min = now.minute
            if utc_hour == 13 and (15 <= utc_min <= 45):
                is_locked = True
                active_event = "08:30 AM EST US Major Data Release"
                min_diff = 13.5 * 60 - (utc_hour * 60 + utc_min)
                lockout_reason = "HEURISTIC NEWS WINDOW: 08:30 AM EST US Major Data Release (CPI/NFP/PPI)."
            elif (utc_hour == 18 and utc_min >= 45) or (utc_hour == 19 and utc_min <= 30):
                if now.weekday() == 2:  # Wednesday FOMC
                    is_locked = True
                    active_event = "02:00 PM EST FOMC Rate Decision"
                    min_diff = 19.0 * 60 - (utc_hour * 60 + utc_min)
                    lockout_reason = "HEURISTIC NEWS WINDOW: 02:00 PM EST FOMC Rate Decision Window."

        if is_locked:
            summary = f"🔴 *News Blackout*: `ACTIVE` ({active_event} | {lockout_reason})"
        else:
            event_count = len(upcoming_today)
            summary = f"🟢 *News Blackout*: `CLEAR` ({event_count} high-impact USD events today)"

        return NewsStatus(
            is_locked=is_locked,
            active_event=active_event,
            minutes_to_event=min_diff,
            reason=lockout_reason if is_locked else "Clear of high-impact news events.",
            summary=summary,
            upcoming_events_today=upcoming_today,
        )

    def is_news_lockout(self, current_dt: datetime | None = None) -> tuple[bool, str]:
        """
        Backward-compatible check returning (is_locked, reason).
        """
        status = self.get_news_status(current_dt)
        return status.is_locked, status.reason
