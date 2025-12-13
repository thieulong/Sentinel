import re
from datetime import datetime, timedelta
from typing import Optional, Tuple

def norm(text: str) -> str:
    if not text:
        text = "node"
    out = "".join(c if c.isalnum() else "_" for c in text)
    if not out:
        out = "node"
    if out[0].isdigit():
        out = "id_" + out
    return out[:60]


def parse_iso_ts(ts: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        return None


def detect_time_window(
    question: str,
    now: Optional[datetime] = None
) -> Optional[Tuple[datetime, datetime]]:

    if now is None:
        now = datetime.now()

    q = question.lower().replace(".", ":")

    base_date = None
    if "yesterday" in q:
        base_date = (now - timedelta(days=1)).date()
    elif "today" in q:
        base_date = now.date()

    hour_match = re.search(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)", q)
    explicit_hour = None
    explicit_minute = 0
    if hour_match:
        h = int(hour_match.group(1))
        m = hour_match.group(2)
        ampm = hour_match.group(3)

        if m is not None:
            explicit_minute = min(max(int(m), 0), 59)

        if ampm == "pm" and h != 12:
            h += 12
        if ampm == "am" and h == 12:
            h = 0

        explicit_hour = min(max(h, 0), 23)

    morning = "morning" in q
    afternoon = "afternoon" in q
    evening = "evening" in q
    night = "night" in q

    if base_date is None and (morning or afternoon or evening or night):
        base_date = now.date()

    if base_date is None and explicit_hour is not None:
        base_date = now.date()

    if base_date is None and explicit_hour is None:
        return None

    if explicit_hour is not None:
        center = datetime.combine(base_date, datetime.min.time()).replace(
            hour=explicit_hour, minute=explicit_minute, second=0, microsecond=0
        )
        start = center - timedelta(hours=1)
        end = center + timedelta(hours=1)

        day_start = datetime.combine(base_date, datetime.min.time())
        day_end = day_start + timedelta(days=1) - timedelta(microseconds=1)

        if start < day_start:
            start = day_start
        if end > day_end:
            end = day_end

        return start, end

    if morning:
        start = datetime.combine(base_date, datetime.min.time()).replace(hour=5, minute=0, second=0, microsecond=0)
        end = datetime.combine(base_date, datetime.min.time()).replace(hour=12, minute=0, second=0, microsecond=0)
        return start, end

    if afternoon:
        start = datetime.combine(base_date, datetime.min.time()).replace(hour=12, minute=0, second=0, microsecond=0)
        end = datetime.combine(base_date, datetime.min.time()).replace(hour=18, minute=0, second=0, microsecond=0)
        return start, end

    if evening or night:
        start = datetime.combine(base_date, datetime.min.time()).replace(hour=18, minute=0, second=0, microsecond=0)
        end = datetime.combine(base_date, datetime.min.time()).replace(hour=23, minute=59, second=59, microsecond=999999)
        return start, end

    if base_date is not None:
        start = datetime.combine(base_date, datetime.min.time())
        end = start + timedelta(days=1) - timedelta(microseconds=1)
        return start, end

    return None