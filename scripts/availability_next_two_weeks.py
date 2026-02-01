#!/usr/bin/env python3
"""
Get availability for a doctor for the next two weeks. Prints each day as:
  Sunday 2026-01-30: 9am-5pm
  Monday 2026-01-31: 9am-12pm, 2pm-5pm
  Tuesday 2026-02-01: —

Usage:
  python scripts/availability_next_two_weeks.py
  DOCTOR_ID=xxx CHECK_AVAIL_DATE=2026-01-29 python scripts/availability_next_two_weeks.py

Docker:
  docker compose -f docker-compose.local.yml run --rm patient-ai-service-v3 \
    python /app/scripts/availability_next_two_weeks.py
"""

import os
import re
import sys
from datetime import datetime, timedelta

try:
    from dotenv import load_dotenv
    root = os.path.join(os.path.dirname(__file__), "..")
    load_dotenv(os.path.join(root, ".env")) or load_dotenv(os.path.join(root, "..", ".env"))
except Exception:
    pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from patient_ai_service.agents.appointment_manager import AppointmentManagerAgent

DOCTOR_ID = os.environ.get("DOCTOR_ID", "d3333333-d333-d333-d333-d33333333333")
NUM_DAYS = 14


def _to_12h(s: str) -> str:
    """'09:00' -> '9am', '17:00' -> '5pm', '14:30' -> '2:30pm'."""
    s = s.strip()
    m = re.match(r"^(\d{1,2}):(\d{2})$", s)
    if not m:
        return s
    h, mm = int(m.group(1)), int(m.group(2))
    if h == 0:
        return "12am" if mm == 0 else f"12:{mm:02d}am"
    if h == 12:
        return "12pm" if mm == 0 else f"12:{mm:02d}pm"
    if h < 12:
        return f"{h}am" if mm == 0 else f"{h}:{mm:02d}am"
    h = h - 12
    return f"{h}pm" if mm == 0 else f"{h}:{mm:02d}pm"


def _format_ranges(ranges: list[str]) -> str:
    """['09:00-17:00'] -> '9am-5pm'; ['09:00-12:00','14:00-17:00'] -> '9am-12pm, 2pm-5pm'."""
    if not ranges:
        return "—"
    out = []
    for r in ranges:
        parts = r.split("-", 1)
        if len(parts) != 2:
            out.append(r)
            continue
        start, end = parts[0].strip(), parts[1].strip()
        out.append(f"{_to_12h(start)}-{_to_12h(end)}")
    return ", ".join(out)


def main() -> None:
    date_cfg = os.environ.get("CHECK_AVAIL_DATE")
    if date_cfg and len(date_cfg) == 10:
        start = datetime.strptime(date_cfg, "%Y-%m-%d").date()
    else:
        start = datetime.now().date()

    print("=" * 60)
    print("Availability — next two weeks")
    print("=" * 60)
    print(f"Doctor ID: {DOCTOR_ID}")
    print(f"From: {start} ({start.strftime('%A')})")
    print()

    agent = AppointmentManagerAgent()

    for i in range(NUM_DAYS):
        d = start + timedelta(days=i)
        date_str = d.strftime("%Y-%m-%d")
        weekday = d.strftime("%A")

        r = agent.tool_check_availability(
            session_id="availability-two-weeks",
            doctor_id=DOCTOR_ID,
            date=date_str,
        )

        available = r.get("available", False)
        ranges = r.get("availability_ranges") or []
        label = _format_ranges(ranges)
        print(f"{weekday} {date_str}: {label}")


if __name__ == "__main__":
    main()
