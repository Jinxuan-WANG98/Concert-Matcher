from __future__ import annotations

import re


def _compact(value: str | None) -> str:
    return (
        str(value or "")
        .replace("\uff0f", "/")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\uff0d", "-")
        .replace("\u2212", "-")
        .replace(" ", "")
        .strip()
    )


def parse_date_text(value: str | None) -> tuple[int, int] | None:
    compact = _compact(value)
    match = re.search(r"(\d{1,2})/(\d{1,2})", compact)
    if not match:
        match = re.search(r"(\d{1,2})\.(\d{1,2})", compact)
    if not match:
        return None
    month = int(match.group(1))
    day = int(match.group(2))
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return None
    return month, day


def _parse_range_end(value: str | None, start_month: int, start_day: int) -> tuple[int, int] | None:
    compact = _compact(value)
    if not re.search(r"\d", compact):
        return None

    full = parse_date_text(compact)
    if full:
        return full

    day_match = re.search(r"(\d{1,2})$", compact)
    if not day_match:
        return None
    day = int(day_match.group(1))
    if not (1 <= day <= 31) or day == start_day:
        return None
    return start_month, day


def merge_date_range(start_date: str, range_text: str | None) -> str:
    start = parse_date_text(start_date)
    if not start:
        return str(start_date).strip()
    start_month, start_day = start
    end = _parse_range_end(range_text, start_month, start_day)
    if not end:
        return f"{start_month}.{start_day}"
    end_month, end_day = end
    if end_month == start_month:
        return f"{start_month}.{start_day}-{end_day}"
    return f"{start_month}.{start_day}-{end_month}.{end_day}"


def format_date_for_display(date_text: str) -> str:
    text = str(date_text or "").strip()
    match = re.match(r"^(\d{1,2})\.(\d{1,2})(?:-(?:(\d{1,2})\.)?(\d{1,2}))?$", text)
    if not match:
        return text
    start_month = int(match.group(1))
    start_day = int(match.group(2))
    if not match.group(4):
        return f"{start_month}\u6708{start_day}\u65e5"
    end_month = int(match.group(3)) if match.group(3) else start_month
    end_day = int(match.group(4))
    if end_month == start_month:
        return f"{start_month}\u6708{start_day}\u65e5-{end_day}\u65e5"
    return f"{start_month}\u6708{start_day}\u65e5-{end_month}\u6708{end_day}\u65e5"
