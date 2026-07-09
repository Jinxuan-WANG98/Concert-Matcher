from __future__ import annotations

import re

_MONTH_DAYS = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


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


def _is_valid_month_day(month: int, day: int) -> bool:
    return 1 <= month <= 12 and 1 <= day <= _MONTH_DAYS[month - 1]


def _month_day_to_ordinal(month: int, day: int) -> int:
    return sum(_MONTH_DAYS[: month - 1]) + day


def parse_date_range(value: str | None) -> tuple[int, int] | None:
    compact = _compact(value)
    match = re.match(
        r"^(\d{1,2})[/\.](\d{1,2})(?:-(?:(\d{1,2})[/\.])?(\d{1,2}))?$",
        compact,
    )
    if not match:
        parsed = parse_date_text(compact)
        if not parsed:
            return None
        month, day = parsed
        if not _is_valid_month_day(month, day):
            return None
        ordinal = _month_day_to_ordinal(month, day)
        return ordinal, ordinal

    start_month = int(match.group(1))
    start_day = int(match.group(2))
    end_month = int(match.group(3)) if match.group(3) else start_month
    end_day = int(match.group(4)) if match.group(4) else start_day
    if not _is_valid_month_day(start_month, start_day) or not _is_valid_month_day(end_month, end_day):
        return None
    start = _month_day_to_ordinal(start_month, start_day)
    end = _month_day_to_ordinal(end_month, end_day)
    if end < start:
        end += sum(_MONTH_DAYS)
    return start, end


def _range_variants(date_range: tuple[int, int]) -> list[tuple[int, int]]:
    variants = [date_range]
    year_days = sum(_MONTH_DAYS)
    start, end = date_range
    if start <= 31:
        variants.append((start + year_days, end + year_days))
    if end > year_days:
        variants.append((start - year_days, end - year_days))
    return variants


def date_ranges_overlap(left: str | None, right: str | None) -> bool:
    left_range = parse_date_range(left)
    right_range = parse_date_range(right)
    if not left_range or not right_range:
        return _compact(left) == _compact(right)
    for left_start, left_end in _range_variants(left_range):
        for right_start, right_end in _range_variants(right_range):
            if max(left_start, right_start) <= min(left_end, right_end):
                return True
    return False


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
