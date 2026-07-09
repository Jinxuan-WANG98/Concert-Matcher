from __future__ import annotations

import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from services.date_parser import merge_date_range, parse_date_text
from services.models import EventRow

_OCR_LOCAL = threading.local()
_OCR_MAX_WORKERS = max(1, min(4, int(os.environ.get("OCR_MAX_WORKERS", "4"))))
_OCR_MAX_WIDTH = int(os.environ.get("OCR_MAX_WIDTH", "1080"))


@dataclass(frozen=True)
class OcrLine:
    text: str
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2


@dataclass(frozen=True)
class OcrImage:
    image_name: str
    width: int
    height: int
    lines: list[OcrLine]


ROW_Y_TOLERANCE = 18
GAP_ROW_MIN = 40
GAP_ROW_MAX = 72
SUMMARY_DATE_COLUMN_TOLERANCE = 70


def clean_ocr_text(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    for _ in range(3):
        text = re.sub(r"([\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", r"\1", text)
    text = re.sub(r"([\u4e00-\u9fff])\s+([A-Za-z0-9])", r"\1\2", text)
    text = re.sub(r"\s*([/&+\u00d7])\s*", r"\1", text)
    return text.replace("M AO", "MAO").replace("7 HERTZ", "7HERTZ").strip()


def _layout_metrics(width: int) -> dict[str, float]:
    scale = (width or 1080) / 1080.0
    return {
        "date_max": 110 * scale,
        "range_min": 105 * scale,
        "range_max": 210 * scale,
        "performer_min": 210 * scale,
        "performer_max": 910 * scale,
        "venue_min": 900 * scale,
    }


KNOWN_VENUE_ABBREVS = {
    "东体",
    "梅奔",
    "虹馆",
    "红石",
    "摩登",
    "静安",
    "瓦肆",
    "JZ",
    "Jz",
    "MAO",
    "MOSH",
    "CAVE",
    "Coven",
    "Legacy",
    "Encore",
    "Cream",
    "MeHub",
    "WEST",
    "翡声",
    "霏声",
    "乐无界",
}

VENUE_HINT = re.compile(
    r"(?:馆|堂|空间|球场|体育场|体育馆|音乐厅|足球场|剧院|演艺|"
    r"Livehouse|Legacy|Encore|Cream|MeHub|CAVE|Coven|WEST|Wast|est|"
    r"MAO|MOSH|JZ|"
    r"梅奔|东体|虹馆|红石|摩登|静安|瓦肆|育音堂|新歌空间|翡声|霏声|乐无界|"
    r"星在|GT\.?CH|虹口|东艺)",
    re.I,
)


def _looks_like_venue(text: str) -> bool:
    cleaned = clean_ocr_text(text)
    if not cleaned or cleaned in {"地点", "场馆", "声"}:
        return False
    normalized = re.sub(r"\s+", "", cleaned)
    if normalized in KNOWN_VENUE_ABBREVS or normalized.lower() in {value.lower() for value in KNOWN_VENUE_ABBREVS}:
        return True
    return bool(VENUE_HINT.search(cleaned))


def _cluster_index_for_line(clusters: list[list[OcrLine]], line: OcrLine) -> int | None:
    for index, cluster in enumerate(clusters):
        if any(item is line for item in cluster):
            return index
    return None


def _date_lines_in_cluster(cluster: list[OcrLine], metrics: dict[str, float]) -> list[OcrLine]:
    return [line for line in cluster if line.x1 < metrics["date_max"] and parse_date_text(line.text)]


def _range_lines_in_cluster(cluster: list[OcrLine], metrics: dict[str, float]) -> list[OcrLine]:
    return [
        line
        for line in cluster
        if metrics["range_min"] <= line.x1 <= metrics["range_max"] and re.search(r"\d", line.text)
    ]


def _resolve_venue_text(line: OcrLine, cluster: list[OcrLine], metrics: dict[str, float]) -> str:
    text = clean_ocr_text(line.text)
    if text == "声":
        for other in cluster:
            if other.x1 >= metrics["venue_min"] and clean_ocr_text(other.text) == "霏":
                return "霏声"
        for other in cluster:
            if other.x1 >= metrics["venue_min"] and clean_ocr_text(other.text) == "翡":
                return "翡声"
    return text


def _venue_lines_in_cluster(cluster: list[OcrLine], metrics: dict[str, float]) -> list[OcrLine]:
    right_lines = [line for line in cluster if line.x1 >= metrics["venue_min"]]
    right_texts = {clean_ocr_text(line.text) for line in right_lines}
    if {"霏", "声"}.issubset(right_texts):
        return [line for line in right_lines if clean_ocr_text(line.text) == "声"]

    venues: list[OcrLine] = []
    for line in right_lines:
        if _looks_like_venue(_resolve_venue_text(line, cluster, metrics)):
            venues.append(line)
    return venues


def _date_for_performer(
    performer: OcrLine,
    clusters: list[list[OcrLine]],
    metrics: dict[str, float],
    venue_line: OcrLine | None,
    venue_method: str,
) -> tuple[OcrLine | None, OcrLine | None, str]:
    performer_cluster_idx = _cluster_index_for_line(clusters, performer)
    if performer_cluster_idx is None:
        return None, None, "performer_cluster_missing"

    cluster_order: list[int] = [performer_cluster_idx]
    if venue_method == "gap_row" and venue_line is not None:
        venue_cluster_idx = _cluster_index_for_line(clusters, venue_line)
        if venue_cluster_idx is not None and venue_cluster_idx not in cluster_order:
            cluster_order.insert(0, venue_cluster_idx)

    for cluster_idx in cluster_order:
        cluster = clusters[cluster_idx]
        dates = _date_lines_in_cluster(cluster, metrics)
        if not dates:
            continue
        date_line = min(dates, key=lambda item: abs(item.cy - performer.cy))
        ranges = _range_lines_in_cluster(cluster, metrics)
        range_line = min(ranges, key=lambda item: abs(item.cy - performer.cy)) if ranges else None
        source = "gap_row_date" if cluster_idx != performer_cluster_idx else "same_row_date"
        return date_line, range_line, source

    return None, None, "no_date"


def _cluster_rows(lines: list[OcrLine], y_tolerance: float = ROW_Y_TOLERANCE) -> list[list[OcrLine]]:
    if not lines:
        return []
    ordered = sorted(lines, key=lambda item: item.cy)
    clusters: list[list[OcrLine]] = [[ordered[0]]]
    for line in ordered[1:]:
        cluster_cy = sum(item.cy for item in clusters[-1]) / len(clusters[-1])
        if abs(line.cy - cluster_cy) <= y_tolerance:
            clusters[-1].append(line)
        else:
            clusters.append([line])
    return clusters


def _cluster_center(cluster: list[OcrLine]) -> float:
    return sum(line.cy for line in cluster) / len(cluster)


def _is_performer_line(line: OcrLine, metrics: dict[str, float]) -> bool:
    text = clean_ocr_text(line.text)
    if not text or not re.search(r"[A-Za-z\u4e00-\u9fff]", text):
        return False
    if re.search(r"\u65e5\u671f|\u573a\u5730|\u9635\u5bb9|\u4ec5\u4f9b\u53c2\u8003|\u5b9e\u9645\u6f14\u51fa\u4e3a\u51c6", text):
        return False
    if line.x1 < metrics["performer_min"] or line.x2 > metrics["performer_max"]:
        return False
    compact = re.sub(r"[^\w\u4e00-\u9fff]", "", text)
    return len(compact) >= 2


def _is_standalone_weekday(text: str) -> bool:
    cleaned = clean_ocr_text(text)
    return cleaned in {
        "\u4e00",
        "\u4e8c",
        "\u4e09",
        "\u56db",
        "\u4e94",
        "\u516d",
        "\u65e5",
        "\u5929",
        "\u5468\u4e00",
        "\u5468\u4e8c",
        "\u5468\u4e09",
        "\u5468\u56db",
        "\u5468\u4e94",
        "\u5468\u516d",
        "\u5468\u65e5",
        "\u5468\u5929",
        "\u661f\u671f\u4e00",
        "\u661f\u671f\u4e8c",
        "\u661f\u671f\u4e09",
        "\u661f\u671f\u56db",
        "\u661f\u671f\u4e94",
        "\u661f\u671f\u516d",
        "\u661f\u671f\u65e5",
        "\u661f\u671f\u5929",
    }


def _is_summary_performer_line(line: OcrLine) -> bool:
    text = clean_ocr_text(line.text)
    if not text or not re.search(r"[A-Za-z\u4e00-\u9fff]", text):
        return False
    if parse_date_text(text) or _is_standalone_weekday(text):
        return False
    if re.search(r"\u65e5\u671f|\u6b4c\u624b|\u9635\u5bb9|\u573a\u5730|\u5468|week|artist", text, re.I):
        return False
    compact = re.sub(r"[^\w\u4e00-\u9fff]", "", text)
    return len(compact) >= 2


def _date_column_starts(lines: list[OcrLine]) -> list[float]:
    starts: list[float] = []
    for line in sorted(lines, key=lambda item: item.x1):
        text = clean_ocr_text(line.text)
        if text.startswith(("-", "\uff0d", "\u2013", "\u2014")) or not parse_date_text(text):
            continue
        if all(abs(line.x1 - start) > SUMMARY_DATE_COLUMN_TOLERANCE for start in starts):
            starts.append(line.x1)
    return starts


def _summary_next_column_start(date_line: OcrLine, date_columns: list[float], width: int) -> float:
    for start in sorted(date_columns):
        if start > date_line.x1 + SUMMARY_DATE_COLUMN_TOLERANCE:
            return start
    return float(width or 1080) + 1


def _parse_summary_page_events(image: OcrImage, clusters: list[list[OcrLine]], date_columns: list[float]) -> list[EventRow]:
    events: list[EventRow] = []
    width = image.width or 1080
    min_performer_gap = max(52.0, width * 0.04)

    for cluster in clusters:
        date_lines = sorted(
            [
                line
                for line in cluster
                if parse_date_text(line.text)
                and any(abs(line.x1 - start) <= SUMMARY_DATE_COLUMN_TOLERANCE for start in date_columns)
            ],
            key=lambda item: item.x1,
        )
        for date_line in date_lines:
            next_column_start = _summary_next_column_start(date_line, date_columns, width)
            left_edge = date_line.x2 + min_performer_gap
            right_edge = next_column_start - max(16.0, width * 0.015)
            performers = [
                line
                for line in cluster
                if left_edge <= line.x1 < right_edge and _is_summary_performer_line(line)
            ]
            for performer in performers:
                events.append(
                    EventRow(
                        date_text=merge_date_range(date_line.text, None),
                        performer=clean_ocr_text(performer.text),
                        venue="",
                        image_name=image.image_name,
                    )
                )
    return events


def _event_dedupe_key(event: EventRow) -> tuple[str, str]:
    performer_key = re.sub(r"[^\w\u4e00-\u9fff]", "", clean_ocr_text(event.performer)).lower()
    return event.date_text, performer_key


def _dedupe_events(events: list[EventRow]) -> list[EventRow]:
    ordered: list[EventRow] = []
    indexes: dict[tuple[str, str], int] = {}
    for event in events:
        key = _event_dedupe_key(event)
        if not key[1]:
            continue
        if key not in indexes:
            indexes[key] = len(ordered)
            ordered.append(event)
            continue
        current_index = indexes[key]
        current = ordered[current_index]
        if event.venue and not current.venue:
            ordered[current_index] = event
    return ordered


def _venue_for_performer(
    performer: OcrLine,
    clusters: list[list[OcrLine]],
    metrics: dict[str, float],
) -> tuple[OcrLine | None, str]:
    performer_cluster_idx = _cluster_index_for_line(clusters, performer)
    if performer_cluster_idx is None:
        return None, "performer_cluster_missing"

    same_cluster_venues = _venue_lines_in_cluster(clusters[performer_cluster_idx], metrics)
    if same_cluster_venues:
        return min(same_cluster_venues, key=lambda item: abs(item.cy - performer.cy)), "same_row"

    if performer_cluster_idx + 1 < len(clusters):
        next_cluster = clusters[performer_cluster_idx + 1]
        dy = _cluster_center(next_cluster) - performer.cy
        if GAP_ROW_MIN <= dy <= GAP_ROW_MAX:
            gap_venues = _venue_lines_in_cluster(next_cluster, metrics)
            if gap_venues:
                return min(gap_venues, key=lambda item: abs(item.cy - performer.cy)), "gap_row"

    return None, "no_venue"


def parse_ocr_events(ocr_images: list[OcrImage]) -> list[EventRow]:
    events: list[EventRow] = []
    for image in ocr_images:
        metrics = _layout_metrics(image.width)
        clusters = _cluster_rows(image.lines)
        date_columns = _date_column_starts(image.lines)
        if len(date_columns) >= 2:
            events.extend(_parse_summary_page_events(image, clusters, date_columns))
            continue
        for line in image.lines:
            if not _is_performer_line(line, metrics):
                continue
            venue_line, venue_method = _venue_for_performer(line, clusters, metrics)
            date_line, range_line, _date_method = _date_for_performer(
                line,
                clusters,
                metrics,
                venue_line,
                venue_method,
            )
            if not date_line:
                continue
            venue_text = ""
            if venue_line is not None:
                venue_cluster_idx = _cluster_index_for_line(clusters, venue_line)
                venue_cluster = clusters[venue_cluster_idx] if venue_cluster_idx is not None else []
                venue_text = _resolve_venue_text(venue_line, venue_cluster, metrics)
            events.append(
                EventRow(
                    date_text=merge_date_range(date_line.text, range_line.text if range_line else None),
                    performer=clean_ocr_text(line.text),
                    venue=venue_text,
                    image_name=image.image_name,
                )
            )
    return _dedupe_events(events)


def _get_ocr_engine():
    engine = getattr(_OCR_LOCAL, "engine", None)
    if engine is None:
        from rapidocr_onnxruntime import RapidOCR

        engine = RapidOCR()
        _OCR_LOCAL.engine = engine
    return engine


def _ocr_image_path(image_path: Path) -> OcrImage:
    with Image.open(image_path) as img:
        width, height = img.size
        ocr_source = image_path
        if width > _OCR_MAX_WIDTH:
            resized = img.copy()
            resized.thumbnail((_OCR_MAX_WIDTH, _OCR_MAX_WIDTH * 4), Image.Resampling.LANCZOS)
            temp_path = image_path.with_suffix(".ocr.jpg")
            resized.save(temp_path, format="JPEG", quality=90)
            ocr_source = temp_path
            width, height = resized.size

    result, _ = _get_ocr_engine()(str(ocr_source))
    if ocr_source != image_path and ocr_source.exists():
        ocr_source.unlink(missing_ok=True)

    lines: list[OcrLine] = []
    for item in result or []:
        points, text = item[0], item[1]
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
        lines.append(OcrLine(text=str(text), x1=min(xs), y1=min(ys), x2=max(xs), y2=max(ys)))
    return OcrImage(image_name=image_path.name, width=width, height=height, lines=lines)


def ocr_images_with_rapidocr(image_paths: list[Path]) -> list[OcrImage]:
    try:
        from rapidocr_onnxruntime import RapidOCR  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "\u5f53\u524d\u73af\u5883\u6ca1\u6709\u5b89\u88c5 RapidOCR\uff1b"
            "\u8bf7\u5728\u90e8\u7f72\u73af\u5883\u5b89\u88c5 rapidocr-onnxruntime\uff0c"
            "\u6216\u4e0a\u4f20\u5df2 OCR \u7684\u6570\u636e\u3002"
        ) from exc

    if not image_paths:
        return []
    if len(image_paths) == 1:
        return [_ocr_image_path(image_paths[0])]

    workers = min(_OCR_MAX_WORKERS, len(image_paths))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(_ocr_image_path, image_paths))
