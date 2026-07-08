from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image

from services.date_parser import merge_date_range, parse_date_text
from services.models import EventRow


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


def clean_ocr_text(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    for _ in range(3):
        text = re.sub(r"([\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", r"\1", text)
    text = re.sub(r"([\u4e00-\u9fff])\s+([A-Za-z0-9])", r"\1\2", text)
    text = re.sub(r"\s*([/&+\u00d7])\s*", r"\1", text)
    return text.replace("M AO", "MAO").replace("7 HERTZ", "7HERTZ").strip()


def _nearest(lines: Iterable[OcrLine], y: float, max_distance: float) -> OcrLine | None:
    best: tuple[float, OcrLine] | None = None
    for line in lines:
        distance = abs(line.cy - y)
        if distance <= max_distance and (best is None or distance < best[0]):
            best = (distance, line)
    return best[1] if best else None


def _is_performer_line(line: OcrLine) -> bool:
    text = clean_ocr_text(line.text)
    if not text or not re.search(r"[A-Za-z\u4e00-\u9fff]", text):
        return False
    if re.search(r"\u65e5\u671f|\u573a\u5730|\u9635\u5bb9|\u4ec5\u4f9b\u53c2\u8003|\u5b9e\u9645\u6f14\u51fa\u4e3a\u51c6", text):
        return False
    if line.x1 < 210 or line.x2 > 910:
        return False
    compact = re.sub(r"[^\w\u4e00-\u9fff]", "", text)
    return len(compact) >= 2


def parse_ocr_events(ocr_images: list[OcrImage]) -> list[EventRow]:
    events: list[EventRow] = []
    for image in ocr_images:
        date_lines = [line for line in image.lines if line.x1 < 110 and parse_date_text(line.text)]
        range_lines = [line for line in image.lines if 105 <= line.x1 and line.x2 <= 210 and re.search(r"\d", line.text)]
        venue_lines = [
            line
            for line in image.lines
            if line.x1 >= 900 and clean_ocr_text(line.text) and "\u573a\u5730" not in line.text
        ]
        for line in image.lines:
            if not _is_performer_line(line):
                continue
            date_line = _nearest(date_lines, line.cy, 60)
            if not date_line:
                continue
            range_line = _nearest(range_lines, line.cy, 38)
            venue_line = _nearest(venue_lines, line.cy, 38)
            events.append(
                EventRow(
                    date_text=merge_date_range(date_line.text, range_line.text if range_line else None),
                    performer=clean_ocr_text(line.text),
                    venue=clean_ocr_text(venue_line.text) if venue_line else "",
                    image_name=image.image_name,
                )
            )
    return events


def ocr_images_with_rapidocr(image_paths: list[Path]) -> list[OcrImage]:
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError as exc:
        raise RuntimeError(
            "\u5f53\u524d\u73af\u5883\u6ca1\u6709\u5b89\u88c5 RapidOCR\uff1b"
            "\u8bf7\u5728\u90e8\u7f72\u73af\u5883\u5b89\u88c5 rapidocr-onnxruntime\uff0c"
            "\u6216\u4e0a\u4f20\u5df2 OCR \u7684\u6570\u636e\u3002"
        ) from exc

    engine = RapidOCR()
    images: list[OcrImage] = []
    for image_path in image_paths:
        with Image.open(image_path) as img:
            width, height = img.size
        result, _ = engine(str(image_path))
        lines: list[OcrLine] = []
        for item in result or []:
            points, text = item[0], item[1]
            xs = [float(point[0]) for point in points]
            ys = [float(point[1]) for point in points]
            lines.append(OcrLine(text=str(text), x1=min(xs), y1=min(ys), x2=max(xs), y2=max(ys)))
        images.append(OcrImage(image_name=image_path.name, width=width, height=height, lines=lines))
    return images
