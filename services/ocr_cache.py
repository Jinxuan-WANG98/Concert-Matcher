from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from urllib import parse

from services.models import EventRow
from services.ocr import OcrImage, OcrLine

CACHE_VERSION = 1
_CACHE_ENABLED = os.environ.get("OCR_CACHE_ENABLED", "true").lower() in {"1", "true", "yes", "on"}


def cache_enabled() -> bool:
    return _CACHE_ENABLED


def cache_root(output_root: Path | None = None) -> Path:
    root = output_root or Path(os.environ.get("OUTPUT_ROOT", "outputs/webapp"))
    return root / "ocr_cache"


def normalize_xhs_url(url: str) -> str:
    text = str(url or "").strip()
    parsed = parse.urlparse(text)
    scheme = parsed.scheme or "https"
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip("/") or "/"
    return parse.urlunparse((scheme, netloc, path, "", "", ""))


def cache_key_for_xhs(url: str) -> str:
    normalized = normalize_xhs_url(url)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def fingerprint_image_urls(urls: list[str]) -> str:
    payload = "\n".join(_normalize_image_url_for_fingerprint(url) for url in urls)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalize_image_url_for_fingerprint(url: str) -> str:
    text = str(url or "").strip()
    parsed = parse.urlparse(text)
    path = parsed.path or ""
    marker = "/spectrum/"
    if marker in path:
        stable = path.split(marker, 1)[1].split("!", 1)[0].strip("/")
        if stable:
            return f"xhs-spectrum:{stable.lower()}"

    scheme = parsed.scheme.lower() or "https"
    netloc = parsed.netloc.lower()
    normalized_path = path.rstrip("/")
    return parse.urlunparse((scheme, netloc, normalized_path, "", "", ""))


def _recognition_source_key(recognition_source: str) -> str:
    return hashlib.sha256(str(recognition_source or "").encode("utf-8")).hexdigest()[:16]


def _ocr_line_to_dict(line: OcrLine) -> dict:
    return {"text": line.text, "x1": line.x1, "y1": line.y1, "x2": line.x2, "y2": line.y2}


def _ocr_line_from_dict(data: dict) -> OcrLine:
    return OcrLine(
        text=str(data["text"]),
        x1=float(data["x1"]),
        y1=float(data["y1"]),
        x2=float(data["x2"]),
        y2=float(data["y2"]),
    )


def _ocr_image_to_dict(image: OcrImage) -> dict:
    return {
        "image_name": image.image_name,
        "width": image.width,
        "height": image.height,
        "lines": [_ocr_line_to_dict(line) for line in image.lines],
    }


def _ocr_image_from_dict(data: dict) -> OcrImage:
    return OcrImage(
        image_name=str(data["image_name"]),
        width=int(data["width"]),
        height=int(data["height"]),
        lines=[_ocr_line_from_dict(line) for line in data.get("lines", [])],
    )


def _event_to_dict(event: EventRow) -> dict:
    return {
        "date_text": event.date_text,
        "performer": event.performer,
        "venue": event.venue,
        "image_name": event.image_name,
        "source_note": event.source_note,
    }


def _event_from_dict(data: dict) -> EventRow:
    return EventRow(
        date_text=str(data["date_text"]),
        performer=str(data["performer"]),
        venue=str(data.get("venue", "")),
        image_name=str(data.get("image_name", "")),
        source_note=str(data.get("source_note", "")),
    )


def _cache_path(cache_dir: Path, xhs_url: str) -> Path:
    return cache_dir / f"{cache_key_for_xhs(xhs_url)}.json"


def _xhs_event_cache_path(cache_dir: Path, xhs_url: str, recognition_source: str) -> Path:
    return cache_dir / f"{cache_key_for_xhs(xhs_url)}-{_recognition_source_key(recognition_source)}-events.json"


def _uploaded_event_cache_path(cache_dir: Path, image_paths: list[Path], recognition_source: str) -> Path | None:
    fingerprint = fingerprint_image_files(image_paths)
    if not fingerprint:
        return None
    return cache_dir / f"uploaded-{fingerprint}-{_recognition_source_key(recognition_source)}-events.json"


def fingerprint_image_files(image_paths: list[Path]) -> str:
    if not image_paths:
        return ""
    digest = hashlib.sha256()
    for image_path in image_paths:
        try:
            digest.update(hashlib.sha256(image_path.read_bytes()).hexdigest().encode("utf-8"))
        except OSError:
            return ""
        digest.update(b"\n")
    return digest.hexdigest()


def load_xhs_ocr_cache(
    xhs_url: str,
    image_urls: list[str],
    output_root: Path | None = None,
) -> list[OcrImage] | None:
    if not cache_enabled() or not xhs_url.strip() or not image_urls:
        return None

    cache_dir = cache_root(output_root)
    path = _cache_path(cache_dir, xhs_url)
    if not path.exists():
        return None

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    if payload.get("cache_version") != CACHE_VERSION:
        return None
    if payload.get("xhs_url") != normalize_xhs_url(xhs_url):
        return None
    if payload.get("image_urls_fingerprint") != fingerprint_image_urls(image_urls):
        return None

    images = payload.get("images")
    if not isinstance(images, list) or not images:
        return None

    try:
        return [_ocr_image_from_dict(item) for item in images]
    except (KeyError, TypeError, ValueError):
        return None


def save_xhs_ocr_cache(
    xhs_url: str,
    image_urls: list[str],
    ocr_images: list[OcrImage],
    output_root: Path | None = None,
) -> None:
    if not cache_enabled() or not xhs_url.strip() or not image_urls or not ocr_images:
        return

    cache_dir = cache_root(output_root)
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "cache_version": CACHE_VERSION,
        "xhs_url": normalize_xhs_url(xhs_url),
        "image_urls_fingerprint": fingerprint_image_urls(image_urls),
        "image_urls": image_urls,
        "images": [_ocr_image_to_dict(image) for image in ocr_images],
    }
    path = _cache_path(cache_dir, xhs_url)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def load_xhs_event_cache(
    xhs_url: str,
    image_urls: list[str],
    recognition_source: str,
    output_root: Path | None = None,
) -> list[EventRow] | None:
    if not cache_enabled() or not xhs_url.strip() or not image_urls or not recognition_source:
        return None

    cache_dir = cache_root(output_root)
    path = _xhs_event_cache_path(cache_dir, xhs_url, recognition_source)
    if not path.exists():
        return None

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    if payload.get("cache_version") != CACHE_VERSION:
        return None
    if payload.get("xhs_url") != normalize_xhs_url(xhs_url):
        return None
    if payload.get("image_urls_fingerprint") != fingerprint_image_urls(image_urls):
        return None
    if payload.get("recognition_source") != recognition_source:
        return None

    events = payload.get("events")
    if not isinstance(events, list) or not events:
        return None

    try:
        return [_event_from_dict(item) for item in events]
    except (KeyError, TypeError, ValueError):
        return None


def save_xhs_event_cache(
    xhs_url: str,
    image_urls: list[str],
    recognition_source: str,
    events: list[EventRow],
    output_root: Path | None = None,
) -> None:
    if not cache_enabled() or not xhs_url.strip() or not image_urls or not recognition_source or not events:
        return

    cache_dir = cache_root(output_root)
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "cache_version": CACHE_VERSION,
        "xhs_url": normalize_xhs_url(xhs_url),
        "image_urls_fingerprint": fingerprint_image_urls(image_urls),
        "recognition_source": recognition_source,
        "events": [_event_to_dict(event) for event in events],
    }
    path = _xhs_event_cache_path(cache_dir, xhs_url, recognition_source)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def load_uploaded_event_cache(
    image_paths: list[Path],
    recognition_source: str,
    output_root: Path | None = None,
) -> list[EventRow] | None:
    if not cache_enabled() or not image_paths or not recognition_source:
        return None

    cache_dir = cache_root(output_root)
    path = _uploaded_event_cache_path(cache_dir, image_paths, recognition_source)
    if path is None or not path.exists():
        return None

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    if payload.get("cache_version") != CACHE_VERSION:
        return None
    if payload.get("image_files_fingerprint") != fingerprint_image_files(image_paths):
        return None
    if payload.get("recognition_source") != recognition_source:
        return None

    events = payload.get("events")
    if not isinstance(events, list) or not events:
        return None

    try:
        return [_event_from_dict(item) for item in events]
    except (KeyError, TypeError, ValueError):
        return None


def save_uploaded_event_cache(
    image_paths: list[Path],
    recognition_source: str,
    events: list[EventRow],
    output_root: Path | None = None,
) -> None:
    if not cache_enabled() or not image_paths or not recognition_source or not events:
        return

    cache_dir = cache_root(output_root)
    path = _uploaded_event_cache_path(cache_dir, image_paths, recognition_source)
    if path is None:
        return

    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "cache_version": CACHE_VERSION,
        "image_files_fingerprint": fingerprint_image_files(image_paths),
        "recognition_source": recognition_source,
        "events": [_event_to_dict(event) for event in events],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
