from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError

from PIL import Image

from services.ai_errors import describe_ai_exception
from services.ai_ocr import AiOcrConfig, extract_events_with_ai_ocr
from services.ai_matcher import AiArtistReviewer, AiMatchConfig
from services.export_excel import write_matches_xlsx
from services.matcher import match_events_to_artists
from services.models import EventRow, PipelineResult, PlaylistArtist
from services.netease import fetch_playlist_artists
from services.ocr import OcrImage, ocr_images_with_rapidocr, parse_ocr_events
from services.ocr_cache import (
    load_uploaded_event_cache,
    load_xhs_event_cache,
    load_xhs_ocr_cache,
    save_uploaded_event_cache,
    save_xhs_event_cache,
    save_xhs_ocr_cache,
)
from services.xhs import download_note_images, fetch_note_image_urls


def run_match_pipeline_from_data(
    artists: list[PlaylistArtist],
    events: list[EventRow],
    ai_reviewer=None,
    ai_only: bool = False,
) -> PipelineResult:
    matches = match_events_to_artists(events, artists, ai_reviewer=ai_reviewer, ai_only=ai_only)
    return PipelineResult(matches=matches, playlist_artist_count=len(artists), event_count=len(events))


def _load_playlist_artists(netease_url: str) -> list[PlaylistArtist]:
    try:
        return fetch_playlist_artists(netease_url)
    except HTTPError as exc:
        if exc.code == 403:
            raise RuntimeError(
                "网易云歌单读取被拒绝：网易云返回 403。"
                "请确认歌单是公开的，或换一个可公开访问的分享链接后重试。"
            ) from exc
        raise RuntimeError(f"网易云歌单读取失败：HTTP {exc.code} {exc.reason}") from exc


class _SafeAiReviewer:
    def __init__(self, reviewer, warnings: list[str]):
        self._reviewer = reviewer
        self._warnings = warnings
        self._failed = False

    def _warn_once(self, exc: Exception) -> None:
        if self._failed:
            return
        self._failed = True
        self._warnings.append(f"AI 匹配失败：{describe_ai_exception(exc)}")

    def review(self, event: EventRow, artist: PlaylistArtist):
        if self._failed:
            return None
        try:
            return self._reviewer.review(event, artist)
        except Exception as exc:
            self._warn_once(exc)
            return None

    def find_best_match(self, event: EventRow, artists: list[PlaylistArtist]):
        if self._failed or not hasattr(self._reviewer, "find_best_match"):
            return None
        try:
            return self._reviewer.find_best_match(event, artists)
        except Exception as exc:
            self._warn_once(exc)
            return None

    def find_best_matches(self, events: list[EventRow], artists: list[PlaylistArtist]):
        if self._failed or not hasattr(self._reviewer, "find_best_matches"):
            return {}
        try:
            return self._reviewer.find_best_matches(events, artists)
        except Exception as exc:
            self._warn_once(exc)
            return {}


def _load_ocr_images_for_xhs(
    xhs_url: str,
    job_dir: Path,
    output_root: Path,
    warnings: list[str],
) -> list[OcrImage]:
    note_image_urls = fetch_note_image_urls(xhs_url)
    if not note_image_urls:
        return []

    cached_images = load_xhs_ocr_cache(xhs_url, note_image_urls, output_root=output_root)
    if cached_images is not None:
        warnings.append(
            "\u5df2\u547d\u4e2d OCR \u7f13\u5b58\uff0c\u8df3\u8fc7\u5c0f\u7ea2\u4e66\u56fe\u7247\u4e0b\u8f7d\u548c\u8bc6\u522b\u3002"
        )
        return cached_images

    image_paths = download_note_images(xhs_url, job_dir / "xhs_images", image_urls=note_image_urls)
    if not image_paths:
        return []

    ocr_images = ocr_images_with_rapidocr(image_paths)
    save_xhs_ocr_cache(xhs_url, note_image_urls, ocr_images, output_root=output_root)
    return ocr_images


def _rapidocr_events_from_paths(image_paths: list[Path]) -> list[EventRow]:
    return parse_ocr_events(ocr_images_with_rapidocr(image_paths))


def _load_events_for_uploaded_images(
    image_paths: list[Path],
    output_root: Path,
    warnings: list[str],
) -> list[EventRow]:
    ai_config = AiOcrConfig.from_env()
    if ai_config.enabled:
        cached_ai_events = load_uploaded_event_cache(image_paths, ai_config.cache_source, output_root=output_root)
        if cached_ai_events is not None:
            warnings.append("已命中 AI 识别缓存，跳过图片重新识别。")
            return cached_ai_events

        warning_count = len(warnings)
        ai_events = extract_events_with_ai_ocr(image_paths, warnings)
        if ai_events:
            if len(warnings) == warning_count:
                warnings.append("AI 识别已用于图片读取。")
            save_uploaded_event_cache(image_paths, ai_config.cache_source, ai_events, output_root=output_root)
            return ai_events
        if not ai_config.local_fallback_enabled:
            if len(warnings) == warning_count:
                warnings.append("AI 识别未返回可用行，未自动调用本地 OCR。")
            return []
        warnings.append("AI 识别未返回可用行，已按配置回退本地 OCR。")
        if len(warnings) == warning_count:
            warnings.append("AI 识别未返回可用行，已回退本地 OCR。")

    cached_local_events = load_uploaded_event_cache(image_paths, "rapidocr", output_root=output_root)
    if cached_local_events is not None:
        warnings.append("已命中上传图片 OCR 缓存，跳过图片重新识别。")
        return cached_local_events

    events = _rapidocr_events_from_paths(image_paths)
    save_uploaded_event_cache(image_paths, "rapidocr", events, output_root=output_root)
    return events


def _load_events_for_xhs(
    xhs_url: str,
    job_dir: Path,
    output_root: Path,
    warnings: list[str],
) -> list[EventRow]:
    note_image_urls = fetch_note_image_urls(xhs_url)
    if not note_image_urls:
        return []

    ai_config = AiOcrConfig.from_env()
    if ai_config.enabled:
        cached_ai_events = load_xhs_event_cache(
            xhs_url,
            note_image_urls,
            ai_config.cache_source,
            output_root=output_root,
        )
        if cached_ai_events is not None:
            warnings.append("已命中 AI 识别缓存，跳过小红书图片下载和识别。")
            return cached_ai_events

    cached_images = load_xhs_ocr_cache(xhs_url, note_image_urls, output_root=output_root)
    cached_local_events = load_xhs_event_cache(xhs_url, note_image_urls, "rapidocr", output_root=output_root)
    if not ai_config.enabled:
        if cached_local_events is not None:
            warnings.append("已命中 OCR 缓存，跳过小红书图片下载和识别。")
            return cached_local_events
        if cached_images is not None:
            warnings.append("已命中 OCR 缓存，跳过小红书图片下载和识别。")
            events = parse_ocr_events(cached_images)
            save_xhs_event_cache(xhs_url, note_image_urls, "rapidocr", events, output_root=output_root)
            return events

    image_paths = download_note_images(xhs_url, job_dir / "xhs_images", image_urls=note_image_urls)
    if not image_paths:
        return []

    if ai_config.enabled:
        warning_count = len(warnings)
        ai_events = extract_events_with_ai_ocr(image_paths, warnings)
        if ai_events:
            if len(warnings) == warning_count:
                warnings.append("AI 识别已用于图片读取。")
            save_xhs_event_cache(xhs_url, note_image_urls, ai_config.cache_source, ai_events, output_root=output_root)
            return ai_events
        if not ai_config.local_fallback_enabled:
            if len(warnings) == warning_count:
                warnings.append("AI 识别未返回可用行，未自动调用本地 OCR。")
            return []
        warnings.append("AI 识别未返回可用行，已按配置回退本地 OCR。")
        if len(warnings) == warning_count:
            warnings.append("AI 识别未返回可用行，已回退本地 OCR。")
        if cached_local_events is not None:
            warnings.append("已命中本地 OCR 缓存，AI 回退时跳过重新识别。")
            return cached_local_events
        if cached_images is not None:
            warnings.append("已命中本地 OCR 缓存，AI 回退时跳过重新识别。")
            events = parse_ocr_events(cached_images)
            save_xhs_event_cache(xhs_url, note_image_urls, "rapidocr", events, output_root=output_root)
            return events

    ocr_images = ocr_images_with_rapidocr(image_paths)
    save_xhs_ocr_cache(xhs_url, note_image_urls, ocr_images, output_root=output_root)
    events = parse_ocr_events(ocr_images)
    save_xhs_event_cache(xhs_url, note_image_urls, "rapidocr", events, output_root=output_root)
    return events


def run_match_pipeline(
    netease_url: str,
    xhs_url: str,
    uploaded_images: list[Path] | None = None,
    output_root: Path | None = None,
    use_ai: bool = False,
) -> PipelineResult:
    warnings: list[str] = []
    output_root = output_root or Path("outputs/webapp")
    job_dir = output_root / uuid.uuid4().hex
    job_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list(uploaded_images or [])
    events: list[EventRow] | None = None
    artists: list[PlaylistArtist]

    if not image_paths and xhs_url.strip():
        with ThreadPoolExecutor(max_workers=2) as executor:
            artists_future = executor.submit(_load_playlist_artists, netease_url)
            events_future = executor.submit(_load_events_for_xhs, xhs_url, job_dir, output_root, warnings)
            try:
                events = events_future.result()
            except Exception as exc:
                warnings.append(
                    "\u5c0f\u7ea2\u4e66\u94fe\u63a5\u6293\u53d6\u5931\u8d25\uff1a"
                    f"{exc}\u3002\u8bf7\u4e0a\u4f20\u56fe\u7247\u7ee7\u7eed\u8bc6\u522b\u3002"
                )
                events = []
            artists = artists_future.result()
    else:
        artists = _load_playlist_artists(netease_url)

    if image_paths:
        events = _load_events_for_uploaded_images(image_paths, output_root, warnings)

    if events is None:
        events = []

    if not image_paths and not events and xhs_url.strip():
        warnings.append(
            "\u6ca1\u6709\u4ece\u5c0f\u7ea2\u4e66\u94fe\u63a5\u4e2d\u8bfb\u5230\u53ef\u7528\u56fe\u7247\uff0c"
            "\u8bf7\u76f4\u63a5\u4e0a\u4f20\u7b14\u8bb0\u56fe\u7247\u3002"
        )

    if not image_paths and not events:
        return PipelineResult(matches=[], playlist_artist_count=len(artists), event_count=0, warnings=warnings)

    ai_reviewer = None
    ai_only = False
    if use_ai:
        ai_config = AiMatchConfig.from_env()
        if ai_config.enabled:
            ai_reviewer = _SafeAiReviewer(AiArtistReviewer(ai_config, output_root=output_root), warnings)
            ai_only = ai_config.mode == "ai_only"
        else:
            warnings.append(
                "AI \u590d\u6838\u672a\u542f\u7528\uff1a"
                "\u670d\u52a1\u5668\u6ca1\u6709\u914d\u7f6e AI API Key\uff0c"
                "\u5df2\u4f7f\u7528\u672c\u5730\u5339\u914d\u3002"
            )
    result = run_match_pipeline_from_data(artists, events, ai_reviewer=ai_reviewer, ai_only=ai_only)
    result = PipelineResult(
        matches=result.matches,
        playlist_artist_count=result.playlist_artist_count,
        event_count=result.event_count,
        warnings=warnings,
    )
    excel_path = write_matches_xlsx(result, job_dir / "matches.xlsx")
    return PipelineResult(
        matches=result.matches,
        playlist_artist_count=result.playlist_artist_count,
        event_count=result.event_count,
        excel_path=excel_path,
        warnings=result.warnings,
    )


def save_uploaded_images(files, upload_dir: Path) -> list[Path]:
    upload_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    saved_count = 0
    for file in files:
        raw_filename = (getattr(file, "filename", "") or "").strip()
        if not raw_filename:
            continue
        ext = Path(raw_filename).suffix.lower()
        if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        saved_count += 1
        path = upload_dir / f"upload_{saved_count:02d}{ext}"
        file.save(path)
        file_size = path.stat().st_size if path.exists() else -1
        if file_size <= 0:
            path.unlink(missing_ok=True)
            continue
        try:
            with Image.open(path) as img:
                img.verify()
        except Exception:
            path.unlink(missing_ok=True)
            continue
        paths.append(path)
    return paths
