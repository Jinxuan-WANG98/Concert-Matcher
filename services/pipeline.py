from __future__ import annotations

import uuid
from pathlib import Path

from services.ai_matcher import AiArtistReviewer, AiMatchConfig
from services.export_excel import write_matches_xlsx
from services.matcher import match_events_to_artists
from services.models import EventRow, PipelineResult, PlaylistArtist
from services.netease import fetch_playlist_artists
from services.ocr import ocr_images_with_rapidocr, parse_ocr_events
from services.xhs import download_note_images


def run_match_pipeline_from_data(artists: list[PlaylistArtist], events: list[EventRow], ai_reviewer=None) -> PipelineResult:
    matches = match_events_to_artists(events, artists, ai_reviewer=ai_reviewer)
    return PipelineResult(matches=matches, playlist_artist_count=len(artists), event_count=len(events))


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

    artists = fetch_playlist_artists(netease_url)
    image_paths = list(uploaded_images or [])

    if not image_paths and xhs_url.strip():
        try:
            image_paths = download_note_images(xhs_url, job_dir / "xhs_images")
        except Exception as exc:
            warnings.append(
                "\u5c0f\u7ea2\u4e66\u94fe\u63a5\u6293\u53d6\u5931\u8d25\uff1a"
                f"{exc}\u3002\u8bf7\u4e0a\u4f20\u56fe\u7247\u7ee7\u7eed\u8bc6\u522b\u3002"
            )
        if not image_paths:
            warnings.append(
                "\u6ca1\u6709\u4ece\u5c0f\u7ea2\u4e66\u94fe\u63a5\u4e2d\u8bfb\u5230\u53ef\u7528\u56fe\u7247\uff0c"
                "\u8bf7\u76f4\u63a5\u4e0a\u4f20\u7b14\u8bb0\u56fe\u7247\u3002"
            )

    if not image_paths:
        return PipelineResult(matches=[], playlist_artist_count=len(artists), event_count=0, warnings=warnings)

    ocr_images = ocr_images_with_rapidocr(image_paths)
    events = parse_ocr_events(ocr_images)
    ai_reviewer = None
    if use_ai:
        ai_config = AiMatchConfig.from_env()
        if ai_config.enabled:
            ai_reviewer = AiArtistReviewer(ai_config)
        else:
            warnings.append(
                "AI \u590d\u6838\u672a\u542f\u7528\uff1a"
                "\u670d\u52a1\u5668\u6ca1\u6709\u914d\u7f6e AI API Key\uff0c"
                "\u5df2\u4f7f\u7528\u672c\u5730\u5339\u914d\u3002"
            )
    result = run_match_pipeline_from_data(artists, events, ai_reviewer=ai_reviewer)
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
    for index, file in enumerate(files, start=1):
        filename = getattr(file, "filename", "") or f"upload_{index}.jpg"
        ext = Path(filename).suffix.lower()
        if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        path = upload_dir / f"upload_{index:02d}{ext}"
        file.save(path)
        paths.append(path)
    return paths
