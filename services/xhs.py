from __future__ import annotations

import html
import json
import os
import re
import stat
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib import request

HEADERS = {
    "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) AppleWebKit/605.1.15 Mobile/15E148",
    "Referer": "https://www.xiaohongshu.com/",
}

_DEFAULT_WORKERS = 2 if sys.platform == "win32" else 4
_XHS_MAX_WORKERS = max(1, min(6, int(os.environ.get("XHS_MAX_WORKERS", str(_DEFAULT_WORKERS)))))
_XHS_MAX_IMAGES = max(1, min(20, int(os.environ.get("XHS_MAX_IMAGES", "20"))))

# #region agent log
_DEBUG_LOG_PATH = Path(__file__).resolve().parents[1] / "debug-1c03fb.log"


def _agent_log(hypothesis_id: str, location: str, message: str, data: dict) -> None:
    try:
        payload = {
            "sessionId": "1c03fb",
            "runId": "xhs-download-pre-fix",
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with _DEBUG_LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


# #endregion


def extract_note_image_urls(note_html: str) -> list[str]:
    tags = re.findall(r"<img\b[^>]*data-xhs-img[^>]*>", note_html or "", flags=re.I)
    urls: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        match = re.search(r'\bsrc="([^"]+)"', tag)
        if not match:
            continue
        url = html.unescape(match.group(1)).replace("http://", "https://", 1)
        if url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


def fetch_note_html(note_url: str, timeout: int = 25) -> str:
    req = request.Request(note_url, headers=HEADERS)
    with request.urlopen(req, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def fetch_note_image_urls(note_url: str, max_images: int | None = None) -> list[str]:
    limit = _XHS_MAX_IMAGES if max_images is None else max(1, min(20, max_images))
    return extract_note_image_urls(fetch_note_html(note_url))[:limit]


def _remove_existing_variants(output_dir: Path, index: int) -> None:
    stem = f"xhs_note_image_{index:02d}"
    for existing in output_dir.glob(f"{stem}.*"):
        try:
            existing.chmod(stat.S_IWRITE)
        except OSError:
            pass
        existing.unlink(missing_ok=True)


def _safe_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".part")
    if temp_path.exists():
        temp_path.unlink(missing_ok=True)
    temp_path.write_bytes(data)
    if path.exists():
        try:
            path.chmod(stat.S_IWRITE)
        except OSError:
            pass
        path.unlink(missing_ok=True)
    temp_path.replace(path)


def _download_one_image(args: tuple[int, str, Path]) -> Path:
    index, url, output_dir = args
    output_dir.mkdir(parents=True, exist_ok=True)
    # #region agent log
    _agent_log(
        "A",
        "xhs.py:_download_one_image:start",
        "download start",
        {
            "index": index,
            "output_dir": str(output_dir),
            "platform": sys.platform,
            "worker_count": _XHS_MAX_WORKERS,
        },
    )
    # #endregion
    try:
        req = request.Request(url, headers={**HEADERS, "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8"})
        with request.urlopen(req, timeout=30) as response:
            content_type = response.headers.get("content-type", "")
            ext = ".png" if "png" in content_type else ".webp" if "webp" in content_type else ".jpg"
            path = output_dir / f"xhs_note_image_{index:02d}{ext}"
            _remove_existing_variants(output_dir, index)
            data = response.read()
            _safe_write_bytes(path, data)
        # #region agent log
        _agent_log(
            "B",
            "xhs.py:_download_one_image:success",
            "download success",
            {"index": index, "path": str(path), "bytes": len(data)},
        )
        # #endregion
        return path
    except Exception as exc:
        target = output_dir / f"xhs_note_image_{index:02d}.jpg"
        exists = target.exists()
        readonly = False
        if exists:
            try:
                readonly = not os.access(target, os.W_OK)
            except OSError:
                readonly = True
        # #region agent log
        _agent_log(
            "C",
            "xhs.py:_download_one_image:error",
            "download failed",
            {
                "index": index,
                "path": str(target),
                "error_type": type(exc).__name__,
                "error": str(exc)[:300],
                "exists": exists,
                "readonly": readonly,
                "dir_writable": os.access(output_dir, os.W_OK),
            },
        )
        # #endregion
        raise


def download_note_images(
    note_url: str,
    output_dir: Path,
    max_images: int = 20,
    image_urls: list[str] | None = None,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    urls = (image_urls or fetch_note_image_urls(note_url, max_images=max_images))[:max_images]
    # #region agent log
    _agent_log(
        "D",
        "xhs.py:download_note_images",
        "download batch",
        {
            "output_dir": str(output_dir),
            "url_count": len(urls),
            "workers": min(_XHS_MAX_WORKERS, len(urls)) if urls else 0,
            "reused_urls": image_urls is not None,
        },
    )
    # #endregion
    if not urls:
        return []

    tasks = [(index, url, output_dir) for index, url in enumerate(urls, start=1)]
    if len(tasks) == 1:
        return [_download_one_image(tasks[0])]

    workers = min(_XHS_MAX_WORKERS, len(tasks))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(_download_one_image, tasks))
