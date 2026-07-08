from __future__ import annotations

import html
import re
from pathlib import Path
from urllib import request

HEADERS = {
    "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) AppleWebKit/605.1.15 Mobile/15E148",
    "Referer": "https://www.xiaohongshu.com/",
}


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


def download_note_images(note_url: str, output_dir: Path, max_images: int = 20) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    note_html = fetch_note_html(note_url)
    urls = extract_note_image_urls(note_html)[:max_images]
    paths: list[Path] = []
    for index, url in enumerate(urls, start=1):
        req = request.Request(url, headers={**HEADERS, "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8"})
        with request.urlopen(req, timeout=30) as response:
            content_type = response.headers.get("content-type", "")
            ext = ".png" if "png" in content_type else ".webp" if "webp" in content_type else ".jpg"
            path = output_dir / f"xhs_note_image_{index:02d}{ext}"
            path.write_bytes(response.read())
            paths.append(path)
    return paths
