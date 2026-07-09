# Public Concert Matcher Web App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a public Flask web app where users submit a NetEase playlist link and a Xiaohongshu note link, then see matched Shanghai concert results in a cute browser UI.

**Architecture:** Flask serves one HTML page and JSON endpoints. The backend is split into focused services for playlist fetching, Xiaohongshu image fetching, OCR parsing, artist matching, pipeline orchestration, and Excel export. Link scraping can fail independently; uploaded images are the fallback path.

**Tech Stack:** Python 3, Flask, Gunicorn, openpyxl, Pillow, optional RapidOCR, optional rapidfuzz, Render/Fly deployment files.

## Global Constraints

- Public use must not depend on Codex runtime or Codex quota.
- The UI result table must show: 序号, 日期, 歌手, 演出场所, 歌单出现次数, 歌单代表歌曲, 置信度.
- Xiaohongshu link extraction must have an uploaded-image fallback.
- Matching must be fuzzy and support common bilingual/alias names.
- Date parsing must support adjacent-column range endings such as `7/20 -21`.
- Do not show or require an access code in the UI.

---

### Task 1: Core Date And Matching Logic

**Files:**
- Create: `services/models.py`
- Create: `services/date_parser.py`
- Create: `services/matcher.py`
- Test: `tests/test_date_parser.py`
- Test: `tests/test_matcher.py`

**Interfaces:**
- Produces: `format_date_for_display(date_text: str) -> str`
- Produces: `merge_date_range(start_date: str, range_text: str | None) -> str`
- Produces: `match_events_to_artists(events: list[EventRow], artists: list[PlaylistArtist]) -> list[MatchResult]`

- [ ] Write failing tests for date range parsing and fuzzy/alias matching.
- [ ] Run tests and verify failures are caused by missing modules.
- [ ] Implement dataclasses, date parsing, normalization, alias matching, and confidence scoring.
- [ ] Run tests and verify they pass.

### Task 2: Data Fetching And OCR Boundaries

**Files:**
- Create: `services/netease.py`
- Create: `services/xhs.py`
- Create: `services/ocr.py`
- Test: `tests/test_fetch_parsers.py`

**Interfaces:**
- Produces: `extract_playlist_id(url: str) -> str`
- Produces: `extract_note_image_urls(html: str) -> list[str]`
- Produces: `parse_ocr_events(ocr_images: list[OcrImage]) -> list[EventRow]`

- [ ] Write failing tests for URL/id extraction, Xiaohongshu image HTML extraction, and OCR row parsing.
- [ ] Run tests and verify failures are caused by missing implementations.
- [ ] Implement standard-library HTTP helpers and parser functions.
- [ ] Run tests and verify they pass.

### Task 3: Pipeline, Excel Export, And Flask Routes

**Files:**
- Create: `services/pipeline.py`
- Create: `services/export_excel.py`
- Create: `app.py`
- Create: `tests/test_pipeline.py`

**Interfaces:**
- Produces: `run_match_pipeline(netease_url: str, xhs_url: str, uploaded_images: list[Path]) -> PipelineResult`
- Produces: `write_matches_xlsx(result: PipelineResult, output_path: Path) -> Path`
- Produces: Flask routes `GET /`, `POST /api/match`, `GET /download/<job_id>`

- [ ] Write failing pipeline test using in-memory sample artists/events.
- [ ] Implement pipeline orchestration and Excel export.
- [ ] Implement Flask routes with optional access code and file size limits.
- [ ] Run tests and verify they pass.

### Task 4: Cute Browser UI

**Files:**
- Create: `templates/index.html`
- Create: `static/styles.css`
- Create: `static/app.js`

**Interfaces:**
- Consumes: `POST /api/match`
- Produces: Result table with the required columns and Excel download link.

- [ ] Build a single-page form with NetEase URL, Xiaohongshu URL, optional access code, and optional image uploads.
- [ ] Add loading, error, empty, and result states.
- [ ] Style with soft pastel colors, rounded panels, and compact mobile-friendly layout.
- [ ] Verify page renders locally through Flask when dependencies are installed.

### Task 5: Deployment Packaging

**Files:**
- Create: `requirements.txt`
- Create: `Procfile`
- Create: `render.yaml`
- Create: `runtime.txt`
- Create: `README.md`

**Interfaces:**
- Produces: Render-compatible start command `gunicorn app:app`.

- [ ] Add production dependencies and deployment config.
- [ ] Document local run, Render deploy, optional access code, and image-upload fallback.
- [ ] Run unit tests.
- [ ] Run import checks for core modules.
- [ ] Summarize any dependency that cannot be verified locally because it is installed on deployment.
