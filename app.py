from __future__ import annotations

import os
import re
import shutil
import time
import uuid
from pathlib import Path
from urllib.error import HTTPError

try:
    from flask import Flask, jsonify, render_template, request, send_file
except ImportError as exc:
    raise RuntimeError("Flask is not installed. Run `pip install -r requirements.txt` before starting the web app.") from exc

from services.debug_timing import debug_log
from services.job_manager import JobBusyError, MatchJobManager
from services.pipeline import run_match_pipeline, save_uploaded_images


FORBIDDEN_EXTERNAL_MESSAGE = "外部服务拒绝访问（403），请检查链接是否公开，或稍后重试。"
JOB_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


def load_local_env_file(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_local_env_file()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("MAX_UPLOAD_MB", "30")) * 1024 * 1024

OUTPUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", "outputs/webapp"))
_job_manager = MatchJobManager(
    OUTPUT_ROOT / "jobs",
    OUTPUT_ROOT,
    ttl_seconds=int(os.environ.get("JOB_TTL_SECONDS", "86400")),
    max_concurrent_jobs=int(os.environ.get("JOB_MAX_CONCURRENT", "2")),
)


@app.errorhandler(413)
def request_too_large(error):
    return jsonify({"error": "上传内容过大，请减少图片数量或压缩图片后重试。"}), 413


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/status")
def api_status():
    return jsonify({"match_in_progress": _job_manager.is_busy()})


@app.get("/api/jobs/<job_id>")
def api_job(job_id: str):
    snapshot = _job_manager.get(job_id)
    if snapshot is None:
        return jsonify({"error": "任务不存在或已过期"}), 404
    return jsonify(snapshot)


@app.post("/api/jobs/<job_id>/cancel")
def api_cancel_job(job_id: str):
    snapshot = _job_manager.cancel(job_id)
    if snapshot is None:
        return jsonify({"error": "任务不存在或已过期"}), 404
    return jsonify(snapshot)


@app.post("/api/match")
def api_match():
    netease_url = (request.form.get("netease_url") or "").strip()
    xhs_url = (request.form.get("xhs_url") or "").strip()
    uploaded_files = request.files.getlist("images")
    use_ai_values = request.form.getlist("use_ai")
    use_ai = True if not use_ai_values else any(
        value.lower() in {"1", "true", "yes", "on"} for value in use_ai_values
    )

    if not netease_url:
        return jsonify({"error": "请填写网易云歌单链接"}), 400

    job_id = uuid.uuid4().hex
    upload_dir = OUTPUT_ROOT / job_id / "uploads"
    uploaded = save_uploaded_images(uploaded_files, upload_dir)

    if not xhs_url and not uploaded:
        shutil.rmtree(OUTPUT_ROOT / job_id, ignore_errors=True)
        return jsonify({"error": "请填写小红书链接，或上传小红书图片"}), 400

    request_started = time.perf_counter()
    debug_log(
        "app.py:api_match",
        "match job accepted",
        {
            "jobId": job_id,
            "uploadedImageCount": len(uploaded),
            "hasXhsUrl": bool(xhs_url),
            "useAi": use_ai,
        },
        hypothesis_id="H5",
    )

    def run_job(progress_callback):
        try:
            result = run_match_pipeline(
                netease_url,
                xhs_url,
                uploaded_images=uploaded,
                output_root=OUTPUT_ROOT,
                use_ai=use_ai,
                job_id=job_id,
                progress_callback=progress_callback,
            )
        except HTTPError as exc:
            if exc.code == 403:
                raise RuntimeError(FORBIDDEN_EXTERNAL_MESSAGE) from exc
            raise RuntimeError(f"外部服务请求失败：HTTP {exc.code}") from exc
        except Exception as exc:
            debug_log(
                "app.py:api_match",
                "match job failed",
                {"jobId": job_id, "error": str(exc)},
                hypothesis_id="H5",
            )
            raise

        elapsed_ms = int((time.perf_counter() - request_started) * 1000)
        debug_log(
            "app.py:api_match",
            "match job completed",
            {
                "jobId": job_id,
                "elapsedMs": elapsed_ms,
                "eventCount": result.event_count,
                "matchCount": len(result.matches),
                "warningCount": len(result.warnings),
            },
            hypothesis_id="H5",
        )
        return _serialize_pipeline_result(result, total_elapsed_ms=elapsed_ms)

    try:
        created = _job_manager.start(job_id, run_job)
    except JobBusyError as exc:
        shutil.rmtree(OUTPUT_ROOT / job_id, ignore_errors=True)
        debug_log(
            "app.py:api_match",
            "match request rejected busy",
            {"activeJobCount": exc.active_job_count, "maxConcurrentJobs": exc.max_concurrent_jobs},
            hypothesis_id="H8",
        )
        return (
            jsonify({"error": "当前匹配容量已满，请稍后再试。"}),
            409,
        )

    return (
        jsonify(
            {
                "job_id": job_id,
                "state": created["state"],
                "status_url": f"/api/jobs/{job_id}",
            }
        ),
        202,
    )


def _serialize_pipeline_result(result, *, total_elapsed_ms: int) -> dict:
    download_url = f"/download/{result.excel_path.parent.name}" if result.excel_path else ""
    return {
        "matches": [
            {
                "index": match.index,
                "date": match.date_display,
                "artist": match.artist_name,
                "venue": match.venue,
                "playlist_song_count": match.playlist_song_count,
                "sample_songs": match.sample_songs,
                "confidence": match.confidence,
            }
            for match in result.matches
        ],
        "playlist_artist_count": result.playlist_artist_count,
        "event_count": result.event_count,
        "warnings": result.warnings,
        "download_url": download_url,
        "phase_timings": result.phase_timings,
        "total_elapsed_ms": total_elapsed_ms,
    }


@app.get("/download/<job_id>")
def download(job_id: str):
    if not JOB_ID_PATTERN.fullmatch(job_id):
        return jsonify({"error": "文件不存在或已过期"}), 404
    path = OUTPUT_ROOT / job_id / "matches.xlsx"
    if not path.exists():
        return jsonify({"error": "文件不存在或已过期"}), 404
    return send_file(path, as_attachment=True, download_name="演出匹配结果.xlsx")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5050")), debug=True)
