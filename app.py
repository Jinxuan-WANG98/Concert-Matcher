from __future__ import annotations

import os
import uuid
from pathlib import Path
from urllib.error import HTTPError

try:
    from flask import Flask, jsonify, render_template, request, send_file
except ImportError as exc:
    raise RuntimeError("Flask is not installed. Run `pip install -r requirements.txt` before starting the web app.") from exc

from services.pipeline import run_match_pipeline, save_uploaded_images


FORBIDDEN_EXTERNAL_MESSAGE = "外部服务拒绝访问（403），请检查链接是否公开，或稍后重试。"


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
app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("MAX_UPLOAD_MB", "80")) * 1024 * 1024

OUTPUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", "outputs/webapp"))


@app.get("/")
def index():
    return render_template("index.html")


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
        return jsonify({"error": "\u8bf7\u586b\u5199\u7f51\u6613\u4e91\u6b4c\u5355\u94fe\u63a5"}), 400

    upload_dir = OUTPUT_ROOT / "uploads" / uuid.uuid4().hex
    uploaded = save_uploaded_images(uploaded_files, upload_dir)

    if not xhs_url and not uploaded:
        return jsonify({"error": "\u8bf7\u586b\u5199\u5c0f\u7ea2\u4e66\u94fe\u63a5\uff0c\u6216\u4e0a\u4f20\u5c0f\u7ea2\u4e66\u56fe\u7247"}), 400

    try:
        result = run_match_pipeline(
            netease_url,
            xhs_url,
            uploaded_images=uploaded,
            output_root=OUTPUT_ROOT,
            use_ai=use_ai,
        )
    except HTTPError as exc:
        if exc.code == 403:
            return jsonify({"error": FORBIDDEN_EXTERNAL_MESSAGE}), 502
        return jsonify({"error": f"外部服务请求失败：HTTP {exc.code}"}), 502
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    download_url = f"/download/{result.excel_path.parent.name}" if result.excel_path else ""
    return jsonify(
        {
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
        }
    )


@app.get("/download/<job_id>")
def download(job_id: str):
    path = OUTPUT_ROOT / job_id / "matches.xlsx"
    if not path.exists():
        return jsonify({"error": "\u6587\u4ef6\u4e0d\u5b58\u5728\u6216\u5df2\u8fc7\u671f"}), 404
    return send_file(path, as_attachment=True, download_name="\u6f14\u51fa\u5339\u914d\u7ed3\u679c.xlsx")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5050")), debug=True)
