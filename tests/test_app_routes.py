import html
import io
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from urllib.error import HTTPError

import app
from services.job_manager import MatchJobManager
from services.models import MatchResult, PipelineResult


class AppRoutesTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_output_root = app.OUTPUT_ROOT
        self.original_job_manager = getattr(app, "_job_manager", None)
        self.original_pipeline = app.run_match_pipeline
        app.OUTPUT_ROOT = Path(self.temp_dir.name)
        app._job_manager = MatchJobManager(
            app.OUTPUT_ROOT / "jobs",
            app.OUTPUT_ROOT,
            ttl_seconds=3600,
        )
        self.client = app.app.test_client()

    def tearDown(self):
        deadline = time.monotonic() + 3
        while app._job_manager.is_busy() and time.monotonic() < deadline:
            time.sleep(0.01)
        app.run_match_pipeline = self.original_pipeline
        app.OUTPUT_ROOT = self.original_output_root
        if self.original_job_manager is not None:
            app._job_manager = self.original_job_manager
        self.temp_dir.cleanup()

    def wait_for_job(self, job_id: str, state: str, timeout: float = 3.0) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            response = self.client.get(f"/api/jobs/{job_id}")
            if response.status_code == 200 and response.json["state"] == state:
                return response.json
            time.sleep(0.01)
        raise AssertionError(f"job {job_id} did not reach {state}")

    def test_index_has_styled_ui_ai_switch_and_no_access_code(self):
        response = self.client.get("/")
        body = response.get_data(as_text=True)
        unescaped = html.unescape(body)

        self.assertEqual(response.status_code, 200)
        self.assertNotIn("access_code", body)
        self.assertNotIn("url_for", body)
        self.assertIn("../static/styles.css", body)
        self.assertIn("../static/app.js", body)
        self.assertIn("近期有你喜欢的歌手在演出", unescaped)
        self.assertIn("图片上传（可选）", unescaped)
        self.assertNotIn("小红书图片兜底上传", unescaped)
        self.assertIn("AI 智能匹配", unescaped)
        self.assertNotIn('name="use_ai"', body)
        self.assertIn('id="result-summary"', body)

    def test_status_endpoint_reports_idle_by_default(self):
        response = self.client.get("/api/status")

        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json["match_in_progress"])
        self.assertNotIn("job_id", response.json)

    def test_download_rejects_non_job_directory_names(self):
        invalid_dir = app.OUTPUT_ROOT / "not-a-job"
        invalid_dir.mkdir(parents=True)
        (invalid_dir / "matches.xlsx").write_bytes(b"not a real workbook")

        response = self.client.get("/download/not-a-job")
        status_code = response.status_code
        response.close()

        self.assertEqual(status_code, 404)

    def test_match_returns_accepted_and_exposes_result_only_after_success(self):
        started = threading.Event()
        release = threading.Event()

        def slow_pipeline(*args, output_root=None, job_id=None, progress_callback=None, **kwargs):
            progress_callback("ai_match", 65, "正在进行 AI 匹配")
            started.set()
            release.wait(timeout=2)
            return PipelineResult(
                matches=[
                    MatchResult(
                        index=1,
                        date_text="8.27",
                        date_display="8月27日",
                        performer="PREP",
                        venue="MAO",
                        artist_name="PREP",
                        playlist_song_count=2,
                        sample_songs=["Cheapest Flight"],
                        confidence="高",
                        score=1.0,
                        match_method="AI",
                        matched_alias="PREP",
                    )
                ],
                playlist_artist_count=1,
                event_count=1,
                excel_path=Path(output_root) / job_id / "matches.xlsx",
            )

        app.run_match_pipeline = slow_pipeline
        started_at = time.monotonic()
        try:
            response = self.client.post(
                "/api/match",
                data={
                    "netease_url": "https://music.163.com/#/playlist?id=1",
                    "xhs_url": "https://www.xiaohongshu.com/explore/1",
                },
            )
            elapsed = time.monotonic() - started_at
            self.assertEqual(response.status_code, 202)
            self.assertLess(elapsed, 0.5)
            self.assertEqual(response.json["state"], "queued")
            job_id = response.json["job_id"]
            self.assertEqual(response.json["status_url"], f"/api/jobs/{job_id}")
            self.assertTrue(started.wait(timeout=1))

            running = self.client.get(response.json["status_url"])
            self.assertEqual(running.status_code, 200)
            self.assertEqual(running.json["state"], "running")
            self.assertEqual(running.json["stage"], "ai_match")
            self.assertNotIn("result", running.json)

            status = self.client.get("/api/status")
            self.assertTrue(status.json["match_in_progress"])
            self.assertNotIn("job_id", status.json)
        finally:
            release.set()

        succeeded = self.wait_for_job(job_id, "succeeded")
        self.assertEqual(succeeded["result"]["matches"][0]["artist"], "PREP")
        self.assertEqual(succeeded["result"]["download_url"], f"/download/{job_id}")

    def test_match_endpoint_rejects_concurrent_requests(self):
        started = threading.Event()
        release = threading.Event()
        first_response = {}

        def slow_pipeline(*args, **kwargs):
            started.set()
            release.wait(timeout=2)
            return PipelineResult(matches=[], playlist_artist_count=0, event_count=0, warnings=[])

        app.run_match_pipeline = slow_pipeline
        first_client = app.app.test_client()
        first = threading.Thread(
            target=lambda: first_response.setdefault(
                "response",
                first_client.post(
                    "/api/match",
                    data={
                        "netease_url": "https://music.163.com/#/playlist?id=1",
                        "xhs_url": "https://www.xiaohongshu.com/explore/1",
                    },
                ),
            )
        )
        first.start()
        try:
            self.assertTrue(started.wait(timeout=1))
            busy = self.client.post(
                "/api/match",
                data={
                    "netease_url": "https://music.163.com/#/playlist?id=1",
                    "xhs_url": "https://www.xiaohongshu.com/explore/1",
                },
            )
            self.assertEqual(busy.status_code, 409)
            self.assertIn("进行中", busy.json["error"])
            self.assertNotIn("active_job_id", busy.json)
        finally:
            release.set()
            first.join(timeout=3)
        self.assertEqual(first_response["response"].status_code, 202)

    def test_match_endpoint_returns_clear_error_for_missing_playlist(self):
        response = self.client.post("/api/match", data={})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json["error"], "请填写网易云歌单链接")

    def test_match_endpoint_rejects_missing_event_source_without_artifacts(self):
        response = self.client.post(
            "/api/match",
            data={
                "netease_url": "https://music.163.com/#/playlist?id=1",
                "images": (io.BytesIO(b"not an image"), "bad.jpg"),
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json["error"], "请填写小红书链接，或上传小红书图片")
        artifact_names = [path.name for path in app.OUTPUT_ROOT.iterdir() if path.name != "jobs"]
        self.assertEqual(artifact_names, [])

    def test_oversized_request_returns_json_413(self):
        original_limit = app.app.config["MAX_CONTENT_LENGTH"]
        app.app.config["MAX_CONTENT_LENGTH"] = 64
        try:
            response = self.client.post(
                "/api/match",
                data={
                    "netease_url": "https://music.163.com/#/playlist?id=1",
                    "xhs_url": "https://www.xiaohongshu.com/explore/1",
                    "images": (io.BytesIO(b"x" * 256), "note.jpg"),
                },
                content_type="multipart/form-data",
            )
        finally:
            app.app.config["MAX_CONTENT_LENGTH"] = original_limit

        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.json["error"], "上传内容过大，请减少图片数量或压缩图片后重试。")

    def test_match_endpoint_passes_job_and_ai_choice_to_pipeline(self):
        captured = {}

        def fake_pipeline(
            netease_url,
            xhs_url,
            uploaded_images=None,
            output_root=None,
            use_ai=False,
            job_id=None,
            progress_callback=None,
        ):
            captured.update(
                netease_url=netease_url,
                xhs_url=xhs_url,
                use_ai=use_ai,
                job_id=job_id,
                has_progress_callback=callable(progress_callback),
            )
            return PipelineResult(matches=[], playlist_artist_count=0, event_count=0)

        app.run_match_pipeline = fake_pipeline
        response = self.client.post(
            "/api/match",
            data={
                "netease_url": "https://music.163.com/#/playlist?id=1",
                "xhs_url": "https://www.xiaohongshu.com/explore/1",
                "use_ai": "on",
            },
        )

        self.assertEqual(response.status_code, 202)
        self.wait_for_job(response.json["job_id"], "succeeded")
        self.assertEqual(captured["netease_url"], "https://music.163.com/#/playlist?id=1")
        self.assertEqual(captured["xhs_url"], "https://www.xiaohongshu.com/explore/1")
        self.assertTrue(captured["use_ai"])
        self.assertEqual(captured["job_id"], response.json["job_id"])
        self.assertTrue(captured["has_progress_callback"])

    def test_background_forbidden_request_becomes_failed_job(self):
        app.run_match_pipeline = lambda *args, **kwargs: (_ for _ in ()).throw(
            HTTPError(url="https://api.example.com", code=403, msg="Forbidden", hdrs=None, fp=None)
        )
        response = self.client.post(
            "/api/match",
            data={
                "netease_url": "https://music.163.com/#/playlist?id=1",
                "xhs_url": "https://www.xiaohongshu.com/explore/1",
            },
        )

        self.assertEqual(response.status_code, 202)
        failed = self.wait_for_job(response.json["job_id"], "failed")
        self.assertEqual(failed["error"], "外部服务拒绝访问（403），请检查链接是否公开，或稍后重试。")
        self.assertNotIn("result", failed)

    def test_unknown_job_returns_404(self):
        response = self.client.get(f"/api/jobs/{'9' * 32}")

        self.assertEqual(response.status_code, 404)

    def test_static_assets_define_confidence_color_classes(self):
        js = Path("static/app.js").read_text(encoding="utf-8")
        css = Path("static/styles.css").read_text(encoding="utf-8")

        self.assertIn("confidenceClass", js)
        self.assertIn("confidence-high", js)
        self.assertIn("confidence-medium", js)
        self.assertIn("confidence-low", js)
        self.assertIn(".confidence-high", css)
        self.assertIn(".confidence-medium", css)
        self.assertIn(".confidence-low", css)

    def test_static_assets_translate_fetch_disconnect_message(self):
        js = Path("static/app.js").read_text(encoding="utf-8")

        self.assertIn("networkFailed", js)
        self.assertIn("Failed to fetch", js)
        self.assertIn("copy.networkFailed", js)

    def test_frontend_polls_persisted_job_and_renders_only_terminal_result(self):
        js = Path("static/app.js").read_text(encoding="utf-8")

        self.assertIn('const jobStorageKey = "concertMatchJobId"', js)
        self.assertIn("localStorage.getItem(jobStorageKey)", js)
        self.assertIn("localStorage.setItem(jobStorageKey, jobId)", js)
        self.assertIn("`/api/jobs/${jobId}`", js)
        self.assertIn('data.state === "succeeded"', js)
        self.assertIn("renderResults(data.result)", js)
        self.assertNotIn("renderResults(data);", js)
        self.assertNotIn("setInterval(refreshMatchStatus", js)

    def test_frontend_resumes_stored_job_without_discovering_another_users_job(self):
        js = Path("static/app.js").read_text(encoding="utf-8")

        self.assertIn("resumeStoredJob", js)
        self.assertIn("scheduleJobPoll", js)
        self.assertIn("setSubmitting(false)", js)
        self.assertIn("clearActiveJob", js)
        self.assertNotIn('fetch("/api/status"', js)
        self.assertNotIn("response.status === 409 && data.active_job_id", js)

    def test_frontend_stops_polling_for_every_terminal_job_state(self):
        js = Path("static/app.js").read_text(encoding="utf-8")

        self.assertIn("function failActiveJob", js)
        self.assertIn('failActiveJob("任务已完成，但服务器没有返回最终结果。请重新提交。")', js)
        self.assertIn("failActiveJob(data.error || copy.failed)", js)

    def test_frontend_explains_interrupted_remembered_job(self):
        js = Path("static/app.js").read_text(encoding="utf-8")

        self.assertIn("jobInterrupted", js)
        self.assertIn("response.status === 404", js)
        self.assertIn("failActiveJob(copy.jobInterrupted)", js)
        self.assertNotIn("renderResults(copy.jobInterrupted)", js)

    def test_load_local_env_file_reads_values_without_overriding_existing_env(self):
        old_existing = os.environ.get("LOCAL_ENV_EXISTING")
        old_new = os.environ.pop("LOCAL_ENV_NEW", None)
        os.environ["LOCAL_ENV_EXISTING"] = "keep"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / ".env"
                path.write_text(
                    "LOCAL_ENV_EXISTING=replace\nLOCAL_ENV_NEW=from-file\n# ignored\n",
                    encoding="utf-8",
                )

                app.load_local_env_file(path)

            self.assertEqual(os.environ["LOCAL_ENV_EXISTING"], "keep")
            self.assertEqual(os.environ["LOCAL_ENV_NEW"], "from-file")
        finally:
            if old_existing is None:
                os.environ.pop("LOCAL_ENV_EXISTING", None)
            else:
                os.environ["LOCAL_ENV_EXISTING"] = old_existing
            if old_new is None:
                os.environ.pop("LOCAL_ENV_NEW", None)
            else:
                os.environ["LOCAL_ENV_NEW"] = old_new


if __name__ == "__main__":
    unittest.main()
