import html
import os
import tempfile
import unittest
from pathlib import Path
from urllib.error import HTTPError

import app
from services.models import PipelineResult


class AppRoutesTest(unittest.TestCase):
    def test_index_has_styled_ui_ai_switch_and_no_access_code(self):
        client = app.app.test_client()

        response = client.get("/")
        body = response.get_data(as_text=True)
        unescaped = html.unescape(body)

        self.assertEqual(response.status_code, 200)
        self.assertNotIn("access_code", body)
        self.assertNotIn("url_for", body)
        self.assertIn("../static/styles.css", body)
        self.assertIn("../static/app.js", body)
        self.assertIn("\u8fd1\u671f\u6709\u4f60\u559c\u6b22\u7684\u6b4c\u624b\u5728\u6f14\u51fa", unescaped)
        self.assertIn("\u56fe\u7247\u4e0a\u4f20\uff08\u53ef\u9009\uff09", unescaped)
        self.assertNotIn("\u5c0f\u7ea2\u4e66\u56fe\u7247\u515c\u5e95\u4e0a\u4f20", unescaped)
        self.assertIn("AI \u590d\u6838", unescaped)
        self.assertIn('name="use_ai"', body)
        self.assertIn('name="use_ai" type="checkbox" checked', body)

    def test_match_endpoint_returns_clear_error_for_missing_playlist(self):
        client = app.app.test_client()

        response = client.post("/api/match", data={})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json["error"], "\u8bf7\u586b\u5199\u7f51\u6613\u4e91\u6b4c\u5355\u94fe\u63a5")

    def test_match_endpoint_passes_ai_choice_to_pipeline(self):
        client = app.app.test_client()
        captured = {}
        original = app.run_match_pipeline

        def fake_pipeline(netease_url, xhs_url, uploaded_images=None, output_root=None, use_ai=False):
            captured["netease_url"] = netease_url
            captured["xhs_url"] = xhs_url
            captured["use_ai"] = use_ai
            return PipelineResult(matches=[], playlist_artist_count=0, event_count=0, warnings=[])

        app.run_match_pipeline = fake_pipeline
        try:
            response = client.post(
                "/api/match",
                data={
                    "netease_url": "https://music.163.com/#/playlist?id=1",
                    "xhs_url": "https://www.xiaohongshu.com/explore/1",
                    "use_ai": "on",
                },
            )
        finally:
            app.run_match_pipeline = original

        self.assertEqual(response.status_code, 200)
        self.assertEqual(captured["netease_url"], "https://music.163.com/#/playlist?id=1")
        self.assertEqual(captured["xhs_url"], "https://www.xiaohongshu.com/explore/1")
        self.assertTrue(captured["use_ai"])

    def test_match_endpoint_returns_clear_message_for_forbidden_external_request(self):
        client = app.app.test_client()
        original = app.run_match_pipeline
        app.run_match_pipeline = lambda *args, **kwargs: (_ for _ in ()).throw(
            HTTPError(url="https://api.example.com", code=403, msg="Forbidden", hdrs=None, fp=None)
        )
        try:
            response = client.post(
                "/api/match",
                data={
                    "netease_url": "https://music.163.com/#/playlist?id=1",
                    "xhs_url": "https://www.xiaohongshu.com/explore/1",
                },
            )
        finally:
            app.run_match_pipeline = original

        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.json["error"], "\u5916\u90e8\u670d\u52a1\u62d2\u7edd\u8bbf\u95ee\uff08403\uff09\uff0c\u8bf7\u68c0\u67e5\u94fe\u63a5\u662f\u5426\u516c\u5f00\uff0c\u6216\u7a0d\u540e\u91cd\u8bd5\u3002")

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
