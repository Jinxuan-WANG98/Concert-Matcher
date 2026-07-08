import html
import unittest

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
        self.assertIn("AI \u590d\u6838", unescaped)
        self.assertIn('name="use_ai"', body)

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


if __name__ == "__main__":
    unittest.main()
