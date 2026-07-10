from pathlib import Path
import unittest


class RenderConfigTest(unittest.TestCase):
    def test_render_defaults_are_memory_conservative(self):
        text = Path("render.yaml").read_text(encoding="utf-8")

        self.assertIn("--workers 1", text)
        self.assertIn("key: WEB_CONCURRENCY", text)
        self.assertIn('value: "1"', text)
        self.assertIn("key: OCR_MAX_WORKERS", text)
        self.assertIn("key: AI_OCR_IMAGE_WORKERS", text)
        self.assertIn("key: AI_MATCH_EVENT_WORKERS", text)
        self.assertIn("key: AI_OCR_LOCAL_FALLBACK", text)
        self.assertIn("key: AI_OCR_MAX_TOKENS", text)
        self.assertIn('key: AI_OCR_PROVIDER_2_ENABLED', text)
        self.assertIn('value: "true"', text)
        self.assertIn("Qwen/Qwen3-VL-32B-Instruct", text)
        self.assertIn("GLM-4V-Flash", text)
