import re
import unittest
from pathlib import Path


class RenderConfigTest(unittest.TestCase):
    def assert_env_value(self, text: str, key: str, value: str) -> None:
        pattern = rf"- key: {re.escape(key)}\s+value: [\"']?{re.escape(value)}[\"']?"
        self.assertRegex(text, pattern)

    def test_render_defaults_are_memory_conservative_and_bounded(self):
        text = Path("render.yaml").read_text(encoding="utf-8")

        self.assertIn("--worker-class gthread", text)
        self.assertIn("--threads 2", text)
        self.assertIn("--workers 1", text)
        self.assertIn("--timeout 1800", text)
        expected = {
            "WEB_CONCURRENCY": "1",
            "MAX_UPLOAD_MB": "30",
            "MAX_IMAGE_FILE_MB": "12",
            "MAX_IMAGE_PIXELS": "12000000",
            "MAX_IMAGE_DIMENSION": "12000",
            "JOB_TTL_SECONDS": "86400",
            "OCR_MAX_WORKERS": "1",
            "PIPELINE_PARALLEL_LOAD": "false",
            "AI_MATCH_CANDIDATE_LIMIT": "200",
            "AI_MATCH_EVENT_BATCH_SIZE": "40",
            "AI_MATCH_EVENT_WORKERS": "2",
            "AI_MATCH_MAX_CALLS": "20",
            "AI_MATCH_MAX_ELAPSED_SECONDS": "600",
            "AI_OCR_IMAGE_BATCH_SIZE": "1",
            "AI_OCR_IMAGE_WORKERS": "3",
            "AI_OCR_MAX_WIDTH": "1200",
            "AI_OCR_PROVIDER_FALLBACK": "true",
            "AI_OCR_TRANSIENT_RETRY_ATTEMPTS": "1",
            "AI_OCR_LOCAL_FALLBACK": "false",
        }
        for key, value in expected.items():
            self.assert_env_value(text, key, value)

        self.assertIn("Qwen/Qwen3-VL-32B-Instruct", text)
        self.assertIn("glm-4.6v", text)

    def test_env_example_matches_production_safety_limits(self):
        values = {}
        for line in Path(".env.example").read_text(encoding="utf-8").splitlines():
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                values[key] = value

        expected = {
            "MAX_UPLOAD_MB": "30",
            "MAX_IMAGE_FILE_MB": "12",
            "MAX_IMAGE_PIXELS": "12000000",
            "MAX_IMAGE_DIMENSION": "12000",
            "PIPELINE_PARALLEL_LOAD": "false",
            "AI_MATCH_CANDIDATE_LIMIT": "200",
            "AI_MATCH_EVENT_BATCH_SIZE": "40",
            "AI_MATCH_EVENT_WORKERS": "2",
            "AI_MATCH_MAX_CALLS": "20",
            "AI_MATCH_MAX_ELAPSED_SECONDS": "600",
            "AI_OCR_IMAGE_BATCH_SIZE": "1",
            "AI_OCR_IMAGE_WORKERS": "3",
            "AI_OCR_MAX_WIDTH": "1200",
            "AI_OCR_PROVIDER_FALLBACK": "true",
            "AI_OCR_TRANSIENT_RETRY_ATTEMPTS": "1",
            "AI_OCR_LOCAL_FALLBACK": "false",
        }
        for key, value in expected.items():
            self.assertEqual(values.get(key), value, key)


if __name__ == "__main__":
    unittest.main()
