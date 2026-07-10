import os
import tempfile
import unittest
from pathlib import Path
from urllib.error import HTTPError
from unittest.mock import patch

from PIL import Image

import services.ai_ocr as ai_ocr
from services.ai_ocr import (
    AiOcrClient,
    AiOcrConfig,
    AiOcrProviderConfig,
    AiOcrProviderResult,
    build_ai_ocr_payload,
    parse_ai_ocr_events,
    select_ai_ocr_events,
)
from services.models import EventRow


class AiOcrTest(unittest.TestCase):
    def _clear_ai_ocr_env(self):
        names = [
            "AI_OCR_ENABLED",
            "AI_OCR_API_KEY",
            "AI_OCR_BASE_URL",
            "AI_OCR_MODEL",
            "AI_OCR_PROVIDER_NAME",
            "AI_OCR_TIMEOUT_SECONDS",
            "AI_OCR_MAX_WIDTH",
            "AI_OCR_IMAGE_BATCH_SIZE",
            "AI_OCR_IMAGE_WORKERS",
            "AI_OCR_MAX_TOKENS",
            "AI_OCR_LOCAL_FALLBACK",
            "AI_OCR_MIN_AGREEMENT_RATIO",
            "AI_OCR_MIN_EVENTS",
        ]
        for index in range(1, 5):
            names.extend(
                [
                    f"AI_OCR_PROVIDER_{index}_ENABLED",
                    f"AI_OCR_PROVIDER_{index}_NAME",
                    f"AI_OCR_PROVIDER_{index}_API_KEY",
                    f"AI_OCR_PROVIDER_{index}_BASE_URL",
                    f"AI_OCR_PROVIDER_{index}_MODEL",
                ]
            )
        return {name: os.environ.pop(name, None) for name in names}

    def _restore_env(self, old_values):
        for name, value in old_values.items():
            if value is not None:
                os.environ[name] = value
            else:
                os.environ.pop(name, None)

    def test_config_is_disabled_without_model(self):
        old_values = self._clear_ai_ocr_env()
        try:
            os.environ["AI_OCR_ENABLED"] = "true"
            os.environ["AI_OCR_API_KEY"] = "test-key"
            config = AiOcrConfig.from_env()
        finally:
            self._restore_env(old_values)

        self.assertFalse(config.enabled)

    def test_config_loads_two_ai_ocr_providers(self):
        old_values = self._clear_ai_ocr_env()
        try:
            os.environ["AI_OCR_ENABLED"] = "true"
            os.environ["AI_OCR_PROVIDER_1_NAME"] = "siliconflow"
            os.environ["AI_OCR_PROVIDER_1_API_KEY"] = "sf-key"
            os.environ["AI_OCR_PROVIDER_1_BASE_URL"] = "https://api.siliconflow.cn/v1"
            os.environ["AI_OCR_PROVIDER_1_MODEL"] = "sf-vl"
            os.environ["AI_OCR_PROVIDER_2_NAME"] = "zhipu"
            os.environ["AI_OCR_PROVIDER_2_API_KEY"] = "zp-key"
            os.environ["AI_OCR_PROVIDER_2_BASE_URL"] = "https://open.bigmodel.cn/api/paas/v4"
            os.environ["AI_OCR_PROVIDER_2_MODEL"] = "glm-v"

            config = AiOcrConfig.from_env()
        finally:
            self._restore_env(old_values)

        self.assertTrue(config.enabled)
        self.assertEqual([provider.name for provider in config.providers], ["siliconflow", "zhipu"])
        self.assertIn("siliconflow:sf-vl", config.cache_source)
        self.assertIn("zhipu:glm-v", config.cache_source)

    def test_config_skips_disabled_ai_ocr_provider(self):
        old_values = self._clear_ai_ocr_env()
        try:
            os.environ["AI_OCR_ENABLED"] = "true"
            os.environ["AI_OCR_PROVIDER_1_NAME"] = "siliconflow"
            os.environ["AI_OCR_PROVIDER_1_API_KEY"] = "sf-key"
            os.environ["AI_OCR_PROVIDER_1_BASE_URL"] = "https://api.siliconflow.cn/v1"
            os.environ["AI_OCR_PROVIDER_1_MODEL"] = "sf-vl"
            os.environ["AI_OCR_PROVIDER_1_ENABLED"] = "false"
            os.environ["AI_OCR_PROVIDER_2_NAME"] = "zhipu"
            os.environ["AI_OCR_PROVIDER_2_API_KEY"] = "zp-key"
            os.environ["AI_OCR_PROVIDER_2_BASE_URL"] = "https://open.bigmodel.cn/api/paas/v4"
            os.environ["AI_OCR_PROVIDER_2_MODEL"] = "glm-v"

            config = AiOcrConfig.from_env()
        finally:
            self._restore_env(old_values)

        self.assertTrue(config.enabled)
        self.assertEqual([provider.name for provider in config.providers], ["zhipu"])

    def test_config_reuses_ai_match_key_for_zhipu_ocr_provider(self):
        old_values = self._clear_ai_ocr_env()
        old_match_key = os.environ.pop("AI_MATCH_API_KEY", None)
        old_match_base = os.environ.pop("AI_MATCH_BASE_URL", None)
        try:
            os.environ["AI_OCR_ENABLED"] = "true"
            os.environ["AI_OCR_PROVIDER_1_NAME"] = "siliconflow"
            os.environ["AI_OCR_PROVIDER_1_API_KEY"] = "sf-key"
            os.environ["AI_OCR_PROVIDER_1_BASE_URL"] = "https://api.siliconflow.cn/v1"
            os.environ["AI_OCR_PROVIDER_1_MODEL"] = "sf-vl"
            os.environ["AI_OCR_PROVIDER_2_NAME"] = "zhipu"
            os.environ["AI_OCR_PROVIDER_2_BASE_URL"] = "https://open.bigmodel.cn/api/paas/v4"
            os.environ["AI_OCR_PROVIDER_2_MODEL"] = "glm-v"
            os.environ["AI_MATCH_API_KEY"] = "shared-zhipu-key"
            os.environ["AI_MATCH_BASE_URL"] = "https://open.bigmodel.cn/api/paas/v4"

            config = AiOcrConfig.from_env()
        finally:
            self._restore_env(old_values)
            if old_match_key is not None:
                os.environ["AI_MATCH_API_KEY"] = old_match_key
            else:
                os.environ.pop("AI_MATCH_API_KEY", None)
            if old_match_base is not None:
                os.environ["AI_MATCH_BASE_URL"] = old_match_base
            else:
                os.environ.pop("AI_MATCH_BASE_URL", None)

        self.assertEqual([provider.name for provider in config.providers], ["siliconflow", "zhipu"])
        self.assertEqual(config.providers[1].api_key, "shared-zhipu-key")

    def test_image_is_resized_before_rgb_conversion(self):
        calls = []

        class FakeImage:
            width = 3200

            def thumbnail(self, size, resample):
                calls.append("thumbnail")

            def convert(self, mode):
                if calls != ["thumbnail"]:
                    raise AssertionError("image must be resized before RGB conversion")
                calls.append("convert")
                return self

            def save(self, output, format, quality):
                calls.append("save")
                output.write(b"compressed-image")

        class FakeImageContext:
            def __enter__(self):
                return FakeImage()

            def __exit__(self, exc_type, exc, traceback):
                return False

        with patch.object(ai_ocr.Image, "open", return_value=FakeImageContext()):
            data_url = ai_ocr._image_path_to_data_url(Path("large.jpg"), max_width=1600)

        self.assertTrue(data_url.startswith("data:image/jpeg;base64,"))
        self.assertEqual(calls, ["thumbnail", "convert", "save"])

    def test_parse_ai_ocr_events_accepts_strict_json(self):
        raw = """
        ```json
        {
          "events": [
            {"date_text": "7 / 11", "performer": "Thomas Bergersen", "venue": "MAO"},
            {"date_text": "7 / 12", "performer": "Hanser", "venue": ""}
          ]
        }
        ```
        """

        events = parse_ai_ocr_events(raw, image_name="note.jpg")

        self.assertEqual(
            [(event.date_text, event.performer, event.venue) for event in events],
            [
                ("7.11", "Thomas Bergersen", "MAO"),
                ("7.12", "Hanser", ""),
            ],
        )
        self.assertEqual(events[0].image_name, "note.jpg")

    def test_parse_ai_ocr_events_ignores_rows_without_date_or_performer(self):
        raw = '{"events":[{"date_text":"", "performer":"Header", "venue":""}, {"date_text":"7/13", "performer":"PREP"}]}'

        events = parse_ai_ocr_events(raw, image_name="summary.jpg")

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].date_text, "7.13")
        self.assertEqual(events[0].performer, "PREP")

    def test_parse_ai_ocr_events_accepts_trailing_text_after_json(self):
        raw = '{"events":[{"date_text":"7.14", "performer":"LakiPak", "venue":""}]}\nextra explanation'

        events = parse_ai_ocr_events(raw, image_name="summary.jpg")

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].date_text, "7.14")
        self.assertEqual(events[0].performer, "LakiPak")

    def test_build_ai_ocr_payload_uses_image_content_part(self):
        payload = build_ai_ocr_payload(
            model="Qwen/Qwen2.5-VL-72B-Instruct",
            image_data_url="data:image/jpeg;base64,abc",
        )

        message = payload["messages"][1]
        self.assertEqual(payload["model"], "Qwen/Qwen2.5-VL-72B-Instruct")
        self.assertEqual(payload["max_tokens"], 8192)
        self.assertEqual(message["content"][0]["type"], "image_url")
        self.assertEqual(message["content"][0]["image_url"]["detail"], "high")
        self.assertIn("JSON", message["content"][1]["text"])
        self.assertIn("穷尽", message["content"][1]["text"])

    def test_select_ai_ocr_events_merges_agreeing_provider_results(self):
        siliconflow = AiOcrProviderResult(
            provider_name="siliconflow",
            events=[
                EventRow(date_text="7.11", performer="Thomas Bergersen", venue=""),
                EventRow(date_text="7.12", performer="Hanser", venue=""),
            ],
        )
        zhipu = AiOcrProviderResult(
            provider_name="zhipu",
            events=[
                EventRow(date_text="7.11", performer="Thomas Bergersen", venue="MAO"),
                EventRow(date_text="7.12", performer="Hanser", venue=""),
                EventRow(date_text="7.13", performer="PREP", venue=""),
            ],
        )

        selected = select_ai_ocr_events([siliconflow, zhipu])

        self.assertEqual(
            [(event.date_text, event.performer, event.venue) for event in selected],
            [
                ("7.11", "Thomas Bergersen", "MAO"),
                ("7.12", "Hanser", ""),
                ("7.13", "PREP", ""),
            ],
        )

    def test_select_ai_ocr_events_merges_distributed_provider_results_without_agreement(self):
        siliconflow = AiOcrProviderResult(
            provider_name="siliconflow",
            events=[
                EventRow(date_text="7.11", performer="Thomas Bergersen", venue=""),
                EventRow(date_text="7.12", performer="Hanser", venue=""),
            ],
        )
        zhipu = AiOcrProviderResult(
            provider_name="zhipu",
            events=[
                EventRow(date_text="9.24", performer="DAKOOKA", venue=""),
                EventRow(date_text="10.3", performer="\u91cd\u8fd4\u672a\u6765\uff1a1999", venue=""),
            ],
        )

        selected = select_ai_ocr_events([siliconflow, zhipu])

        self.assertEqual(
            [(event.date_text, event.performer) for event in selected],
            [
                ("7.11", "Thomas Bergersen"),
                ("7.12", "Hanser"),
                ("9.24", "DAKOOKA"),
                ("10.3", "\u91cd\u8fd4\u672a\u6765\uff1a1999"),
            ],
        )

    def test_extract_events_with_ai_ocr_uses_friendly_warning_for_forbidden_provider(self):
        old_values = self._clear_ai_ocr_env()
        original_extract = ai_ocr.AiOcrClient._extract_batch_events
        warnings = []
        try:
            os.environ["AI_OCR_ENABLED"] = "true"
            os.environ["AI_OCR_PROVIDER_1_NAME"] = "zhipu"
            os.environ["AI_OCR_PROVIDER_1_API_KEY"] = "test-key"
            os.environ["AI_OCR_PROVIDER_1_BASE_URL"] = "https://open.bigmodel.cn/api/paas/v4"
            os.environ["AI_OCR_PROVIDER_1_MODEL"] = "GLM-4V-Flash"

            def forbidden_extract(self, image_paths):
                raise HTTPError(url="https://api.example.com", code=403, msg="Forbidden", hdrs=None, fp=None)

            ai_ocr.AiOcrClient._extract_batch_events = forbidden_extract

            events = ai_ocr.extract_events_with_ai_ocr([Path("note.jpg")], warnings)
        finally:
            ai_ocr.AiOcrClient._extract_batch_events = original_extract
            self._restore_env(old_values)

        self.assertEqual(events, [])
        self.assertIn("AI \u8bc6\u522b\u5931\u8d25", " ".join(warnings))
        self.assertIn("403", " ".join(warnings))
        self.assertNotIn("HTTP Error 403: Forbidden", " ".join(warnings))

    def test_client_sends_multiple_images_in_one_request_when_batch_size_allows(self):
        original_urlopen = ai_ocr.request.urlopen
        request_payloads = []

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                content = '{"events":[{"date_text":"7.11","performer":"PREP","venue":""}]}'
                payload = {"choices": [{"message": {"content": content}}]}
                return ai_ocr.json.dumps(payload).encode("utf-8")

        def fake_urlopen(req, timeout):
            request_payloads.append(ai_ocr.json.loads(req.data.decode("utf-8")))
            return FakeResponse()

        with tempfile.TemporaryDirectory() as tmp:
            image_paths = []
            for index in range(3):
                path = Path(tmp) / f"image_{index}.jpg"
                Image.new("RGB", (10, 10), color=(index, index, index)).save(path)
                image_paths.append(path)

            client = AiOcrClient(
                AiOcrProviderConfig(
                    name="test",
                    api_key="key",
                    base_url="https://api.example.com/v1",
                    model="vision-model",
                ),
                timeout_seconds=1,
                max_width=16,
            )
            client.image_batch_size = 10
            ai_ocr.request.urlopen = fake_urlopen
            try:
                events = client.extract_events(image_paths)
            finally:
                ai_ocr.request.urlopen = original_urlopen

        self.assertEqual(len(events), 1)
        self.assertEqual(len(request_payloads), 1)
        content = request_payloads[0]["messages"][1]["content"]
        image_parts = [part for part in content if part["type"] == "image_url"]
        self.assertEqual(len(image_parts), 3)

    def test_client_repairs_malformed_ai_ocr_json_with_ai(self):
        original_urlopen = ai_ocr.request.urlopen
        request_payloads = []

        class FakeResponse:
            def __init__(self, content):
                self.content = content

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                payload = {"choices": [{"message": {"content": self.content}}]}
                return ai_ocr.json.dumps(payload).encode("utf-8")

        responses = [
            FakeResponse("events: date 7/11 performer PREP venue MAO"),
            FakeResponse('{"events":[{"date_text":"7.11","performer":"PREP","venue":"MAO"}]}'),
        ]

        def fake_urlopen(req, timeout):
            request_payloads.append(ai_ocr.json.loads(req.data.decode("utf-8")))
            return responses.pop(0)

        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "image.jpg"
            Image.new("RGB", (10, 10)).save(image_path)
            client = AiOcrClient(
                AiOcrProviderConfig(
                    name="test",
                    api_key="key",
                    base_url="https://api.example.com/v1",
                    model="vision-model",
                ),
                timeout_seconds=1,
                max_width=16,
            )
            ai_ocr.request.urlopen = fake_urlopen
            try:
                events = client.extract_events([image_path])
            finally:
                ai_ocr.request.urlopen = original_urlopen

        self.assertEqual([(event.date_text, event.performer, event.venue) for event in events], [("7.11", "PREP", "MAO")])
        self.assertEqual(len(request_payloads), 2)
        self.assertIn("修复", request_payloads[1]["messages"][0]["content"])

    def test_extract_events_with_ai_ocr_distributes_batches_across_providers(self):
        old_values = self._clear_ai_ocr_env()
        original_extract_batch = ai_ocr.AiOcrClient._extract_batch_events
        calls = []
        warnings = []
        try:
            os.environ["AI_OCR_ENABLED"] = "true"
            os.environ["AI_OCR_IMAGE_BATCH_SIZE"] = "2"
            os.environ["AI_OCR_PROVIDER_1_NAME"] = "siliconflow"
            os.environ["AI_OCR_PROVIDER_1_API_KEY"] = "sf-key"
            os.environ["AI_OCR_PROVIDER_1_BASE_URL"] = "https://api.example.com/v1"
            os.environ["AI_OCR_PROVIDER_1_MODEL"] = "sf-vl"
            os.environ["AI_OCR_PROVIDER_2_NAME"] = "zhipu"
            os.environ["AI_OCR_PROVIDER_2_API_KEY"] = "zp-key"
            os.environ["AI_OCR_PROVIDER_2_BASE_URL"] = "https://api.example.com/v1"
            os.environ["AI_OCR_PROVIDER_2_MODEL"] = "zp-vl"

            def fake_extract_batch(self, batch):
                calls.append((self.provider.name, [path.name for path in batch]))
                return [EventRow(date_text="7.11", performer=self.provider.name, venue="", image_name="+".join(path.name for path in batch))]

            ai_ocr.AiOcrClient._extract_batch_events = fake_extract_batch
            image_paths = [Path(f"image_{index}.jpg") for index in range(4)]

            events = ai_ocr.extract_events_with_ai_ocr(image_paths, warnings)
        finally:
            ai_ocr.AiOcrClient._extract_batch_events = original_extract_batch
            self._restore_env(old_values)

        self.assertEqual([call[0] for call in calls], ["siliconflow", "zhipu"])
        self.assertEqual([event.performer for event in events], ["siliconflow", "zhipu"])
        self.assertIn("分批并行", " ".join(warnings))


if __name__ == "__main__":
    unittest.main()
