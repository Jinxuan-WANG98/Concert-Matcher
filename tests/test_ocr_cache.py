import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from services.ocr import OcrImage, OcrLine
from services.ocr_cache import (
    cache_key_for_xhs,
    fingerprint_image_urls,
    load_uploaded_event_cache,
    load_xhs_event_cache,
    load_xhs_ocr_cache,
    normalize_xhs_url,
    save_uploaded_event_cache,
    save_xhs_event_cache,
    save_xhs_ocr_cache,
)
from services.models import EventRow, PlaylistArtist
import services.pipeline as pipeline
from services.pipeline import run_match_pipeline


class OcrCacheTest(unittest.TestCase):
    def test_normalize_xhs_url_strips_tracking_query(self):
        raw = "https://www.xiaohongshu.com/explore/abc123?xsec_token=foo&xsec_source=pc"
        self.assertEqual(normalize_xhs_url(raw), "https://www.xiaohongshu.com/explore/abc123")

    def test_save_and_load_roundtrip(self):
        urls = ["https://example.com/a.jpg", "https://example.com/b.jpg"]
        images = [
            OcrImage(
                image_name="xhs_note_image_01.jpg",
                width=1080,
                height=1440,
                lines=[OcrLine(text="7 / 11", x1=16, y1=80, x2=83, y2=90)],
            )
        ]
        xhs_url = "https://www.xiaohongshu.com/explore/note-1"

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_xhs_ocr_cache(xhs_url, urls, images, output_root=root)
            loaded = load_xhs_ocr_cache(xhs_url, urls, output_root=root)

        self.assertIsNotNone(loaded)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].image_name, "xhs_note_image_01.jpg")
        self.assertEqual(loaded[0].lines[0].text, "7 / 11")

    def test_cache_miss_when_image_urls_change(self):
        xhs_url = "https://www.xiaohongshu.com/explore/note-2"
        images = [OcrImage(image_name="xhs_note_image_01.jpg", width=100, height=100, lines=[])]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_xhs_ocr_cache(xhs_url, ["https://example.com/a.jpg"], images, output_root=root)
            loaded = load_xhs_ocr_cache(xhs_url, ["https://example.com/b.jpg"], output_root=root)

        self.assertIsNone(loaded)

    def test_xhs_image_fingerprint_uses_stable_spectrum_id(self):
        first = [
            "https://sns-webpic-qc.xhscdn.com/20260708/spectrum/abc123!nd_dft_wlteh_webp_3?imageView2/2/w/1080"
        ]
        second = [
            "https://sns-webpic-qc.xhscdn.com/20260709/spectrum/abc123!nd_dft_wlteh_webp_3?imageView2/2/w/1440"
        ]

        self.assertEqual(fingerprint_image_urls(first), fingerprint_image_urls(second))

    def test_xhs_cache_hits_when_only_cdn_url_expiry_changes(self):
        xhs_url = "https://www.xiaohongshu.com/explore/note-2b"
        images = [OcrImage(image_name="xhs_note_image_01.jpg", width=100, height=100, lines=[])]
        first = [
            "https://sns-webpic-qc.xhscdn.com/20260708/spectrum/abc123!nd_dft_wlteh_webp_3?imageView2/2/w/1080"
        ]
        second = [
            "https://sns-webpic-qc.xhscdn.com/20260709/spectrum/abc123!nd_dft_wlteh_webp_3?imageView2/2/w/1440"
        ]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_xhs_ocr_cache(xhs_url, first, images, output_root=root)
            loaded = load_xhs_ocr_cache(xhs_url, second, output_root=root)

        self.assertIsNotNone(loaded)

    def test_xhs_event_cache_roundtrip(self):
        xhs_url = "https://www.xiaohongshu.com/explore/note-events"
        urls = ["https://example.com/a.jpg"]
        events = [EventRow(date_text="7.11", performer="Thomas Bergersen", venue="MAO", image_name="a.jpg")]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_xhs_event_cache(xhs_url, urls, "ai-ocr:model", events, output_root=root)
            loaded = load_xhs_event_cache(xhs_url, urls, "ai-ocr:model", output_root=root)

        self.assertEqual(loaded, events)

    def test_uploaded_event_cache_uses_file_content_hash(self):
        events = [EventRow(date_text="7.12", performer="Hanser", venue="", image_name="upload.jpg")]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_a = root / "first.jpg"
            image_b = root / "second.jpg"
            image_a.write_bytes(b"same image bytes")
            image_b.write_bytes(b"same image bytes")

            save_uploaded_event_cache([image_a], "rapidocr", events, output_root=root)
            loaded = load_uploaded_event_cache([image_b], "rapidocr", output_root=root)

        self.assertEqual(loaded, events)

    def test_cache_key_is_stable_for_same_note(self):
        a = "https://www.xiaohongshu.com/explore/note-3?foo=1"
        b = "https://www.xiaohongshu.com/explore/note-3?bar=2"
        self.assertEqual(cache_key_for_xhs(a), cache_key_for_xhs(b))


class PipelineOcrCacheTest(unittest.TestCase):
    def _disable_ai_ocr_env(self):
        names = [
            "AI_OCR_ENABLED",
            "AI_OCR_API_KEY",
            "AI_OCR_BASE_URL",
            "AI_OCR_MODEL",
            "AI_OCR_PROVIDER_NAME",
        ]
        for index in range(1, 5):
            names.extend([
                f"AI_OCR_PROVIDER_{index}_NAME",
                f"AI_OCR_PROVIDER_{index}_API_KEY",
                f"AI_OCR_PROVIDER_{index}_BASE_URL",
                f"AI_OCR_PROVIDER_{index}_MODEL",
            ])
        old_values = {name: os.environ.pop(name, None) for name in names}
        os.environ["AI_OCR_ENABLED"] = "false"
        return old_values

    def _restore_env(self, old_values):
        os.environ.pop("AI_OCR_ENABLED", None)
        for name, value in old_values.items():
            if value is not None:
                os.environ[name] = value
            else:
                os.environ.pop(name, None)

    def test_pipeline_uses_xhs_ocr_cache_on_second_run(self):
        urls = ["https://example.com/a.jpg"]
        cached_images = [
            OcrImage(
                image_name="xhs_note_image_01.jpg",
                width=1080,
                height=1440,
                lines=[
                    OcrLine(text="8 / 27", x1=16, y1=80, x2=83, y2=90),
                    OcrLine(text="ZeIIa Day", x1=500, y1=80, x2=700, y2=90),
                    OcrLine(text="MAO", x1=950, y1=80, x2=1000, y2=90),
                ],
            )
        ]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_xhs_ocr_cache("https://www.xiaohongshu.com/explore/cache-note", urls, cached_images, output_root=root)
            old_ai_ocr_env = self._disable_ai_ocr_env()

            original_fetch = pipeline.fetch_playlist_artists
            original_download = pipeline.download_note_images
            original_ocr = pipeline.ocr_images_with_rapidocr
            ocr_calls = {"count": 0}

            def fake_ocr(paths):
                ocr_calls["count"] += 1
                return []

            pipeline.fetch_playlist_artists = lambda url: [
                PlaylistArtist(name="Zella Day", song_count=1, sample_songs=["Hypnotic"])
            ]
            pipeline.download_note_images = lambda url, output_dir, image_urls=None: (_ for _ in ()).throw(AssertionError("should not download"))
            pipeline.ocr_images_with_rapidocr = fake_ocr

            try:
                with patch("services.pipeline.fetch_note_image_urls", return_value=urls):
                    result = run_match_pipeline(
                        "https://music.163.com/#/playlist?id=1",
                        "https://www.xiaohongshu.com/explore/cache-note?x=1",
                        output_root=root,
                    )
            finally:
                pipeline.fetch_playlist_artists = original_fetch
                pipeline.download_note_images = original_download
                pipeline.ocr_images_with_rapidocr = original_ocr
                self._restore_env(old_ai_ocr_env)

        self.assertEqual(ocr_calls["count"], 0)
        self.assertEqual(len(result.matches), 1)
        self.assertEqual(result.matches[0].artist_name, "Zella Day")
        self.assertIn("\u5df2\u547d\u4e2d OCR \u7f13\u5b58", " ".join(result.warnings))


if __name__ == "__main__":
    unittest.main()
