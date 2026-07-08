import io
import os
import tempfile
import unittest
from pathlib import Path

from openpyxl import load_workbook
from werkzeug.datastructures import FileStorage

from services.export_excel import write_matches_xlsx
from services.models import EventRow, PlaylistArtist
import services.pipeline as pipeline
from services.pipeline import run_match_pipeline, run_match_pipeline_from_data, save_uploaded_images


class PipelineTest(unittest.TestCase):
    def test_pipeline_matches_and_exports_required_columns(self):
        artists = [
            PlaylistArtist(name="Jackson Wang", song_count=7, sample_songs=["WOLO", "LMLY"]),
            PlaylistArtist(name="\u6797\u5ba5\u5609", song_count=4, sample_songs=["\u6b8b\u9177\u6708\u5149"]),
        ]
        events = [
            EventRow(date_text="10.24-25", performer="\u738b\u5609\u5c14", venue="\u6885\u5954"),
            EventRow(date_text="8.15-16", performer="\u6797\u5ba5\u5609", venue="\u8679\u53e3\u8db3\u7403\u573a"),
        ]

        result = run_match_pipeline_from_data(artists, events)

        self.assertEqual(len(result.matches), 2)
        self.assertEqual(result.matches[0].date_display, "8\u670815\u65e5-16\u65e5")
        self.assertEqual(result.matches[1].artist_name, "Jackson Wang")

        with tempfile.TemporaryDirectory() as tmp:
            output = write_matches_xlsx(result, Path(tmp) / "matches.xlsx")
            workbook = load_workbook(output)
            sheet = workbook.active
            headers = [cell.value for cell in sheet[1]]

        self.assertEqual(
            headers,
            [
                "\u5e8f\u53f7",
                "\u65e5\u671f",
                "\u6b4c\u624b",
                "\u6f14\u51fa\u573a\u6240",
                "\u6b4c\u5355\u51fa\u73b0\u6b21\u6570",
                "\u6b4c\u5355\u4ee3\u8868\u6b4c\u66f2",
                "\u7f6e\u4fe1\u5ea6",
            ],
        )

    def test_pipeline_warns_when_ai_switch_is_on_without_api_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "note.jpg"
            image_path.write_bytes(b"fake image content")
            original_fetch = pipeline.fetch_playlist_artists
            original_ocr = pipeline.ocr_images_with_rapidocr
            original_parse = pipeline.parse_ocr_events
            pipeline.fetch_playlist_artists = lambda url: [PlaylistArtist(name="Zella Day", song_count=1, sample_songs=["Hypnotic"])]
            pipeline.ocr_images_with_rapidocr = lambda images: []
            pipeline.parse_ocr_events = lambda images: [EventRow(date_text="8.27", performer="ZeIIa Day", venue="MAO")]
            old_key = os.environ.pop("AI_MATCH_API_KEY", None)
            old_enabled = os.environ.pop("AI_MATCH_ENABLED", None)
            try:
                result = run_match_pipeline(
                    "https://music.163.com/#/playlist?id=1",
                    "",
                    uploaded_images=[image_path],
                    output_root=Path(tmp) / "out",
                    use_ai=True,
                )
            finally:
                pipeline.fetch_playlist_artists = original_fetch
                pipeline.ocr_images_with_rapidocr = original_ocr
                pipeline.parse_ocr_events = original_parse
                if old_key is not None:
                    os.environ["AI_MATCH_API_KEY"] = old_key
                if old_enabled is not None:
                    os.environ["AI_MATCH_ENABLED"] = old_enabled

        self.assertEqual(len(result.matches), 1)
        self.assertIn(
            "AI \u590d\u6838\u672a\u542f\u7528\uff1a\u670d\u52a1\u5668\u6ca1\u6709\u914d\u7f6e AI API Key",
            " ".join(result.warnings),
        )

    def test_save_uploaded_images_ignores_empty_file_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            upload_dir = Path(tmp)
            empty = FileStorage(stream=io.BytesIO(b""), filename="", content_type="application/octet-stream")
            paths = save_uploaded_images([empty], upload_dir)
            self.assertEqual(paths, [])
            self.assertEqual(list(upload_dir.iterdir()), [])

    def test_save_uploaded_images_keeps_real_file_when_empty_input_present(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            upload_dir = Path(tmp)
            image_bytes = io.BytesIO()
            Image.new("RGB", (1, 1)).save(image_bytes, format="JPEG")
            empty = FileStorage(stream=io.BytesIO(b""), filename="", content_type="application/octet-stream")
            real = FileStorage(
                stream=io.BytesIO(image_bytes.getvalue()),
                filename="note.jpg",
                content_type="image/jpeg",
            )
            paths = save_uploaded_images([empty, real], upload_dir)
            self.assertEqual(len(paths), 1)
            self.assertEqual(paths[0].name, "upload_01.jpg")
            self.assertGreater(paths[0].stat().st_size, 0)

    def test_save_uploaded_images_rejects_invalid_image_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            upload_dir = Path(tmp)
            invalid = FileStorage(stream=io.BytesIO(b"not an image"), filename="bad.jpg", content_type="image/jpeg")
            paths = save_uploaded_images([invalid], upload_dir)
            self.assertEqual(paths, [])
            self.assertEqual(list(upload_dir.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
