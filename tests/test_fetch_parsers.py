import unittest

from services.netease import extract_playlist_id
from services.ocr import OcrImage, OcrLine, parse_ocr_events
from services.xhs import extract_note_image_urls


class FetchParserTest(unittest.TestCase):
    def test_extract_playlist_id_from_hash_url(self):
        url = "https://music.163.com/#/playlist?app_version=9.5.45&id=167320827&userid=132918937"

        self.assertEqual(extract_playlist_id(url), "167320827")

    def test_extract_playlist_id_from_plain_url(self):
        self.assertEqual(extract_playlist_id("https://music.163.com/playlist?id=12345"), "12345")

    def test_extract_note_image_urls_from_xhs_html(self):
        html = """
        <img data-xhs-img src="http://sns-webpic-qc.xhscdn.com/a.jpg?x=1&amp;y=2">
        <img data-xhs-img src="https://sns-webpic-qc.xhscdn.com/b.webp">
        <img src="https://example.com/not-note.jpg">
        """

        urls = extract_note_image_urls(html)

        self.assertEqual(
            urls,
            [
                "https://sns-webpic-qc.xhscdn.com/a.jpg?x=1&y=2",
                "https://sns-webpic-qc.xhscdn.com/b.webp",
            ],
        )

    def test_parse_ocr_events_merges_adjacent_range_end(self):
        image = OcrImage(
            image_name="note_05.jpg",
            width=1080,
            height=1439,
            lines=[
                OcrLine(text="7 / 20", x1=16, y1=1200, x2=83, y2=1230),
                OcrLine(text="-21", x1=135, y1=1200, x2=182, y2=1230),
                OcrLine(text="BADBADNOTGOOD", x1=418, y1=1200, x2=703, y2=1230),
                OcrLine(text="\u74e6\u8086Vast", x1=943, y1=1200, x2=1051, y2=1230),
            ],
        )

        events = parse_ocr_events([image])

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].date_text, "7.20-21")
        self.assertEqual(events[0].performer, "BADBADNOTGOOD")
        self.assertEqual(events[0].venue, "\u74e6\u8086Vast")


if __name__ == "__main__":
    unittest.main()
