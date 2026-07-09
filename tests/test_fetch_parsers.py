import unittest

from services.netease import extract_playlist_id
from services.ocr import OcrImage, OcrLine, _looks_like_venue, parse_ocr_events
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

    def test_parse_ocr_events_ignores_artist_names_in_right_column(self):
        image = OcrImage(
            image_name="note_01.jpg",
            width=1080,
            height=1440,
            lines=[
                OcrLine(text="7 / 11", x1=16, y1=80, x2=83, y2=110),
                OcrLine(text="LakiPak", x1=529, y1=80, x2=620, y2=110),
                OcrLine(text="\u8fc8\u514b\u5b66\u6447\u6eda", x1=912, y1=80, x2=1029, y2=110),
                OcrLine(text="7 / 12", x1=16, y1=250, x2=83, y2=280),
                OcrLine(text="Hanser", x1=531, y1=250, x2=618, y2=280),
                OcrLine(text="\u674e\u8363\u6d69", x1=932, y1=250, x2=1010, y2=280),
            ],
        )

        events = parse_ocr_events([image])

        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].venue, "")
        self.assertEqual(events[1].venue, "")

    def test_parse_ocr_events_reads_summary_page_multiple_date_columns(self):
        image = OcrImage(
            image_name="summary.jpg",
            width=1080,
            height=1440,
            lines=[
                OcrLine(text="7 / 11", x1=20, y1=200, x2=90, y2=230),
                OcrLine(text="\u5468\u4e94", x1=120, y1=200, x2=170, y2=230),
                OcrLine(text="Thomas Bergersen", x1=205, y1=200, x2=350, y2=230),
                OcrLine(text="7 / 12", x1=390, y1=200, x2=460, y2=230),
                OcrLine(text="\u5468\u516d", x1=490, y1=200, x2=540, y2=230),
                OcrLine(text="Hanser", x1=575, y1=200, x2=650, y2=230),
                OcrLine(text="7 / 13", x1=740, y1=200, x2=810, y2=230),
                OcrLine(text="\u5468\u65e5", x1=835, y1=200, x2=885, y2=230),
                OcrLine(text="PREP", x1=920, y1=200, x2=990, y2=230),
            ],
        )

        events = parse_ocr_events([image])

        self.assertEqual([(event.date_text, event.performer, event.venue) for event in events], [
            ("7.11", "Thomas Bergersen", ""),
            ("7.12", "Hanser", ""),
            ("7.13", "PREP", ""),
        ])

    def test_parse_ocr_events_prefers_detail_row_with_venue_over_summary_duplicate(self):
        summary = OcrImage(
            image_name="summary.jpg",
            width=1080,
            height=1440,
            lines=[
                OcrLine(text="7 / 11", x1=20, y1=200, x2=90, y2=230),
                OcrLine(text="\u5468\u4e94", x1=120, y1=200, x2=170, y2=230),
                OcrLine(text="Thomas Bergersen", x1=205, y1=200, x2=350, y2=230),
            ],
        )
        detail = OcrImage(
            image_name="detail.jpg",
            width=1080,
            height=1440,
            lines=[
                OcrLine(text="7 / 11", x1=16, y1=320, x2=83, y2=350),
                OcrLine(text="Thomas Bergersen", x1=418, y1=320, x2=703, y2=350),
                OcrLine(text="MAO", x1=943, y1=320, x2=1000, y2=350),
            ],
        )

        events = parse_ocr_events([summary, detail])

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].date_text, "7.11")
        self.assertEqual(events[0].performer, "Thomas Bergersen")
        self.assertEqual(events[0].venue, "MAO")

    def test_parse_ocr_events_matches_venue_on_gap_row(self):
        image = OcrImage(
            image_name="note_11.jpg",
            width=1080,
            height=1439,
            lines=[
                OcrLine(text="6 / 6", x1=16, y1=540, x2=83, y2=554),
                OcrLine(text="\u5c71\u5f62\u745e\u79cbRachael Yamagata", x1=365, y1=540, x2=754, y2=554),
                OcrLine(text="\u661f\u57282F", x1=945, y1=590, x2=1050, y2=617),
            ],
        )

        events = parse_ocr_events([image])

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].venue, "\u661f\u57282F")

    def test_parse_ocr_events_uses_gap_row_date_when_venue_is_on_gap_row(self):
        image = OcrImage(
            image_name="note_11.jpg",
            width=1080,
            height=1439,
            lines=[
                OcrLine(text="6 / 6", x1=16, y1=540, x2=83, y2=554),
                OcrLine(text="\u5c71\u5f62\u745e\u79cbRachael Yamagata", x1=365, y1=540, x2=754, y2=554),
                OcrLine(text="9 / 6", x1=16, y1=590, x2=83, y2=617),
                OcrLine(text="\u661f\u57282F", x1=945, y1=590, x2=1050, y2=617),
            ],
        )

        events = parse_ocr_events([image])

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].date_text, "9.6")
        self.assertEqual(events[0].venue, "\u661f\u57282F")

    def test_parse_ocr_events_merges_split_fei_sheng_venue(self):
        image = OcrImage(
            image_name="venue_split.jpg",
            width=1080,
            height=1440,
            lines=[
                OcrLine(text="7 / 10", x1=16, y1=300, x2=83, y2=310),
                OcrLine(text="\u5eb7\u58eb\u5766\u7684\u53d8\u5316\u7403", x1=400, y1=300, x2=700, y2=310),
                OcrLine(text="\u970f", x1=930, y1=300, x2=955, y2=310),
                OcrLine(text="\u58f0", x1=956, y1=300, x2=990, y2=310),
            ],
        )

        events = parse_ocr_events([image])

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].venue, "\u970f\u58f0")

    def test_parse_ocr_events_rejects_artist_name_on_same_row(self):
        image = OcrImage(
            image_name="note_01.jpg",
            width=1080,
            height=1440,
            lines=[
                OcrLine(text="7 / 17", x1=16, y1=370, x2=83, y2=377),
                OcrLine(text="Zella Day", x1=518, y1=370, x2=630, y2=377),
                OcrLine(text="Quinn\u8471\u56e0", x1=910, y1=370, x2=1031, y2=377),
            ],
        )

        events = parse_ocr_events([image])

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].venue, "")

    def test_looks_like_venue(self):
        self.assertTrue(_looks_like_venue("\u6885\u5954"))
        self.assertTrue(_looks_like_venue("JZ"))
        self.assertTrue(_looks_like_venue("\u74e6\u8086Vast"))
        self.assertTrue(_looks_like_venue("\u970f\u58f0"))
        self.assertFalse(_looks_like_venue("\u58f0"))
        self.assertFalse(_looks_like_venue("\u674e\u8363\u6d69"))
        self.assertFalse(_looks_like_venue("PREP"))
        self.assertFalse(_looks_like_venue("DAKOOKA"))


if __name__ == "__main__":
    unittest.main()
