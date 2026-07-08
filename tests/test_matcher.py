import unittest

from services.matcher import match_events_to_artists
from services.models import EventRow, PlaylistArtist


class MatcherTest(unittest.TestCase):
    def test_matches_exact_chinese_artist(self):
        events = [
            EventRow(
                date_text="8.15-16",
                performer="\u6797\u5ba5\u5609",
                venue="\u8679\u53e3\u8db3\u7403\u573a",
                image_name="image09.jpg",
            )
        ]
        artists = [PlaylistArtist(name="\u6797\u5ba5\u5609", song_count=4, sample_songs=["\u6b8b\u9177\u6708\u5149"])]

        matches = match_events_to_artists(events, artists)

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].artist_name, "\u6797\u5ba5\u5609")
        self.assertEqual(matches[0].confidence, "\u9ad8")

    def test_matches_common_alias(self):
        events = [EventRow(date_text="10.24-25", performer="\u738b\u5609\u5c14", venue="\u6885\u5954", image_name="image13.jpg")]
        artists = [PlaylistArtist(name="Jackson Wang", song_count=7, sample_songs=["WOLO"])]

        matches = match_events_to_artists(events, artists)

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].artist_name, "Jackson Wang")
        self.assertIn("\u522b\u540d", matches[0].match_method)

    def test_matches_ocr_latin_confusion(self):
        events = [EventRow(date_text="8.27", performer="ZeIIa Day", venue="MAO", image_name="image10.jpg")]
        artists = [PlaylistArtist(name="Zella Day", song_count=1, sample_songs=["Hypnotic"])]

        matches = match_events_to_artists(events, artists)

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].artist_name, "Zella Day")
        self.assertIn("OCR", matches[0].match_method)

    def test_splits_collaboration_or_guest_line(self):
        events = [
            EventRow(
                date_text="7.17",
                performer="\u5468\u83f2\u6208 / \u5609\u5bbe\uff1a\u6797\u5fd7\u70ab",
                venue="\u56de\u54cd\u4e4b\u5730",
                image_name="image04.jpg",
            )
        ]
        artists = [PlaylistArtist(name="\u6797\u5fd7\u70ab", song_count=2, sample_songs=["\u5355\u8eab\u60c5\u6b4c"])]

        matches = match_events_to_artists(events, artists)

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].matched_alias, "\u6797\u5fd7\u70ab")


if __name__ == "__main__":
    unittest.main()
