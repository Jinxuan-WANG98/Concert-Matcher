import unittest

from services.ai_matcher import AiDecision
from services.matcher import match_events_to_artists
from services.models import EventRow, PlaylistArtist


class FakeReviewer:
    def __init__(self, decision):
        self.decision = decision
        self.calls = 0

    def review(self, event, artist):
        self.calls += 1
        return self.decision


class MatcherAiReviewTest(unittest.TestCase):
    def test_ai_reviewer_can_reject_medium_confidence_match(self):
        events = [EventRow(date_text="10.31", performer="Stratovarius \u7075\u4e91", venue="MAO")]
        artists = [PlaylistArtist(name="Stratovarius", song_count=1, sample_songs=["Forever"])]
        reviewer = FakeReviewer(AiDecision(is_match=False, confidence="\u4f4e", reason="\u5019\u9009\u4e0d\u662f\u540c\u4e00\u827a\u4eba"))

        matches = match_events_to_artists(events, artists, ai_reviewer=reviewer)

        self.assertEqual(matches, [])
        self.assertEqual(reviewer.calls, 1)

    def test_ai_reviewer_can_upgrade_medium_confidence_match(self):
        events = [EventRow(date_text="10.31", performer="Stratovarius \u7075\u4e91", venue="MAO")]
        artists = [PlaylistArtist(name="Stratovarius", song_count=1, sample_songs=["Forever"])]
        reviewer = FakeReviewer(AiDecision(is_match=True, confidence="\u9ad8", reason="\u827a\u540d\u4e3b\u4f53\u4e00\u81f4"))

        matches = match_events_to_artists(events, artists, ai_reviewer=reviewer)

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].confidence, "\u9ad8")
        self.assertIn("AI\u590d\u6838", matches[0].match_method)


if __name__ == "__main__":
    unittest.main()
