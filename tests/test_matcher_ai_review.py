import unittest

import services.matcher as matcher
from services.ai_matcher import AiDecision, AiMatchSuggestion
from services.matcher import match_events_to_artists
from services.models import EventRow, PlaylistArtist


class FakeReviewer:
    def __init__(self, decision, suggestion=None):
        self.decision = decision
        self.suggestion = suggestion
        self.calls = 0
        self.find_calls = 0

    def review(self, event, artist):
        self.calls += 1
        return self.decision

    def find_best_match(self, event, artists):
        self.find_calls += 1
        return self.suggestion


class FakeBatchReviewer:
    def __init__(self, suggestions):
        self.suggestions = suggestions
        self.batch_calls = 0
        self.find_calls = 0
        self.calls = 0
        self.last_events = None
        self.last_artists = None
        self.last_event_indices = None

    def review(self, event, artist):
        self.calls += 1
        raise AssertionError("review should not run in AI-only batch mode")

    def find_best_match(self, event, artists):
        self.find_calls += 1
        return None

    def find_best_matches(self, events, artists, event_indices=None):
        self.batch_calls += 1
        self.last_events = events
        self.last_artists = artists
        self.last_event_indices = event_indices
        return self.suggestions


class FakeEmptyBatchReviewer:
    def __init__(self, suggestion):
        self.suggestion = suggestion
        self.batch_calls = 0
        self.find_calls = 0

    def find_best_matches(self, events, artists, event_indices=None):
        self.batch_calls += 1
        return {}

    def find_best_match(self, event, artists):
        self.find_calls += 1
        return self.suggestion


class MatcherAiReviewTest(unittest.TestCase):
    def test_ai_only_exact_anchor_prevents_vox_low_wrong_row(self):
        events = [
            EventRow(date_text="9.6", performer="叶琼琳", venue="新歌空间", image_name="image05.jpg"),
            EventRow(date_text="9.12", performer="VoX LoW", venue="星在", image_name="image06.jpg"),
        ]
        artists = [PlaylistArtist(name="VOX LOW", song_count=7, sample_songs=["We Walk"])]
        reviewer = FakeBatchReviewer(
            {0: AiMatchSuggestion(artist_name="VOX LOW", confidence="高", reason="wrong row")}
        )

        matches = match_events_to_artists(events, artists, ai_reviewer=reviewer, ai_only=True)

        self.assertEqual(
            [(match.artist_name, match.date_text, match.venue) for match in matches],
            [("VOX LOW", "9.12", "星在")],
        )
        self.assertEqual(reviewer.batch_calls, 0)

    def test_ai_only_sends_only_unresolved_events_and_unanchored_artists(self):
        events = [
            EventRow(date_text="8.27", performer="PREP", venue="MAO"),
            EventRow(date_text="8.28", performer="泽拉黛", venue="MAO"),
        ]
        artists = [
            PlaylistArtist(name="PREP", song_count=1, sample_songs=[]),
            PlaylistArtist(name="Zella Day", song_count=1, sample_songs=[]),
        ]
        reviewer = FakeBatchReviewer(
            {1: AiMatchSuggestion(artist_name="Zella Day", confidence="高", reason="中文译名")}
        )

        matches = match_events_to_artists(events, artists, ai_reviewer=reviewer, ai_only=True)

        self.assertEqual([event.performer for event in reviewer.last_events], ["泽拉黛"])
        self.assertEqual([artist.name for artist in reviewer.last_artists], ["Zella Day"])
        self.assertEqual(reviewer.last_event_indices, [1])
        self.assertEqual([match.artist_name for match in matches], ["PREP", "Zella Day"])

    def test_ai_only_does_not_anchor_non_unique_cleaned_artist_names(self):
        events = [EventRow(date_text="9.12", performer="VoX LoW", venue="星在")]
        artists = [
            PlaylistArtist(name="VOX LOW", song_count=1, sample_songs=[]),
            PlaylistArtist(name="Vox Low", song_count=1, sample_songs=[]),
        ]
        reviewer = FakeBatchReviewer({})

        match_events_to_artists(events, artists, ai_reviewer=reviewer, ai_only=True)

        self.assertEqual(reviewer.batch_calls, 1)
        self.assertEqual(reviewer.last_event_indices, [0])
        self.assertEqual([artist.name for artist in reviewer.last_artists], ["VOX LOW", "Vox Low"])

    def test_ai_only_anchor_does_not_split_collaboration_line(self):
        events = [EventRow(date_text="9.12", performer="PREP / Guest", venue="MAO")]
        artists = [PlaylistArtist(name="PREP", song_count=1, sample_songs=[])]
        reviewer = FakeBatchReviewer(
            {0: AiMatchSuggestion(artist_name="PREP", confidence="高", reason="主艺人")}
        )

        matches = match_events_to_artists(events, artists, ai_reviewer=reviewer, ai_only=True)

        self.assertEqual(reviewer.batch_calls, 1)
        self.assertEqual([match.artist_name for match in matches], ["PREP"])

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

    def test_ai_reviewer_can_fill_no_local_candidate_match(self):
        events = [EventRow(date_text="8.27", performer="\u6cfd\u62c9\u9edb", venue="MAO")]
        artists = [PlaylistArtist(name="Zella Day", song_count=1, sample_songs=["Hypnotic"])]
        reviewer = FakeReviewer(
            decision=None,
            suggestion=AiMatchSuggestion(artist_name="Zella Day", confidence="\u4e2d", reason="\u4e2d\u6587\u97f3\u8bd1\u6307\u5411\u540c\u4e00\u827a\u4eba"),
        )

        matches = match_events_to_artists(events, artists, ai_reviewer=reviewer)

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].artist_name, "Zella Day")
        self.assertEqual(matches[0].confidence, "\u4e2d")
        self.assertIn("AI\u5019\u9009\u8865\u5145", matches[0].match_method)
        self.assertEqual(reviewer.find_calls, 1)

    def test_ai_only_mode_anchors_exact_name_without_fuzzy_scorer(self):
        events = [EventRow(date_text="8.27", performer="Zella Day", venue="MAO")]
        artists = [PlaylistArtist(name="Zella Day", song_count=1, sample_songs=["Hypnotic"])]
        reviewer = FakeReviewer(
            decision=None,
            suggestion=AiMatchSuggestion(artist_name="Zella Day", confidence="\u9ad8", reason="AI \u76f4\u63a5\u786e\u8ba4"),
        )
        original_score_pair = matcher._score_pair
        matcher._score_pair = lambda alias, artist_name: (_ for _ in ()).throw(
            AssertionError("local scorer should not run")
        )

        try:
            matches = match_events_to_artists(events, artists, ai_reviewer=reviewer, ai_only=True)
        finally:
            matcher._score_pair = original_score_pair

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].artist_name, "Zella Day")
        self.assertEqual(matches[0].confidence, "\u9ad8")
        self.assertEqual(matches[0].match_method, "精确名称锚定")
        self.assertEqual(reviewer.calls, 0)
        self.assertEqual(reviewer.find_calls, 0)

    def test_ai_only_mode_anchors_all_exact_names_without_batch_reviewer(self):
        events = [
            EventRow(date_text="8.27", performer="Zella Day", venue="MAO"),
            EventRow(date_text="9.01", performer="PREP", venue="Modern Sky"),
        ]
        artists = [
            PlaylistArtist(name="Zella Day", song_count=1, sample_songs=["Hypnotic"]),
            PlaylistArtist(name="PREP", song_count=2, sample_songs=["Cheapest Flight"]),
        ]
        reviewer = FakeBatchReviewer(
            {
                0: AiMatchSuggestion(artist_name="Zella Day", confidence="\u9ad8", reason="\u540c\u540d"),
                1: AiMatchSuggestion(artist_name="PREP", confidence="\u9ad8", reason="\u540c\u540d"),
            }
        )
        original_score_pair = matcher._score_pair
        matcher._score_pair = lambda alias, artist_name: (_ for _ in ()).throw(
            AssertionError("local scorer should not run")
        )

        try:
            matches = match_events_to_artists(events, artists, ai_reviewer=reviewer, ai_only=True)
        finally:
            matcher._score_pair = original_score_pair

        self.assertEqual(len(matches), 2)
        self.assertEqual([match.artist_name for match in matches], ["Zella Day", "PREP"])
        self.assertEqual(reviewer.batch_calls, 0)
        self.assertEqual(reviewer.find_calls, 0)
        self.assertEqual(reviewer.calls, 0)

    def test_ai_only_mode_falls_back_to_single_match_when_batch_has_no_suggestion(self):
        events = [EventRow(date_text="8.27", performer="泽拉黛", venue="MAO")]
        artists = [PlaylistArtist(name="Zella Day", song_count=1, sample_songs=["Hypnotic"])]
        reviewer = FakeEmptyBatchReviewer(
            AiMatchSuggestion(artist_name="Zella Day", confidence="\u9ad8", reason="\u5355\u6761\u515c\u5e95")
        )

        matches = match_events_to_artists(events, artists, ai_reviewer=reviewer, ai_only=True)

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].artist_name, "Zella Day")
        self.assertEqual(reviewer.batch_calls, 1)
        self.assertEqual(reviewer.find_calls, 1)

    def test_ai_only_mode_skips_per_event_fallback_after_large_batch(self):
        events = [
            EventRow(date_text=f"8.{index + 1}", performer=f"Artist {index}", venue="MAO")
            for index in range(21)
        ]
        artists = [PlaylistArtist(name="Zella Day", song_count=1, sample_songs=["Hypnotic"])]
        reviewer = FakeEmptyBatchReviewer(
            AiMatchSuggestion(artist_name="Zella Day", confidence="高", reason="should not run")
        )

        matches = match_events_to_artists(events, artists, ai_reviewer=reviewer, ai_only=True)

        self.assertEqual(matches, [])
        self.assertEqual(reviewer.batch_calls, 1)
        self.assertEqual(reviewer.find_calls, 0)

    def test_ai_only_mode_maps_bilingual_ai_name_to_single_playlist_candidate(self):
        events = [EventRow(date_text="10.24", performer="Jackson Wang \u738b\u5609\u5c14", venue="")]
        artists = [PlaylistArtist(name="\u738b\u5609\u5c14", song_count=1, sample_songs=["LMLY"])]
        reviewer = FakeReviewer(
            decision=None,
            suggestion=AiMatchSuggestion(
                artist_name="Jackson Wang \u738b\u5609\u5c14",
                confidence="\u9ad8",
                reason="\u4e2d\u82f1\u6587\u540c\u4e00\u6b4c\u624b",
            ),
        )

        matches = match_events_to_artists(events, artists, ai_reviewer=reviewer, ai_only=True)

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].artist_name, "\u738b\u5609\u5c14")
        self.assertEqual(matches[0].matched_alias, "Jackson Wang \u738b\u5609\u5c14")

    def test_ai_only_mode_deduplicates_same_date_artist_and_prefers_venue(self):
        events = [
            EventRow(date_text="8.27", performer="ZeIIa Day", venue="", image_name="summary.jpg"),
            EventRow(date_text="8.27", performer="Zella Day", venue="MAO", image_name="detail.jpg"),
        ]
        artists = [PlaylistArtist(name="Zella Day", song_count=1, sample_songs=["Hypnotic"])]
        reviewer = FakeBatchReviewer(
            {
                0: AiMatchSuggestion(artist_name="Zella Day", confidence="\u9ad8", reason="\u603b\u89c8\u9875"),
                1: AiMatchSuggestion(artist_name="Zella Day", confidence="\u9ad8", reason="\u8be6\u60c5\u9875"),
            }
        )

        matches = match_events_to_artists(events, artists, ai_reviewer=reviewer, ai_only=True)

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].artist_name, "Zella Day")
        self.assertEqual(matches[0].venue, "MAO")
        self.assertEqual(matches[0].image_name, "detail.jpg")

    def test_ai_only_mode_deduplicates_overlapping_date_range_and_single_day(self):
        events = [
            EventRow(date_text="7.31-8.2", performer="\u5f90\u826f", venue="\u6885\u5954", image_name="detail.jpg"),
            EventRow(date_text="7.31", performer="\u5f90\u826f", venue="", image_name="summary.jpg"),
        ]
        artists = [PlaylistArtist(name="\u5f90\u826f", song_count=4, sample_songs=["\u574f\u5973\u5b69"])]
        reviewer = FakeBatchReviewer(
            {
                0: AiMatchSuggestion(artist_name="\u5f90\u826f", confidence="\u9ad8", reason="\u8be6\u60c5\u9875"),
                1: AiMatchSuggestion(artist_name="\u5f90\u826f", confidence="\u9ad8", reason="\u603b\u89c8\u9875"),
            }
        )

        matches = match_events_to_artists(events, artists, ai_reviewer=reviewer, ai_only=True)

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].date_text, "7.31-8.2")
        self.assertEqual(matches[0].venue, "\u6885\u5954")

    def test_ai_only_mode_prefers_complete_date_range_over_higher_confidence_single_day(self):
        events = [
            EventRow(date_text="7.31", performer="徐良 (summary)", venue="", image_name="summary.jpg"),
            EventRow(date_text="7.31-8.2", performer="徐良 (detail)", venue="梅奔", image_name="detail.jpg"),
        ]
        artists = [PlaylistArtist(name="\u5f90\u826f", song_count=4, sample_songs=["\u574f\u5973\u5b69"])]
        reviewer = FakeBatchReviewer(
            {
                0: AiMatchSuggestion(artist_name="\u5f90\u826f", confidence="\u9ad8", reason="\u603b\u89c8\u9875"),
                1: AiMatchSuggestion(artist_name="\u5f90\u826f", confidence="\u4e2d", reason="\u8be6\u60c5\u9875"),
            }
        )

        matches = match_events_to_artists(events, artists, ai_reviewer=reviewer, ai_only=True)

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].date_text, "7.31-8.2")
        self.assertEqual(matches[0].venue, "\u6885\u5954")
        self.assertEqual(matches[0].image_name, "detail.jpg")

    def test_ai_only_mode_fills_blank_venue_from_same_artist_overlapping_date_event(self):
        events = [
            EventRow(date_text="7.31", performer="徐良 (summary)", venue="", image_name="summary.jpg"),
            EventRow(date_text="7.31-8.2", performer="徐良 (detail)", venue="梅奔", image_name="detail.jpg"),
        ]
        artists = [PlaylistArtist(name="\u5f90\u826f", song_count=4, sample_songs=["\u574f\u5973\u5b69"])]
        reviewer = FakeBatchReviewer(
            {
                0: AiMatchSuggestion(artist_name="\u5f90\u826f", confidence="\u9ad8", reason="\u603b\u89c8\u9875"),
            }
        )

        matches = match_events_to_artists(events, artists, ai_reviewer=reviewer, ai_only=True)

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].date_text, "7.31")
        self.assertEqual(matches[0].venue, "\u6885\u5954")
        self.assertEqual(matches[0].image_name, "summary.jpg")

    def test_ai_only_mode_does_not_fill_blank_venue_from_different_artist_same_date(self):
        events = [
            EventRow(date_text="7.17", performer="Zella Day", venue="", image_name="summary.jpg"),
            EventRow(date_text="7.17-18", performer="\u5355\u4f9d\u7eaf", venue="\u6885\u5954", image_name="detail.jpg"),
        ]
        artists = [PlaylistArtist(name="Zella Day", song_count=1, sample_songs=["Hypnotic"])]
        reviewer = FakeBatchReviewer(
            {
                0: AiMatchSuggestion(artist_name="Zella Day", confidence="\u9ad8", reason="\u603b\u89c8\u9875"),
            }
        )

        matches = match_events_to_artists(events, artists, ai_reviewer=reviewer, ai_only=True)

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].venue, "")

    def test_ai_only_mode_fills_blank_venue_from_reverse_alias_event(self):
        events = [
            EventRow(date_text="10.24", performer="\u738b\u5609\u5c14", venue="", image_name="summary.jpg"),
            EventRow(date_text="10.24-25", performer="Jackson Wang", venue="\u6885\u5954", image_name="detail.jpg"),
        ]
        artists = [PlaylistArtist(name="\u738b\u5609\u5c14", song_count=7, sample_songs=["LMLY"])]
        reviewer = FakeBatchReviewer(
            {
                0: AiMatchSuggestion(artist_name="\u738b\u5609\u5c14", confidence="\u9ad8", reason="\u603b\u89c8\u9875"),
            }
        )

        matches = match_events_to_artists(events, artists, ai_reviewer=reviewer, ai_only=True)

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].venue, "\u6885\u5954")


if __name__ == "__main__":
    unittest.main()
