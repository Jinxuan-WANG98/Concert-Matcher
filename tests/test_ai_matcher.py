import json
import os
import tempfile
import threading
import unittest
from pathlib import Path

from services.ai_matcher import (
    AiDecision,
    AiMatchConfig,
    AiMatchSuggestion,
    build_batch_artist_pick_payload,
    build_artist_pick_payload,
    build_review_payload,
    parse_ai_batch_match_suggestions,
    parse_ai_decision,
    parse_ai_match_suggestion,
    AiArtistReviewer,
)
from services.models import EventRow, PlaylistArtist


class AiMatcherTest(unittest.TestCase):
    def test_config_is_disabled_without_key(self):
        old_key = os.environ.pop("AI_MATCH_API_KEY", None)
        old_enabled = os.environ.pop("AI_MATCH_ENABLED", None)
        try:
            config = AiMatchConfig.from_env()
        finally:
            if old_key is not None:
                os.environ["AI_MATCH_API_KEY"] = old_key
            if old_enabled is not None:
                os.environ["AI_MATCH_ENABLED"] = old_enabled

        self.assertFalse(config.enabled)

    def test_config_loads_ai_only_mode(self):
        old_values = {
            "AI_MATCH_API_KEY": os.environ.pop("AI_MATCH_API_KEY", None),
            "AI_MATCH_ENABLED": os.environ.pop("AI_MATCH_ENABLED", None),
            "AI_MATCH_MODE": os.environ.pop("AI_MATCH_MODE", None),
        }
        try:
            os.environ["AI_MATCH_ENABLED"] = "true"
            os.environ["AI_MATCH_API_KEY"] = "test-key"
            os.environ["AI_MATCH_MODE"] = "ai_only"

            config = AiMatchConfig.from_env()
        finally:
            for name, value in old_values.items():
                if value is not None:
                    os.environ[name] = value
                else:
                    os.environ.pop(name, None)

        self.assertTrue(config.enabled)
        self.assertEqual(config.mode, "ai_only")

    def test_config_loads_event_batch_size(self):
        old_values = {
            "AI_MATCH_API_KEY": os.environ.pop("AI_MATCH_API_KEY", None),
            "AI_MATCH_ENABLED": os.environ.pop("AI_MATCH_ENABLED", None),
            "AI_MATCH_EVENT_BATCH_SIZE": os.environ.pop("AI_MATCH_EVENT_BATCH_SIZE", None),
        }
        try:
            os.environ["AI_MATCH_ENABLED"] = "true"
            os.environ["AI_MATCH_API_KEY"] = "test-key"
            os.environ["AI_MATCH_EVENT_BATCH_SIZE"] = "12"

            config = AiMatchConfig.from_env()
        finally:
            for name, value in old_values.items():
                if value is not None:
                    os.environ[name] = value
                else:
                    os.environ.pop(name, None)

        self.assertEqual(config.event_batch_size, 12)

    def test_config_loads_event_workers(self):
        old_values = {
            "AI_MATCH_API_KEY": os.environ.pop("AI_MATCH_API_KEY", None),
            "AI_MATCH_ENABLED": os.environ.pop("AI_MATCH_ENABLED", None),
            "AI_MATCH_EVENT_WORKERS": os.environ.pop("AI_MATCH_EVENT_WORKERS", None),
        }
        try:
            os.environ["AI_MATCH_ENABLED"] = "true"
            os.environ["AI_MATCH_API_KEY"] = "test-key"
            os.environ["AI_MATCH_EVENT_WORKERS"] = "4"

            config = AiMatchConfig.from_env()
        finally:
            for name, value in old_values.items():
                if value is not None:
                    os.environ[name] = value
                else:
                    os.environ.pop(name, None)

        self.assertEqual(config.event_workers, 4)

    def test_config_can_reuse_ocr_provider_key(self):
        names = [
            "AI_MATCH_API_KEY",
            "AI_MATCH_ENABLED",
            "AI_MATCH_PROVIDER_INDEX",
            "AI_MATCH_BASE_URL",
            "AI_MATCH_MODEL",
            "AI_OCR_PROVIDER_2_API_KEY",
            "AI_OCR_PROVIDER_2_BASE_URL",
        ]
        old_values = {name: os.environ.pop(name, None) for name in names}
        try:
            os.environ["AI_MATCH_ENABLED"] = "true"
            os.environ["AI_MATCH_PROVIDER_INDEX"] = "2"
            os.environ["AI_MATCH_API_KEY"] = "siliconflow-key"
            os.environ["AI_MATCH_BASE_URL"] = "https://api.siliconflow.cn/v1"
            os.environ["AI_MATCH_MODEL"] = "GLM-4-Flash"
            os.environ["AI_OCR_PROVIDER_2_API_KEY"] = "zhipu-key"
            os.environ["AI_OCR_PROVIDER_2_BASE_URL"] = "https://open.bigmodel.cn/api/paas/v4"

            config = AiMatchConfig.from_env()
        finally:
            for name, value in old_values.items():
                if value is not None:
                    os.environ[name] = value
                else:
                    os.environ.pop(name, None)

        self.assertTrue(config.enabled)
        self.assertEqual(config.api_key, "zhipu-key")
        self.assertEqual(config.base_url, "https://open.bigmodel.cn/api/paas/v4")
        self.assertEqual(config.model, "GLM-4-Flash")

    def test_build_review_payload_contains_only_candidate_context(self):
        event = EventRow(date_text="10.24-25", performer="\u738b\u5609\u5c14", venue="\u6885\u5954")
        artist = PlaylistArtist(name="Jackson Wang", song_count=7, sample_songs=["WOLO"])

        payload = build_review_payload(event, artist)

        content = payload["messages"][-1]["content"]
        self.assertIn("\u738b\u5609\u5c14", content)
        self.assertIn("Jackson Wang", content)
        self.assertIn("WOLO", content)
        self.assertIn("JSON", payload["messages"][0]["content"])
        self.assertIn("\u540c\u4e00\u4f4d\u6b4c\u624b", payload["messages"][0]["content"])

    def test_parse_ai_decision_accepts_strict_json(self):
        decision = parse_ai_decision(
            '{"is_match": true, "confidence": "\u9ad8", "reason": "\u540c\u4e00\u827a\u540d\u7684\u4e2d\u82f1\u6587\u540d"}'
        )

        self.assertEqual(
            decision,
            AiDecision(is_match=True, confidence="\u9ad8", reason="\u540c\u4e00\u827a\u540d\u7684\u4e2d\u82f1\u6587\u540d"),
        )

    def test_parse_ai_decision_rejects_unknown_confidence(self):
        with self.assertRaises(ValueError):
            parse_ai_decision('{"is_match": true, "confidence": "maybe", "reason": "x"}')

    def test_build_artist_pick_payload_contains_playlist_candidates(self):
        event = EventRow(date_text="8.27", performer="ZeIIa Day", venue="MAO")
        artists = [
            PlaylistArtist(name="Zella Day", song_count=1, sample_songs=["Hypnotic"]),
            PlaylistArtist(name="PREP", song_count=2, sample_songs=["Cheapest Flight"]),
        ]

        payload = build_artist_pick_payload(event, artists, model="vision-model")

        content = payload["messages"][-1]["content"]
        self.assertEqual(payload["model"], "vision-model")
        self.assertIn("ZeIIa Day", content)
        self.assertIn("Zella Day", content)
        self.assertIn("artist_name", payload["messages"][0]["content"])

    def test_parse_ai_match_suggestion_accepts_null_artist(self):
        suggestion = parse_ai_match_suggestion('{"artist_name": null, "confidence": "\u4f4e", "reason": "\u4e0d\u786e\u5b9a"}')

        self.assertIsNone(suggestion)

    def test_parse_ai_match_suggestion_accepts_artist_name(self):
        suggestion = parse_ai_match_suggestion(
            '{"artist_name": "Zella Day", "confidence": "\u9ad8", "reason": "OCR \u628a ll \u8bc6\u522b\u6210 II"}'
        )

        self.assertEqual(
            suggestion,
            AiMatchSuggestion(artist_name="Zella Day", confidence="\u9ad8", reason="OCR \u628a ll \u8bc6\u522b\u6210 II"),
        )

    def test_build_batch_artist_pick_payload_contains_events_and_candidates(self):
        events = [
            EventRow(date_text="8.27", performer="ZeIIa Day", venue="MAO"),
            EventRow(date_text="9.01", performer="PREP", venue="Modern Sky"),
        ]
        artists = [
            PlaylistArtist(name="Zella Day", song_count=1, sample_songs=["Hypnotic"]),
            PlaylistArtist(name="PREP", song_count=2, sample_songs=["Cheapest Flight"]),
        ]

        payload = build_batch_artist_pick_payload(events, artists, model="text-model")

        content = payload["messages"][-1]["content"]
        self.assertEqual(payload["model"], "text-model")
        self.assertIn("event_index", content)
        self.assertIn("ZeIIa Day", content)
        self.assertIn("Zella Day", content)
        self.assertIn("matches", payload["messages"][0]["content"])

    def test_parse_ai_batch_match_suggestions_accepts_matches(self):
        suggestions = parse_ai_batch_match_suggestions(
            """
            ```json
            {
              "matches": [
                {"event_index": 0, "artist_name": "Zella Day", "confidence": "\u9ad8", "reason": "OCR \u5b57\u5f62\u6df7\u6dc6"},
                {"event_index": 1, "artist_name": null, "confidence": "\u4f4e", "reason": "\u65e0\u628a\u63e1"}
              ]
            }
            ```
            """
        )

        self.assertEqual(
            suggestions,
            {0: AiMatchSuggestion(artist_name="Zella Day", confidence="\u9ad8", reason="OCR \u5b57\u5f62\u6df7\u6dc6")},
        )

    def test_reviewer_batches_event_matching_requests(self):
        config = AiMatchConfig(
            enabled=True,
            api_key="test-key",
            base_url="https://api.example.com/v1",
            model="text-model",
            event_batch_size=2,
            event_workers=1,
        )
        reviewer = AiArtistReviewer(config)
        calls = []

        def fake_chat(payload):
            calls.append(payload)
            if len(calls) == 1:
                return (
                    '{"matches":['
                    '{"event_index":0,"artist_name":"Zella Day","confidence":"\u9ad8","reason":"same"},'
                    '{"event_index":1,"artist_name":"PREP","confidence":"\u9ad8","reason":"same"}'
                    "]}"
                )
            return '{"matches":[{"event_index":2,"artist_name":"Hanser","confidence":"\u4e2d","reason":"alias"}]}'

        reviewer._chat_content = fake_chat
        events = [
            EventRow(date_text="8.27", performer="Zella Day", venue="MAO"),
            EventRow(date_text="9.01", performer="PREP", venue="Modern Sky"),
            EventRow(date_text="9.02", performer="Hanser", venue="MAO"),
        ]
        artists = [
            PlaylistArtist(name="Zella Day", song_count=1, sample_songs=[]),
            PlaylistArtist(name="PREP", song_count=1, sample_songs=[]),
            PlaylistArtist(name="Hanser", song_count=1, sample_songs=[]),
        ]

        suggestions = reviewer.find_best_matches(events, artists)

        self.assertEqual(len(calls), 2)
        self.assertEqual(suggestions[0].artist_name, "Zella Day")
        self.assertEqual(suggestions[1].artist_name, "PREP")
        self.assertEqual(suggestions[2].artist_name, "Hanser")

    def test_reviewer_runs_event_batches_in_parallel(self):
        config = AiMatchConfig(
            enabled=True,
            api_key="test-key",
            base_url="https://api.example.com/v1",
            model="text-model",
            event_batch_size=1,
            event_workers=2,
        )
        reviewer = AiArtistReviewer(config)
        lock = threading.Lock()
        both_started = threading.Event()
        active_count = 0
        max_active_count = 0

        def fake_batch(batch, artists, start_index):
            nonlocal active_count, max_active_count
            with lock:
                active_count += 1
                max_active_count = max(max_active_count, active_count)
                if active_count == 2:
                    both_started.set()
            both_started.wait(timeout=1)
            with lock:
                active_count -= 1
            return {
                start_index: AiMatchSuggestion(
                    artist_name=artists[start_index].name,
                    confidence="\u9ad8",
                    reason="\u5e76\u884c\u6279\u6b21",
                )
            }

        reviewer._find_best_matches_batch = fake_batch
        events = [
            EventRow(date_text="8.27", performer="Zella Day", venue="MAO"),
            EventRow(date_text="9.01", performer="PREP", venue="Modern Sky"),
        ]
        artists = [
            PlaylistArtist(name="Zella Day", song_count=1, sample_songs=[]),
            PlaylistArtist(name="PREP", song_count=1, sample_songs=[]),
        ]

        suggestions = reviewer.find_best_matches(events, artists)

        self.assertEqual(max_active_count, 2)
        self.assertEqual(set(suggestions), {0, 1})

    def test_reviewer_keeps_successful_parallel_batches_when_one_batch_fails(self):
        config = AiMatchConfig(
            enabled=True,
            api_key="test-key",
            base_url="https://api.example.com/v1",
            model="text-model",
            event_batch_size=1,
            event_workers=2,
        )
        reviewer = AiArtistReviewer(config)

        def fake_batch(batch, artists, start_index):
            if start_index == 0:
                raise RuntimeError("provider rejected this batch")
            return {
                start_index: AiMatchSuggestion(
                    artist_name="PREP",
                    confidence="高",
                    reason="other batch succeeded",
                )
            }

        reviewer._find_best_matches_batch = fake_batch
        events = [
            EventRow(date_text="8.27", performer="Zella Day", venue="MAO"),
            EventRow(date_text="9.01", performer="PREP", venue="Modern Sky"),
        ]
        artists = [
            PlaylistArtist(name="Zella Day", song_count=1, sample_songs=[]),
            PlaylistArtist(name="PREP", song_count=1, sample_songs=[]),
        ]

        suggestions = reviewer.find_best_matches(events, artists)

        self.assertEqual(set(suggestions), {1})
        self.assertEqual(suggestions[1].artist_name, "PREP")
        self.assertEqual(len(reviewer.last_failures), 1)

    def test_reviewer_does_not_cache_partial_batch_results(self):
        config = AiMatchConfig(
            enabled=True,
            api_key="test-key",
            base_url="https://api.example.com/v1",
            model="text-model",
            event_batch_size=1,
            event_workers=1,
        )
        events = [
            EventRow(date_text="8.27", performer="Zella Day", venue="MAO"),
            EventRow(date_text="9.01", performer="PREP", venue="Modern Sky"),
        ]
        artists = [
            PlaylistArtist(name="Zella Day", song_count=1, sample_songs=[]),
            PlaylistArtist(name="PREP", song_count=1, sample_songs=[]),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            reviewer = AiArtistReviewer(config, output_root=Path(tmp))

            def partially_failing_batch(batch, artists, start_index):
                if start_index == 0:
                    raise RuntimeError("provider rejected this batch")
                return {
                    1: AiMatchSuggestion(
                        artist_name="PREP",
                        confidence="高",
                        reason="second batch succeeded",
                    )
                }

            reviewer._find_best_matches_batch = partially_failing_batch
            first = reviewer.find_best_matches(events, artists)

            calls = []

            def successful_batch(batch, artists, start_index):
                calls.append(start_index)
                return {
                    start_index: AiMatchSuggestion(
                        artist_name=artists[start_index].name,
                        confidence="高",
                        reason="retry succeeded",
                    )
                }

            reviewer._find_best_matches_batch = successful_batch
            second = reviewer.find_best_matches(events, artists)

        self.assertEqual(set(first), {1})
        self.assertEqual(calls, [0, 1])
        self.assertEqual(set(second), {0, 1})

    def test_reviewer_reuses_cached_batch_suggestions(self):
        config = AiMatchConfig(
            enabled=True,
            api_key="test-key",
            base_url="https://api.example.com/v1",
            model="text-model",
            event_batch_size=2,
            event_workers=1,
        )
        events = [
            EventRow(date_text="8.27", performer="Zella Day", venue="MAO"),
            EventRow(date_text="9.01", performer="PREP", venue="Modern Sky"),
        ]
        artists = [
            PlaylistArtist(name="Zella Day", song_count=1, sample_songs=["Hypnotic"]),
            PlaylistArtist(name="PREP", song_count=2, sample_songs=["Cheapest Flight"]),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            reviewer = AiArtistReviewer(config, output_root=Path(tmp))
            calls = []

            def fake_chat(payload):
                calls.append(payload)
                return (
                    '{"matches":['
                    '{"event_index":0,"artist_name":"Zella Day","confidence":"\u9ad8","reason":"same"},'
                    '{"event_index":1,"artist_name":"PREP","confidence":"\u9ad8","reason":"same"}'
                    "]}"
                )

            reviewer._chat_content = fake_chat
            first = reviewer.find_best_matches(events, artists)

            reviewer_again = AiArtistReviewer(config, output_root=Path(tmp))
            reviewer_again._chat_content = lambda payload: (_ for _ in ()).throw(
                AssertionError("cached suggestions should not call AI again")
            )
            second = reviewer_again.find_best_matches(events, artists)

        self.assertEqual(len(calls), 1)
        self.assertEqual(first, second)
        self.assertEqual(second[0].artist_name, "Zella Day")

    def test_reviewer_checks_artist_candidate_chunks_beyond_first_limit(self):
        config = AiMatchConfig(
            enabled=True,
            api_key="test-key",
            base_url="https://api.example.com/v1",
            model="text-model",
            candidate_limit=2,
            event_batch_size=1,
            event_workers=1,
        )
        reviewer = AiArtistReviewer(config)
        candidate_batches = []

        def fake_chat(payload):
            content = payload["messages"][-1]["content"]
            data = json.loads(content.split("JSON:\n", 1)[1])
            candidates = [item["name"] for item in data["playlist_candidates"]]
            candidate_batches.append(candidates)
            if "Late Artist" not in candidates:
                return '{"matches":[{"event_index":0,"artist_name":null,"confidence":"\u4f4e","reason":"no"}]}'
            return '{"matches":[{"event_index":0,"artist_name":"Late Artist","confidence":"\u9ad8","reason":"later chunk"}]}'

        reviewer._chat_content = fake_chat
        events = [EventRow(date_text="8.27", performer="Late Artist", venue="MAO")]
        artists = [
            PlaylistArtist(name="Artist 1", song_count=1, sample_songs=[]),
            PlaylistArtist(name="Artist 2", song_count=1, sample_songs=[]),
            PlaylistArtist(name="Artist 3", song_count=1, sample_songs=[]),
            PlaylistArtist(name="Late Artist", song_count=1, sample_songs=[]),
        ]

        suggestions = reviewer.find_best_matches(events, artists)

        self.assertEqual(
            candidate_batches,
            [["Artist 1", "Artist 2"], ["Artist 3", "Late Artist"]],
        )
        self.assertEqual(suggestions[0].artist_name, "Late Artist")

    def test_reviewer_splits_artist_candidates_when_single_event_times_out(self):
        config = AiMatchConfig(
            enabled=True,
            api_key="test-key",
            base_url="https://api.example.com/v1",
            model="text-model",
            candidate_limit=4,
            event_batch_size=1,
            event_workers=1,
        )
        reviewer = AiArtistReviewer(config)
        candidate_batches = []

        def fake_chat(payload):
            content = payload["messages"][-1]["content"]
            data = json.loads(content.split("JSON:\n", 1)[1])
            candidates = [item["name"] for item in data["playlist_candidates"]]
            candidate_batches.append(candidates)
            if len(candidates) == 4:
                raise TimeoutError("The read operation timed out")
            if "Late Artist" not in candidates:
                return '{"matches":[{"event_index":0,"artist_name":null,"confidence":"\u4f4e","reason":"no"}]}'
            return '{"matches":[{"event_index":0,"artist_name":"Late Artist","confidence":"\u9ad8","reason":"split candidates"}]}'

        reviewer._chat_content = fake_chat
        events = [EventRow(date_text="8.27", performer="Late Artist", venue="MAO")]
        artists = [
            PlaylistArtist(name="Artist 1", song_count=1, sample_songs=[]),
            PlaylistArtist(name="Artist 2", song_count=1, sample_songs=[]),
            PlaylistArtist(name="Artist 3", song_count=1, sample_songs=[]),
            PlaylistArtist(name="Late Artist", song_count=1, sample_songs=[]),
        ]

        suggestions = reviewer.find_best_matches(events, artists)

        self.assertEqual(
            candidate_batches,
            [
                ["Artist 1", "Artist 2", "Artist 3", "Late Artist"],
                ["Artist 1", "Artist 2"],
                ["Artist 3", "Late Artist"],
            ],
        )
        self.assertEqual(suggestions[0].artist_name, "Late Artist")

    def test_reviewer_repairs_malformed_batch_match_json_with_ai(self):
        config = AiMatchConfig(
            enabled=True,
            api_key="test-key",
            base_url="https://api.example.com/v1",
            model="text-model",
            event_batch_size=2,
        )
        reviewer = AiArtistReviewer(config)
        calls = []

        def fake_chat(payload):
            calls.append(payload)
            if len(calls) == 1:
                return "event 0 -> Zella Day, high confidence, same artist"
            return '{"matches":[{"event_index":0,"artist_name":"Zella Day","confidence":"\u9ad8","reason":"same artist"}]}'

        reviewer._chat_content = fake_chat
        events = [EventRow(date_text="8.27", performer="ZeIIa Day", venue="MAO")]
        artists = [PlaylistArtist(name="Zella Day", song_count=1, sample_songs=["Hypnotic"])]

        suggestions = reviewer.find_best_matches(events, artists)

        self.assertEqual(suggestions[0].artist_name, "Zella Day")
        self.assertEqual(len(calls), 2)
        self.assertIn("修复", calls[1]["messages"][0]["content"])

    def test_reviewer_splits_timed_out_batch_and_retries_smaller_batches(self):
        config = AiMatchConfig(
            enabled=True,
            api_key="test-key",
            base_url="https://api.example.com/v1",
            model="text-model",
            event_batch_size=4,
            event_workers=1,
        )
        reviewer = AiArtistReviewer(config)
        calls = []

        def fake_chat(payload):
            content = payload["messages"][-1]["content"]
            batch = json.loads(content.split("JSON:\n", 1)[1])["events"]
            indexes = [item["event_index"] for item in batch]
            calls.append(indexes)
            if indexes == [0, 1, 2, 3]:
                raise TimeoutError("The read operation timed out")
            matches = [
                {
                    "event_index": index,
                    "artist_name": f"Artist {index}",
                    "confidence": "\u9ad8",
                    "reason": "\u62c6\u5206\u540e\u6210\u529f",
                }
                for index in indexes
            ]
            return json.dumps({"matches": matches}, ensure_ascii=False)

        reviewer._chat_content = fake_chat
        events = [EventRow(date_text=f"8.{index + 1}", performer=f"Artist {index}", venue="MAO") for index in range(4)]
        artists = [PlaylistArtist(name=f"Artist {index}", song_count=1, sample_songs=[]) for index in range(4)]

        suggestions = reviewer.find_best_matches(events, artists)

        self.assertEqual(calls, [[0, 1, 2, 3], [0, 1], [2, 3]])
        self.assertEqual(set(suggestions.keys()), {0, 1, 2, 3})
        self.assertEqual(suggestions[3].artist_name, "Artist 3")


if __name__ == "__main__":
    unittest.main()
