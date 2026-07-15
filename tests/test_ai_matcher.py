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
    build_event_candidate_shortlists,
    _merge_suggestions,
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

    def test_config_uses_current_prompt_cache_version(self):
        config = AiMatchConfig(
            enabled=True,
            api_key="test-key",
            base_url="https://api.example.com/v1",
            model="text-model",
        )

        self.assertTrue(config.cache_source.startswith("ai-match:v9:"))

    def test_config_defaults_bound_calls_and_use_two_workers(self):
        config = AiMatchConfig()

        self.assertEqual(config.candidate_limit, 200)
        self.assertEqual(config.timeout_seconds, 90)
        self.assertEqual(config.event_batch_size, 20)
        self.assertEqual(config.shortlist_per_event, 5)
        self.assertEqual(config.event_workers, 2)
        self.assertEqual(config.max_calls, 30)
        self.assertEqual(config.max_elapsed_seconds, 600)

    def test_config_loads_shortlist_per_event(self):
        old_value = os.environ.pop("AI_MATCH_SHORTLIST_PER_EVENT", None)
        try:
            os.environ["AI_MATCH_SHORTLIST_PER_EVENT"] = "7"
            config = AiMatchConfig.from_env()
        finally:
            if old_value is None:
                os.environ.pop("AI_MATCH_SHORTLIST_PER_EVENT", None)
            else:
                os.environ["AI_MATCH_SHORTLIST_PER_EVENT"] = old_value

        self.assertEqual(config.shortlist_per_event, 7)

    def test_shortlists_prioritize_alias_and_ocr_equivalent_names(self):
        events = [
            EventRow(date_text="9.01", performer="周华健", venue="MAO"),
            EventRow(date_text="9.12", performer="VoX LoW", venue="星在"),
        ]
        artists = [
            PlaylistArtist(name="Unrelated Artist", song_count=1, sample_songs=[]),
            PlaylistArtist(name="Wakin Chau", song_count=10, sample_songs=[]),
            PlaylistArtist(name="Vox Low", song_count=8, sample_songs=[]),
            PlaylistArtist(name="Another Guest", song_count=1, sample_songs=[]),
        ]

        shortlists = build_event_candidate_shortlists(
            events,
            artists,
            event_indices=[11, 12],
            per_event_limit=2,
        )

        self.assertEqual(shortlists[11][0].name, "Wakin Chau")
        self.assertEqual(shortlists[12][0].name, "Vox Low")
        self.assertEqual(len(shortlists[11]), 2)
        self.assertEqual(len(shortlists[12]), 2)

    def test_shortlists_expand_known_bilingual_aliases_in_both_directions(self):
        event = EventRow(date_text="9.01", performer="Wakin Chau", venue="MAO")
        artists = [
            PlaylistArtist(name=name, song_count=1, sample_songs=[])
            for name in ["Charlie", "Echo", "Alpha", "Bravo", "Delta", "周华健"]
        ]

        shortlist = build_event_candidate_shortlists([event], artists, [0], 5)[0]

        self.assertEqual(shortlist[0].name, "周华健")

    def test_shortlists_keep_adjacent_transposition_spelling_candidate(self):
        event = EventRow(date_text="9.01", performer="ABCD", venue="MAO")
        artists = [
            PlaylistArtist(name=name, song_count=1, sample_songs=[])
            for name in ["AAAA", "AAAB", "AABB", "AACC", "AADD", "ACBD"]
        ]

        shortlist = build_event_candidate_shortlists([event], artists, [0], 5)[0]

        self.assertEqual(shortlist[0].name, "ACBD")

    def test_shortlists_are_deterministic_when_similarity_is_equal(self):
        event = EventRow(date_text="9.01", performer="No Lexical Signal", venue="MAO")
        artists = [
            PlaylistArtist(name="Zulu", song_count=1, sample_songs=[]),
            PlaylistArtist(name="Alpha", song_count=1, sample_songs=[]),
            PlaylistArtist(name="Bravo", song_count=1, sample_songs=[]),
        ]

        first = build_event_candidate_shortlists([event], artists, [0], 2)
        second = build_event_candidate_shortlists([event], list(reversed(artists)), [0], 2)

        self.assertEqual([artist.name for artist in first[0]], [artist.name for artist in second[0]])
        self.assertEqual(len(first[0]), 2)

    def test_config_loads_call_and_elapsed_budgets(self):
        names = [
            "AI_MATCH_API_KEY",
            "AI_MATCH_ENABLED",
            "AI_MATCH_MAX_CALLS",
            "AI_MATCH_MAX_ELAPSED_SECONDS",
        ]
        old_values = {name: os.environ.pop(name, None) for name in names}
        try:
            os.environ["AI_MATCH_ENABLED"] = "true"
            os.environ["AI_MATCH_API_KEY"] = "test-key"
            os.environ["AI_MATCH_MAX_CALLS"] = "9"
            os.environ["AI_MATCH_MAX_ELAPSED_SECONDS"] = "240"

            config = AiMatchConfig.from_env()
        finally:
            for name, value in old_values.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

        self.assertEqual(config.max_calls, 9)
        self.assertEqual(config.max_elapsed_seconds, 240)

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

    def test_batch_payload_can_scope_candidates_per_event(self):
        events = [
            EventRow(date_text="8.27", performer="ZeIIa Day", venue="MAO"),
            EventRow(date_text="9.01", performer="PREP", venue="Modern Sky"),
        ]
        artists = [
            PlaylistArtist(name="Zella Day", song_count=1, sample_songs=[]),
            PlaylistArtist(name="PREP", song_count=2, sample_songs=[]),
        ]

        payload = build_batch_artist_pick_payload(
            events,
            artists,
            model="text-model",
            event_indices=[10, 11],
            candidate_names_by_event_index={10: ["Zella Day"], 11: ["PREP"]},
        )
        user = json.loads(payload["messages"][-1]["content"].split("JSON:\n", 1)[1])

        self.assertNotIn("playlist_candidates", user)
        self.assertEqual(user["events"][0]["candidate_names"], ["Zella Day"])
        self.assertEqual(user["events"][1]["candidate_names"], ["PREP"])
        self.assertIn("candidate_names", payload["messages"][0]["content"])

    def test_batch_payload_uses_explicit_original_event_indices(self):
        events = [
            EventRow(date_text="9.6", performer="叶琼琳", venue="新歌空间"),
            EventRow(date_text="9.12", performer="VoX LoW", venue="星在"),
        ]
        payload = build_batch_artist_pick_payload(
            events,
            [PlaylistArtist(name="VOX LOW", song_count=1, sample_songs=[])],
            event_indices=[101, 116],
        )

        user = json.loads(payload["messages"][-1]["content"].split("JSON:\n", 1)[1])

        self.assertEqual([item["event_index"] for item in user["events"]], [101, 116])
        self.assertIn("event_performer", payload["messages"][0]["content"])

    def test_batch_payload_allows_clear_alias_and_ocr_identity_matches(self):
        payload = build_batch_artist_pick_payload(
            [EventRow(date_text="10.24", performer="Jackson Wang", venue="")],
            [PlaylistArtist(name="王嘉尔", song_count=1, sample_songs=["LMLY"])],
        )

        system = payload["messages"][0]["content"]

        self.assertIn("艺名", system)
        self.assertIn("中英文", system)
        self.assertIn("简繁", system)
        self.assertIn("OCR", system)
        self.assertIn("中", system)
        self.assertIn("只返回确定命中", system)

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

    def test_parse_ai_batch_match_suggestion_keeps_echoed_performer(self):
        suggestions = parse_ai_batch_match_suggestions(
            '{"matches":[{"event_index":116,"event_performer":"VoX LoW",'
            '"artist_name":"VOX LOW","confidence":"\u9ad8","reason":"\u540c\u540d"}]}'
        )

        self.assertEqual(suggestions[116].event_performer, "VoX LoW")

    def test_parse_ai_batch_match_suggestions_accepts_glm_nested_event_matches(self):
        suggestions = parse_ai_batch_match_suggestions(
            json.dumps(
                {
                    "events": [
                        {
                            "event_index": 101,
                            "event_performer": "山形瑞秋Rachael Yamagata",
                            "candidate_names": ["Rachael Yamagata", "AGA"],
                            "matches": [
                                {
                                    "artist_name": "Rachael Yamagata",
                                    "confidence": "高",
                                    "reason": "same identity",
                                }
                            ],
                        }
                    ]
                },
                ensure_ascii=False,
            )
        )

        self.assertEqual(suggestions[101].artist_name, "Rachael Yamagata")
        self.assertEqual(suggestions[101].event_performer, "山形瑞秋Rachael Yamagata")

    def test_reviewer_rejects_unknown_index_and_mismatched_performer(self):
        config = AiMatchConfig(
            enabled=True,
            api_key="test-key",
            base_url="https://api.example.com/v1",
            model="text-model",
            event_batch_size=2,
            event_workers=1,
        )
        reviewer = AiArtistReviewer(config)
        reviewer._chat_content = lambda payload: json.dumps(
            {
                "matches": [
                    {
                        "event_index": 999,
                        "event_performer": "VoX LoW",
                        "artist_name": "VOX LOW",
                        "confidence": "高",
                        "reason": "wrong batch",
                    },
                    {
                        "event_index": 101,
                        "event_performer": "VoX LoW",
                        "artist_name": "VOX LOW",
                        "confidence": "高",
                        "reason": "wrong row",
                    },
                ]
            },
            ensure_ascii=False,
        )

        suggestions = reviewer.find_best_matches(
            [EventRow(date_text="9.6", performer="叶琼琳", venue="新歌空间")],
            [PlaylistArtist(name="VOX LOW", song_count=1, sample_songs=[])],
            event_indices=[101],
        )

        self.assertEqual(suggestions, {})

    def test_reviewer_rejects_artist_outside_the_events_shortlist(self):
        config = AiMatchConfig(
            enabled=True,
            api_key="test-key",
            base_url="https://api.example.com/v1",
            model="text-model",
            shortlist_per_event=1,
            event_batch_size=2,
            event_workers=1,
        )
        reviewer = AiArtistReviewer(config)

        def cross_event_answer(payload):
            user = json.loads(payload["messages"][-1]["content"].split("JSON:\n", 1)[1])
            first_candidate = user["events"][0]["candidate_names"][0]
            second_candidate = user["events"][1]["candidate_names"][0]
            self.assertNotEqual(first_candidate, second_candidate)
            return json.dumps(
                {
                    "matches": [
                        {
                            "event_index": 10,
                            "event_performer": "Alpha",
                            "artist_name": second_candidate,
                            "confidence": "高",
                            "reason": "candidate from another event",
                        }
                    ]
                },
                ensure_ascii=False,
            )

        reviewer._chat_content = cross_event_answer
        suggestions = reviewer.find_best_matches(
            [
                EventRow(date_text="9.01", performer="Alpha", venue="MAO"),
                EventRow(date_text="9.02", performer="Beta", venue="MAO"),
            ],
            [
                PlaylistArtist(name="Alpha", song_count=1, sample_songs=[]),
                PlaylistArtist(name="Beta", song_count=1, sample_songs=[]),
                PlaylistArtist(name="Noise", song_count=1, sample_songs=[]),
            ],
            event_indices=[10, 11],
        )

        self.assertEqual(suggestions, {})

    def test_large_reviewer_plan_uses_one_shortlisted_call_per_event_batch(self):
        config = AiMatchConfig(
            enabled=True,
            api_key="test-key",
            base_url="https://api.example.com/v1",
            model="text-model",
            shortlist_per_event=5,
            candidate_limit=200,
            event_batch_size=20,
            event_workers=1,
            max_calls=30,
        )
        reviewer = AiArtistReviewer(config)
        payloads = []

        def fake_chat(payload):
            payloads.append(json.loads(payload["messages"][-1]["content"].split("JSON:\n", 1)[1]))
            return '{"matches":[]}'

        reviewer._chat_content = fake_chat
        events = [
            EventRow(date_text=f"9.{index + 1}", performer=f"Artist {index}", venue="MAO")
            for index in range(177)
        ]
        artists = [
            PlaylistArtist(name=f"Artist {index}", song_count=1, sample_songs=[])
            for index in range(592)
        ]

        reviewer.find_best_matches(events, artists)

        self.assertEqual(len(payloads), 9)
        self.assertEqual(sum(len(payload["events"]) for payload in payloads), 177)
        self.assertTrue(
            all(len(event["candidate_names"]) == 5 for payload in payloads for event in payload["events"])
        )
        self.assertTrue(all("playlist_candidates" not in payload for payload in payloads))

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

    def test_reviewer_sends_event_specific_candidates_once_per_large_event_batch(self):
        config = AiMatchConfig(
            enabled=True,
            api_key="test-key",
            base_url="https://api.example.com/v1",
            model="text-model",
            candidate_limit=1000,
            event_batch_size=2,
            event_workers=1,
            max_calls=30,
        )
        reviewer = AiArtistReviewer(config)
        calls = []

        def fake_chat(payload):
            content = payload["messages"][-1]["content"]
            calls.append(json.loads(content.split("JSON:\n", 1)[1]))
            return '{"matches":[{"event_index":4,"artist_name":"Artist 4","confidence":"高","reason":"same"}]}'

        reviewer._chat_content = fake_chat
        events = [
            EventRow(date_text=f"8.{index + 1}", performer=f"Artist {index}", venue="MAO")
            for index in range(51)
        ]
        artists = [
            PlaylistArtist(name=f"Artist {index}", song_count=1, sample_songs=["Song"])
            for index in range(5)
        ]

        suggestions = reviewer.find_best_matches(events, artists)

        # 51 events / batch 2 => 26 calls; every event carries only its own shortlist.
        self.assertEqual(len(calls), 26)
        self.assertTrue(all(len(item["events"]) <= 2 for item in calls))
        self.assertNotIn("playlist_candidates", calls[0])
        self.assertEqual(len(calls[0]["events"][0]["candidate_names"]), 5)
        self.assertEqual(calls[0]["events"][0]["candidate_names"][0], "Artist 0")
        self.assertEqual([item["event_index"] for item in calls[0]["events"]], [0, 1])
        self.assertEqual(suggestions[4].artist_name, "Artist 4")

    def test_reviewer_uses_event_specific_candidates_for_normal_event_batches(self):
        config = AiMatchConfig(
            enabled=True,
            api_key="test-key",
            base_url="https://api.example.com/v1",
            model="text-model",
            candidate_limit=1000,
            event_batch_size=40,
            event_workers=1,
        )
        reviewer = AiArtistReviewer(config)
        payloads = []

        def fake_chat(payload):
            content = payload["messages"][-1]["content"]
            payloads.append(json.loads(content.split("JSON:\n", 1)[1]))
            return '{"matches":[]}'

        reviewer._chat_content = fake_chat
        reviewer.find_best_matches(
            [EventRow(date_text="8.27", performer="PREP", venue="MAO")],
            [PlaylistArtist(name="PREP", song_count=20, sample_songs=["Cheapest Flight"])],
        )

        self.assertNotIn("playlist_candidates", payloads[0])
        self.assertEqual(payloads[0]["events"][0]["candidate_names"], ["PREP"])

    def test_timeout_candidate_split_stops_after_one_recovery_level(self):
        config = AiMatchConfig(
            enabled=True,
            api_key="test-key",
            base_url="https://api.example.com/v1",
            model="text-model",
            candidate_limit=4,
            event_batch_size=1,
            event_workers=1,
            max_calls=2,
            max_elapsed_seconds=600,
        )
        reviewer = AiArtistReviewer(config)
        calls = []

        def always_timeout(payload):
            calls.append(payload)
            raise TimeoutError("The read operation timed out")

        reviewer._chat_content = always_timeout
        events = [EventRow(date_text="8.27", performer="Late Artist", venue="MAO")]
        artists = [
            PlaylistArtist(name=f"Artist {index}", song_count=1, sample_songs=[])
            for index in range(4)
        ]

        with self.assertRaisesRegex(TimeoutError, "timed out"):
            reviewer.find_best_matches(events, artists)

        self.assertEqual(len(calls), 2)

    def test_timeout_event_batch_splits_once_without_recursive_fanout(self):
        config = AiMatchConfig(
            enabled=True,
            api_key="test-key",
            base_url="https://api.example.com/v1",
            model="text-model",
            candidate_limit=10,
            event_batch_size=4,
            event_workers=1,
            max_calls=30,
            max_elapsed_seconds=600,
        )
        reviewer = AiArtistReviewer(config)
        call_sizes = []

        def timeout(payload):
            content = payload["messages"][-1]["content"]
            body = json.loads(content.split("JSON:\n", 1)[1])
            call_sizes.append(len(body["events"]))
            raise TimeoutError("The read operation timed out")

        reviewer._chat_content = timeout
        events = [
            EventRow(date_text=f"8.{index + 1}", performer=f"Artist {index}", venue="MAO")
            for index in range(4)
        ]
        artists = [PlaylistArtist(name="Artist 0", song_count=1, sample_songs=[])]

        with self.assertRaisesRegex(TimeoutError, "timed out"):
            reviewer.find_best_matches(events, artists)

        self.assertEqual(call_sizes, [4, 2])

    def test_terminal_single_event_single_artist_timeout_is_not_treated_as_no_match(self):
        config = AiMatchConfig(
            enabled=True,
            api_key="test-key",
            base_url="https://api.example.com/v1",
            model="text-model",
            candidate_limit=1,
            event_batch_size=1,
            event_workers=1,
        )
        reviewer = AiArtistReviewer(config)
        reviewer._chat_content = lambda payload: (_ for _ in ()).throw(
            TimeoutError("The read operation timed out")
        )

        with self.assertRaisesRegex(TimeoutError, "timed out"):
            reviewer.find_best_matches(
                [EventRow(date_text="8.27", performer="PREP", venue="MAO")],
                [PlaylistArtist(name="PREP", song_count=1, sample_songs=[])],
            )

    def test_elapsed_budget_stops_before_provider_call(self):
        config = AiMatchConfig(
            enabled=True,
            api_key="test-key",
            base_url="https://api.example.com/v1",
            model="text-model",
            event_batch_size=1,
            event_workers=1,
            max_calls=20,
            max_elapsed_seconds=0,
        )
        reviewer = AiArtistReviewer(config)
        calls = []
        reviewer._chat_content = lambda payload: calls.append(payload) or '{"matches":[]}'

        with self.assertRaisesRegex(RuntimeError, "总时限"):
            reviewer.find_best_matches(
                [EventRow(date_text="8.27", performer="PREP", venue="MAO")],
                [PlaylistArtist(name="PREP", song_count=1, sample_songs=[])],
            )

        self.assertEqual(calls, [])

    def test_equal_confidence_merge_is_deterministic(self):
        alpha = {0: AiMatchSuggestion(artist_name="Alpha", confidence="高", reason="first")}
        zeta = {0: AiMatchSuggestion(artist_name="Zeta", confidence="高", reason="second")}

        forward = _merge_suggestions(alpha, zeta)
        reverse = _merge_suggestions(zeta, alpha)

        self.assertEqual(forward, reverse)
        self.assertEqual(forward[0].artist_name, "Alpha")

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

        def fake_batch(batch, artists, start_index, event_indices=None, require_event_performer=False):
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

        def fake_batch(batch, artists, start_index, event_indices=None, require_event_performer=False):
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

            def partially_failing_batch(batch, artists, start_index, event_indices=None, require_event_performer=False):
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

            def successful_batch(batch, artists, start_index, event_indices=None, require_event_performer=False):
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

    def test_reviewer_shortlist_finds_artist_beyond_candidate_limit(self):
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
            candidates = data["events"][0]["candidate_names"]
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

        self.assertEqual(len(candidate_batches), 1)
        self.assertEqual(candidate_batches[0][0], "Late Artist")
        self.assertEqual(suggestions[0].artist_name, "Late Artist")

    def test_reviewer_single_match_uses_ranked_shortlist(self):
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
            candidates = data["events"][0]["candidate_names"]
            candidate_batches.append(candidates)
            if "Late Artist" not in candidates:
                return '{"matches":[{"event_index":0,"artist_name":null,"confidence":"低","reason":"no"}]}'
            return '{"matches":[{"event_index":0,"artist_name":"Late Artist","confidence":"高","reason":"later chunk"}]}'

        reviewer._chat_content = fake_chat
        event = EventRow(date_text="8.27", performer="Late Artist", venue="MAO")
        artists = [
            PlaylistArtist(name="Artist 1", song_count=1, sample_songs=[]),
            PlaylistArtist(name="Artist 2", song_count=1, sample_songs=[]),
            PlaylistArtist(name="Artist 3", song_count=1, sample_songs=[]),
            PlaylistArtist(name="Late Artist", song_count=1, sample_songs=[]),
        ]

        suggestion = reviewer.find_best_match(event, artists)

        self.assertEqual(len(candidate_batches), 1)
        self.assertEqual(candidate_batches[0][0], "Late Artist")
        self.assertEqual(suggestion.artist_name, "Late Artist")

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
            candidates = data["events"][0]["candidate_names"]
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

        self.assertEqual([len(batch) for batch in candidate_batches], [4, 2, 2])
        self.assertEqual(candidate_batches[0][0], "Late Artist")
        self.assertEqual(candidate_batches[1] + candidate_batches[2], candidate_batches[0])
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
