import os
import unittest

from services.ai_matcher import AiDecision, AiMatchConfig, build_review_payload, parse_ai_decision
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


if __name__ == "__main__":
    unittest.main()
