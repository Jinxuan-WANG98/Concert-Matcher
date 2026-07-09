from __future__ import annotations

import json
import os
import socket
from dataclasses import dataclass
from typing import Any
from urllib import request

from services.models import EventRow, PlaylistArtist


@dataclass(frozen=True)
class AiDecision:
    is_match: bool
    confidence: str
    reason: str


@dataclass(frozen=True)
class AiMatchSuggestion:
    artist_name: str
    confidence: str
    reason: str


@dataclass(frozen=True)
class AiMatchConfig:
    enabled: bool = False
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4.1-mini"
    timeout_seconds: int = 20
    candidate_limit: int = 120
    mode: str = "review"
    event_batch_size: int = 30

    @classmethod
    def from_env(cls) -> "AiMatchConfig":
        enabled = os.environ.get("AI_MATCH_ENABLED", "").lower() in {"1", "true", "yes", "on"}
        provider_index = os.environ.get("AI_MATCH_PROVIDER_INDEX", "").strip()
        provider_prefix = f"AI_OCR_PROVIDER_{provider_index}_" if provider_index else ""
        provider_api_key = os.environ.get(f"{provider_prefix}API_KEY", "").strip() if provider_prefix else ""
        provider_base_url = os.environ.get(f"{provider_prefix}BASE_URL", "").strip() if provider_prefix else ""
        explicit_api_key = os.environ.get("AI_MATCH_API_KEY", "").strip()
        api_key = provider_api_key if provider_index else explicit_api_key
        if not api_key:
            api_key = explicit_api_key or provider_api_key
        mode = os.environ.get("AI_MATCH_MODE", "review").strip().lower().replace("-", "_")
        if mode not in {"review", "ai_only"}:
            mode = "review"
        explicit_base_url = os.environ.get("AI_MATCH_BASE_URL", "").strip()
        base_url = provider_base_url if provider_index else explicit_base_url
        if not base_url:
            base_url = explicit_base_url or provider_base_url or "https://api.openai.com/v1"
        return cls(
            enabled=enabled and bool(api_key),
            api_key=api_key,
            base_url=base_url.rstrip("/"),
            model=os.environ.get("AI_MATCH_MODEL", "gpt-4.1-mini"),
            timeout_seconds=int(os.environ.get("AI_MATCH_TIMEOUT_SECONDS", "20")),
            candidate_limit=int(os.environ.get("AI_MATCH_CANDIDATE_LIMIT", "120")),
            mode=mode,
            event_batch_size=int(os.environ.get("AI_MATCH_EVENT_BATCH_SIZE", "30")),
        )


def build_review_payload(event: EventRow, artist: PlaylistArtist, model: str = "gpt-4.1-mini") -> dict[str, Any]:
    system = (
        "\u4f60\u662f\u6b4c\u624b\u540d\u79f0\u5339\u914d\u590d\u6838\u5668\u3002"
        "\u53ea\u5224\u65ad\u5019\u9009\u6f14\u51fa\u540d\u548c\u5019\u9009\u6b4c\u5355\u6b4c\u624b\u662f\u5426\u6307\u5411\u540c\u4e00\u4f4d\u6b4c\u624b\u6216\u540c\u4e00\u7ec4\u5408\u3002"
        "\u5fc5\u987b\u8fd4\u56de\u4e25\u683c JSON\uff0c\u683c\u5f0f\u4e3a {\"is_match\": boolean, \"confidence\": \"\u9ad8|\u4e2d|\u4f4e\", \"reason\": string}\u3002"
        "\u4e0d\u8981\u8865\u5145 JSON \u4e4b\u5916\u7684\u6587\u5b57\u3002\u4e0d\u8981\u56e0\u4e3a\u98ce\u683c\u76f8\u4f3c\u5c31\u5224\u5b9a\u5339\u914d\u3002"
    )
    user = {
        "event_performer": event.performer,
        "event_date": event.date_text,
        "event_venue": event.venue,
        "playlist_artist": artist.name,
        "playlist_song_count": artist.song_count,
        "playlist_sample_songs": artist.sample_songs[:5],
    }
    return {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": (
                    "\u8bf7\u590d\u6838\u8fd9\u4e2a\u5019\u9009\u5339\u914d\uff0c"
                    f"\u53ea\u8fd4\u56de JSON:\n{json.dumps(user, ensure_ascii=False)}"
                ),
            },
        ],
    }


def build_artist_pick_payload(
    event: EventRow,
    artists: list[PlaylistArtist],
    model: str = "gpt-4.1-mini",
    candidate_limit: int = 120,
) -> dict[str, Any]:
    system = (
        "\u4f60\u662f\u6b4c\u624b\u540d\u79f0\u5339\u914d\u8865\u5145\u5668\u3002"
        "\u4ece\u7ed9\u5b9a\u7684\u6b4c\u5355\u6b4c\u624b\u5019\u9009\u4e2d\uff0c"
        "\u5224\u65ad\u56fe\u7247\u6f14\u51fa\u540d\u662f\u5426\u6307\u5411\u5176\u4e2d\u4e00\u4f4d\u6b4c\u624b\u6216\u7ec4\u5408\u3002"
        "\u5fc5\u987b\u8fd4\u56de\u4e25\u683c JSON\uff0c\u683c\u5f0f\u4e3a "
        '{"artist_name": string|null, "confidence": "\u9ad8|\u4e2d|\u4f4e", "reason": string}\u3002'
        "\u6ca1\u628a\u63e1\u5c31\u8fd4\u56de null\uff0c\u4e0d\u8981\u56e0\u4e3a\u98ce\u683c\u76f8\u4f3c\u5c31\u731c\u3002"
        "\u4e0d\u8981\u8865\u5145 JSON \u4e4b\u5916\u7684\u6587\u5b57\u3002"
    )
    candidates = [
        {
            "name": artist.name,
            "song_count": artist.song_count,
            "sample_songs": artist.sample_songs[:5],
        }
        for artist in artists[:candidate_limit]
    ]
    user = {
        "event_performer": event.performer,
        "event_date": event.date_text,
        "event_venue": event.venue,
        "playlist_candidates": candidates,
    }
    return {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": (
                    "\u8bf7\u53ea\u5728\u5019\u9009\u5217\u8868\u5185\u9009\u62e9\uff0c"
                    f"\u53ea\u8fd4\u56de JSON:\n{json.dumps(user, ensure_ascii=False)}"
                ),
            },
        ],
    }


def build_batch_artist_pick_payload(
    events: list[EventRow],
    artists: list[PlaylistArtist],
    model: str = "gpt-4.1-mini",
    candidate_limit: int = 120,
    start_index: int = 0,
) -> dict[str, Any]:
    system = (
        "\u4f60\u662f\u6b4c\u624b\u540d\u79f0\u6279\u91cf\u5339\u914d\u5668\u3002"
        "\u4ece\u7ed9\u5b9a\u7684\u6b4c\u5355\u6b4c\u624b\u5019\u9009\u4e2d\uff0c"
        "\u4e3a\u6bcf\u4e2a\u6f14\u51fa\u540d\u9009\u62e9\u6700\u53ef\u80fd\u7684\u6b4c\u5355\u6b4c\u624b\u3002"
        "\u5fc5\u987b\u53ea\u8fd4\u56de\u4e25\u683c JSON\uff0c\u683c\u5f0f\u4e3a "
        '{"matches":[{"event_index": number, "artist_name": string|null, "confidence": "\u9ad8|\u4e2d|\u4f4e", "reason": string}]}\u3002'
        "event_index \u5fc5\u987b\u4f7f\u7528\u8f93\u5165\u91cc\u7684 event_index\u3002"
        "\u6ca1\u628a\u63e1\u5c31\u8fd4\u56de artist_name: null \u548c confidence: \"\u4f4e\"\u3002"
        "\u4e0d\u8981\u56e0\u4e3a\u98ce\u683c\u76f8\u4f3c\u5c31\u731c\uff0c\u4e0d\u8981\u8865\u5145 JSON \u4e4b\u5916\u7684\u6587\u5b57\u3002"
    )
    event_items = [
        {
            "event_index": start_index + index,
            "event_performer": event.performer,
            "event_date": event.date_text,
            "event_venue": event.venue,
        }
        for index, event in enumerate(events)
    ]
    candidates = [
        {
            "name": artist.name,
            "song_count": artist.song_count,
            "sample_songs": artist.sample_songs[:5],
        }
        for artist in artists[:candidate_limit]
    ]
    user = {
        "events": event_items,
        "playlist_candidates": candidates,
    }
    return {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": (
                    "\u8bf7\u53ea\u5728\u5019\u9009\u5217\u8868\u5185\u9009\u62e9\uff0c"
                    f"\u53ea\u8fd4\u56de JSON:\n{json.dumps(user, ensure_ascii=False)}"
                ),
            },
        ],
    }


def _strip_json_fence(raw_text: str) -> str:
    text = str(raw_text or "").strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        text = text.removeprefix("json").strip()
    start = text.find("{")
    if start < 0:
        return text

    depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(text[start:], start=start):
        if escaped:
            escaped = False
            continue
        if char == "\\" and in_string:
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return text[start:]


def parse_ai_decision(raw_text: str) -> AiDecision:
    data = json.loads(_strip_json_fence(raw_text))
    confidence = str(data.get("confidence", "")).strip()
    if confidence not in {"\u9ad8", "\u4e2d", "\u4f4e"}:
        raise ValueError(f"Unknown AI confidence: {confidence}")
    return AiDecision(
        is_match=bool(data.get("is_match")),
        confidence=confidence,
        reason=str(data.get("reason", "")).strip(),
    )


def parse_ai_match_suggestion(raw_text: str) -> AiMatchSuggestion | None:
    data = json.loads(_strip_json_fence(raw_text))
    artist_name = data.get("artist_name")
    if artist_name is None or str(artist_name).strip() == "":
        return None
    confidence = str(data.get("confidence", "")).strip()
    if confidence not in {"\u9ad8", "\u4e2d", "\u4f4e"}:
        raise ValueError(f"Unknown AI confidence: {confidence}")
    return AiMatchSuggestion(
        artist_name=str(artist_name).strip(),
        confidence=confidence,
        reason=str(data.get("reason", "")).strip(),
    )


def parse_ai_batch_match_suggestions(raw_text: str) -> dict[int, AiMatchSuggestion]:
    data = json.loads(_strip_json_fence(raw_text))
    raw_matches = data.get("matches", [])
    if not isinstance(raw_matches, list):
        return {}

    suggestions: dict[int, AiMatchSuggestion] = {}
    for item in raw_matches:
        if not isinstance(item, dict):
            continue
        try:
            event_index = int(item.get("event_index"))
        except (TypeError, ValueError):
            continue
        artist_name = item.get("artist_name")
        if artist_name is None or str(artist_name).strip() == "":
            continue
        confidence = str(item.get("confidence", "")).strip()
        if confidence not in {"\u9ad8", "\u4e2d", "\u4f4e"}:
            raise ValueError(f"Unknown AI confidence: {confidence}")
        suggestions[event_index] = AiMatchSuggestion(
            artist_name=str(artist_name).strip(),
            confidence=confidence,
            reason=str(item.get("reason", "")).strip(),
        )
    return suggestions


class AiArtistReviewer:
    def __init__(self, config: AiMatchConfig | None = None):
        self.config = config or AiMatchConfig.from_env()

    def _chat_content(self, payload: dict[str, Any]) -> str:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = request.Request(
            f"{self.config.base_url}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with request.urlopen(req, timeout=self.config.timeout_seconds) as response:
            result = json.loads(response.read().decode("utf-8"))
        return result["choices"][0]["message"]["content"]

    def review(self, event: EventRow, artist: PlaylistArtist) -> AiDecision | None:
        if not self.config.enabled:
            return None

        payload = build_review_payload(event, artist, model=self.config.model)
        return parse_ai_decision(self._chat_content(payload))

    def find_best_match(self, event: EventRow, artists: list[PlaylistArtist]) -> AiMatchSuggestion | None:
        if not self.config.enabled or not artists:
            return None

        payload = build_artist_pick_payload(
            event,
            artists,
            model=self.config.model,
            candidate_limit=self.config.candidate_limit,
        )
        return parse_ai_match_suggestion(self._chat_content(payload))

    def find_best_matches(self, events: list[EventRow], artists: list[PlaylistArtist]) -> dict[int, AiMatchSuggestion]:
        if not self.config.enabled or not events or not artists:
            return {}

        batch_size = max(1, self.config.event_batch_size)
        suggestions: dict[int, AiMatchSuggestion] = {}
        for start_index in range(0, len(events), batch_size):
            batch = events[start_index : start_index + batch_size]
            suggestions.update(self._find_best_matches_batch(batch, artists, start_index=start_index))
        return suggestions

    def _find_best_matches_batch(
        self,
        events: list[EventRow],
        artists: list[PlaylistArtist],
        start_index: int,
    ) -> dict[int, AiMatchSuggestion]:
        payload = build_batch_artist_pick_payload(
            events,
            artists,
            model=self.config.model,
            candidate_limit=self.config.candidate_limit,
            start_index=start_index,
        )
        try:
            return parse_ai_batch_match_suggestions(self._chat_content(payload))
        except Exception as exc:
            if not _is_timeout_error(exc):
                raise
            if len(events) <= 1:
                return {}
            midpoint = len(events) // 2
            suggestions = self._find_best_matches_batch(events[:midpoint], artists, start_index=start_index)
            suggestions.update(
                self._find_best_matches_batch(
                    events[midpoint:],
                    artists,
                    start_index=start_index + midpoint,
                )
            )
            return suggestions


def _is_timeout_error(exc: Exception) -> bool:
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return True
    return "timed out" in str(exc).lower()
