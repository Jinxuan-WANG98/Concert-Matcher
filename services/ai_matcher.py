from __future__ import annotations

import json
import os
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
class AiMatchConfig:
    enabled: bool = False
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4.1-mini"
    timeout_seconds: int = 20

    @classmethod
    def from_env(cls) -> "AiMatchConfig":
        enabled = os.environ.get("AI_MATCH_ENABLED", "").lower() in {"1", "true", "yes", "on"}
        api_key = os.environ.get("AI_MATCH_API_KEY", "").strip()
        return cls(
            enabled=enabled and bool(api_key),
            api_key=api_key,
            base_url=os.environ.get("AI_MATCH_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
            model=os.environ.get("AI_MATCH_MODEL", "gpt-4.1-mini"),
            timeout_seconds=int(os.environ.get("AI_MATCH_TIMEOUT_SECONDS", "20")),
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


def parse_ai_decision(raw_text: str) -> AiDecision:
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.removeprefix("json").strip()
    data = json.loads(text)
    confidence = str(data.get("confidence", "")).strip()
    if confidence not in {"\u9ad8", "\u4e2d", "\u4f4e"}:
        raise ValueError(f"Unknown AI confidence: {confidence}")
    return AiDecision(
        is_match=bool(data.get("is_match")),
        confidence=confidence,
        reason=str(data.get("reason", "")).strip(),
    )


class AiArtistReviewer:
    def __init__(self, config: AiMatchConfig | None = None):
        self.config = config or AiMatchConfig.from_env()

    def review(self, event: EventRow, artist: PlaylistArtist) -> AiDecision | None:
        if not self.config.enabled:
            return None

        payload = build_review_payload(event, artist, model=self.config.model)
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
        content = result["choices"][0]["message"]["content"]
        return parse_ai_decision(content)
