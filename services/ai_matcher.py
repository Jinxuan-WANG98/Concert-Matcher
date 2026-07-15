from __future__ import annotations

import json
import os
import re
import socket
import hashlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import request

from services.debug_timing import PhaseTimer, debug_log
from services.models import EventRow, PlaylistArtist
from services.ocr_cache import cache_enabled, cache_root

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
    event_performer: str = ""


@dataclass(frozen=True)
class AiMatchConfig:
    enabled: bool = False
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4.1-mini"
    timeout_seconds: int = 90
    candidate_limit: int = 200
    shortlist_per_event: int = 5
    minimum_candidate_tier: int = 4
    mode: str = "review"
    event_batch_size: int = 20
    event_workers: int = 2
    max_calls: int = 30
    max_elapsed_seconds: int = 600

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
            timeout_seconds=int(os.environ.get("AI_MATCH_TIMEOUT_SECONDS", "90")),
            candidate_limit=max(1, int(os.environ.get("AI_MATCH_CANDIDATE_LIMIT", "200"))),
            shortlist_per_event=max(1, int(os.environ.get("AI_MATCH_SHORTLIST_PER_EVENT", "5"))),
            minimum_candidate_tier=max(
                1, min(6, int(os.environ.get("AI_MATCH_MIN_CANDIDATE_TIER", "4")))
            ),
            mode=mode,
            event_batch_size=max(1, int(os.environ.get("AI_MATCH_EVENT_BATCH_SIZE", "20"))),
            event_workers=max(1, int(os.environ.get("AI_MATCH_EVENT_WORKERS", "2"))),
            max_calls=max(1, int(os.environ.get("AI_MATCH_MAX_CALLS", "30"))),
            max_elapsed_seconds=max(1, int(os.environ.get("AI_MATCH_MAX_ELAPSED_SECONDS", "600"))),
        )

    @property
    def cache_source(self) -> str:
        return (
            f"ai-match:v10:{self.base_url}:{self.model}:"
            f"c{self.candidate_limit}:s{self.shortlist_per_event}:"
            f"t{self.minimum_candidate_tier}:mode{self.mode}"
        )


def _bounded_spelling_similarity(left: str, right: str, max_distance: int = 2) -> float:
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    effective_max_distance = min(max_distance, 1 if min(len(left), len(right)) < 8 else max_distance)
    length_gap = abs(len(left) - len(right))
    if length_gap > effective_max_distance:
        return 0.0

    if len(left) == len(right):
        mismatches = [index for index, pair in enumerate(zip(left, right)) if pair[0] != pair[1]]
        if (
            len(mismatches) == 2
            and mismatches[1] == mismatches[0] + 1
            and left[mismatches[0]] == right[mismatches[1]]
            and left[mismatches[1]] == right[mismatches[0]]
        ):
            return 1.0 - (1.0 / len(left))
        if len(mismatches) > effective_max_distance:
            return 0.0
        return 1.0 - (len(mismatches) / len(left))

    shorter, longer = (left, right) if len(left) < len(right) else (right, left)
    short_index = 0
    long_index = 0
    edits = 0
    while short_index < len(shorter) and long_index < len(longer):
        if shorter[short_index] == longer[long_index]:
            short_index += 1
            long_index += 1
            continue
        edits += 1
        long_index += 1
        if edits > effective_max_distance:
            return 0.0
    edits += (len(longer) - long_index) + (len(shorter) - short_index)
    if edits > effective_max_distance:
        return 0.0
    return 1.0 - (edits / len(longer))


def _candidate_tokens(value: str) -> frozenset[str]:
    return frozenset(
        part for part in re.findall(r"[a-z0-9\u4e00-\u9fff]+", value.casefold()) if part != "and"
    )


def _candidate_ngrams(value: str) -> frozenset[str]:
    if len(value) <= 2:
        return frozenset({value}) if value else frozenset()
    return frozenset(value[index : index + 2] for index in range(len(value) - 1))


def _candidate_text_features(value: str) -> tuple[str, str, frozenset[str], frozenset[str]]:
    from services.matcher import normalize_name, normalize_ocr_latin

    key = normalize_name(value)
    return key, normalize_ocr_latin(value), _candidate_tokens(value), _candidate_ngrams(key)


def _candidate_relevance_from_features(
    alias_features: list[tuple[str, str, frozenset[str], frozenset[str]]],
    artist_features: tuple[str, str, frozenset[str], frozenset[str]],
) -> tuple[int, float]:
    artist_key, artist_ocr_key, artist_tokens, artist_ngrams = artist_features
    best_tier = 0
    best_score = 0.0
    for alias_key, alias_ocr_key, alias_tokens, alias_ngrams in alias_features:
        if alias_key and alias_key == artist_key:
            tier, score = 6, 1.0
        elif alias_ocr_key and len(alias_ocr_key) >= 4 and alias_ocr_key == artist_ocr_key:
            tier, score = 5, 1.0
        elif (
            alias_key
            and artist_key
            and (alias_key in artist_key or artist_key in alias_key)
            and min(len(alias_key), len(artist_key)) / max(len(alias_key), len(artist_key)) >= 0.6
        ):
            shorter = min(len(alias_key), len(artist_key))
            longer = max(len(alias_key), len(artist_key))
            tier, score = 4, shorter / longer
        elif alias_tokens and artist_tokens and (
            alias_tokens.issubset(artist_tokens) or artist_tokens.issubset(alias_tokens)
        ):
            tier = 3
            score = min(len(alias_tokens), len(artist_tokens)) / max(
                len(alias_tokens), len(artist_tokens)
            )
        else:
            spelling_score = max(
                _bounded_spelling_similarity(alias_key, artist_key),
                _bounded_spelling_similarity(alias_ocr_key, artist_ocr_key),
            )
            if spelling_score > 0:
                tier, score = 2, spelling_score
            else:
                union = alias_ngrams | artist_ngrams
                tier = 1
                score = len(alias_ngrams & artist_ngrams) / len(union) if union else 0.0
        if (tier, score) > (best_tier, best_score):
            best_tier, best_score = tier, score
    return best_tier, best_score


def _candidate_relevance_tier(event_performer: str, artist_name: str) -> int:
    from services.matcher import event_aliases

    alias_features = [_candidate_text_features(alias.value) for alias in event_aliases(event_performer)]
    tier, _score = _candidate_relevance_from_features(
        alias_features,
        _candidate_text_features(artist_name),
    )
    return tier


def build_event_candidate_shortlists(
    events: list[EventRow],
    artists: list[PlaylistArtist],
    event_indices: list[int] | None = None,
    per_event_limit: int = 5,
) -> dict[int, list[PlaylistArtist]]:
    """Return a deterministic, high-recall lexical shortlist for every event."""
    resolved_indices = _resolved_event_indices(events, 0, event_indices)
    limit = max(1, per_event_limit)
    from services.matcher import event_aliases

    artist_features = []
    for artist in artists:
        artist_features.append((artist, *_candidate_text_features(artist.name)))

    shortlists: dict[int, list[PlaylistArtist]] = {}
    for event_index, event in zip(resolved_indices, events):
        alias_features = []
        for alias in event_aliases(event.performer):
            alias_features.append(_candidate_text_features(alias.value))

        def rank(feature) -> tuple[int, float, str, str]:
            artist, artist_key, artist_ocr_key, artist_tokens, artist_ngrams = feature
            best_tier, best_score = _candidate_relevance_from_features(
                alias_features,
                (artist_key, artist_ocr_key, artist_tokens, artist_ngrams),
            )
            return (-best_tier, -best_score, artist_key, artist.name.casefold())

        shortlists[event_index] = [feature[0] for feature in sorted(artist_features, key=rank)[:limit]]
    return shortlists


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
    candidate_limit: int | None = 120,
    start_index: int = 0,
    event_indices: list[int] | None = None,
    compact_candidates: bool = False,
    candidate_names_by_event_index: dict[int, list[str]] | None = None,
) -> dict[str, Any]:
    system = (
        "\u4f60\u662f\u6b4c\u624b\u540d\u79f0\u6279\u91cf\u5339\u914d\u5668\u3002"
        "\u4ece\u7ed9\u5b9a\u7684\u6b4c\u5355\u6b4c\u624b\u5019\u9009\u4e2d\uff0c"
        "\u4e3a\u6bcf\u4e2a\u6f14\u51fa\u540d\u9009\u62e9\u6700\u53ef\u80fd\u7684\u6b4c\u5355\u6b4c\u624b\u3002"
        "\u5fc5\u987b\u53ea\u8fd4\u56de\u4e25\u683c JSON\uff0c\u683c\u5f0f\u4e3a "
        '{"matches":[{"event_index": number, "event_performer": string, "artist_name": string|null, "confidence": "\u9ad8|\u4e2d|\u4f4e", "reason": string}]}\u3002'
        "event_index \u5fc5\u987b\u4f7f\u7528\u8f93\u5165\u91cc\u7684 event_index\u3002"
        "event_performer \u5fc5\u987b\u539f\u6837\u5199\u56de\u540c\u4e00\u4e2a event_index \u7684\u8f93\u5165\u6f14\u51fa\u540d\u3002"
        "\u660e\u786e\u662f\u540c\u4e00\u6b4c\u624b\u6216\u7ec4\u5408\u7684\u4e0d\u540c\u5199\u6cd5\u5fc5\u987b\u9009\u62e9\u8be5\u5019\u9009\uff0c\u5305\u542b\u4e2d\u82f1\u6587/\u62fc\u97f3/\u8bd1\u540d\u3001\u827a\u540d\u4e0e\u672c\u540d\u3001\u7b80\u7e41\u4f53\u3001"
        "\u7a7a\u683c\u6216\u7b26\u53f7\u5dee\u5f02\uff0c\u4ee5\u53ca\u53ef\u660e\u786e\u8fd8\u539f\u7684 OCR \u9519\u5b57\u3002"
        "\u4e0a\u8ff0\u8eab\u4efd\u5bf9\u5e94\u6e05\u6670\u4f46\u5199\u6cd5\u5b58\u5728\u8f7b\u5fae\u6b67\u4e49\u65f6\uff0c\u8fd4\u56de\u4e2d\u7f6e\u4fe1\u5ea6\uff0c\u4e0d\u8981\u76f4\u63a5\u8fd4\u56de null\u3002"
        "\u53ea\u6709\u65e0\u6cd5\u786e\u8ba4\u662f\u540c\u4e00\u827a\u4eba\u65f6\u624d\u8fd4\u56de artist_name: null \u548c confidence: \"\u4f4e\"\u3002"
        "\u4e0d\u8981\u56e0\u4e3a\u98ce\u683c\u76f8\u4f3c\u3001\u540c\u573a\u65e5\u671f\u6216\u5408\u4f5c\u5173\u7cfb\u731c\u6d4b\uff1b\u4e0d\u8981\u8865\u5145 JSON \u4e4b\u5916\u7684\u6587\u5b57\u3002"
    )
    system += (
        "\u53ea\u8fd4\u56de\u786e\u5b9a\u547d\u4e2d\u7684 matches\uff1b"
        "\u65e0\u6cd5\u786e\u8ba4\u7684\u6f14\u51fa\u4e0d\u8981\u8f93\u51fa\u5bf9\u8c61\uff0c"
        "\u4e0d\u8981\u8fd4\u56de artist_name: null \u7684\u884c\u3002"
    )
    if candidate_names_by_event_index is not None:
        system += (
            "\u6bcf\u4e2a events \u5bf9\u8c61\u90fd\u6709\u81ea\u5df1\u7684 candidate_names\uff1b"
            "\u53ea\u80fd\u4ece\u8be5 event_index \u81ea\u5df1\u7684 candidate_names \u4e2d\u9009\u62e9\uff0c"
            "\u4e0d\u5f97\u4f7f\u7528\u5176\u4ed6\u6d3b\u52a8\u7684\u5019\u9009\u4eba\u3002"
            "\u8f93\u51fa\u7684 JSON \u6839\u5bf9\u8c61\u53ea\u80fd\u6709 matches\uff1b"
            "\u4e0d\u8981\u628a events \u6216 candidate_names \u6284\u56de\u8f93\u51fa\uff0c"
            "\u4e0d\u8981\u628a matches \u5d4c\u5957\u5728 events \u91cc\u3002"
        )
    if event_indices is None:
        resolved_event_indices = [start_index + index for index in range(len(events))]
    else:
        if len(event_indices) != len(events):
            raise ValueError("event_indices must match events")
        resolved_event_indices = list(event_indices)

    event_items = []
    for event_index, event in zip(resolved_event_indices, events):
        item = {
            "event_index": event_index,
            "event_performer": event.performer,
            "event_date": event.date_text,
            "event_venue": event.venue,
        }
        if candidate_names_by_event_index is not None:
            item["candidate_names"] = list(candidate_names_by_event_index.get(event_index, []))
        event_items.append(item)
    user = {"events": event_items}
    if candidate_names_by_event_index is None:
        selected_artists = artists if candidate_limit is None else artists[:candidate_limit]
        user["playlist_candidates"] = [
            {"name": artist.name}
            if compact_candidates
            else {
                "name": artist.name,
                "song_count": artist.song_count,
                "sample_songs": artist.sample_songs[:5],
            }
            for artist in selected_artists
        ]
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


def build_ai_match_repair_payload(model: str, raw_text: str, schema: str) -> dict[str, Any]:
    system = (
        "你是 JSON 修复器。只修复格式，不新增、推测或改写原始输出里没有的匹配判断。"
        "把输入整理成程序可解析的严格 JSON。"
    )
    prompt = (
        f"请把下面的 AI 匹配原始输出修复为严格 JSON，格式只能是：{schema}\n"
        "规则：\n"
        "1. 只能使用原始输出里已经出现的匹配判断。\n"
        "2. artist_name 不确定时用 null。\n"
        "3. confidence 只能是 高、中、低。\n"
        "4. 不要返回 Markdown，不要解释，只返回 JSON。\n"
        f"原始输出：\n{raw_text}"
    )
    return {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
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
    if not raw_matches and isinstance(data.get("events"), list):
        nested_matches = []
        for event_item in data["events"]:
            if not isinstance(event_item, dict) or not isinstance(event_item.get("matches"), list):
                continue
            for raw_match in event_item["matches"]:
                if not isinstance(raw_match, dict):
                    continue
                match = dict(raw_match)
                match.setdefault("event_index", event_item.get("event_index"))
                match.setdefault("event_performer", event_item.get("event_performer", ""))
                nested_matches.append(match)
        raw_matches = nested_matches

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
            event_performer=str(item.get("event_performer", "")).strip(),
        )
    return suggestions


MATCH_CACHE_VERSION = 3


def _match_cache_fingerprint(
    events: list[EventRow],
    artists: list[PlaylistArtist],
    cache_source: str,
    event_indices: list[int],
) -> str:
    payload = {
        "source": cache_source,
        "event_indices": event_indices,
        "events": [
            {
                "date_text": event.date_text,
                "performer": event.performer,
                "venue": event.venue,
            }
            for event in events
        ],
        "artists": [
            {
                "name": artist.name,
                "song_count": artist.song_count,
                "sample_songs": artist.sample_songs[:5],
            }
            for artist in artists
        ],
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _match_cache_path(
    events: list[EventRow],
    artists: list[PlaylistArtist],
    cache_source: str,
    event_indices: list[int],
    output_root: Path | None,
) -> Path:
    fingerprint = _match_cache_fingerprint(events, artists, cache_source, event_indices)
    return cache_root(output_root) / f"ai-match-{fingerprint}.json"


def _suggestion_to_dict(suggestion: AiMatchSuggestion) -> dict[str, str]:
    return {
        "artist_name": suggestion.artist_name,
        "confidence": suggestion.confidence,
        "reason": suggestion.reason,
        "event_performer": suggestion.event_performer,
    }


def _suggestion_from_dict(data: dict[str, Any]) -> AiMatchSuggestion:
    return AiMatchSuggestion(
        artist_name=str(data["artist_name"]),
        confidence=str(data["confidence"]),
        reason=str(data.get("reason", "")),
        event_performer=str(data.get("event_performer", "")),
    )


def _load_match_cache(
    events: list[EventRow],
    artists: list[PlaylistArtist],
    cache_source: str,
    event_indices: list[int],
    output_root: Path | None,
) -> dict[int, AiMatchSuggestion] | None:
    if output_root is None or not cache_enabled():
        return None

    path = _match_cache_path(events, artists, cache_source, event_indices, output_root)
    if not path.exists():
        return None

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    if payload.get("cache_version") != MATCH_CACHE_VERSION:
        return None
    if payload.get("cache_source") != cache_source:
        return None
    expected_fingerprint = _match_cache_fingerprint(events, artists, cache_source, event_indices)
    if payload.get("fingerprint") != expected_fingerprint:
        return None

    raw_suggestions = payload.get("suggestions", {})
    if not isinstance(raw_suggestions, dict):
        return None
    try:
        return {int(index): _suggestion_from_dict(value) for index, value in raw_suggestions.items()}
    except (TypeError, ValueError, KeyError):
        return None


def _save_match_cache(
    events: list[EventRow],
    artists: list[PlaylistArtist],
    cache_source: str,
    event_indices: list[int],
    suggestions: dict[int, AiMatchSuggestion],
    output_root: Path | None,
) -> None:
    if output_root is None or not cache_enabled():
        return

    cache_dir = cache_root(output_root)
    cache_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = _match_cache_fingerprint(events, artists, cache_source, event_indices)
    payload = {
        "cache_version": MATCH_CACHE_VERSION,
        "cache_source": cache_source,
        "fingerprint": fingerprint,
        "suggestions": {
            str(index): _suggestion_to_dict(suggestion) for index, suggestion in sorted(suggestions.items())
        },
    }
    _match_cache_path(events, artists, cache_source, event_indices, output_root).write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )


def _confidence_rank(value: str) -> int:
    return {"\u9ad8": 3, "\u4e2d": 2, "\u4f4e": 1}.get(value, 0)


def _merge_suggestions(
    current: dict[int, AiMatchSuggestion],
    new: dict[int, AiMatchSuggestion],
) -> dict[int, AiMatchSuggestion]:
    merged = dict(current)
    for index, suggestion in new.items():
        existing = merged.get(index)
        new_rank = _confidence_rank(suggestion.confidence)
        existing_rank = _confidence_rank(existing.confidence) if existing is not None else -1
        deterministic_key = (suggestion.artist_name.casefold(), suggestion.reason.casefold())
        existing_key = (
            (existing.artist_name.casefold(), existing.reason.casefold())
            if existing is not None
            else ("", "")
        )
        if existing is None or new_rank > existing_rank or (
            new_rank == existing_rank and deterministic_key < existing_key
        ):
            merged[index] = suggestion
    return merged


def _resolved_event_indices(events: list[EventRow], start_index: int, event_indices: list[int] | None) -> list[int]:
    if event_indices is None:
        return [start_index + index for index in range(len(events))]
    if len(event_indices) != len(events):
        raise ValueError("event_indices must match events")
    return list(event_indices)


def _validate_batch_suggestions(
    suggestions: dict[int, AiMatchSuggestion],
    events: list[EventRow],
    event_indices: list[int],
    require_event_performer: bool,
    candidate_names_by_event_index: dict[int, list[str]] | None = None,
    minimum_candidate_tier: int = 0,
) -> dict[int, AiMatchSuggestion]:
    events_by_index = dict(zip(event_indices, events))
    accepted: dict[int, AiMatchSuggestion] = {}
    for event_index, suggestion in suggestions.items():
        event = events_by_index.get(event_index)
        if event is None:
            debug_log(
                "ai_matcher.py:_validate_batch_suggestions",
                "ignored ai match outside current batch",
                {"eventIndex": event_index},
                hypothesis_id="H8",
            )
            continue
        if require_event_performer:
            from services.matcher import normalize_name

            if not suggestion.event_performer or normalize_name(suggestion.event_performer) != normalize_name(event.performer):
                debug_log(
                    "ai_matcher.py:_validate_batch_suggestions",
                    "ignored ai match with mismatched event performer",
                    {"eventIndex": event_index},
                    hypothesis_id="H8",
                )
                continue
        if candidate_names_by_event_index is not None:
            from services.matcher import normalize_name

            allowed_names = {
                normalize_name(name) for name in candidate_names_by_event_index.get(event_index, [])
            }
            if normalize_name(suggestion.artist_name) not in allowed_names:
                debug_log(
                    "ai_matcher.py:_validate_batch_suggestions",
                    "ignored ai match outside event shortlist",
                    {"eventIndex": event_index, "artistName": suggestion.artist_name},
                    hypothesis_id="H9",
                )
                continue
        candidate_tier = _candidate_relevance_tier(event.performer, suggestion.artist_name)
        if candidate_tier < minimum_candidate_tier:
            debug_log(
                "ai_matcher.py:_validate_batch_suggestions",
                "ignored ai match below local relevance threshold",
                {
                    "eventIndex": event_index,
                    "artistName": suggestion.artist_name,
                    "candidateTier": candidate_tier,
                    "minimumTier": minimum_candidate_tier,
                },
                hypothesis_id="H10",
            )
            continue
        accepted[event_index] = suggestion
    return accepted


class AiArtistReviewer:
    def __init__(self, config: AiMatchConfig | None = None, output_root: Path | None = None):
        self.config = config or AiMatchConfig.from_env()
        self.output_root = output_root
        self.last_failures: list[Exception] = []
        self._budget_lock = threading.Lock()
        self._calls_used = 0
        self._deadline: float | None = None

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

    def _guarded_chat_content(self, payload: dict[str, Any]) -> str:
        with self._budget_lock:
            if self._deadline is None:
                self._deadline = time.monotonic() + self.config.max_elapsed_seconds
            if time.monotonic() >= self._deadline:
                raise RuntimeError("AI 匹配超过总时限，任务已停止，未返回不完整结果。")
            if self._calls_used >= self.config.max_calls:
                raise RuntimeError("AI 匹配达到调用次数上限，任务已停止，未返回不完整结果。")
            self._calls_used += 1
        return self._chat_content(payload)

    def _parse_with_ai_repair(self, raw_text: str, parser, schema: str):
        try:
            return parser(raw_text)
        except (json.JSONDecodeError, TypeError, ValueError):
            payload = build_ai_match_repair_payload(self.config.model, raw_text, schema)
            return parser(self._guarded_chat_content(payload))

    def review(self, event: EventRow, artist: PlaylistArtist) -> AiDecision | None:
        if not self.config.enabled:
            return None

        payload = build_review_payload(event, artist, model=self.config.model)
        return self._parse_with_ai_repair(
            self._guarded_chat_content(payload),
            parse_ai_decision,
            '{"is_match": boolean, "confidence": "高|中|低", "reason": string}',
        )

    def find_best_match(self, event: EventRow, artists: list[PlaylistArtist]) -> AiMatchSuggestion | None:
        if not self.config.enabled or not artists:
            return None

        return self._find_best_matches_batch(
            [event],
            artists,
            start_index=0,
            event_indices=[0],
        ).get(0)

    def find_best_matches(
        self,
        events: list[EventRow],
        artists: list[PlaylistArtist],
        event_indices: list[int] | None = None,
    ) -> dict[int, AiMatchSuggestion]:
        if not self.config.enabled or not events or not artists:
            return {}

        resolved_indices = _resolved_event_indices(events, 0, event_indices)
        require_event_performer = event_indices is not None
        with PhaseTimer("ai_matcher.py:find_best_matches", "ai_match_total") as timer:
            timer.data = {
                "eventCount": len(events),
                "artistCount": len(artists),
                "mode": self.config.mode,
            }
            self.last_failures = []
            cached = _load_match_cache(events, artists, self.config.cache_source, resolved_indices, self.output_root)
            if cached is not None:
                timer.data["cacheHit"] = True
                timer.data["suggestionCount"] = len(cached)
                return cached
            with self._budget_lock:
                self._calls_used = 0
                self._deadline = time.monotonic() + self.config.max_elapsed_seconds

            batch_size = max(1, self.config.event_batch_size)
            batches = [
                (events[offset : offset + batch_size], resolved_indices[offset : offset + batch_size])
                for offset in range(0, len(events), batch_size)
            ]
            suggestions: dict[int, AiMatchSuggestion] = {}
            worker_count = min(max(1, self.config.event_workers), len(batches))
            if worker_count <= 1:
                for batch, batch_indices in batches:
                    try:
                        suggestions = _merge_suggestions(
                            suggestions,
                            self._find_best_matches_batch(
                                batch,
                                artists,
                                start_index=batch_indices[0],
                                event_indices=batch_indices,
                                require_event_performer=require_event_performer,
                            ),
                        )
                    except Exception as exc:
                        self.last_failures.append(exc)
                if len(self.last_failures) == len(batches):
                    raise self.last_failures[0]
                if not self.last_failures:
                    _save_match_cache(events, artists, self.config.cache_source, resolved_indices, suggestions, self.output_root)
                timer.data["suggestionCount"] = len(suggestions)
                timer.data["failureCount"] = len(self.last_failures)
                timer.data["batchCount"] = len(batches)
                return suggestions

            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = {
                    executor.submit(
                        self._find_best_matches_batch,
                        batch,
                        artists,
                        batch_indices[0],
                        batch_indices,
                        require_event_performer,
                    ): batch_indices[0]
                    for batch, batch_indices in batches
                }
                for future in as_completed(futures):
                    try:
                        suggestions = _merge_suggestions(suggestions, future.result())
                    except Exception as exc:
                        self.last_failures.append(exc)
            if len(self.last_failures) == len(batches):
                raise self.last_failures[0]
            if not self.last_failures:
                _save_match_cache(events, artists, self.config.cache_source, resolved_indices, suggestions, self.output_root)
            timer.data["suggestionCount"] = len(suggestions)
            timer.data["failureCount"] = len(self.last_failures)
            timer.data["batchCount"] = len(batches)
            return suggestions

    def _find_best_matches_batch(
        self,
        events: list[EventRow],
        artists: list[PlaylistArtist],
        start_index: int,
        event_indices: list[int] | None = None,
        require_event_performer: bool = False,
    ) -> dict[int, AiMatchSuggestion]:
        resolved_indices = _resolved_event_indices(events, start_index, event_indices)
        shortlists = build_event_candidate_shortlists(
            events,
            artists,
            event_indices=resolved_indices,
            per_event_limit=self.config.shortlist_per_event,
        )
        candidate_names_by_event_index = {
            event_index: [artist.name for artist in event_artists]
            for event_index, event_artists in shortlists.items()
        }

        return self._find_best_matches_batch_for_candidates(
            events,
            artists,
            start_index=start_index,
            event_indices=resolved_indices,
            require_event_performer=require_event_performer,
            compact_candidates=True,
            candidate_names_by_event_index=candidate_names_by_event_index,
        )

    def _find_best_matches_batch_for_candidates(
        self,
        events: list[EventRow],
        artists: list[PlaylistArtist],
        start_index: int,
        compact_candidates: bool = False,
        event_indices: list[int] | None = None,
        require_event_performer: bool = False,
        timeout_split_depth: int = 0,
        candidate_names_by_event_index: dict[int, list[str]] | None = None,
    ) -> dict[int, AiMatchSuggestion]:
        resolved_indices = _resolved_event_indices(events, start_index, event_indices)
        payload = build_batch_artist_pick_payload(
            events,
            artists,
            model=self.config.model,
            candidate_limit=self.config.candidate_limit,
            start_index=start_index,
            event_indices=resolved_indices,
            compact_candidates=compact_candidates,
            candidate_names_by_event_index=candidate_names_by_event_index,
        )
        try:
            suggestions = self._parse_with_ai_repair(
                self._guarded_chat_content(payload),
                parse_ai_batch_match_suggestions,
                '{"matches":[{"event_index": number, "event_performer": string, "artist_name": string|null, "confidence": "高|中|低", "reason": string}]}',
            )
            return _validate_batch_suggestions(
                suggestions,
                events,
                resolved_indices,
                require_event_performer,
                candidate_names_by_event_index,
                self.config.minimum_candidate_tier,
            )
        except Exception as exc:
            if not _is_timeout_error(exc):
                raise
            if timeout_split_depth >= 1:
                raise
            if len(events) <= 1:
                if candidate_names_by_event_index is not None:
                    event_index = resolved_indices[0]
                    candidate_names = candidate_names_by_event_index.get(event_index, [])
                    if len(candidate_names) <= 1:
                        raise
                    midpoint = len(candidate_names) // 2
                    left_candidates = {event_index: candidate_names[:midpoint]}
                    right_candidates = {event_index: candidate_names[midpoint:]}
                    suggestions = self._find_best_matches_batch_for_candidates(
                        events,
                        artists,
                        start_index=start_index,
                        compact_candidates=compact_candidates,
                        event_indices=resolved_indices,
                        require_event_performer=require_event_performer,
                        timeout_split_depth=timeout_split_depth + 1,
                        candidate_names_by_event_index=left_candidates,
                    )
                    return _merge_suggestions(
                        suggestions,
                        self._find_best_matches_batch_for_candidates(
                            events,
                            artists,
                            start_index=start_index,
                            compact_candidates=compact_candidates,
                            event_indices=resolved_indices,
                            require_event_performer=require_event_performer,
                            timeout_split_depth=timeout_split_depth + 1,
                            candidate_names_by_event_index=right_candidates,
                        ),
                    )
                if len(artists) <= 1:
                    raise
                midpoint = len(artists) // 2
                suggestions = self._find_best_matches_batch_for_candidates(
                    events,
                    artists[:midpoint],
                    start_index=start_index,
                    compact_candidates=compact_candidates,
                    event_indices=resolved_indices,
                    require_event_performer=require_event_performer,
                    timeout_split_depth=timeout_split_depth + 1,
                    candidate_names_by_event_index=candidate_names_by_event_index,
                )
                return _merge_suggestions(
                    suggestions,
                    self._find_best_matches_batch_for_candidates(
                        events,
                        artists[midpoint:],
                        start_index=start_index,
                        compact_candidates=compact_candidates,
                        event_indices=resolved_indices,
                        require_event_performer=require_event_performer,
                        timeout_split_depth=timeout_split_depth + 1,
                        candidate_names_by_event_index=candidate_names_by_event_index,
                    ),
                )
            midpoint = len(events) // 2
            suggestions = self._find_best_matches_batch_for_candidates(
                events[:midpoint],
                artists,
                start_index=start_index,
                compact_candidates=compact_candidates,
                event_indices=resolved_indices[:midpoint],
                require_event_performer=require_event_performer,
                timeout_split_depth=timeout_split_depth + 1,
                candidate_names_by_event_index=candidate_names_by_event_index,
            )
            return _merge_suggestions(
                suggestions,
                self._find_best_matches_batch_for_candidates(
                    events[midpoint:],
                    artists,
                    start_index=resolved_indices[midpoint],
                    compact_candidates=compact_candidates,
                    event_indices=resolved_indices[midpoint:],
                    require_event_performer=require_event_performer,
                    timeout_split_depth=timeout_split_depth + 1,
                    candidate_names_by_event_index=candidate_names_by_event_index,
                ),
            )


def _is_timeout_error(exc: Exception) -> bool:
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return True
    return "timed out" in str(exc).lower()
