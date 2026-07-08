from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass

from services.date_parser import format_date_for_display
from services.models import EventRow, MatchResult, PlaylistArtist


ALIASES: dict[str, list[str]] = {
    "\u5468\u534e\u5065": ["Wakin Chau", "Emil Chau", "Emil Wakin Chau"],
    "\u5468\u5174\u54f2(Eric)": ["\u5468\u5174\u54f2", "Eric Chou"],
    "\u5468\u5174\u54f2": ["Eric Chou"],
    "\u8fc8\u514b\u5b66\u6447\u6eda": ["Michael Learns To Rock", "MLTR"],
    "Belle&Sebastian": ["Belle and Sebastian", "Belle Sebastian"],
    "Molly Nilsson": ["Molly Nillson"],
    "GAI": ["GAI\u5468\u5ef6", "\u5468\u5ef6"],
    "\u738b\u5609\u5c14": ["Jackson Wang", "Jackson Wang \u738b\u5609\u5c14"],
    "\u5f20\u667a\u9716": ["Julian Cheung", "Chilam Cheung"],
    "\u5468\u4f20\u96c4": ["Steve Chou", "\u5c0f\u521a"],
    "\u6768\u5343\u5b05": ["Miriam Yeung"],
    "\u9f50\u8c6b": ["Chyi Yu"],
    "\u5b89\u6ea5": ["\u5f20\u60ac", "Deserts Chang", "Anpu"],
    "\u6c5f\u7f8e\u742a": ["Maggie Chiang"],
    "\u6797\u5fd7\u70ab": ["Terry Lin"],
    "\u6797\u5ba5\u5609": ["Yoga Lin"],
    "\u674e\u8363\u6d69": ["Ronghao Li"],
    "\u6f58\u73ae\u67cf": ["Wilber Pan"],
    "\u6e38\u9e3f\u660e": ["Chris Yu"],
    "\u5f90\u826f": ["Xu Liang"],
}

WEAK_TOKENS = {
    "and",
    "the",
    "live",
    "concert",
    "show",
    "festival",
    "\u4e0a\u6d77",
    "\u5609\u5bbe",
    "\u4e13\u573a",
}


@dataclass(frozen=True)
class Alias:
    value: str
    reason: str


def _strip_diacritics(value: str) -> str:
    return "".join(
        char for char in unicodedata.normalize("NFKD", str(value or "")) if not unicodedata.combining(char)
    )


def clean_name(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    for _ in range(3):
        text = re.sub(r"([\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", r"\1", text)
    text = re.sub(r"\s*([/&+\u00d7])\s*", r"\1", text)
    return text.strip()


def normalize_name(value: str) -> str:
    text = _strip_diacritics(clean_name(value)).lower()
    text = text.replace("\u4e28", "i").replace("|", "i")
    text = text.replace("&", "and").replace("+", "and")
    return re.sub(
        r"[\s'\"`.,:;!?()\[\]{}<>\u00b7\u2022\-_/\\|\uff08\uff09\uff0c\u3002\u3001\uff01\uff1a\uff1b\u300a\u300b\u3010\u3011]",
        "",
        text,
    ).strip()


def normalize_ocr_latin(value: str) -> str:
    return normalize_name(value).replace("1", "l").replace("i", "l").replace("\u4e28", "l").replace("|", "l").replace("0", "o")


def _has_cjk(value: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", value))


def _has_latin(value: str) -> bool:
    return bool(re.search(r"[a-z]", _strip_diacritics(value), re.I))


def _tokens(value: str) -> list[str]:
    text = _strip_diacritics(clean_name(value)).lower()
    text = text.replace("&", " and ").replace("+", " and ")
    parts = re.split(r"[^a-z0-9\u4e00-\u9fff]+", text)
    return [part for part in parts if part and part not in WEAK_TOKENS]


def _add_alias(aliases: list[Alias], value: str, reason: str) -> None:
    cleaned = clean_name(value)
    if cleaned:
        aliases.append(Alias(cleaned, reason))


def event_aliases(performer: str) -> list[Alias]:
    cleaned = clean_name(performer)
    aliases: list[Alias] = []
    _add_alias(aliases, cleaned, "\u56fe\u7247\u6f14\u51fa\u540d")

    no_paren = re.sub(r"\([^)]*\)", "", cleaned).strip()
    if no_paren and no_paren != cleaned:
        _add_alias(aliases, no_paren, "\u53bb\u62ec\u53f7\u540d\u79f0")

    for inside in re.findall(r"\(([^)]{2,})\)", cleaned):
        _add_alias(aliases, inside, "\u62ec\u53f7\u5185\u540d\u79f0")

    for alias in ALIASES.get(cleaned, []) + ALIASES.get(no_paren, []):
        _add_alias(aliases, alias, "\u5e38\u89c1\u522b\u540d/\u8bd1\u540d")

    split_source = re.sub(r"\u5609\u5bbe[:\uff1a]?", "/", cleaned)
    split_source = re.sub(r"\b(feat\.?|ft\.|with)\b", "/", split_source, flags=re.I)
    if re.search(r"[/\u3001&+\u00d7]|\s+x\s+|\u5609\u5bbe|feat|with|ft", cleaned, re.I) and not re.match(
        r"^belle\s*&\s*sebastian$", cleaned, re.I
    ):
        for part in re.split(r"\s*(?:[/\u3001&+\u00d7]|\s+x\s+)\s*", split_source, flags=re.I):
            if normalize_name(part):
                _add_alias(aliases, part, "\u5408\u4f5c/\u5609\u5bbe\u62c6\u5206")

    seen: set[str] = set()
    deduped: list[Alias] = []
    for alias in aliases:
        key = normalize_name(alias.value)
        if key and key not in seen:
            seen.add(key)
            deduped.append(alias)
    return deduped


def _score_pair(alias: Alias, artist_name: str) -> tuple[float, str, str] | None:
    a = normalize_name(alias.value)
    b = normalize_name(artist_name)
    if not a or not b:
        return None

    if a == b:
        method = "\u522b\u540d/\u8bd1\u540d\u7cbe\u786e\u5339\u914d" if "\u522b\u540d" in alias.reason else "\u6e05\u6d17\u540e\u7cbe\u786e\u5339\u914d"
        return (0.98 if "\u522b\u540d" in alias.reason else 1.0, "\u9ad8", method)

    ao = normalize_ocr_latin(alias.value)
    bo = normalize_ocr_latin(artist_name)
    if ao == bo and min(len(ao), len(bo)) >= 4:
        return (0.94, "\u4e2d", "OCR \u5b57\u5f62\u6df7\u6dc6\u5339\u914d")

    shorter, longer = (a, b) if len(a) < len(b) else (b, a)
    if shorter and shorter in longer:
        cjk = _has_cjk(alias.value) or _has_cjk(artist_name)
        if cjk and len(shorter) >= 2 and len(shorter) / len(longer) >= 0.45:
            return (0.9, "\u4e2d", "\u4e2d\u6587\u5305\u542b\u5339\u914d")
        if not cjk and len(shorter) >= 6 and len(shorter) / len(longer) >= 0.62:
            return (0.88, "\u4e2d", "\u82f1\u6587\u5305\u542b\u5339\u914d")

    alias_tokens = _tokens(alias.value)
    artist_tokens = _tokens(artist_name)
    if len(alias_tokens) >= 2 and len(artist_tokens) >= 2:
        small, big = (alias_tokens, set(artist_tokens)) if len(alias_tokens) <= len(artist_tokens) else (artist_tokens, set(alias_tokens))
        if all(token in big for token in small):
            return (0.87, "\u4e2d", "\u82f1\u6587 token \u96c6\u5408\u5339\u914d")

    if _has_latin(alias.value) and _has_latin(artist_name) and min(len(a), len(b)) >= 7:
        ratio = max(difflib.SequenceMatcher(None, a, b).ratio(), difflib.SequenceMatcher(None, ao, bo).ratio())
        if ratio >= 0.9:
            return (min(0.86, ratio * 0.9), "\u4e2d", "\u8f7b\u5fae\u62fc\u5199\u5dee\u5f02\u5339\u914d")

    return None


def match_events_to_artists(events: list[EventRow], artists: list[PlaylistArtist], ai_reviewer=None) -> list[MatchResult]:
    raw_matches: list[MatchResult] = []
    for event in events:
        aliases = event_aliases(event.performer)
        for artist in artists:
            best: tuple[float, str, str, Alias] | None = None
            for alias in aliases:
                scored = _score_pair(alias, artist.name)
                if not scored:
                    continue
                score, confidence, method = scored
                if best is None or score > best[0]:
                    best = (score, confidence, method, alias)
            if not best:
                continue
            score, confidence, method, alias = best
            if ai_reviewer is not None and confidence != "\u9ad8":
                decision = ai_reviewer.review(event, artist)
                if decision is not None:
                    if not decision.is_match:
                        continue
                    confidence = decision.confidence
                    method = f"{method} + AI\u590d\u6838\uff1a{decision.reason}"
            raw_matches.append(
                MatchResult(
                    index=0,
                    date_text=event.date_text,
                    date_display=format_date_for_display(event.date_text),
                    performer=clean_name(event.performer),
                    venue=event.venue,
                    artist_name=artist.name,
                    playlist_song_count=artist.song_count,
                    sample_songs=artist.sample_songs[:5],
                    confidence=confidence,
                    score=round(score, 3),
                    match_method=method,
                    matched_alias=alias.value,
                    image_name=event.image_name,
                )
            )

    deduped: dict[tuple[str, str, str], MatchResult] = {}
    for match in sorted(raw_matches, key=lambda item: (-item.score, -item.playlist_song_count)):
        key = (match.date_text, normalize_name(match.performer), match.artist_name)
        deduped.setdefault(key, match)

    ordered = sorted(deduped.values(), key=lambda item: (_date_sort_key(item.date_text), item.performer, -item.score))
    return [
        MatchResult(
            index=index,
            date_text=match.date_text,
            date_display=match.date_display,
            performer=match.performer,
            venue=match.venue,
            artist_name=match.artist_name,
            playlist_song_count=match.playlist_song_count,
            sample_songs=match.sample_songs,
            confidence=match.confidence,
            score=match.score,
            match_method=match.match_method,
            matched_alias=match.matched_alias,
            image_name=match.image_name,
        )
        for index, match in enumerate(ordered, start=1)
    ]


def _date_sort_key(value: str) -> int:
    match = re.match(r"^(\d{1,2})\.(\d{1,2})", value or "")
    if not match:
        return 9999
    return int(match.group(1)) * 100 + int(match.group(2))
