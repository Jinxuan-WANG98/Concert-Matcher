from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass, replace

from services.date_parser import date_ranges_overlap, format_date_for_display, parse_date_range
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

AI_ONLY_SINGLE_FALLBACK_EVENT_LIMIT = 20


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


def match_events_to_artists(
    events: list[EventRow],
    artists: list[PlaylistArtist],
    ai_reviewer=None,
    ai_only: bool = False,
) -> list[MatchResult]:
    raw_matches: list[MatchResult] = []
    batch_suggestions = None
    if ai_only and ai_reviewer is not None and hasattr(ai_reviewer, "find_best_matches"):
        batch_suggestions = ai_reviewer.find_best_matches(events, artists)
    allow_single_fallback = batch_suggestions is None or len(events) <= AI_ONLY_SINGLE_FALLBACK_EVENT_LIMIT

    for event_index, event in enumerate(events):
        if ai_only and ai_reviewer is not None and hasattr(ai_reviewer, "find_best_match"):
            suggestion = batch_suggestions.get(event_index) if batch_suggestions is not None else None
            if suggestion is None and allow_single_fallback:
                suggestion = ai_reviewer.find_best_match(event, artists)
            if suggestion is not None and suggestion.confidence != "\u4f4e":
                suggested_artist = _find_artist_by_name(artists, suggestion.artist_name)
                if suggested_artist is not None:
                    raw_matches.append(_match_from_ai_suggestion(event, suggested_artist, suggestion, "\u76f4\u63a5\u5339\u914d"))
            continue

        event_match_count = 0
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
            event_match_count += 1

        if event_match_count == 0 and ai_reviewer is not None and hasattr(ai_reviewer, "find_best_match"):
            suggestion = ai_reviewer.find_best_match(event, artists)
            if suggestion is not None and suggestion.confidence != "\u4f4e":
                suggested_artist = _find_artist_by_name(artists, suggestion.artist_name)
                if suggested_artist is not None:
                    raw_matches.append(_match_from_ai_suggestion(event, suggested_artist, suggestion, "\u5019\u9009\u8865\u5145"))

    deduped: list[MatchResult] = []
    for match in sorted(raw_matches, key=_dedupe_sort_key):
        if any(_is_duplicate_match(match, kept) for kept in deduped):
            continue
        deduped.append(match)

    deduped = _fill_blank_match_venues(deduped, events)
    ordered = sorted(deduped, key=lambda item: (_date_sort_key(item.date_text), item.performer, -item.score))
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


def _confidence_rank(value: str) -> int:
    return {"\u9ad8": 3, "\u4e2d": 2, "\u4f4e": 1}.get(value, 0)


def _date_span_days(value: str) -> int:
    date_range = parse_date_range(value)
    if not date_range:
        return 1
    start, end = date_range
    return max(1, end - start + 1)


def _dedupe_sort_key(match: MatchResult) -> tuple[float, int, int, int, int]:
    return (
        -_date_span_days(match.date_text),
        -match.score,
        -_confidence_rank(match.confidence),
        -int(bool(match.venue.strip())),
        -match.playlist_song_count,
    )


def _is_duplicate_match(candidate: MatchResult, kept: MatchResult) -> bool:
    if normalize_name(candidate.artist_name) != normalize_name(kept.artist_name):
        return False
    return date_ranges_overlap(candidate.date_text, kept.date_text)


def _fill_blank_match_venues(matches: list[MatchResult], events: list[EventRow]) -> list[MatchResult]:
    venue_events = [event for event in events if event.venue.strip()]
    if not venue_events:
        return matches

    enriched: list[MatchResult] = []
    for match in matches:
        if match.venue.strip():
            enriched.append(match)
            continue
        venue = _related_event_venue(match, venue_events)
        enriched.append(replace(match, venue=venue) if venue else match)
    return enriched


def _related_event_venue(match: MatchResult, venue_events: list[EventRow]) -> str:
    candidates = [
        event
        for event in venue_events
        if date_ranges_overlap(match.date_text, event.date_text) and _event_refers_to_match_artist(event, match)
    ]
    if not candidates:
        return ""

    venue_keys = {normalize_name(event.venue) for event in candidates if event.venue.strip()}
    if len(venue_keys) != 1:
        return ""
    candidates = sorted(candidates, key=lambda event: (-_date_span_days(event.date_text), event.image_name))
    return candidates[0].venue


def _event_refers_to_match_artist(event: EventRow, match: MatchResult) -> bool:
    return _names_clearly_same(event.performer, match.performer) or _names_clearly_same(
        event.performer, match.artist_name
    )


def _names_clearly_same(left: str, right: str) -> bool:
    left_key = normalize_name(left)
    right_key = normalize_name(right)
    if not left_key or not right_key:
        return False
    if left_key == right_key:
        return True
    shorter, longer = (left_key, right_key) if len(left_key) < len(right_key) else (right_key, left_key)
    if len(shorter) >= 2 and (_has_cjk(left) or _has_cjk(right)) and shorter in longer:
        return True

    for alias in event_aliases(left):
        if normalize_name(alias.value) == right_key:
            return True
        scored = _score_pair(alias, right)
        if scored and scored[0] >= 0.87:
            return True
    for alias in event_aliases(right):
        if normalize_name(alias.value) == left_key:
            return True
        scored = _score_pair(alias, left)
        if scored and scored[0] >= 0.87:
            return True
    return False


def _find_artist_by_name(artists: list[PlaylistArtist], suggested_name: str) -> PlaylistArtist | None:
    suggested_key = normalize_name(suggested_name)
    if not suggested_key:
        return None
    for artist in artists:
        if normalize_name(artist.name) == suggested_key:
            return artist
    clear_matches = [artist for artist in artists if _names_clearly_same(suggested_name, artist.name)]
    if len(clear_matches) == 1:
        return clear_matches[0]
    return None


def _match_from_ai_suggestion(event: EventRow, artist: PlaylistArtist, suggestion, label: str) -> MatchResult:
    return MatchResult(
        index=0,
        date_text=event.date_text,
        date_display=format_date_for_display(event.date_text),
        performer=clean_name(event.performer),
        venue=event.venue,
        artist_name=artist.name,
        playlist_song_count=artist.song_count,
        sample_songs=artist.sample_songs[:5],
        confidence=suggestion.confidence,
        score=0.84 if suggestion.confidence == "\u9ad8" else 0.76,
        match_method=f"AI{label}\uff1a{suggestion.reason}",
        matched_alias=suggestion.artist_name,
        image_name=event.image_name,
    )
