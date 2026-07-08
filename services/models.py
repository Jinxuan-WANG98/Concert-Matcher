from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class PlaylistArtist:
    name: str
    song_count: int = 0
    sample_songs: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class EventRow:
    date_text: str
    performer: str
    venue: str
    image_name: str = ""
    source_note: str = ""


@dataclass(frozen=True)
class MatchResult:
    index: int
    date_text: str
    date_display: str
    performer: str
    venue: str
    artist_name: str
    playlist_song_count: int
    sample_songs: list[str]
    confidence: str
    score: float
    match_method: str
    matched_alias: str
    image_name: str = ""


@dataclass(frozen=True)
class PipelineResult:
    matches: list[MatchResult]
    playlist_artist_count: int
    event_count: int
    excel_path: Path | None = None
    warnings: list[str] = field(default_factory=list)
