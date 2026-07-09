from __future__ import annotations

import json
import os
import re
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from urllib import parse, request

from services.models import PlaylistArtist

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126 Safari/537.36",
    "Referer": "https://music.163.com/",
}

_NETEASE_MAX_WORKERS = max(1, min(4, int(os.environ.get("NETEASE_MAX_WORKERS", "3"))))


def extract_playlist_id(url: str) -> str:
    text = str(url or "").strip()
    parsed = parse.urlparse(text)
    query = parse.parse_qs(parsed.query)
    if "id" in query and query["id"]:
        return query["id"][0]
    if parsed.fragment:
        fragment = parsed.fragment
        if "?" in fragment:
            query = parse.parse_qs(fragment.split("?", 1)[1])
            if "id" in query and query["id"]:
                return query["id"][0]
    match = re.search(r"(?:playlist\?id=|id=)(\d+)", text)
    if match:
        return match.group(1)
    raise ValueError("\u6ca1\u6709\u5728\u7f51\u6613\u4e91\u94fe\u63a5\u4e2d\u627e\u5230\u6b4c\u5355 id")


def _fetch_json(url: str, timeout: int = 20) -> dict:
    req = request.Request(url, headers=HEADERS)
    with request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _artist_names(song: dict) -> list[str]:
    artists = song.get("artists") or song.get("ar") or []
    return [artist.get("name", "").strip() for artist in artists if artist.get("name")]


def _fetch_song_chunk(id_chunk: list[str]) -> dict[str, dict]:
    ids_param = parse.quote(json.dumps([int(item) for item in id_chunk]))
    detail = _fetch_json(f"https://music.163.com/api/song/detail?ids={ids_param}")
    return {str(song.get("id")): song for song in detail.get("songs", []) if song.get("id")}


def fetch_playlist_artists(playlist_url: str, limit: int = 1000) -> list[PlaylistArtist]:
    playlist_id = extract_playlist_id(playlist_url)
    detail_url = f"https://music.163.com/api/v6/playlist/detail?id={playlist_id}&n={limit}&s=0"
    playlist_data = _fetch_json(detail_url)
    playlist = playlist_data.get("playlist") or {}
    track_ids = [str(item.get("id")) for item in playlist.get("trackIds", []) if item.get("id")]
    fallback_tracks = {str(song.get("id")): song for song in playlist.get("tracks", []) if song.get("id")}

    chunks = [track_ids[index : index + 80] for index in range(0, len(track_ids), 80)]
    detail_tracks: dict[str, dict] = {}
    if len(chunks) <= 1:
        if chunks:
            detail_tracks.update(_fetch_song_chunk(chunks[0]))
    else:
        workers = min(_NETEASE_MAX_WORKERS, len(chunks))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for chunk_result in executor.map(_fetch_song_chunk, chunks):
                detail_tracks.update(chunk_result)

    songs: list[dict] = []
    for song_id in track_ids:
        songs.append(detail_tracks.get(song_id) or fallback_tracks.get(song_id) or {"id": song_id})

    artists: OrderedDict[str, PlaylistArtist] = OrderedDict()
    mutable: dict[str, dict] = {}
    for song in songs:
        song_name = str(song.get("name") or "").strip()
        for artist_name in _artist_names(song):
            if artist_name not in mutable:
                mutable[artist_name] = {"count": 0, "songs": []}
            mutable[artist_name]["count"] += 1
            if song_name and len(mutable[artist_name]["songs"]) < 5 and song_name not in mutable[artist_name]["songs"]:
                mutable[artist_name]["songs"].append(song_name)

    for artist_name, data in mutable.items():
        artists[artist_name] = PlaylistArtist(
            name=artist_name,
            song_count=data["count"],
            sample_songs=data["songs"],
        )
    return list(artists.values())
