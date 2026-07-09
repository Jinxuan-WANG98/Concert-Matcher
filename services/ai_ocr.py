from __future__ import annotations

import base64
import io
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import request

from PIL import Image

from services.ai_errors import describe_ai_exception
from services.date_parser import merge_date_range, parse_date_text
from services.models import EventRow
from services.ocr import clean_ocr_text


@dataclass(frozen=True)
class AiOcrProviderConfig:
    name: str
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    model: str = ""


@dataclass(frozen=True)
class AiOcrProviderResult:
    provider_name: str
    events: list[EventRow]
    error: str = ""


@dataclass(frozen=True)
class AiOcrConfig:
    enabled: bool = False
    providers: tuple[AiOcrProviderConfig, ...] = ()
    timeout_seconds: int = 45
    max_width: int = 1600
    image_batch_size: int = 8
    image_workers: int = 2
    min_agreement_ratio: float = 0.2
    min_events: int = 1

    @classmethod
    def from_env(cls) -> "AiOcrConfig":
        enabled = os.environ.get("AI_OCR_ENABLED", "").lower() in {"1", "true", "yes", "on"}
        providers = tuple(_providers_from_env())
        return cls(
            enabled=enabled and bool(providers),
            providers=providers,
            timeout_seconds=int(os.environ.get("AI_OCR_TIMEOUT_SECONDS", "45")),
            max_width=int(os.environ.get("AI_OCR_MAX_WIDTH", "1600")),
            image_batch_size=max(1, int(os.environ.get("AI_OCR_IMAGE_BATCH_SIZE", "8"))),
            image_workers=max(1, int(os.environ.get("AI_OCR_IMAGE_WORKERS", "2"))),
            min_agreement_ratio=float(os.environ.get("AI_OCR_MIN_AGREEMENT_RATIO", "0.2")),
            min_events=int(os.environ.get("AI_OCR_MIN_EVENTS", "1")),
        )

    @property
    def cache_source(self) -> str:
        provider_source = "|".join(f"{provider.name}:{provider.model}:{provider.base_url}" for provider in self.providers)
        return f"ai-ocr:{provider_source}:w{self.max_width}:agree{self.min_agreement_ratio}:min{self.min_events}"

    @property
    def api_key(self) -> str:
        return self.providers[0].api_key if self.providers else ""

    @property
    def base_url(self) -> str:
        return self.providers[0].base_url if self.providers else "https://api.openai.com/v1"

    @property
    def model(self) -> str:
        return self.providers[0].model if self.providers else ""


def _providers_from_env() -> list[AiOcrProviderConfig]:
    providers: list[AiOcrProviderConfig] = []
    for index in range(1, 5):
        prefix = f"AI_OCR_PROVIDER_{index}_"
        provider_enabled = os.environ.get(f"{prefix}ENABLED", "true").strip().lower()
        if provider_enabled in {"0", "false", "no", "off"}:
            continue
        api_key = os.environ.get(f"{prefix}API_KEY", "").strip()
        base_url = os.environ.get(f"{prefix}BASE_URL", "").strip().rstrip("/")
        model = os.environ.get(f"{prefix}MODEL", "").strip()
        name = os.environ.get(f"{prefix}NAME", f"provider_{index}").strip() or f"provider_{index}"
        if api_key and base_url and model:
            providers.append(AiOcrProviderConfig(name=name, api_key=api_key, base_url=base_url, model=model))

    if providers:
        return providers

    api_key = (os.environ.get("AI_OCR_API_KEY") or os.environ.get("AI_MATCH_API_KEY") or "").strip()
    base_url = (
        os.environ.get("AI_OCR_BASE_URL")
        or os.environ.get("AI_MATCH_BASE_URL")
        or "https://api.openai.com/v1"
    ).rstrip("/")
    model = os.environ.get("AI_OCR_MODEL", "").strip()
    if api_key and model:
        providers.append(
            AiOcrProviderConfig(
                name=os.environ.get("AI_OCR_PROVIDER_NAME", "primary").strip() or "primary",
                api_key=api_key,
                base_url=base_url,
                model=model,
            )
        )
    return providers


def build_ai_ocr_payload(model: str, image_data_url: str | list[str]) -> dict[str, Any]:
    image_data_urls = [image_data_url] if isinstance(image_data_url, str) else list(image_data_url)
    system = (
        "\u4f60\u662f\u6f14\u51fa\u6d77\u62a5\u548c\u8868\u683c\u7684 OCR \u7ed3\u6784\u5316\u52a9\u624b\u3002"
        "\u53ea\u63d0\u53d6\u56fe\u7247\u4e2d\u7684\u6f14\u51fa\u65e5\u671f\u3001\u6b4c\u624b/\u9635\u5bb9\u3001\u573a\u5730\u3002"
        "\u4e0d\u8981\u6839\u636e\u5e38\u8bc6\u8865\u5168\uff0c\u770b\u4e0d\u6e05\u7684\u884c\u53ef\u4ee5\u8df3\u8fc7\u3002"
    )
    prompt = (
        "\u8bf7\u628a\u8fd9\u4e9b\u56fe\u91cc\u6240\u6709\u6f14\u51fa\u884c\u8f6c\u6210\u4e25\u683c JSON\uff0c"
        "\u683c\u5f0f\u53ea\u80fd\u662f\uff1a"
        '{"events":[{"date_text":"7.11","performer":"\u6b4c\u624b\u540d","venue":"\u573a\u5730"}]}\n'
        "\u8981\u6c42\uff1a\n"
        "1. \u540c\u4e00\u9875\u53ef\u80fd\u6709\u591a\u7ec4\u300c\u65e5\u671f/\u5468/\u6b4c\u624b\u300d\u6392\u7248\uff0c\u8981\u5206\u522b\u914d\u5bf9\u3002\n"
        "2. \u8be6\u60c5\u9875\u53ef\u80fd\u662f\u300c\u65e5\u671f/\u9635\u5bb9/\u573a\u5730\u300d\u8868\u683c\uff0c\u573a\u5730\u8981\u5199\u5165 venue\u3002\n"
        "3. \u6ca1\u6709\u573a\u5730\u5c31\u8fd4\u56de\u7a7a\u5b57\u7b26\u4e32\u3002\n"
        "4. \u65e5\u671f\u5c3d\u91cf\u7528 7.11 \u6216 7.20-21 \u683c\u5f0f\u3002\n"
        "5. \u4e0d\u8981\u8fd4\u56de JSON \u4e4b\u5916\u7684\u6587\u5b57\u3002"
    )
    content: list[dict[str, Any]] = [
        {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}}
        for image_url in image_data_urls
    ]
    content.append({"type": "text", "text": prompt})
    return {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
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


def _normalize_ai_date(value: str) -> str:
    text = clean_ocr_text(value).replace("\uff0f", "/")
    text = re.sub(r"\s+", "", text)
    range_match = re.match(
        r"^(\d{1,2})[/\.](\d{1,2})(?:[-~\u2013\u2014\u81f3](?:(\d{1,2})[/\.])?(\d{1,2}))?$",
        text,
    )
    if range_match:
        start_month = int(range_match.group(1))
        start_day = int(range_match.group(2))
        if not range_match.group(4):
            return f"{start_month}.{start_day}"
        end_month = int(range_match.group(3)) if range_match.group(3) else start_month
        end_day = int(range_match.group(4))
        if end_month == start_month:
            return f"{start_month}.{start_day}-{end_day}"
        return f"{start_month}.{start_day}-{end_month}.{end_day}"
    return merge_date_range(text, None)


def parse_ai_ocr_events(raw_text: str, image_name: str = "") -> list[EventRow]:
    data = json.loads(_strip_json_fence(raw_text))
    raw_events = data.get("events", [])
    if not isinstance(raw_events, list):
        return []

    events: list[EventRow] = []
    for item in raw_events:
        if not isinstance(item, dict):
            continue
        date_text = _normalize_ai_date(str(item.get("date_text", "")))
        performer = clean_ocr_text(str(item.get("performer", "")))
        venue = clean_ocr_text(str(item.get("venue", "")))
        if not performer or not parse_date_text(date_text):
            continue
        events.append(EventRow(date_text=date_text, performer=performer, venue=venue, image_name=image_name))
    return events


def _event_key(event: EventRow) -> tuple[str, str]:
    performer_key = re.sub(r"[^\w\u4e00-\u9fff]", "", clean_ocr_text(event.performer)).lower()
    return event.date_text, performer_key


def _merge_event_lists(event_lists: list[list[EventRow]]) -> list[EventRow]:
    merged: list[EventRow] = []
    indexes: dict[tuple[str, str], int] = {}
    for events in event_lists:
        for event in events:
            key = _event_key(event)
            if not key[1]:
                continue
            if key not in indexes:
                indexes[key] = len(merged)
                merged.append(event)
                continue
            current_index = indexes[key]
            current = merged[current_index]
            if event.venue and not current.venue:
                merged[current_index] = event
    return merged


def _is_structurally_complete(events: list[EventRow], min_events: int) -> bool:
    if len(events) < min_events:
        return False
    return all(event.performer.strip() and parse_date_text(event.date_text) for event in events)


def _agreement_ratio(left: list[EventRow], right: list[EventRow]) -> float:
    left_keys = {_event_key(event) for event in left if _event_key(event)[1]}
    right_keys = {_event_key(event) for event in right if _event_key(event)[1]}
    if not left_keys or not right_keys:
        return 0.0
    return len(left_keys & right_keys) / min(len(left_keys), len(right_keys))


def select_ai_ocr_events(
    results: list[AiOcrProviderResult],
    min_agreement_ratio: float = 0.2,
    min_events: int = 1,
) -> list[EventRow]:
    complete_results = [
        result
        for result in results
        if not result.error and _is_structurally_complete(result.events, min_events=min_events)
    ]
    if not complete_results:
        return []
    if len(complete_results) == 1:
        return _merge_event_lists([complete_results[0].events])

    best_agreement = 0.0
    for left_index, left in enumerate(complete_results):
        for right in complete_results[left_index + 1 :]:
            best_agreement = max(best_agreement, _agreement_ratio(left.events, right.events))
    if best_agreement < min_agreement_ratio:
        return []

    return _merge_event_lists([result.events for result in complete_results])


def _image_path_to_data_url(image_path: Path, max_width: int) -> str:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        if image.width > max_width:
            image.thumbnail((max_width, max_width * 4), Image.Resampling.LANCZOS)
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=88)
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


class AiOcrClient:
    def __init__(
        self,
        provider: AiOcrProviderConfig | AiOcrConfig | None = None,
        timeout_seconds: int | None = None,
        max_width: int | None = None,
        image_batch_size: int | None = None,
        image_workers: int | None = None,
    ):
        if isinstance(provider, AiOcrConfig):
            config = provider
            provider = config.providers[0] if config.providers else None
            timeout_seconds = config.timeout_seconds
            max_width = config.max_width
            image_batch_size = config.image_batch_size
            image_workers = config.image_workers
        self.provider = provider
        self.timeout_seconds = timeout_seconds or 45
        self.max_width = max_width or 1600
        self.image_batch_size = max(1, image_batch_size or 8)
        self.image_workers = max(1, image_workers or 2)

    def extract_events(self, image_paths: list[Path]) -> list[EventRow]:
        if self.provider is None:
            return []

        batches = list(_chunked(image_paths, self.image_batch_size))
        if not batches:
            return []

        events: list[EventRow] = []
        worker_count = min(self.image_workers, len(batches))
        if worker_count <= 1:
            for batch in batches:
                events.extend(self._extract_batch_events(batch))
            return events

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(self._extract_batch_events, batch) for batch in batches]
            for future in futures:
                events.extend(future.result())
        return events

    def _extract_batch_events(self, image_paths: list[Path]) -> list[EventRow]:
        image_data_urls = [_image_path_to_data_url(image_path, self.max_width) for image_path in image_paths]
        payload = build_ai_ocr_payload(self.provider.model, image_data_urls)
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = request.Request(
            f"{self.provider.base_url}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.provider.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with request.urlopen(req, timeout=self.timeout_seconds) as response:
            result = json.loads(response.read().decode("utf-8"))
        content = result["choices"][0]["message"]["content"]
        batch_name = "+".join(image_path.name for image_path in image_paths)
        return parse_ai_ocr_events(content, image_name=batch_name)


def _chunked(items: list[Path], size: int) -> list[list[Path]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def extract_events_with_ai_ocr(image_paths: list[Path], warnings: list[str]) -> list[EventRow]:
    config = AiOcrConfig.from_env()
    if not config.enabled:
        return []

    results: list[AiOcrProviderResult] = []
    with ThreadPoolExecutor(max_workers=max(1, len(config.providers))) as executor:
        futures = {
            executor.submit(
                AiOcrClient(
                    provider,
                    timeout_seconds=config.timeout_seconds,
                    max_width=config.max_width,
                    image_batch_size=config.image_batch_size,
                    image_workers=config.image_workers,
                ).extract_events,
                image_paths,
            ): provider
            for provider in config.providers
        }
        for future in as_completed(futures):
            provider = futures[future]
            try:
                events = future.result()
            except Exception as exc:
                error = describe_ai_exception(exc)
                results.append(AiOcrProviderResult(provider_name=provider.name, events=[], error=error))
                warnings.append(f"AI \u8bc6\u522b\u5931\u8d25\uff08{provider.name}\uff09\uff1a{error}")
                continue
            results.append(AiOcrProviderResult(provider_name=provider.name, events=events))

    events = select_ai_ocr_events(
        results,
        min_agreement_ratio=config.min_agreement_ratio,
        min_events=config.min_events,
    )
    if not events:
        warnings.append("AI \u8bc6\u522b\u7ed3\u679c\u7ed3\u6784\u4e0d\u5b8c\u6574\u6216\u4e24\u5bb6\u5dee\u5f02\u8fc7\u5927\uff0c\u5df2\u56de\u9000\u672c\u5730 OCR\u3002")
        return []

    provider_count = len([result for result in results if not result.error])
    if provider_count >= 2:
        warnings.append(f"AI \u8bc6\u522b\u5df2\u5e76\u884c\u8c03\u7528 {provider_count} \u5bb6\u6a21\u578b\u5e76\u5408\u5e76\u7ed3\u679c\u3002")
    else:
        warnings.append("AI \u8bc6\u522b\u5df2\u7528\u4e8e\u56fe\u7247\u8bfb\u53d6\u3002")
    return events
