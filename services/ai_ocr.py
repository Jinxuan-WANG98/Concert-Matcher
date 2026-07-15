from __future__ import annotations

import base64
import io
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import request

from PIL import Image

from services.ai_errors import describe_ai_exception
from services.date_parser import merge_date_range, parse_date_text
from services.debug_timing import PhaseTimer, debug_log
from services.models import EventRow
from services.ocr import clean_ocr_text


class AiOcrIncompleteError(RuntimeError):
    """Raised when at least one required image batch has no AI result."""


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
    local_fallback_enabled: bool = False
    image_detail: str = "auto"
    provider_fallback: bool = False
    dual_provider: bool = False
    jpeg_quality: int = 85
    low_result_threshold: int = 3
    transient_retry_attempts: int = 1

    @classmethod
    def from_env(cls) -> "AiOcrConfig":
        enabled = os.environ.get("AI_OCR_ENABLED", "").lower() in {"1", "true", "yes", "on"}
        providers = _active_ocr_providers_from_env()
        dual_default = "true" if len(providers) > 1 else "false"
        dual_provider = os.environ.get("AI_OCR_DUAL_PROVIDER", dual_default).lower() in {"1", "true", "yes", "on"}
        provider_fallback = os.environ.get("AI_OCR_PROVIDER_FALLBACK", "false").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if not dual_provider and len(providers) > 1:
            primary_index = int(os.environ.get("AI_OCR_PRIMARY_PROVIDER_INDEX", "1")) - 1
            primary_index = max(0, min(primary_index, len(providers) - 1))
            providers = [providers[primary_index]]
            provider_fallback = False
        return cls(
            enabled=enabled and bool(providers),
            providers=tuple(providers),
            timeout_seconds=int(os.environ.get("AI_OCR_TIMEOUT_SECONDS", "45")),
            max_width=int(os.environ.get("AI_OCR_MAX_WIDTH", "1600")),
            image_batch_size=max(1, int(os.environ.get("AI_OCR_IMAGE_BATCH_SIZE", "8"))),
            image_workers=max(1, int(os.environ.get("AI_OCR_IMAGE_WORKERS", "2"))),
            local_fallback_enabled=os.environ.get("AI_OCR_LOCAL_FALLBACK", "").lower() in {"1", "true", "yes", "on"},
            image_detail=_ocr_image_detail(),
            provider_fallback=provider_fallback,
            dual_provider=dual_provider,
            jpeg_quality=max(60, min(95, int(os.environ.get("AI_OCR_JPEG_QUALITY", "88")))),
            low_result_threshold=max(1, int(os.environ.get("AI_OCR_LOW_RESULT_THRESHOLD", "3"))),
            transient_retry_attempts=max(0, min(1, int(os.environ.get("AI_OCR_TRANSIENT_RETRY_ATTEMPTS", "1")))),
        )

    @property
    def cache_source(self) -> str:
        provider_source = "|".join(f"{provider.name}:{provider.model}:{provider.base_url}" for provider in self.providers)
        mode = "dual" if self.dual_provider else "single"
        return (
            f"ai-ocr:v6-{mode}:{provider_source}:w{self.max_width}:"
            f"batch{self.image_batch_size}:low{self.low_result_threshold}"
        )

    @property
    def api_key(self) -> str:
        return self.providers[0].api_key if self.providers else ""

    @property
    def base_url(self) -> str:
        return self.providers[0].base_url if self.providers else "https://api.openai.com/v1"

    @property
    def model(self) -> str:
        return self.providers[0].model if self.providers else ""


def _provider_api_key(prefix: str, base_url: str) -> str:
    api_key = os.environ.get(f"{prefix}API_KEY", "").strip()
    if api_key:
        return api_key
    match_key = os.environ.get("AI_MATCH_API_KEY", "").strip()
    match_base = os.environ.get("AI_MATCH_BASE_URL", "").strip().rstrip("/")
    if match_key and match_base and base_url and match_base == base_url:
        return match_key
    return ""


def _active_ocr_providers_from_env() -> list[AiOcrProviderConfig]:
    return _providers_from_env()


def _providers_from_env() -> list[AiOcrProviderConfig]:
    providers: list[AiOcrProviderConfig] = []
    for index in range(1, 5):
        prefix = f"AI_OCR_PROVIDER_{index}_"
        provider_enabled = os.environ.get(f"{prefix}ENABLED", "true").strip().lower()
        if provider_enabled in {"0", "false", "no", "off"}:
            continue
        base_url = os.environ.get(f"{prefix}BASE_URL", "").strip().rstrip("/")
        api_key = _provider_api_key(prefix, base_url)
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


def _ocr_image_detail() -> str:
    detail = os.environ.get("AI_OCR_IMAGE_DETAIL", "auto").strip().lower()
    return detail if detail in {"high", "low", "auto"} else "auto"


def build_ai_ocr_payload(
    model: str,
    image_data_url: str | list[str],
    image_detail: str | None = None,
) -> dict[str, Any]:
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
        "5. \u5fc5\u987b\u7a77\u5c3d\u63d0\u53d6\u6bcf\u4e00\u884c\u6f14\u51fa\uff0c\u5305\u62ec\u591a\u680f\u6458\u8981\u9875\u91cc\u6240\u6709\u65e5\u671f\u680f\u4e0b\u7684\u6bcf\u4f4d\u6b4c\u624b\uff0c\u4e0d\u8981\u63d0\u524d\u505c\u6b62\u3002\n"
        "6. \u4e0d\u8981\u8fd4\u56de JSON \u4e4b\u5916\u7684\u6587\u5b57\u3002"
    )
    detail = image_detail or _ocr_image_detail()
    content: list[dict[str, Any]] = [
        {"type": "image_url", "image_url": {"url": image_url, "detail": detail}}
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


def build_ai_ocr_repair_payload(model: str, raw_text: str) -> dict[str, Any]:
    system = (
        "你是 JSON 修复器。只修复格式，不新增、推测或改写图片中没有出现的事实。"
        "把输入整理成程序可解析的严格 JSON。"
    )
    prompt = (
        "请把下面的 AI OCR 原始输出修复为严格 JSON，格式只能是："
        '{"events":[{"date_text":"7.11","performer":"歌手名","venue":"场地"}]}\n'
        "规则：\n"
        "1. 只能使用原始输出里已经出现的信息。\n"
        "2. 缺少场地时 venue 用空字符串。\n"
        "3. 无法确认日期或歌手的行不要输出。\n"
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
    if isinstance(data, list):
        raw_events = data
    elif isinstance(data, dict):
        raw_events = data.get("events") or data.get("rows") or data.get("items") or []
    else:
        raw_events = []
    if not isinstance(raw_events, list):
        return []

    events: list[EventRow] = []
    for item in raw_events:
        if not isinstance(item, dict):
            continue
        date_text = _normalize_ai_date(
            _text_field(item, "date_text", "date", "dateText", "演出日期", "日期")
        )
        performer = clean_ocr_text(
            _text_field(item, "performer", "artist", "artist_name", "lineup", "name", "歌手", "阵容")
        )
        venue = clean_ocr_text(_text_field(item, "venue", "location", "place", "场地", "演出场所"))
        if not performer or not parse_date_text(date_text):
            continue
        events.append(EventRow(date_text=date_text, performer=performer, venue=venue, image_name=image_name))
    return events


def _text_field(item: dict, *names: str) -> str:
    for name in names:
        value = item.get(name)
        if value is None:
            continue
        return str(value).strip()
    return ""


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


def select_ai_ocr_events(results: list[AiOcrProviderResult]) -> list[EventRow]:
    usable_results = [result for result in results if not result.error and result.events]
    if not usable_results:
        return []
    return _merge_event_lists([result.events for result in usable_results])


def _image_path_to_data_url(image_path: Path, max_width: int, jpeg_quality: int = 85) -> str:
    with Image.open(image_path) as image:
        if image.width > max_width:
            image.thumbnail((max_width, max_width * 4), Image.Resampling.LANCZOS)
        image = image.convert("RGB")
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=jpeg_quality)
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
        image_detail: str | None = None,
        jpeg_quality: int | None = None,
    ):
        if isinstance(provider, AiOcrConfig):
            config = provider
            provider = config.providers[0] if config.providers else None
            timeout_seconds = config.timeout_seconds
            max_width = config.max_width
            image_batch_size = config.image_batch_size
            image_workers = config.image_workers
            image_detail = config.image_detail
            jpeg_quality = config.jpeg_quality
        self.provider = provider
        self.timeout_seconds = timeout_seconds or 45
        self.max_width = max_width or 1600
        self.image_batch_size = max(1, image_batch_size or 8)
        self.image_workers = max(1, image_workers or 2)
        self.image_detail = image_detail or _ocr_image_detail()
        self.jpeg_quality = max(60, min(95, jpeg_quality or 88))

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
                events.extend(self._extract_batch_events_resilient(batch))
            return events

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(self._extract_batch_events_resilient, batch) for batch in batches]
            for future in futures:
                events.extend(future.result())
        return events

    def _chat_content(self, payload: dict[str, Any], call_kind: str = "ocr") -> str:
        started = time.perf_counter()
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
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        debug_log(
            "ai_ocr.py:_chat_content",
            "ai ocr api call complete",
            {
                "callKind": call_kind,
                "provider": self.provider.name if self.provider else "",
                "model": self.provider.model if self.provider else "",
                "elapsedMs": elapsed_ms,
            },
            hypothesis_id="H1",
        )
        return result["choices"][0]["message"]["content"]

    def _extract_batch_events(self, image_paths: list[Path]) -> list[EventRow]:
        image_data_urls = [
            _image_path_to_data_url(image_path, self.max_width, self.jpeg_quality) for image_path in image_paths
        ]
        payload = build_ai_ocr_payload(
            self.provider.model,
            image_data_urls,
            image_detail=self.image_detail,
        )
        content = self._chat_content(payload, call_kind="ocr")
        batch_name = "+".join(image_path.name for image_path in image_paths)
        events = self._parse_events_with_repair(content, image_name=batch_name)
        # #region agent log
        debug_log(
            "ai_ocr.py:_extract_batch_events",
            "ai ocr batch parsed",
            {
                "imageCount": len(image_paths),
                "imageNames": [image_path.name for image_path in image_paths],
                "eventCount": len(events),
                "rawLength": len(content or ""),
                "imageDetail": self.image_detail,
            },
            hypothesis_id="H1",
        )
        # #endregion
        return events

    def _should_attempt_repair(self, raw_text: str) -> bool:
        text = str(raw_text or "").strip()
        if len(text) < 8:
            return False
        return "{" in text or "events" in text.lower()

    def _parse_events_with_repair(self, raw_text: str, image_name: str) -> list[EventRow]:
        try:
            events = parse_ai_ocr_events(raw_text, image_name=image_name)
        except (json.JSONDecodeError, TypeError, ValueError):
            events = []
        if events:
            return events
        if not self._should_attempt_repair(raw_text):
            debug_log(
                "ai_ocr.py:_parse_events_with_repair",
                "skipped ai ocr repair call",
                {"imageName": image_name, "rawLength": len(str(raw_text or ""))},
                hypothesis_id="H3",
            )
            return []

        repair_payload = build_ai_ocr_repair_payload(self.provider.model, raw_text)
        repaired_content = self._chat_content(repair_payload, call_kind="repair")
        return parse_ai_ocr_events(repaired_content, image_name=image_name)

    def _extract_batch_events_resilient(self, image_paths: list[Path]) -> list[EventRow]:
        try:
            events = self._extract_batch_events(image_paths)
            if events or len(image_paths) <= 1:
                return events
            raise ValueError("AI OCR returned no usable rows")
        except Exception:
            if len(image_paths) <= 1:
                raise
            midpoint = len(image_paths) // 2
            debug_log(
                "ai_ocr.py:_extract_batch_events_resilient",
                "splitting ocr batch after empty result",
                {"imageCount": len(image_paths), "leftCount": midpoint, "rightCount": len(image_paths) - midpoint},
                hypothesis_id="H3",
            )
            return self._extract_batch_events_resilient(image_paths[:midpoint]) + self._extract_batch_events_resilient(
                image_paths[midpoint:]
            )


def _chunked(items: list[Path], size: int) -> list[list[Path]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def _provider_order(providers: tuple[AiOcrProviderConfig, ...], start_index: int) -> list[AiOcrProviderConfig]:
    if not providers:
        return []
    ordered = list(providers)
    offset = start_index % len(ordered)
    return ordered[offset:] + ordered[:offset]


def _event_keys(events: list[EventRow]) -> set[tuple[str, str]]:
    return {_event_key(event) for event in events if _event_key(event)[1]}


def _low_result_attempts_agree(results: list[AiOcrProviderResult]) -> bool:
    if len(results) < 2:
        return False
    reference = _event_keys(results[0].events)
    return bool(reference) and all(_event_keys(result.events) == reference for result in results[1:])


def _is_transient_ocr_error(error: str) -> bool:
    normalized = str(error or "").lower()
    return any(
        marker in normalized
        for marker in (
            "timed out",
            "timeout",
            "connection reset",
            "connection aborted",
            "temporarily unavailable",
            "service unavailable",
            "too many requests",
            "rate limit",
            "http error 5",
        )
    ) or bool(re.search(r"\b5\d{2}\b", normalized))


def _extract_batch_with_provider_fallback(
    batch: list[Path],
    providers: tuple[AiOcrProviderConfig, ...],
    start_index: int,
    config: AiOcrConfig,
) -> AiOcrProviderResult:
    errors: list[str] = []
    low_results: list[AiOcrProviderResult] = []
    provider_list = _provider_order(providers, start_index)
    if not config.provider_fallback:
        provider_list = provider_list[:1]
    for provider in provider_list:
        client = AiOcrClient(
            provider,
            timeout_seconds=config.timeout_seconds,
            max_width=config.max_width,
            image_batch_size=config.image_batch_size,
            image_workers=1,
            image_detail=config.image_detail,
            jpeg_quality=config.jpeg_quality,
        )
        try:
            events = client._extract_batch_events_resilient(batch)
        except Exception as exc:
            errors.append(f"{provider.name}: {describe_ai_exception(exc)}")
            continue

        if not events:
            errors.append(f"{provider.name}: returned no usable rows")
            continue

        result = AiOcrProviderResult(provider_name=provider.name, events=events)
        if len(events) >= config.low_result_threshold:
            if low_results:
                combined = _merge_event_lists([item.events for item in [*low_results, result]])
                return AiOcrProviderResult(
                    provider_name="+".join(item.provider_name for item in [*low_results, result]),
                    events=combined,
                )
            return result

        if not config.provider_fallback:
            return result

        low_results.append(result)
        debug_log(
            "ai_ocr.py:_extract_batch_with_provider_fallback",
            "rechecking low-row ocr batch with next provider",
            {
                "provider": provider.name,
                "eventCount": len(events),
                "threshold": config.low_result_threshold,
                "imageNames": [image_path.name for image_path in batch],
            },
            hypothesis_id="H3",
        )

        if _low_result_attempts_agree(low_results):
            return AiOcrProviderResult(
                provider_name="+".join(item.provider_name for item in low_results),
                events=_merge_event_lists([item.events for item in low_results]),
            )

    if low_results:
        errors.append(
            f"low-row result below {config.low_result_threshold} rows could not be independently confirmed"
        )
    return AiOcrProviderResult(
        provider_name=",".join(provider.name for provider in providers),
        events=[],
        error="; ".join(errors),
    )


def extract_events_with_ai_ocr(image_paths: list[Path], warnings: list[str]) -> list[EventRow]:
    config = AiOcrConfig.from_env()
    if not config.enabled or not image_paths:
        return []

    with PhaseTimer("ai_ocr.py:extract_events_with_ai_ocr", "ai_ocr_total") as timer:
        timer.data = {
            "imageCount": len(image_paths),
            "batchSize": config.image_batch_size,
            "imageWorkers": config.image_workers,
            "imageDetail": config.image_detail,
            "providerFallback": config.provider_fallback,
            "dualProvider": config.dual_provider,
            "maxWidth": config.max_width,
            "jpegQuality": config.jpeg_quality,
        }
        debug_log(
            "ai_ocr.py:extract_events_with_ai_ocr",
            "ai ocr providers loaded",
            {
                "providerNames": [provider.name for provider in config.providers],
                "providerModels": [provider.model for provider in config.providers],
                "providerCount": len(config.providers),
            },
            hypothesis_id="H4",
        )

        results_by_batch: dict[int, AiOcrProviderResult] = {}
        failed_results_by_batch: dict[int, AiOcrProviderResult] = {}
        batches = list(_chunked(image_paths, config.image_batch_size))
        worker_count = min(max(1, config.image_workers), len(batches))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(
                    _extract_batch_with_provider_fallback,
                    batch,
                    config.providers,
                    batch_index,
                    config,
                ): batch_index
                for batch_index, batch in enumerate(batches)
            }
            for future in as_completed(futures):
                batch_index = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    error = describe_ai_exception(exc)
                    result = AiOcrProviderResult(provider_name="batch", events=[], error=error)
                if result.error:
                    failed_results_by_batch[batch_index] = result
                    debug_log(
                        "ai_ocr.py:extract_events_with_ai_ocr",
                        "ai ocr batch requires recovery",
                        {"batchIndex": batch_index, "provider": result.provider_name, "error": result.error},
                        hypothesis_id="H3",
                    )
                    continue
                results_by_batch[batch_index] = result
                debug_log(
                    "ai_ocr.py:extract_events_with_ai_ocr",
                    "ai ocr batch succeeded",
                    {
                        "batchIndex": batch_index,
                        "provider": result.provider_name,
                        "eventCount": len(result.events),
                    },
                    hypothesis_id="H1",
                    )

        retry_count = 0
        for batch_index in sorted(failed_results_by_batch):
            result = failed_results_by_batch[batch_index]
            if retry_count >= config.transient_retry_attempts or not _is_transient_ocr_error(result.error):
                continue
            retry_count += 1
            debug_log(
                "ai_ocr.py:extract_events_with_ai_ocr",
                "retrying transient ocr batch sequentially",
                {
                    "batchIndex": batch_index,
                    "attempt": retry_count,
                    "provider": result.provider_name,
                    "error": result.error,
                },
                hypothesis_id="H3",
            )
            retry_result = _extract_batch_with_provider_fallback(
                batches[batch_index],
                config.providers,
                batch_index + retry_count,
                config,
            )
            if retry_result.error:
                failed_results_by_batch[batch_index] = retry_result
                continue
            results_by_batch[batch_index] = retry_result
            del failed_results_by_batch[batch_index]
            debug_log(
                "ai_ocr.py:extract_events_with_ai_ocr",
                "transient ocr batch recovered",
                {
                    "batchIndex": batch_index,
                    "attempt": retry_count,
                    "provider": retry_result.provider_name,
                    "eventCount": len(retry_result.events),
                },
                hypothesis_id="H3",
            )

        failed_batches = sorted(failed_results_by_batch)
        for batch_index in failed_batches:
            result = failed_results_by_batch[batch_index]
            warnings.append(f"AI \u8bc6\u522b\u5931\u8d25\uff08{result.provider_name}\uff09\uff1a{result.error}")
            debug_log(
                "ai_ocr.py:extract_events_with_ai_ocr",
                "ai ocr batch failed",
                {"batchIndex": batch_index, "provider": result.provider_name, "error": result.error},
                hypothesis_id="H3",
            )

        results = [results_by_batch[index] for index in sorted(results_by_batch)]
        timer.data["successfulBatches"] = len(results)
        timer.data["failedBatches"] = len(failed_batches)
        if failed_batches:
            raise AiOcrIncompleteError(
                f"AI 图片识别未完整完成：{len(failed_batches)}/{len(batches)} 个图片批次失败，"
                "为避免返回部分结果，本次任务已停止，请重试。"
            )
        events = select_ai_ocr_events(results)
        timer.data["mergedEventCount"] = len(events)
        if not events:
            warnings.append("AI \u8bc6\u522b\u672a\u8fd4\u56de\u53ef\u7528\u884c\uff0c\u672a\u81ea\u52a8\u8c03\u7528\u672c\u5730 OCR\u3002")
            return []

        provider_names = sorted({result.provider_name for result in results if result.events})
        if len(provider_names) >= 2:
            warnings.append(f"AI \u8bc6\u522b\u5df2\u5206\u6279\u5e76\u884c\u8c03\u7528 {len(provider_names)} \u5bb6\u6a21\u578b\u5e76\u5408\u5e76\u7ed3\u679c\u3002")
        else:
            warnings.append("AI \u8bc6\u522b\u5df2\u7528\u4e8e\u56fe\u7247\u8bfb\u53d6\u3002")
        debug_log(
            "ai_ocr.py:extract_events_with_ai_ocr",
            "ai ocr extraction complete",
            {
                "imageCount": len(image_paths),
                "batchCount": len(batches),
                "successfulBatches": len(results),
                "mergedEventCount": len(events),
                "batchSize": config.image_batch_size,
                "model": config.model,
            },
            hypothesis_id="H2",
        )
        return events
