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
    max_tokens: int = 8192
    local_fallback_enabled: bool = False
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
            max_tokens=max(512, int(os.environ.get("AI_OCR_MAX_TOKENS", "8192"))),
            local_fallback_enabled=os.environ.get("AI_OCR_LOCAL_FALLBACK", "").lower() in {"1", "true", "yes", "on"},
            min_agreement_ratio=float(os.environ.get("AI_OCR_MIN_AGREEMENT_RATIO", "0.2")),
            min_events=int(os.environ.get("AI_OCR_MIN_EVENTS", "1")),
        )

    @property
    def cache_source(self) -> str:
        provider_source = "|".join(f"{provider.name}:{provider.model}:{provider.base_url}" for provider in self.providers)
        return (
            f"ai-ocr:v3-distributed:{provider_source}:w{self.max_width}:"
            f"batch{self.image_batch_size}:tok{self.max_tokens}:min{self.min_events}"
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


def _debug_log(location: str, message: str, data: dict[str, Any], hypothesis_id: str = "") -> None:
    # #region agent log
    try:
        payload = {
            "sessionId": "c69de8",
            "timestamp": int(time.time() * 1000),
            "location": location,
            "message": message,
            "data": data,
            "hypothesisId": hypothesis_id,
        }
        with Path("debug-c69de8.log").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except OSError:
        pass
    # #endregion


def build_ai_ocr_payload(
    model: str,
    image_data_url: str | list[str],
    max_tokens: int = 8192,
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
    content: list[dict[str, Any]] = [
        {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}}
        for image_url in image_data_urls
    ]
    content.append({"type": "text", "text": prompt})
    return {
        "model": model,
        "temperature": 0,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ],
    }


def build_ai_ocr_repair_payload(model: str, raw_text: str, max_tokens: int = 8192) -> dict[str, Any]:
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
        "max_tokens": max_tokens,
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
    return _merge_event_lists([result.events for result in complete_results])


def _image_path_to_data_url(image_path: Path, max_width: int) -> str:
    with Image.open(image_path) as image:
        if image.width > max_width:
            image.thumbnail((max_width, max_width * 4), Image.Resampling.LANCZOS)
        image = image.convert("RGB")
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
        max_tokens: int | None = None,
    ):
        if isinstance(provider, AiOcrConfig):
            config = provider
            provider = config.providers[0] if config.providers else None
            timeout_seconds = config.timeout_seconds
            max_width = config.max_width
            image_batch_size = config.image_batch_size
            image_workers = config.image_workers
            max_tokens = config.max_tokens
        self.provider = provider
        self.timeout_seconds = timeout_seconds or 45
        self.max_width = max_width or 1600
        self.image_batch_size = max(1, image_batch_size or 8)
        self.image_workers = max(1, image_workers or 2)
        self.max_tokens = max(512, max_tokens or 8192)

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

    def _chat_content(self, payload: dict[str, Any]) -> str:
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
        return result["choices"][0]["message"]["content"]

    def _extract_batch_events(self, image_paths: list[Path]) -> list[EventRow]:
        image_data_urls = [_image_path_to_data_url(image_path, self.max_width) for image_path in image_paths]
        payload = build_ai_ocr_payload(self.provider.model, image_data_urls, max_tokens=self.max_tokens)
        content = self._chat_content(payload)
        batch_name = "+".join(image_path.name for image_path in image_paths)
        events = self._parse_events_with_repair(content, image_name=batch_name)
        # #region agent log
        _debug_log(
            "ai_ocr.py:_extract_batch_events",
            "ai ocr batch parsed",
            {
                "imageCount": len(image_paths),
                "imageNames": [image_path.name for image_path in image_paths],
                "eventCount": len(events),
                "rawLength": len(content or ""),
                "maxTokens": self.max_tokens,
            },
            hypothesis_id="H1",
        )
        # #endregion
        return events

    def _parse_events_with_repair(self, raw_text: str, image_name: str) -> list[EventRow]:
        try:
            events = parse_ai_ocr_events(raw_text, image_name=image_name)
        except (json.JSONDecodeError, TypeError, ValueError):
            events = []
        if events:
            return events

        repair_payload = build_ai_ocr_repair_payload(self.provider.model, raw_text, max_tokens=self.max_tokens)
        repaired_content = self._chat_content(repair_payload)
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


def _extract_batch_with_provider_fallback(
    batch: list[Path],
    providers: tuple[AiOcrProviderConfig, ...],
    start_index: int,
    config: AiOcrConfig,
) -> AiOcrProviderResult:
    errors: list[str] = []
    for provider in _provider_order(providers, start_index):
        client = AiOcrClient(
            provider,
            timeout_seconds=config.timeout_seconds,
            max_width=config.max_width,
            image_batch_size=config.image_batch_size,
            image_workers=1,
            max_tokens=config.max_tokens,
        )
        try:
            events = client._extract_batch_events_resilient(batch)
        except Exception as exc:
            errors.append(f"{provider.name}: {describe_ai_exception(exc)}")
            continue
        return AiOcrProviderResult(provider_name=provider.name, events=events)
    return AiOcrProviderResult(
        provider_name=",".join(provider.name for provider in providers),
        events=[],
        error="; ".join(errors),
    )


def extract_events_with_ai_ocr(image_paths: list[Path], warnings: list[str]) -> list[EventRow]:
    config = AiOcrConfig.from_env()
    if not config.enabled or not image_paths:
        return []

    results_by_batch: dict[int, AiOcrProviderResult] = {}
    batches = list(_chunked(image_paths, config.image_batch_size))
    worker_count = min(max(1, config.image_workers * len(config.providers)), len(batches))
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
                warnings.append(f"AI \u8bc6\u522b\u5931\u8d25\uff08{result.provider_name}\uff09\uff1a{result.error}")
                # #region agent log
                _debug_log(
                    "ai_ocr.py:extract_events_with_ai_ocr",
                    "ai ocr batch failed",
                    {"batchIndex": batch_index, "provider": result.provider_name, "error": result.error},
                    hypothesis_id="H3",
                )
                # #endregion
                continue
            results_by_batch[batch_index] = result
            # #region agent log
            _debug_log(
                "ai_ocr.py:extract_events_with_ai_ocr",
                "ai ocr batch succeeded",
                {
                    "batchIndex": batch_index,
                    "provider": result.provider_name,
                    "eventCount": len(result.events),
                },
                hypothesis_id="H1",
            )
            # #endregion

    results = [results_by_batch[index] for index in sorted(results_by_batch)]

    events = select_ai_ocr_events(
        results,
        min_agreement_ratio=config.min_agreement_ratio,
        min_events=config.min_events,
    )
    if not events:
        warnings.append("AI \u8bc6\u522b\u672a\u8fd4\u56de\u53ef\u7528\u884c\uff0c\u672a\u81ea\u52a8\u8c03\u7528\u672c\u5730 OCR\u3002")
        return []

    provider_names = sorted({result.provider_name for result in results if result.events})
    if len(provider_names) >= 2:
        warnings.append(f"AI \u8bc6\u522b\u5df2\u5206\u6279\u5e76\u884c\u8c03\u7528 {len(provider_names)} \u5bb6\u6a21\u578b\u5e76\u5408\u5e76\u7ed3\u679c\u3002")
    else:
        warnings.append("AI \u8bc6\u522b\u5df2\u7528\u4e8e\u56fe\u7247\u8bfb\u53d6\u3002")
    # #region agent log
    _debug_log(
        "ai_ocr.py:extract_events_with_ai_ocr",
        "ai ocr extraction complete",
        {
            "imageCount": len(image_paths),
            "batchCount": len(batches),
            "successfulBatches": len(results),
            "mergedEventCount": len(events),
            "batchSize": config.image_batch_size,
            "maxTokens": config.max_tokens,
            "model": config.model,
        },
        hypothesis_id="H2",
    )
    # #endregion
    return events
