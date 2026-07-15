# AI OCR Transient Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recover one transiently failed AI OCR image batch without allowing partial matching results.

**Architecture:** Retain the parallel OCR pass. After it completes, retry only transiently failed batches one at a time, rotating the first provider; unresolved batches retain the existing all-or-nothing error. The web client treats a missing remembered job as an interrupted task rather than a result.

**Tech Stack:** Python 3.12, Flask, `unittest`, browser JavaScript, Render Blueprint YAML.

## Global Constraints

- Keep OCR at one image per batch and three parallel initial image workers.
- Keep the 90-second individual provider timeout.
- Permit exactly one retry only for a combined transient provider failure; do not retry malformed, forbidden, or empty-result failures.
- Never render or return matches unless every required OCR batch and later AI analysis has completed.
- Do not add dependencies or increase normal-path memory use.

---

### Task 1: Add bounded OCR recovery tests and configuration

**Files:**

- Modify: `tests/test_ai_ocr.py`
- Modify: `tests/test_render_config.py`
- Modify: `.env.example`
- Modify: `render.yaml`

**Interfaces:**

- Consumes: `extract_events_with_ai_ocr(image_paths, warnings)` and `_extract_batch_with_provider_fallback(batch, providers, start_index, config)`.
- Produces: failing behavioral tests for `AiOcrConfig.transient_retry_attempts` and sequential rotated retry behavior.

- [ ] **Step 1: Write the failing tests**

```python
def test_extract_events_with_ai_ocr_recovers_one_transient_failed_batch(self):
    calls = []
    def fake_extract(batch, providers, start_index, config):
        calls.append(start_index)
        if len(calls) == 1:
            return AiOcrProviderResult("siliconflow,zhipu", [], "The read operation timed out")
        return AiOcrProviderResult("zhipu", [EventRow("9.12", "VoX LoW", "星在")])
    # patch helper, invoke one image, assert calls == [0, 1] and event is returned

def test_extract_events_with_ai_ocr_does_not_retry_non_transient_failure(self):
    # return "HTTP Error 403" and assert one call plus AiOcrIncompleteError
```

Add assertions that Blueprint and example environment values contain `AI_OCR_TRANSIENT_RETRY_ATTEMPTS=1`.

- [ ] **Step 2: Run the focused tests and confirm RED**

Run: `python -m unittest tests.test_ai_ocr.AiOcrTest.test_extract_events_with_ai_ocr_recovers_one_transient_failed_batch tests.test_ai_ocr.AiOcrTest.test_extract_events_with_ai_ocr_does_not_retry_non_transient_failure tests.test_render_config -v`

Expected: failures because the config field and recovery pass do not exist.

- [ ] **Step 3: Add only the configuration entries**

Add `AI_OCR_TRANSIENT_RETRY_ATTEMPTS=1` to both checked configuration sources and expose it as a non-negative `AiOcrConfig` field.

- [ ] **Step 4: Commit the test/config checkpoint**

Run: `git add tests/test_ai_ocr.py tests/test_render_config.py .env.example render.yaml services/ai_ocr.py && git commit -m "test: cover transient OCR recovery"`

### Task 2: Implement sequential transient recovery

**Files:**

- Modify: `services/ai_ocr.py`
- Test: `tests/test_ai_ocr.py`

**Interfaces:**

- Consumes: `AiOcrProviderResult.error`, `_extract_batch_with_provider_fallback`, and the retry limit from `AiOcrConfig`.
- Produces: complete `results_by_batch` or the existing `AiOcrIncompleteError`.

- [ ] **Step 1: Implement the smallest retry classifier**

```python
def _is_transient_ocr_error(error: str) -> bool:
    normalized = str(error or "").lower()
    return any(marker in normalized for marker in (
        "timed out", "timeout", "connection reset", "temporarily unavailable", "too many requests", "http 5",
    ))
```

- [ ] **Step 2: Recover only failed transient batches after the executor closes**

```python
for batch_index in list(failed_batches):
    if not _is_transient_ocr_error(error_by_batch[batch_index]):
        continue
    for attempt in range(config.transient_retry_attempts):
        retry = _extract_batch_with_provider_fallback(
            batches[batch_index], config.providers, batch_index + attempt + 1, config
        )
        if not retry.error:
            results_by_batch[batch_index] = retry
            failed_batches.remove(batch_index)
            break
```

Preserve detailed debug logs and leave unresolved results in the existing error path.

- [ ] **Step 3: Run focused tests and confirm GREEN**

Run: `python -m unittest tests.test_ai_ocr tests.test_render_config -v`

Expected: PASS.

- [ ] **Step 4: Commit the behavior**

Run: `git add services/ai_ocr.py tests/test_ai_ocr.py tests/test_render_config.py .env.example render.yaml && git commit -m "fix: recover transient OCR batch failures"`

### Task 3: Make interrupted jobs explicit in the browser

**Files:**

- Modify: `static/app.js`
- Modify: `tests/test_app_routes.py`

**Interfaces:**

- Consumes: a non-success HTTP 404 from `GET /api/jobs/<remembered-id>`.
- Produces: a cleared remembered id, enabled submit button, and an error explaining that no result was returned.

- [ ] **Step 1: Write the failing static-asset assertion**

```python
def test_frontend_explains_interrupted_remembered_job(self):
    js = Path("static/app.js").read_text(encoding="utf-8")
    self.assertIn("jobInterrupted", js)
    self.assertIn("response.status === 404", js)
```

- [ ] **Step 2: Run the focused test and confirm RED**

Run: `python -m unittest tests.test_app_routes.AppRouteTest.test_frontend_explains_interrupted_remembered_job -v`

Expected: FAIL because `jobInterrupted` is absent.

- [ ] **Step 3: Add the explicit 404 copy and use it before clearing the job**

```javascript
if (response.status === 404) {
  failActiveJob(copy.jobInterrupted);
  return;
}
```

Do not call `renderResults` in this path.

- [ ] **Step 4: Run focused tests and commit**

Run: `python -m unittest tests.test_app_routes -v`

Run: `git add static/app.js tests/test_app_routes.py && git commit -m "fix: explain interrupted match jobs"`

### Task 4: Verify and deploy

**Files:**

- Verify: repository test suite and Render production service

- [ ] **Step 1: Run compilation, all tests, and diff checks**

Run: `python -m compileall app.py services tests && python -m unittest discover -s tests -v && git diff --check`

Expected: all tests pass and no whitespace errors.

- [ ] **Step 2: Run the existing public-source local smoke test**

Assert the final result contains `VoX LoW`, `9月12日`, and `星在`, with no `9月6日`/`新歌空间` row.

- [ ] **Step 3: Push and deploy the commits**

Run: `git push origin main`

Trigger a Render deploy of the pushed commit and verify it reaches `live`.

- [ ] **Step 4: Submit one public-source production job**

While running, assert the job payload has no `result`. At completion, assert the expected VoX row and the absence of the incorrect row. Check Render memory stays below 512 MB and logs contain no new application error.
