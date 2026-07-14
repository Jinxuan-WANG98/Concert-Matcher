# Exact Match Anchors Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent AI batch-index mistakes from assigning a playlist artist to the wrong concert row, while reducing AI workload and preserving all-or-nothing result display.

**Architecture:** `services.matcher` will establish unique whole-name exact anchors before AI-only matching, remove anchored events and artists from the AI work set, and merge anchors only after the remaining AI call succeeds. `services.ai_matcher` will carry explicit original event indices through batching, timeout splitting, parsing, validation, and caching; every accepted batch suggestion must echo the performer belonging to an index in that exact batch.

**Tech Stack:** Python 3.11, dataclasses, `unittest`, existing Flask job pipeline, OpenAI-compatible OCR/matching APIs, Render.

## Global Constraints

- All required AI analysis must finish before the page displays the final result once.
- Any OCR or AI matching batch failure must prevent partial display and partial export.
- Do not add provider calls, raise OCR/image concurrency, or change current timeout/environment settings for this fix.
- Render memory must remain significantly below the 512 MB limit.
- Only unique cleaned whole-name equality may create an anchor; collaboration splitting, fuzzy similarity, and model knowledge may not create anchors.
- Invalidate the old AI match cache so the incorrect VOX LOW suggestion cannot be reused.

---

### Task 1: Preserve and validate original event identities in AI batches

**Files:**
- Modify: `services/ai_matcher.py:19-850`
- Test: `tests/test_ai_matcher.py:20-870`

**Interfaces:**
- Produces: `AiArtistReviewer.find_best_matches(events, artists, event_indices=None) -> dict[int, AiMatchSuggestion]`.
- Produces: `build_batch_artist_pick_payload(..., event_indices=None, start_index=0) -> dict[str, Any]`.
- Produces: batch `AiMatchSuggestion.event_performer`, populated by `parse_ai_batch_match_suggestions` and required by batch validation.
- Consumes: existing `EventRow`, `PlaylistArtist`, cache helpers, timeout splitting, and parallel batch execution.

- [ ] **Step 1: Write failing payload and parser tests for explicit identities**

Add tests that request non-contiguous original indices and require an echoed performer:

```python
def test_batch_payload_uses_explicit_original_event_indices(self):
    events = [
        EventRow(date_text="9.6", performer="叶琼琳", venue="新歌空间"),
        EventRow(date_text="9.12", performer="VoX LoW", venue="星在"),
    ]
    payload = build_batch_artist_pick_payload(
        events,
        [PlaylistArtist(name="VOX LOW")],
        event_indices=[101, 116],
    )
    user = json.loads(payload["messages"][-1]["content"].split("JSON:\n", 1)[1])

    self.assertEqual([item["event_index"] for item in user["events"]], [101, 116])
    self.assertIn("event_performer", payload["messages"][0]["content"])


def test_parse_batch_suggestion_keeps_echoed_performer(self):
    suggestions = parse_ai_batch_match_suggestions(
        '{"matches":[{"event_index":116,"event_performer":"VoX LoW",'
        '"artist_name":"VOX LOW","confidence":"高","reason":"同名"}]}'
    )

    self.assertEqual(suggestions[116].event_performer, "VoX LoW")
```

- [ ] **Step 2: Run the two tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest `
  tests.test_ai_matcher.AiMatcherTest.test_batch_payload_uses_explicit_original_event_indices `
  tests.test_ai_matcher.AiMatcherTest.test_parse_batch_suggestion_keeps_echoed_performer -v
```

Expected: FAIL because the payload has no `event_indices` argument and `AiMatchSuggestion` has no `event_performer` field.

- [ ] **Step 3: Add explicit IDs and performer echoes to the batch schema**

Implement the public defaults without changing single-event review behavior:

```python
@dataclass(frozen=True)
class AiMatchSuggestion:
    artist_name: str
    confidence: str
    reason: str
    event_performer: str = ""


def _resolved_event_indices(events, start_index, event_indices):
    if event_indices is None:
        return [start_index + index for index in range(len(events))]
    if len(event_indices) != len(events):
        raise ValueError("event_indices must match events")
    return list(event_indices)
```

Use `_resolved_event_indices` in `build_batch_artist_pick_payload`, add `event_performer` to the required JSON schema, and populate the new field in `parse_ai_batch_match_suggestions`. Leave `parse_ai_match_suggestion` on its default empty echo because it is not a batch response.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run the command from Step 2.

Expected: both tests PASS.

- [ ] **Step 5: Write failing tests for batch membership and performer validation**

Add a reviewer test whose model response contains one cross-batch index and one mismatched echo:

```python
def test_reviewer_rejects_unknown_index_and_mismatched_performer(self):
    reviewer = AiArtistReviewer(AiMatchConfig(
        enabled=True,
        api_key="test-key",
        base_url="https://api.example.com/v1",
        model="text-model",
        event_batch_size=2,
        event_workers=1,
    ))
    reviewer._chat_content = lambda payload: json.dumps({"matches": [
        {"event_index": 999, "event_performer": "VoX LoW", "artist_name": "VOX LOW", "confidence": "高", "reason": "wrong batch"},
        {"event_index": 101, "event_performer": "VoX LoW", "artist_name": "VOX LOW", "confidence": "高", "reason": "wrong row"},
    ]}, ensure_ascii=False)

    suggestions = reviewer.find_best_matches(
        [EventRow(date_text="9.6", performer="叶琼琳", venue="新歌空间")],
        [PlaylistArtist(name="VOX LOW")],
        event_indices=[101],
    )

    self.assertEqual(suggestions, {})
```

Extend the existing timeout-split test so the input indices are `[101, 102, 116, 117]`, every fake response echoes the corresponding performer, and the observed split calls remain those exact IDs.

- [ ] **Step 6: Run validation tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest `
  tests.test_ai_matcher.AiMatcherTest.test_reviewer_rejects_unknown_index_and_mismatched_performer `
  tests.test_ai_matcher.AiMatcherTest.test_reviewer_splits_timed_out_batch_and_retries_smaller_batches -v
```

Expected: FAIL because reviewer batching still rebuilds contiguous indices and accepts every parsed suggestion.

- [ ] **Step 7: Carry IDs through batching, recursion, validation, and cache fingerprints**

Change reviewer batching to operate on paired indices and events:

```python
indexed_events = list(zip(resolved_indices, events))
batches = _chunked(indexed_events, batch_size)
```

Every `_find_best_matches_*` method must accept the corresponding `event_indices` list. Timeout recursion must split `events` and `event_indices` at the same midpoint. After parsing a response, retain only suggestions whose index exists in `dict(zip(event_indices, events))` and whose echoed performer satisfies `normalize_name(echo) == normalize_name(event.performer)`; import `normalize_name` inside the validation helper to avoid module initialization cycles. Log only the rejected index and reason.

Include `event_indices` in `_match_cache_fingerprint`, `_match_cache_path`, `_load_match_cache`, and `_save_match_cache`. Raise `MATCH_CACHE_VERSION` from `2` to `3` and change `AiMatchConfig.cache_source` from `ai-match:v7` to `ai-match:v8`.

- [ ] **Step 8: Update existing batch response fixtures to echo performers**

For every fake batch JSON in `tests/test_ai_matcher.py`, include the exact input performer:

```python
{
    "event_index": index,
    "event_performer": events_by_index[index].performer,
    "artist_name": f"Artist {index}",
    "confidence": "高",
    "reason": "same",
}
```

Update fake `_find_best_matches_batch` functions to accept the new `event_indices` argument, and change the cache-source assertion to `ai-match:v8:`.

- [ ] **Step 9: Run the complete AI matcher suite**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_ai_matcher -v
```

Expected: all AI matcher tests PASS; no warning or unhandled thread exception appears.

- [ ] **Step 10: Commit Task 1**

```powershell
git add services/ai_matcher.py tests/test_ai_matcher.py
git commit -m "fix: validate AI match event identities"
```

---

### Task 2: Anchor unique exact matches before AI-only matching

**Files:**
- Modify: `services/matcher.py:25-350`
- Test: `tests/test_matcher_ai_review.py:1-220`

**Interfaces:**
- Consumes: `AiArtistReviewer.find_best_matches(events, artists, event_indices=None)` from Task 1.
- Produces: `_find_unique_exact_anchor(event, artists) -> PlaylistArtist | None` using only the complete performer name.
- Produces: AI-only matching that sends only unresolved events and unanchored artists, but returns anchors and AI matches through the existing `list[MatchResult]` interface.

- [ ] **Step 1: Write the VOX LOW regression test first**

Update `FakeBatchReviewer.find_best_matches` to accept and record `event_indices`. Add:

```python
def test_ai_only_exact_anchor_prevents_vox_low_wrong_row(self):
    events = [
        EventRow(date_text="9.6", performer="叶琼琳", venue="新歌空间", image_name="image05.jpg"),
        EventRow(date_text="9.12", performer="VoX LoW", venue="星在", image_name="image06.jpg"),
    ]
    artists = [PlaylistArtist(name="VOX LOW", song_count=7, sample_songs=["We Walk"])]
    reviewer = FakeBatchReviewer({
        0: AiMatchSuggestion(artist_name="VOX LOW", confidence="高", reason="wrong row")
    })

    matches = match_events_to_artists(events, artists, ai_reviewer=reviewer, ai_only=True)

    self.assertEqual([(m.artist_name, m.date_text, m.venue) for m in matches], [
        ("VOX LOW", "9.12", "星在")
    ])
    self.assertEqual(reviewer.batch_calls, 0)
```

This test deliberately supplies the old wrong suggestion; the reviewer must not be called because the only candidate artist is already anchored.

- [ ] **Step 2: Run the regression test and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest `
  tests.test_matcher_ai_review.MatcherAiReviewTest.test_ai_only_exact_anchor_prevents_vox_low_wrong_row -v
```

Expected: FAIL with the old `VOX LOW / 9.6 / 新歌空间` behavior or an unexpected reviewer call.

- [ ] **Step 3: Implement unique whole-name anchors**

Add a helper that does not call `_score_pair` or split collaborators:

```python
def _find_unique_exact_anchor(event: EventRow, artists: list[PlaylistArtist]) -> PlaylistArtist | None:
    performer_key = normalize_name(event.performer)
    if not performer_key:
        return None
    matches = [artist for artist in artists if normalize_name(artist.name) == performer_key]
    if len(matches) == 1:
        return matches[0]

    if _has_latin(event.performer) and len(performer_key) >= 4:
        ocr_key = normalize_ocr_latin(event.performer)
        ocr_matches = [artist for artist in artists if normalize_ocr_latin(artist.name) == ocr_key]
        if len(ocr_matches) == 1:
            return ocr_matches[0]
    return None
```

At the start of the `ai_only` branch, build `anchors: dict[int, PlaylistArtist]`, `unresolved_events`, `unresolved_indices`, and `remaining_artists`. Call the batch reviewer only when both unresolved lists are non-empty. Pass `event_indices=unresolved_indices`. Add anchored matches with confidence `高`, score `1.0`, method `精确名称锚定`, and the original event's date, venue, performer, and image.

- [ ] **Step 4: Run the regression test and verify GREEN**

Run the command from Step 2.

Expected: PASS with only `VOX LOW / 9.12 / 星在`.

- [ ] **Step 5: Add filtering and ambiguity tests**

Add tests proving:

```python
def test_ai_only_sends_only_unresolved_events_and_unanchored_artists(self):
    # PREP is anchored at original index 0; 泽拉黛 remains unresolved at index 1.
    # The reviewer receives only 泽拉黛, only Zella Day, and event_indices=[1].

def test_ai_only_does_not_anchor_non_unique_cleaned_artist_names(self):
    # Two playlist candidates with the same normalized name make the event unresolved.

def test_ai_only_anchor_does_not_split_collaboration_line(self):
    # "PREP / Guest" is not a whole-name equality and remains AI work.
```

Use assertions on `FakeBatchReviewer.last_events`, `last_artists`, and `last_event_indices`, then assert the final merged result keeps original dates and venues.

- [ ] **Step 6: Run new tests and verify RED, then implement minimal filtering**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_matcher_ai_review -v
```

Expected before implementation completion: the three new tests FAIL. Complete only the anchor/filter logic needed for them, rerun, and expect the entire module to PASS.

- [ ] **Step 7: Update old AI-only expectations**

Rename `test_ai_only_mode_skips_local_scorer` to `test_ai_only_mode_anchors_exact_name_without_fuzzy_scorer`; continue monkeypatching `_score_pair` to prove fuzzy scoring is not used, but assert no AI call and `精确名称锚定`. Adjust other exact-name fixtures so they expect anchors, while translation/alias fixtures still exercise AI.

- [ ] **Step 8: Commit Task 2**

```powershell
git add services/matcher.py tests/test_matcher_ai_review.py
git commit -m "fix: anchor exact artists before AI matching"
```

---

### Task 3: Preserve strict all-or-nothing behavior through the safe reviewer

**Files:**
- Modify: `services/pipeline.py:55-105`
- Test: `tests/test_pipeline.py:20-90`
- Test: `tests/test_matcher_ai_review.py`

**Interfaces:**
- Consumes: Task 1's optional `event_indices` parameter.
- Produces: `_SafeAiReviewer.find_best_matches(events, artists, event_indices=None)` that forwards stable IDs and still raises the first partial-batch failure in strict mode.

- [ ] **Step 1: Write failing forwarding and strict-failure tests**

```python
def test_safe_ai_reviewer_forwards_original_event_indices(self):
    class CapturingReviewer:
        last_failures = []
        def find_best_matches(self, events, artists, event_indices=None):
            self.event_indices = event_indices
            return {}

    inner = CapturingReviewer()
    safe = pipeline._SafeAiReviewer(inner, [], strict=True)
    safe.find_best_matches([EventRow("9.12", "VoX LoW", "星在")], [PlaylistArtist("VOX LOW")], event_indices=[116])
    self.assertEqual(inner.event_indices, [116])


def test_ai_only_does_not_return_anchor_when_remaining_ai_fails(self):
    class FailingReviewer:
        def find_best_matches(self, events, artists, event_indices=None):
            raise RuntimeError("required AI batch failed")

    with self.assertRaisesRegex(RuntimeError, "required AI batch failed"):
        match_events_to_artists(
            [EventRow("9.12", "VoX LoW", "星在"), EventRow("9.13", "未知译名", "MAO")],
            [PlaylistArtist("VOX LOW"), PlaylistArtist("Unknown Artist")],
            ai_reviewer=FailingReviewer(),
            ai_only=True,
        )
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest `
  tests.test_pipeline.PipelineTest.test_safe_ai_reviewer_forwards_original_event_indices `
  tests.test_matcher_ai_review.MatcherAiReviewTest.test_ai_only_does_not_return_anchor_when_remaining_ai_fails -v
```

Expected: the forwarding test FAILS because `_SafeAiReviewer` does not accept `event_indices`; the strict matching test must raise rather than return the anchor.

- [ ] **Step 3: Forward IDs without weakening strict failure handling**

```python
def find_best_matches(self, events, artists, event_indices=None):
    if not hasattr(self._reviewer, "find_best_matches"):
        return {}
    try:
        suggestions = self._reviewer.find_best_matches(
            events,
            artists,
            event_indices=event_indices,
        )
        failures = getattr(self._reviewer, "last_failures", [])
        if failures:
            if self._strict:
                raise failures[0]
            self._warn_once(failures[0])
        return suggestions
    except Exception as exc:
        if self._strict:
            raise
        self._warn_once(exc)
        return {}
```

Update `PartialReviewer` and other fake batch reviewers in pipeline tests to accept the optional keyword.

- [ ] **Step 4: Run pipeline and matcher tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_pipeline tests.test_matcher_ai_review -v
```

Expected: all tests PASS and strict mode still raises after any required batch failure.

- [ ] **Step 5: Commit Task 3**

```powershell
git add services/pipeline.py tests/test_pipeline.py tests/test_matcher_ai_review.py
git commit -m "fix: preserve strict AI matching with stable event ids"
```

---

### Task 4: Full regression, real-link verification, and Render deployment

**Files:**
- Verify: `services/ai_matcher.py`
- Verify: `services/matcher.py`
- Verify: `services/pipeline.py`
- Verify: `tests/test_ai_matcher.py`
- Verify: `tests/test_matcher_ai_review.py`
- Verify: `tests/test_pipeline.py`
- No production environment-variable changes.

**Interfaces:**
- Consumes: Tasks 1-3 as a complete matching pipeline.
- Produces: a deployable commit whose UI still commits results only after the job is complete.

- [ ] **Step 1: Run syntax, diff, and full automated verification**

```powershell
.\.venv\Scripts\python.exe -m compileall -q app.py services tests
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
git diff --check
git status --short
```

Expected: compile command exits `0`; all tests PASS; `git diff --check` prints nothing; only intentional files are modified.

- [ ] **Step 2: Run the local nine-image regression with the known playlist**

Use playlist `167320827`, which contains VoX LoW, and the user-provided XHS link. Load `.env` without printing secrets and align only the existing production values for this process:

```powershell
@'
import os, uuid
from pathlib import Path
from app import load_local_env_file
from services.pipeline import run_match_pipeline

load_local_env_file(Path('.env'))
os.environ.update({
    'AI_OCR_IMAGE_BATCH_SIZE': '1',
    'AI_OCR_IMAGE_WORKERS': '3',
    'AI_OCR_TIMEOUT_SECONDS': '90',
    'AI_OCR_PROVIDER_FALLBACK': 'true',
    'AI_OCR_DUAL_PROVIDER': 'true',
    'AI_OCR_LOCAL_FALLBACK': 'false',
    'AI_MATCH_MODE': 'ai_only',
    'AI_MATCH_EVENT_BATCH_SIZE': '40',
    'AI_MATCH_EVENT_WORKERS': '2',
})
result = run_match_pipeline(
    'https://music.163.com/#/playlist?id=167320827',
    'http://xhslink.com/o/3UIAQIybL21',
    output_root=Path('work/diagnostics/vox-low-fixed'),
    job_id=uuid.uuid4().hex,
)
vox = [(m.artist_name, m.date_text, m.venue) for m in result.matches if 'vox' in m.artist_name.casefold()]
print({'vox': vox, 'events': result.event_count, 'matches': len(result.matches)})
assert ('VoX LoW', '9.12', '星在') in vox or ('VOX LOW', '9.12', '星在') in vox
assert not any(date == '9.6' and venue == '新歌空间' for _, date, venue in vox)
'@ | & '.\.venv\Scripts\python.exe' -
```

Expected: the command completes only after all OCR and AI work; the printed `vox` list contains `9.12 / 星在` and no `9.6 / 新歌空间`.

- [ ] **Step 3: Review runtime evidence**

Record total duration, AI call/batch count, and process memory peak from existing timing/debug logs. Confirm the new flow makes no extra provider calls and does not retain image copies. If the real-link task fails, diagnose and fix before deployment; do not waive the all-or-nothing assertion.

- [ ] **Step 4: Commit any final fixture-only corrections**

If Step 1 or Step 2 required an intentional correction, rerun the full suite and commit only those files:

```powershell
git add services/ai_matcher.py services/matcher.py services/pipeline.py tests/test_ai_matcher.py tests/test_matcher_ai_review.py tests/test_pipeline.py
git commit -m "test: cover exact-anchor concert regression"
```

If there are no remaining changes, do not create an empty commit.

- [ ] **Step 5: Push and deploy the exact tested commit**

```powershell
git push origin main
```

Trigger the existing Render service deployment for the pushed commit without clearing cache. Do not change Render environment variables.

- [ ] **Step 6: Verify Render health and memory**

Wait for the deployment to become `live`, then verify:

- `GET /` returns `200`.
- `GET /api/status` returns `200` and valid JSON.
- Deployment and runtime logs show no new 5xx, uncaught exception, or OOM.
- Render memory remains significantly below `512 MB`; compare the task peak with the previous observed peak near `128 MiB`.
- The production task retains the all-or-nothing UI behavior; no partial match rows are exposed while status is queued/running.

- [ ] **Step 7: Report the deployed commit and evidence**

Report the commit SHA, Render deploy ID/status, automated test count, real-link VOX LOW tuple, request-duration comparison, and memory peak. Explicitly state that concurrency and timeout environment variables were not changed.

