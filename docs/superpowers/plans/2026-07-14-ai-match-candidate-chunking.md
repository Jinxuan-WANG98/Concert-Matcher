# AI Match Candidate Chunking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep full AI candidate coverage while preventing large candidate directories from causing production read-timeout cascades.

**Architecture:** Candidate batching already occurs in `AiArtistReviewer` through `AiMatchConfig.candidate_limit`. Change its safe default and the Render/example configuration from 1000 to 200; the existing large-event planner then produces three candidate chunks for roughly 600 artists and remains below the twenty-call budget. No matching algorithm changes are required.

**Tech Stack:** Python 3.12, unittest, Render Blueprint environment variables.

## Global Constraints

- Keep `AI_MATCH_EVENT_BATCH_SIZE=40`, `AI_MATCH_EVENT_WORKERS=2`, `AI_MATCH_MAX_CALLS=20`, and `AI_MATCH_MAX_ELAPSED_SECONDS=600` unchanged.
- Retain all-candidate coverage and all-or-nothing result handling.
- Do not add concurrency or increase the Render 512MB memory envelope.

---

### Task 1: Set and document the safe candidate chunk size

**Files:**

- Modify: `services/ai_matcher.py:30-80`
- Modify: `.env.example:10-20`
- Modify: `render.yaml:30-45`
- Modify: `tests/test_ai_matcher.py:70-85`
- Modify: `tests/test_render_config.py:10-55`

**Interfaces:**

- Consumes: `AI_MATCH_CANDIDATE_LIMIT` from the environment.
- Produces: `AiMatchConfig.candidate_limit == 200` when the environment does not set a value; Render and example configurations also set `200`.

- [ ] **Step 1: Write the failing test**

```python
def test_config_defaults_bound_calls_and_use_two_workers(self):
    config = AiMatchConfig()
    self.assertEqual(config.candidate_limit, 200)
    self.assertEqual(config.event_batch_size, 40)
    self.assertEqual(config.event_workers, 2)
    self.assertEqual(config.max_calls, 20)
    self.assertEqual(config.max_elapsed_seconds, 600)
```

Change each `expected` mapping in `tests/test_render_config.py` to contain:

```python
"AI_MATCH_CANDIDATE_LIMIT": "200",
```

- [ ] **Step 2: Run the focused tests to verify failure**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_ai_matcher.AiMatcherTest.test_config_defaults_bound_calls_and_use_two_workers tests.test_render_config -v
```

Expected: failures reporting that the value is still `1000`.

- [ ] **Step 3: Change the default and checked-in environment values**

Use these exact values:

```python
candidate_limit: int = 200
candidate_limit=max(1, int(os.environ.get("AI_MATCH_CANDIDATE_LIMIT", "200")))
```

```text
AI_MATCH_CANDIDATE_LIMIT=200
```

```yaml
- key: AI_MATCH_CANDIDATE_LIMIT
  value: "200"
```

- [ ] **Step 4: Run focused tests to verify success**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_ai_matcher tests.test_render_config -v
```

Expected: all selected tests pass.

- [ ] **Step 5: Run complete verification and commit**

```powershell
.\.venv\Scripts\python.exe -m compileall -q app.py services tests
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
git diff --check
git add services/ai_matcher.py .env.example render.yaml tests/test_ai_matcher.py tests/test_render_config.py
git commit -m "perf: chunk AI match candidates"
```

Expected: compilation succeeds, every test passes, and `git diff --check` prints nothing.

### Task 2: Apply and verify the production setting

**Files:**

- No repository files.

**Interfaces:**

- Consumes: Render service `srv-d9714b7avr4c73crv2l0` and environment key `AI_MATCH_CANDIDATE_LIMIT`.
- Produces: production value `200` before triggering a deploy.

- [ ] **Step 1: Update the production variable**

Use the Render API to set `AI_MATCH_CANDIDATE_LIMIT` to `200` for `srv-d9714b7avr4c73crv2l0`.

- [ ] **Step 2: Push and deploy**

Push `main`, trigger a Render deploy without clearing the build cache, and wait until the deploy status is `live`.

- [ ] **Step 3: Verify the live service**

Confirm root HTTP status is `200`, the Render environment reports `AI_MATCH_CANDIDATE_LIMIT=200`, and current memory usage is below 512MB.
