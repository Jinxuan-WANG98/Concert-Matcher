# Async Match Jobs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the fragile long-running `/api/match` response with a single-job asynchronous workflow that only exposes results after every required AI batch completes, while keeping Render memory below 512 MB.

**Architecture:** A process-local `MatchJobManager` owns one daemon worker thread, persists atomic JSON snapshots under `outputs/webapp/jobs`, and exposes queued/running/succeeded/failed states. Flask returns `202` with a job ID, the browser privately stores and polls that job resource, and the pipeline reports coarse stage progress. AI OCR and matching wait for every required batch, compact artist lists limit prompt size, and any partial AI completion fails the job instead of publishing a partial result.

**Tech Stack:** Python 3.12, Flask, Gunicorn gthread, `unittest`, browser JavaScript, Render Blueprint YAML.

## Global Constraints

- Show no match rows until every required AI OCR and AI matching batch has completed successfully.
- Run at most one match job per Render instance.
- Default AI OCR concurrency to 3 and match concurrency to 2; never enable local RapidOCR fallback on Render.
- Keep one Gunicorn process and remain below the Render Free 512 MB memory limit.
- Preserve and integrate the existing uncommitted `static/app.js` status-polling changes.
- Persist only JSON-serializable job snapshots using atomic file replacement; the design must not claim durability across Render restarts.
- Use TDD for every behavior change and run the complete suite before completion.

---

### Task 1: Persistent single-job state manager

**Files:**
- Create: `services/job_manager.py`
- Create: `tests/test_job_manager.py`

**Interfaces:**
- Produces: `JobBusyError`, `MatchJobManager(state_dir, artifact_root, ttl_seconds)`, `start(job_id, operation)`, `get(job_id)`, `latest()`, and `is_busy()`.
- `operation` receives `progress(stage: str, progress: int, message: str)` and returns the final JSON result dictionary.

- [x] **Step 1: Write failing tests** for immediate queued/running snapshots, terminal success visibility, failure messages, busy rejection, atomic disk recovery, and TTL cleanup restricted to 32-character hexadecimal job IDs.
- [x] **Step 2: Run** `.venv\Scripts\python.exe -m unittest tests.test_job_manager -v` and verify imports fail because `services.job_manager` does not exist.
- [x] **Step 3: Implement** a locked in-memory registry, daemon worker, atomic `*.tmp` to `*.json` replacement, terminal result/error storage, latest-job lookup, and safe expired-artifact cleanup.
- [x] **Step 4: Re-run** `.venv\Scripts\python.exe -m unittest tests.test_job_manager -v` and verify all Task 1 tests pass.

### Task 2: Asynchronous Flask API and pipeline progress

**Files:**
- Modify: `app.py`
- Modify: `services/pipeline.py`
- Modify: `tests/test_app_routes.py`
- Modify: `tests/test_pipeline.py`

**Interfaces:**
- `POST /api/match` returns `202 {job_id, state, status_url}` and never waits for the pipeline.
- `GET /api/jobs/<job_id>` exposes progress but includes `result` only for `succeeded`.
- `GET /api/status` exposes only whether the worker is active; page reload recovery uses the submitting browser's private `localStorage` job ID so other visitors cannot discover a job.
- `run_match_pipeline(..., job_id=None, progress_callback=None)` writes output to the supplied job directory and emits playlist/OCR/match/export stages.

- [x] **Step 1: Replace route tests with failing async-contract tests** covering `202`, no intermediate result, final-only result, failure state, `409`, private stored-job recovery, and deterministic download ID.
- [x] **Step 2: Add a failing pipeline test** proving playlist failure occurs before uploaded-image OCR begins.
- [x] **Step 3: Run** `.venv\Scripts\python.exe -m unittest tests.test_app_routes tests.test_pipeline -v` and verify failures describe the old synchronous contract and parallel load.
- [x] **Step 4: Implement** the manager-backed routes, result serializer, friendly background errors, job-ID pipeline output, progress callback, and playlist-first loading.
- [x] **Step 5: Re-run** the two test modules and verify they pass without exposing partial results.

### Task 3: Browser polling and final-only rendering

**Files:**
- Modify: `static/app.js`
- Modify: `tests/test_app_routes.py`

**Interfaces:**
- Browser stores `concertMatchJobId`, polls `/api/jobs/<id>` every 5 seconds, resumes that stored ID after refresh, displays stage/progress text, and calls `renderResults` only for `state === "succeeded"`.

- [x] **Step 1: Add failing static-contract tests** requiring job-ID storage, job-resource polling, terminal-only rendering, retryable network polling, and button re-enable on idle/terminal states.
- [x] **Step 2: Run** `.venv\Scripts\python.exe -m unittest tests.test_app_routes -v` and verify the old synchronous fetch flow fails the assertions.
- [x] **Step 3: Refactor the existing user JavaScript diff** into a single non-overlapping poll loop; retain its Chinese busy/network messages and confidence rendering.
- [x] **Step 4: Re-run** the route/static tests and verify they pass.

### Task 4: Bounded complete-candidate AI matching

**Files:**
- Modify: `services/ai_matcher.py`
- Modify: `services/pipeline.py`
- Modify: `tests/test_ai_matcher.py`
- Modify: `tests/test_pipeline.py`

**Interfaces:**
- `AiMatchConfig` adds `max_calls` and `max_elapsed_seconds` from `AI_MATCH_MAX_CALLS` and `AI_MATCH_MAX_ELAPSED_SECONDS`.
- Every event batch receives compact names for up to `candidate_limit` artists; with 607 artists and limit 1000 there is one candidate group.
- AI-only pipeline wrapping is strict: any required failed batch fails the job rather than returning partial matches.

- [x] **Step 1: Add failing tests** showing 51 events × 5 artists with limit 1000 uses only event-batch-count calls, default workers are 2, call/deadline budgets stop recursive retries, merge ties are deterministic, and AI-only partial batch failure propagates.
- [x] **Step 2: Run** `.venv\Scripts\python.exe -m unittest tests.test_ai_matcher tests.test_pipeline -v` and confirm the tests fail against Cartesian candidate batching and swallowed partial failures.
- [x] **Step 3: Implement** unified compact batching, guarded AI calls, monotonic deadline checks, deterministic merge, and strict `_SafeAiReviewer` behavior for AI-only mode.
- [x] **Step 4: Re-run** both modules and verify the reduced-call and complete-analysis contracts pass.

### Task 5: Upload and Render resource boundaries

**Files:**
- Modify: `services/pipeline.py`
- Modify: `tests/test_pipeline.py`
- Modify: `render.yaml`
- Modify: `tests/test_render_config.py`
- Modify: `README.md`

**Interfaces:**
- Upload defaults: total request 30 MB, each file 12 MB, each image 12 million pixels; image validation reads dimensions and calls `verify()` before full decode.
- Render defaults: one process, two threads, local OCR fallback false, OCR workers 3, match workers 2, candidate limit 1000, event batch 40, max AI calls 20, match deadline 600 seconds, playlist-first load.

- [x] **Step 1: Add failing tests** for oversized individual files, excessive dimensions/pixels, and exact Blueprint values.
- [x] **Step 2: Run** `.venv\Scripts\python.exe -m unittest tests.test_pipeline tests.test_render_config -v` and verify the new bounds fail against current defaults.
- [x] **Step 3: Implement** streaming-safe upload checks, update `render.yaml`, and document asynchronous API behavior and Free-instance restart limitations.
- [x] **Step 4: Run** the focused tests, then `.venv\Scripts\python.exe -m unittest discover -s tests -v`, `.venv\Scripts\python.exe -m compileall -q app.py services tests`, and a local Flask async smoke test.
- [x] **Step 5: Inspect** `git diff --check`, `git status --short`, and the final diff to ensure the pre-existing JavaScript work was preserved and no output/cache files were added.
