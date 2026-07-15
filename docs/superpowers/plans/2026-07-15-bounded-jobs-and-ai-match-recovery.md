# Bounded Concurrent Jobs and AI Match Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task.

**Goal:** Let two independent users submit bounded match jobs, let a refreshed browser abandon its own job, and recover slow AI match calls without exposing partial results.

**Architecture:** The job manager owns a two-entry executing set rather than a global singleton.  Cancellation changes the observable job state immediately but holds the physical slot until the worker exits.  The frontend cancels, clears, and enables rather than restoring a remembered job.  Matching uses a 90-second request threshold, a 30-call/600-second budget, and a one-level-only timeout split.

**Tech Stack:** Python 3.12, Flask, browser JavaScript, `unittest`, Render Blueprint YAML.

## Constraints

- Do not run more than two full jobs in one Render process.
- Keep OCR at one image per request and three OCR workers per job; keep AI matching at two workers per job.
- Never publish a result for `queued`, `running`, `cancelled`, or failed jobs.
- Do not expose an active job id or result to another user.
- Do not allow timeout splitting below one child level; retain the 600-second total match deadline.

## Task 1: Specify and test bounded job lifecycle

**Files:** `tests/test_job_manager.py`, `tests/test_app_routes.py`

1. Add failing manager tests for two simultaneous operations, a generic third-job capacity error, cancellation without result, and retention of the execution slot until a blocked operation returns.
2. Add failing route tests for two `202` submissions, third `409`, and `POST /api/jobs/<id>/cancel`.
3. Add static frontend assertions for cancellation on `pagehide` and clearing an old remembered id rather than polling it.
4. Run focused job and route tests; they must fail before implementation.

## Task 2: Implement job manager and browser cancellation

**Files:** `services/job_manager.py`, `app.py`, `static/app.js`, `.env.example`, `render.yaml`

1. Introduce `max_concurrent_jobs`, executing-job tracking, capacity-only `JobBusyError`, and a persisted `cancel` transition.
2. Ensure `_run`, success, and failure preserve a cancelled snapshot and release the executing slot in `finally`.
3. Remove the global preflight busy check, add the cancel route, and return generic capacity text.
4. On browser page hide, best-effort cancel the active id.  On load, cancel/clear any remembered id and enable a fresh submission.  Handle `cancelled` as a terminal no-result state.
5. Set `JOB_MAX_CONCURRENT=2` in checked configuration sources.
6. Run focused tests until green.

## Task 3: Bound AI timeout recovery

**Files:** `tests/test_ai_matcher.py`, `tests/test_pipeline.py`, `services/ai_matcher.py`, `.env.example`, `render.yaml`, `tests/test_render_config.py`

1. Add failing tests showing one timed-out multi-event batch becomes exactly two child calls, child timeout is terminal, and strict pipeline mode refuses any partial batch completion.
2. Add a split-depth argument to the candidate batch method; only depth zero may split a timed-out event batch.
3. Raise the individual match timeout to 90 seconds and the request budget to 30 while retaining the 600-second aggregate deadline.
4. Run matcher, pipeline, and configuration tests until green.

## Task 4: Verify, publish, and observe

1. Run `python -m compileall app.py services tests`, `python -m unittest discover -s tests -v`, and `git diff --check`.
2. Commit the complete bounded-concurrency, cancellation, and timeout-recovery change; push `main`.
3. Update the same Render production environment values, deploy the commit, and wait for `live`.
4. Submit a production job.  Verify it has no `result` before success, its final output keeps `VoX LoW / 9月12日 / 星在`, and logs/memory remain inside the configured limits.
