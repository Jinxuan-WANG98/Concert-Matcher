# Per-event AI Match Shortlisting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task.

**Goal:** Replace large all-to-all artist matching calls with small event-specific candidate lists while preserving complete AI analysis and all-or-nothing output.

**Architecture:** A pure local ranker selects five candidates for every unresolved event. Batch payloads carry the candidate names alongside their event, and response validation enforces that boundary. Existing two-worker execution and strict pipeline failure handling remain unchanged.

**Tech Stack:** Python 3.12, Flask, `unittest`, Render Blueprint YAML.

## Task 1: Lock behavior with failing tests

**Files:** `tests/test_ai_matcher.py`, `tests/test_pipeline.py`, `tests/test_render_config.py`

1. Test shortlist defaults and environment loading.
2. Test exact, alias, OCR-normalized, and fallback shortlist ranking with deterministic limits.
3. Test event-specific payloads and rejection of a model response that selects another event's candidate.
4. Test that 177 events produce nine calls with at most five candidates per event.
5. Retain tests proving a failed batch is not cached and strict pipeline mode publishes no partial result.
6. Run focused tests and confirm the new assertions fail before implementation.

## Task 2: Implement shortlisting and guarded payloads

**Files:** `services/ai_matcher.py`, `.env.example`, `render.yaml`

1. Add `shortlist_per_event` configuration and a new cache namespace.
2. Implement deterministic local ranking using existing matcher normalization and alias rules.
3. Extend batch payload construction with per-event candidate names.
4. Replace the large-set candidate cross product with event batches over their shortlists.
5. Reject suggestions outside the corresponding event shortlist.
6. Keep one-level timeout splitting, call/time budgets, worker count, and strict failure reporting.
7. Run focused tests until green.

## Task 3: Verify and deploy

1. Run full unit tests, Python compilation, JavaScript syntax, and `git diff --check`.
2. Commit and push `main`.
3. Update Render to batch size 20 and shortlist size 5, deploy, and wait for `live`.
4. Submit the real playlist and Xiaohongshu link. Verify there is no intermediate result, the task succeeds only after all AI calls complete, and the final `VoX LoW` date and venue are correct.
5. Check production logs and memory before reporting completion.
