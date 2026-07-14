# AI OCR Provider Fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable one alternate-provider retry for failed AI OCR image batches while preserving three-worker concurrency, the 90-second per-request timeout, the 512 MB memory boundary, and all-or-nothing final results.

**Architecture:** The application already rotates single-image batches between SiliconFlow and Zhipu and already contains the alternate-provider retry path. This change enables that path through deployment configuration only, keeps the strict `AiOcrIncompleteError` gate unchanged, and synchronizes the Render Blueprint, example environment, and README.

**Tech Stack:** Python 3, unittest, Flask application configuration, Render Blueprint/API, Git.

## Global Constraints

- Keep `AI_OCR_IMAGE_BATCH_SIZE=1`.
- Keep `AI_OCR_IMAGE_WORKERS=3` and do not increase concurrency to 4.
- Keep `AI_OCR_DUAL_PROVIDER=true`.
- Keep `AI_OCR_TIMEOUT_SECONDS=90`.
- Keep `AI_OCR_LOCAL_FALLBACK=false`.
- Require every image OCR batch and every AI matching batch to finish before the frontend displays the final result.
- Keep observed Render memory below the 512 MB service limit.

---

### Task 1: Enable and document the bounded provider fallback

**Files:**
- Modify: `tests/test_render_config.py`
- Modify: `render.yaml`
- Modify: `.env.example`
- Modify: `README.md`

**Interfaces:**
- Consumes: Existing `AiOcrConfig.from_env()` handling of `AI_OCR_PROVIDER_FALLBACK` and `_extract_batch_with_provider_fallback()`.
- Produces: Repository deployment defaults where `AI_OCR_PROVIDER_FALLBACK=true`, with no Python runtime behavior changes.

- [ ] **Step 1: Write the failing configuration assertions**

Add `"AI_OCR_PROVIDER_FALLBACK": "true"` to the `expected` dictionaries in both `RenderConfigTest.test_render_defaults_are_memory_conservative_and_bounded` and `RenderConfigTest.test_env_example_matches_production_safety_limits`:

```python
expected = {
    # existing bounded settings remain unchanged
    "AI_OCR_PROVIDER_FALLBACK": "true",
}
```

- [ ] **Step 2: Run the targeted test and verify RED**

Run:

```powershell
& .\.venv\Scripts\python.exe -m unittest tests.test_render_config -v
```

Expected: two assertion failures showing the current repository values are `false` instead of the required `true`.

- [ ] **Step 3: Apply the minimal configuration change**

Change only these values:

```yaml
# render.yaml
- key: AI_OCR_PROVIDER_FALLBACK
  value: "true"
```

```dotenv
# .env.example
AI_OCR_PROVIDER_FALLBACK=true
```

Update the README example to `AI_OCR_PROVIDER_FALLBACK=true` and state that only a failed batch is retried once using the alternate provider. Do not change batch size, worker count, timeout, or local OCR fallback.

- [ ] **Step 4: Run the targeted test and verify GREEN**

Run:

```powershell
& .\.venv\Scripts\python.exe -m unittest tests.test_render_config -v
```

Expected: both Render configuration tests pass.

- [ ] **Step 5: Run the complete test suite**

Run:

```powershell
& .\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Expected: exit code 0 with no failures or errors.

- [ ] **Step 6: Commit the repository configuration**

```powershell
git add tests/test_render_config.py render.yaml .env.example README.md docs/superpowers/plans/2026-07-14-ai-ocr-provider-fallback.md
git commit -m "fix: retry failed OCR batches across providers"
```

### Task 2: Publish and deploy the production setting

**Files:**
- Verify only: `render.yaml`

**Interfaces:**
- Consumes: Git remote `origin`, branch `main`, Render service `srv-d9714b7avr4c73crv2l0`, and environment variable `RENDER_API_KEY`.
- Produces: A live Render deployment of the committed configuration with `AI_OCR_PROVIDER_FALLBACK=true`.

- [ ] **Step 1: Push the configuration commit**

Run:

```powershell
git push origin main
```

Expected: the new commit is accepted by `origin/main`.

- [ ] **Step 2: Update the existing Render service environment variable**

Send an authenticated Render API update for service `srv-d9714b7avr4c73crv2l0` so that:

```text
AI_OCR_PROVIDER_FALLBACK=true
```

Read the variable back immediately and require the returned value to be `true` before continuing.

- [ ] **Step 3: Ensure the deployed revision is the new Git commit**

Inspect the latest deployments for `srv-d9714b7avr4c73crv2l0`. If the environment update did not already deploy the new `main` revision, trigger a deployment without clearing the build cache. Wait until its status is `live`; treat `build_failed`, `update_failed`, `canceled`, or `deactivated` as failure.

- [ ] **Step 4: Verify production health and resource safety**

Require all of the following:

```text
GET https://concert-matcher.onrender.com -> HTTP 200
GET https://concert-matcher.onrender.com/api/status -> HTTP 200
latest deploy commit == local HEAD
recent error log count == 0
recent HTTP 5xx count == 0
recent memory usage < 536870912 bytes
```

- [ ] **Step 5: Confirm repository cleanliness**

Run:

```powershell
git status --short
```

Expected: no output.
