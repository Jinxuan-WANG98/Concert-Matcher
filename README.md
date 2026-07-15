# Concert Matcher Web App

A deployable Flask web app that compares a NetEase Cloud Music playlist with a Xiaohongshu concert note. It extracts playlist artists, reads concert images with OCR, matches artists, shows results in the browser, and exports an Excel file.

## Features

- NetEase playlist URL input.
- Xiaohongshu note URL input.
- Image upload fallback when Xiaohongshu scraping fails.
- OCR-based concert row parsing.
- Date range parsing, including adjacent-column endings like `7/20 -21`.
- Alias, fuzzy, OCR-tolerant, and optional AI-reviewed artist matching.
- Result table with index, date, artist, venue, playlist count, sample songs, and confidence.
- Excel download.
- Asynchronous match jobs with resumable browser polling and final-only result rendering.

## Local Run

Double-click `start_concert_matcher.bat`, or run manually:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
flask --app app run --host 127.0.0.1 --port 5050
```

Open `http://127.0.0.1:5050`.

## Render Deployment

1. Push this project to GitHub.
2. Create a Render Web Service and connect the repository.
3. Build Command: `pip install -r requirements.txt`
4. Start Command: `gunicorn --worker-class gthread --threads 2 --workers 1 --timeout 1800 app:app`
5. Choose the Free instance for testing, or use the included `render.yaml`.
6. Use the generated `onrender.com` URL after deployment.

## Optional AI Review

The page includes an AI review switch. The switch only works when the deployment has an OpenAI-compatible API key configured. Without these environment variables, the app still runs with local fuzzy and alias matching.

```text
AI_MATCH_ENABLED=true
AI_MATCH_API_KEY=your_api_key
AI_MATCH_BASE_URL=https://api.openai.com/v1
AI_MATCH_MODEL=gpt-4.1-mini
AI_MATCH_CANDIDATE_LIMIT=200
AI_MATCH_SHORTLIST_PER_EVENT=5
AI_MATCH_EVENT_BATCH_SIZE=20
AI_MATCH_EVENT_WORKERS=2
AI_MATCH_MAX_CALLS=30
AI_MATCH_MAX_ELAPSED_SECONDS=600
```

`AI_MATCH_BASE_URL` can point to any OpenAI-compatible Chat Completions endpoint. AI review checks medium-confidence matches and can also fill a missing match when local rules find no candidate, but only from the playlist artists you provide.

For AI-only matching, every unresolved event receives a deterministic local shortlist before AI review. Each request carries at most `AI_MATCH_EVENT_BATCH_SIZE` events and only `AI_MATCH_SHORTLIST_PER_EVENT` candidates for each event, so the model never receives the full playlist cross product. Timed-out requests can split once into smaller batches, but all requests—including repair and split calls—share `AI_MATCH_MAX_CALLS` and `AI_MATCH_MAX_ELAPSED_SECONDS`. If any required batch fails, the job fails and does not publish a partial result.

On the 512 MB Render instance, keep `WEB_CONCURRENCY=1`, `OCR_MAX_WORKERS=1`, `AI_OCR_IMAGE_BATCH_SIZE=1`, `AI_OCR_IMAGE_WORKERS=3`, `AI_MATCH_EVENT_WORKERS=2`, and `AI_OCR_LOCAL_FALLBACK=false`. Three concurrent worst-case 12-megapixel image encodes peaked at about 248 MB in local measurement; local RapidOCR can create a much larger peak and is intentionally disabled in the Blueprint.

## Asynchronous Job API

`POST /api/match` validates and stores the uploaded images, starts one background job, and returns HTTP `202` with a `job_id` and `status_url`. It does not keep the original HTTP request open while OCR and matching run.

Poll `GET /api/jobs/<job_id>` until `state` is `succeeded` or `failed`. Queued and running responses contain stage/progress only. The `result` object appears only in `succeeded`, after every required AI OCR and AI matching batch finishes successfully. The browser stores its own unguessable job ID in `localStorage`, so a refresh resumes that job without exposing another visitor's job through the global status endpoint. `GET /api/status` reports only whether the single worker is busy.

The browser polls every five seconds while a job is active. This supplies inbound traffic during a Free-instance task, but the process-local worker and ephemeral files cannot survive a Render restart or a new deployment. Durable execution across restarts requires an external job store and a paid background worker.

Only one job runs per instance. Finished state and output files expire after `JOB_TTL_SECONDS` (default 86400). The Render upload defaults are 30 MB per request, 12 MB per image, 12 million pixels per image, and 12000 pixels on either edge.

## Optional AI Image Recognition

The app can also use vision models before local OCR. This is useful for Xiaohongshu notes whose first page and later pages use different layouts. Configure these values on the server, or copy `.env.example` to `.env` for local runs. Do not put real keys in frontend code.

When two providers are configured with `AI_OCR_DUAL_PROVIDER=true`, the app rotates image batches across them in parallel (for example batch 1 to SiliconFlow, batch 2 to Zhipu), repairs malformed AI JSON with AI, merges rows by date and performer, and only uses local RapidOCR when `AI_OCR_LOCAL_FALLBACK=true`. With `AI_OCR_PROVIDER_FALLBACK=true`, only a failed image batch is retried once by the alternate provider. It does not require both models to agree on the same image.

```text
AI_OCR_ENABLED=true
AI_OCR_PROVIDER_1_NAME=siliconflow
AI_OCR_PROVIDER_1_API_KEY=your_siliconflow_key
AI_OCR_PROVIDER_1_BASE_URL=https://api.siliconflow.cn/v1
AI_OCR_PROVIDER_1_MODEL=Qwen/Qwen3-VL-32B-Instruct
AI_OCR_PROVIDER_2_NAME=zhipu
AI_OCR_PROVIDER_2_ENABLED=true
AI_OCR_PROVIDER_2_API_KEY=your_zhipu_key
AI_OCR_PROVIDER_2_BASE_URL=https://open.bigmodel.cn/api/paas/v4
AI_OCR_PROVIDER_2_MODEL=glm-4.6v
AI_OCR_IMAGE_BATCH_SIZE=1
AI_OCR_IMAGE_WORKERS=3
AI_OCR_DUAL_PROVIDER=true
AI_OCR_PROVIDER_FALLBACK=true
AI_OCR_LOCAL_FALLBACK=false
```

If AI image recognition returns malformed JSON, the app first asks AI to repair the response into the required structure. If a large image batch still fails, the app retries smaller AI batches before giving up. Local RapidOCR is not called automatically unless `AI_OCR_LOCAL_FALLBACK=true`.

For China-accessible OpenAI-compatible providers:

- SiliconFlow: `AI_MATCH_BASE_URL` / `AI_OCR_BASE_URL` = `https://api.siliconflow.cn/v1`
- Zhipu BigModel: `AI_MATCH_BASE_URL` / `AI_OCR_BASE_URL` = `https://open.bigmodel.cn/api/paas/v4`

## Notes

Xiaohongshu scraping may fail because of login requirements, anti-bot behavior, or page changes. When that happens, upload the note images directly.
