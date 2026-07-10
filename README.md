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
4. Start Command: `gunicorn --workers 1 --timeout 600 app:app`
5. Choose the Free instance for testing, or use the included `render.yaml`.
6. Use the generated `onrender.com` URL after deployment.

## Optional AI Review

The page includes an AI review switch. The switch only works when the deployment has an OpenAI-compatible API key configured. Without these environment variables, the app still runs with local fuzzy and alias matching.

```text
AI_MATCH_ENABLED=true
AI_MATCH_API_KEY=your_api_key
AI_MATCH_BASE_URL=https://api.openai.com/v1
AI_MATCH_MODEL=gpt-4.1-mini
AI_MATCH_CANDIDATE_LIMIT=120
AI_MATCH_EVENT_BATCH_SIZE=30
AI_MATCH_EVENT_WORKERS=3
```

`AI_MATCH_BASE_URL` can point to any OpenAI-compatible Chat Completions endpoint. AI review checks medium-confidence matches and can also fill a missing match when local rules find no candidate, but only from the playlist artists you provide.

For AI-only matching, `AI_MATCH_EVENT_BATCH_SIZE` controls how many concert rows are sent in one request, and `AI_MATCH_EVENT_WORKERS` controls how many event batches run at the same time. `AI_MATCH_CANDIDATE_LIMIT` is treated as the candidate chunk size per request; if a playlist has more artists than that, the app continues with later chunks instead of truncating the playlist. Timed-out requests are retried with smaller event batches, and if a single-event request is still too large, the app splits the playlist candidates further.

On small Render instances, keep `WEB_CONCURRENCY=1`, `OCR_MAX_WORKERS=1`, and `AI_OCR_IMAGE_WORKERS=1` to avoid memory restarts. If memory is stable in Render logs, `AI_OCR_IMAGE_WORKERS=2` can improve image recognition speed.

## Optional AI Image Recognition

The app can also use vision models before local OCR. This is useful for Xiaohongshu notes whose first page and later pages use different layouts. Configure these values on the server, or copy `.env.example` to `.env` for local runs. Do not put real keys in frontend code.

When two providers are configured, the app splits image batches across them in parallel, repairs malformed AI JSON with AI, merges usable structured rows, and only uses local RapidOCR when `AI_OCR_LOCAL_FALLBACK=true`. AI matching results are cached for the same recognized events, playlist artists, model, and endpoint, so repeated runs can skip the matching calls.

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
AI_OCR_IMAGE_BATCH_SIZE=8
AI_OCR_IMAGE_WORKERS=2
AI_OCR_LOCAL_FALLBACK=false
AI_OCR_MIN_AGREEMENT_RATIO=0.2
AI_OCR_MIN_EVENTS=1
```

If AI image recognition returns malformed JSON, the app first asks AI to repair the response into the required structure. If a large image batch still fails, the app retries smaller AI batches before giving up. Local RapidOCR is not called automatically unless `AI_OCR_LOCAL_FALLBACK=true`.

For China-accessible OpenAI-compatible providers:

- SiliconFlow: `AI_MATCH_BASE_URL` / `AI_OCR_BASE_URL` = `https://api.siliconflow.cn/v1`
- Zhipu BigModel: `AI_MATCH_BASE_URL` / `AI_OCR_BASE_URL` = `https://open.bigmodel.cn/api/paas/v4`

## Notes

Xiaohongshu scraping may fail because of login requirements, anti-bot behavior, or page changes. When that happens, upload the note images directly.
