# Bounded Concurrent Jobs and AI Match Recovery Design

## Evidence

The production task `26111e89766d448c83d480d572b4f5e7` completed OCR for all nine images (190 events) but stopped during AI matching.  Its match plan contained 177 unresolved events, 592 playlist artists, five event batches, three candidate batches, and therefore 15 planned requests.  The task stopped after 512 seconds with ten failed work items because timeout-triggered recursive splitting consumed the 20-call guard.

This rules out increasing `AI_MATCH_CANDIDATE_LIMIT` from 200 to 250 as a remedy: both values produce three candidate batches for 592 artists.

The web server currently stores exactly one active job id.  It rejects every second request with `409`, even when it belongs to a different browser.  The browser also restores its remembered job on refresh, so a user cannot start over after refreshing the page.

One production job has observed a process-memory peak of about 136 MiB.  Two concurrent jobs are conservatively below 272 MiB before shared-process savings, leaving headroom below Render's 512 MiB limit.  Four such jobs are not safe.

## Selected design

### Job ownership and bounded capacity

`MatchJobManager` will replace its singleton active-job id with a set of executing job ids and a `max_concurrent_jobs` limit.  The default and Render value will be `2`.

- Any two independent submissions can start and have distinct job ids.
- A third request receives a generic capacity error; it never exposes another user's job id.
- `GET /api/status` remains only a boolean health/capacity signal and does not identify jobs.
- The process retains a cancelled worker in its executing set until its current operation returns.  This is essential: cancellation must not create a third concurrently consuming worker during a slow provider request.

### Refresh and cancellation

The browser will treat a refresh as abandoning its own remembered job.  On `pagehide`, it sends a best-effort `POST /api/jobs/<id>/cancel` with `keepalive`; on a subsequent page load, it repeats that request if a remembered id remains, clears local storage, enables the form, and states that the old task returned no result.

The cancel endpoint changes only an active job to `cancelled`, removes any result, and persists the snapshot.  A cancelled job never later transitions to `succeeded` or exposes its eventual background result.  If it was still queued, its worker exits before beginning the pipeline.  If a provider call is already in progress, Python cannot safely interrupt it; the worker is discarded as soon as that call returns and remains counted against the two-job ceiling in the meantime.

This implements the user-visible requirement to terminate the old submission without risking unsafe thread termination or memory oversubscription.

### AI match timeout recovery

The matching configuration will use the already proven higher individual read threshold of 90 seconds.  The current 60-second threshold turns slow-but-valid provider responses into recursive split trees.

The recursive split fallback will receive an explicit `split_depth` and may split an original timed-out event batch once only.  Its two child batches must not split again.  This prevents an outage from fanning out exponentially.  The request budget will be raised from 20 to 30, which covers the 15 normal planned calls and up to seven single-depth split recoveries, while the existing 600-second total deadline remains the hard wall-clock bound.  Any exhausted budget, second-level timeout, or failed work item still fails the whole task; `_SafeAiReviewer(strict=True)` prevents partial matches from reaching the browser.

The change does not increase worker counts: image OCR remains three requests per job and AI matching remains two requests per job.  With two jobs, the hard process-level maximum is six OCR or four matching requests, while each job is bounded by its existing upload and deadline limits.

## Rejected alternatives

1. Four concurrent jobs: the measured single-job peak makes its simple worst-case estimate exceed Render's 512 MiB allowance.
2. Increasing candidate limit to 250: produces the same three batches for the failed production input, so it cannot reduce calls.
3. Unbounded recursive retries or simply removing the call guard: either can exhaust time, provider quota, or process resources during an outage.
4. Killing Python threads on refresh: Python cannot safely kill a blocked network thread.  Persistent cancellation plus retained capacity is deterministic and safe.

## Verification

- Manager tests will prove two jobs run, the third is rejected, a cancelled job hides its result, and its actual slot is retained until its operation returns.
- Route tests will prove two users receive `202`, the third receives a generic `409`, and cancellation returns a result-free `cancelled` snapshot.
- Browser assertions will prove page-hide cancellation and refresh cleanup replace job resumption, without rendering a cancelled result.
- Matcher tests will prove a timed-out batch splits once, never at depth two, and exhausted recovery produces no partial result.
- Configuration tests will keep `.env.example` and `render.yaml` aligned at two jobs, 90 seconds, 30 calls, and the existing conservative worker counts.
- Full tests, compilation, production deployment, a live job result, and Render memory/log checks are required before completion.
