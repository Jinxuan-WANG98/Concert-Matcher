# AI OCR Transient Recovery Design

## Problem

The Render production logs for 14 July show that a nine-image task completed eight OCR batches, but one batch timed out at both providers after the existing 90-second per-provider limit.  The job correctly failed without exposing partial matches, but the user had to resubmit the entire task.

The same logs show a peak process memory of about 146 MB with three concurrent single-image OCR requests.  Four concurrent requests would add upstream pressure without addressing a sequential timeout at both providers.

## Considered approaches

1. Raise the timeout for every OCR call to 120 seconds.  This increases normal-path tail latency and still leaves each image as a single point of failure.
2. Raise image concurrency to four.  It can shorten a healthy three-wave job, but it worsens the observed provider-pressure failure mode.
3. Retry only a batch that exhausted both providers with a transient transport/provider error.  Run one such retry only after the parallel pass completes, so it is isolated at concurrency one.  This keeps the normal path unchanged and bounds failure recovery.

Approach 3 is selected.

## Design

`AiOcrConfig` gains `transient_retry_attempts`, read from `AI_OCR_TRANSIENT_RETRY_ATTEMPTS` with a production default of `1` and clamped to a non-negative value.

During the normal pass, all image batches retain the existing three-worker concurrency and provider fallback behavior.  Once that pass finishes, the code examines only failed batches.  A failed batch is eligible only when its combined provider error is transient: timeout, connection reset, temporary unavailability, rate limiting, or an HTTP 5xx indication.  To keep the job bounded during a provider outage, each task retries at most one eligible batch and does so once, sequentially, with the provider order rotated so the other provider gets the first attempt.  A successful retry replaces the failed batch result.

Non-transient errors, retry exhaustion, and failed retry results remain failures.  The function continues to raise `AiOcrIncompleteError` if any batch has no verified OCR result, so matching never runs and the web UI never receives a partial result.

The web client will recognize a 404 while polling its own remembered job as an interrupted job (for example, after a free-instance restart).  It clears the local job id, reenables submission, and states that no result was returned and a new submission is needed.

## Constraints and verification

- Keep `AI_OCR_IMAGE_WORKERS=3`, `AI_OCR_IMAGE_BATCH_SIZE=1`, and the current 90-second provider timeout.
- Recovery is sequential and performs no additional image decoding beyond the retry itself, so it does not raise the normal-path process-memory peak.
- Update `render.yaml` and `.env.example` to document the retry limit.
- Tests must prove: a timeout result is recovered by exactly one rotated retry; no more than one failed batch is retried per task; a non-transient failed batch is not retried; an exhausted retry still raises the existing all-or-nothing error; and the frontend clears an interrupted job without rendering results.
- Run focused tests, the full suite, a local real-data smoke test, then a Render deployment and production job verification.
