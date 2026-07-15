# Per-event AI Match Shortlisting Design

## Evidence

The first production verification after bounded job recovery completed OCR for all nine images, but AI matching still failed after about 614 seconds. The input contained 177 unresolved events and 592 playlist artists. With 40 events and 200 artists per request, the matcher created fifteen large all-to-all calls. A terminal child call then hit the 90-second read timeout, so the strict pipeline correctly returned no partial result.

A controlled 20-event, 100-candidate request with five known matches completed on the current `GLM-4-Flash` provider in about 10 seconds. The same request took about 28 seconds on SiliconFlow `deepseek-ai/DeepSeek-V4-Flash`. The model swap therefore does not solve the root problem; the candidate payload and cross-product plan do.

## Selected design

For every unresolved event, build a deterministic local shortlist of five playlist artists. Exact, bidirectional known-alias, OCR-normalized, containment, token, bounded spelling, and character-fragment similarities are ranked in that order. Every event receives a shortlist, including events without a strong lexical signal, so every unresolved event is still sent through AI analysis.

Each AI payload contains batches of at most 20 events. Every event carries only its own `candidate_names`. The model may select only from that event-specific list, and server-side validation rejects a returned artist outside it. The previous artist-chunk × event-batch cross product is removed from the normal batch path.

For 177 unresolved events, the normal plan becomes nine calls rather than fifteen, and each call carries at most 100 event-candidate pairs rather than up to 8,000 all-to-all combinations. Two calls may run concurrently, so the change reduces provider payload and wall time without increasing process concurrency or memory.

## Correctness and failure semantics

- Unique exact and OCR-normalized anchors still bypass AI, preserving the verified `VoX LoW / 9月12日 / 星在` result.
- Known bilingual aliases participate in local ranking before generic edit similarity.
- An AI suggestion is accepted only when its event index, event performer, and artist name all belong to the current event-specific input.
- The prompt requires a root-level `matches` array. If GLM echoes the input and nests that same array under `events`, the parser may flatten the explicit nested decisions before applying all three validation boundaries; it never invents a match.
- A timeout may still split one event batch once. Child failure remains terminal.
- `_SafeAiReviewer(strict=True)` continues to convert any failed AI batch into a whole-task failure. The browser receives a result only after every OCR and AI batch succeeds.
- The cache namespace changes so no result from the former all-to-all prompt is reused.

## Production configuration

- Keep text matching on the existing GLM provider after the faster controlled comparison.
- Set `AI_MATCH_EVENT_BATCH_SIZE=20` and `AI_MATCH_SHORTLIST_PER_EVENT=5`.
- Keep `AI_MATCH_EVENT_WORKERS=2`, `AI_MATCH_TIMEOUT_SECONDS=90`, `AI_MATCH_MAX_CALLS=30`, and `AI_MATCH_MAX_ELAPSED_SECONDS=600`.
- Keep two full jobs maximum and three one-image OCR workers per job. No memory-affecting concurrency is added.

## Verification

Tests must prove deterministic shortlist ranking, bilingual/OCR recall, per-event payload isolation, rejection of cross-event candidates, nine-call planning for 177 events, one-level timeout recovery, and strict all-or-nothing pipeline behavior. Full unit tests, compilation, JavaScript syntax, configuration checks, deployment, a real-link production task, final `VoX LoW` fields, and Render memory/log observation are required before completion.
