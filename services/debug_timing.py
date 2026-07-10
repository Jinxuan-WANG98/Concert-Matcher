from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

SESSION_ID = "c69de8"
LOG_PATH = Path("debug-c69de8.log")
_phase_timings: list[dict[str, Any]] = []


def reset_phase_timings() -> None:
    _phase_timings.clear()


def get_phase_timings() -> list[dict[str, Any]]:
    return list(_phase_timings)


def debug_log(
    location: str,
    message: str,
    data: dict[str, Any],
    hypothesis_id: str = "",
    run_id: str = "",
) -> None:
    # #region agent log
    payload = {
        "sessionId": SESSION_ID,
        "timestamp": int(time.time() * 1000),
        "location": location,
        "message": message,
        "data": data,
        "hypothesisId": hypothesis_id,
    }
    if run_id:
        payload["runId"] = run_id
    try:
        with LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except OSError:
        pass
    try:
        print(f"[debug:{SESSION_ID}] {json.dumps(payload, ensure_ascii=False)}", file=sys.stderr, flush=True)
    except OSError:
        pass
    # #endregion


class PhaseTimer:
    def __init__(self, location: str, phase: str, hypothesis_id: str = "H6"):
        self._location = location
        self._phase = phase
        self._hypothesis_id = hypothesis_id
        self._start = time.perf_counter()
        self.data: dict[str, Any] = {}

    def __enter__(self) -> "PhaseTimer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        elapsed_ms = int((time.perf_counter() - self._start) * 1000)
        payload = {"phase": self._phase, "elapsedMs": elapsed_ms, **self.data}
        if exc is not None:
            payload["error"] = str(exc)
        _phase_timings.append(payload)
        debug_log(self._location, "pipeline phase complete", payload, hypothesis_id=self._hypothesis_id)
