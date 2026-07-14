from __future__ import annotations

import json
import re
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


_ACTIVE_STATES = {"queued", "running"}
_JOB_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


class JobBusyError(RuntimeError):
    def __init__(self, active_job_id: str):
        super().__init__("a match job is already running")
        self.active_job_id = active_job_id


class MatchJobManager:
    def __init__(self, state_dir: Path, artifact_root: Path, ttl_seconds: int = 86400):
        self.state_dir = Path(state_dir)
        self.artifact_root = Path(artifact_root)
        self.ttl_seconds = max(1, int(ttl_seconds))
        self._lock = threading.RLock()
        self._jobs: dict[str, dict] = {}
        self._active_job_id: str | None = None
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.cleanup_expired()
        self._load_persisted_jobs()

    def start(self, job_id: str, operation: Callable[[Callable[[str, int, str], None]], dict]) -> dict:
        if not _JOB_ID_PATTERN.fullmatch(job_id):
            raise ValueError("job_id must be 32 lowercase hexadecimal characters")
        self.cleanup_expired()
        now = _utc_now()
        with self._lock:
            if self._active_job_id:
                active = self._jobs.get(self._active_job_id)
                if active and active.get("state") in _ACTIVE_STATES:
                    raise JobBusyError(self._active_job_id)
                self._active_job_id = None
            snapshot = {
                "job_id": job_id,
                "state": "queued",
                "stage": "queued",
                "progress": 0,
                "message": "任务已进入队列",
                "created_at": now,
                "updated_at": now,
            }
            self._jobs[job_id] = snapshot
            self._active_job_id = job_id
            self._persist_locked(snapshot)
            created = dict(snapshot)

        worker = threading.Thread(
            target=self._run,
            args=(job_id, operation),
            name=f"match-job-{job_id[:8]}",
            daemon=True,
        )
        worker.start()
        return created

    def get(self, job_id: str) -> dict | None:
        if not _JOB_ID_PATTERN.fullmatch(job_id):
            return None
        self.cleanup_expired()
        with self._lock:
            snapshot = self._jobs.get(job_id)
            if snapshot is None:
                snapshot = self._read_state(job_id)
                if snapshot is not None:
                    self._jobs[job_id] = snapshot
            return dict(snapshot) if snapshot is not None else None

    def latest(self) -> dict | None:
        self.cleanup_expired()
        with self._lock:
            if not self._jobs:
                return None
            snapshot = max(
                self._jobs.values(),
                key=lambda item: (str(item.get("updated_at", "")), str(item.get("job_id", ""))),
            )
            return dict(snapshot)

    def is_busy(self) -> bool:
        with self._lock:
            if not self._active_job_id:
                return False
            active = self._jobs.get(self._active_job_id)
            return bool(active and active.get("state") in _ACTIVE_STATES)

    def cleanup_expired(self) -> None:
        cutoff = time.time() - self.ttl_seconds
        root = self.artifact_root.resolve()
        with self._lock:
            active_job_id = self._active_job_id
            for state_path in self.state_dir.glob("*.json"):
                job_id = state_path.stem
                if job_id == active_job_id or not _JOB_ID_PATTERN.fullmatch(job_id):
                    continue
                try:
                    expired = state_path.stat().st_mtime < cutoff
                except OSError:
                    continue
                if not expired:
                    continue
                state_path.unlink(missing_ok=True)
                self._jobs.pop(job_id, None)
                artifact_path = (root / job_id).resolve()
                if artifact_path.parent == root and artifact_path.is_dir():
                    shutil.rmtree(artifact_path)

    def _run(self, job_id: str, operation) -> None:
        self._update(job_id, state="running", stage="starting", progress=1, message="任务正在启动")

        def report_progress(stage: str, progress: int, message: str) -> None:
            self._update(
                job_id,
                state="running",
                stage=str(stage),
                progress=max(1, min(99, int(progress))),
                message=str(message),
            )

        try:
            result = operation(report_progress)
        except Exception as exc:
            self._finish_failed(job_id, str(exc) or exc.__class__.__name__)
            return
        self._finish_succeeded(job_id, result)

    def _update(self, job_id: str, **changes) -> None:
        with self._lock:
            snapshot = self._jobs[job_id]
            if snapshot.get("state") not in _ACTIVE_STATES:
                return
            if "progress" in changes:
                changes["progress"] = max(int(snapshot.get("progress", 0)), int(changes["progress"]))
            snapshot.update(changes)
            snapshot["updated_at"] = _utc_now()
            snapshot.pop("result", None)
            snapshot.pop("error", None)
            self._persist_locked(snapshot)

    def _finish_succeeded(self, job_id: str, result: dict) -> None:
        with self._lock:
            snapshot = self._jobs[job_id]
            snapshot.update(
                state="succeeded",
                stage="completed",
                progress=100,
                message="匹配完成",
                result=result,
                updated_at=_utc_now(),
            )
            snapshot.pop("error", None)
            self._persist_locked(snapshot)
            if self._active_job_id == job_id:
                self._active_job_id = None

    def _finish_failed(self, job_id: str, error: str) -> None:
        with self._lock:
            snapshot = self._jobs[job_id]
            snapshot.update(
                state="failed",
                stage="failed",
                message="匹配失败",
                error=error,
                updated_at=_utc_now(),
            )
            snapshot.pop("result", None)
            self._persist_locked(snapshot)
            if self._active_job_id == job_id:
                self._active_job_id = None

    def _persist_locked(self, snapshot: dict) -> None:
        job_id = snapshot["job_id"]
        state_path = self.state_dir / f"{job_id}.json"
        temp_path = self.state_dir / f"{job_id}.tmp"
        temp_path.write_text(json.dumps(snapshot, ensure_ascii=False), encoding="utf-8")
        temp_path.replace(state_path)

    def _read_state(self, job_id: str) -> dict | None:
        path = self.state_dir / f"{job_id}.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return None
        return data if isinstance(data, dict) and data.get("job_id") == job_id else None

    def _load_persisted_jobs(self) -> None:
        with self._lock:
            for path in self.state_dir.glob("*.json"):
                job_id = path.stem
                if not _JOB_ID_PATTERN.fullmatch(job_id):
                    continue
                snapshot = self._read_state(job_id)
                if snapshot is None:
                    continue
                if snapshot.get("state") in _ACTIVE_STATES:
                    snapshot.update(
                        state="failed",
                        stage="failed",
                        message="匹配失败",
                        error="服务已重启，原任务无法继续，请重新提交。",
                        updated_at=_utc_now(),
                    )
                    snapshot.pop("result", None)
                    self._persist_locked(snapshot)
                self._jobs[job_id] = snapshot


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
