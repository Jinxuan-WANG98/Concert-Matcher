import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path

from services.job_manager import JobBusyError, MatchJobManager


def wait_for_state(manager: MatchJobManager, job_id: str, state: str, timeout: float = 2.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snapshot = manager.get(job_id)
        if snapshot and snapshot["state"] == state:
            return snapshot
        time.sleep(0.01)
    raise AssertionError(f"job {job_id} did not reach {state}")


class MatchJobManagerTest(unittest.TestCase):
    def test_result_is_hidden_until_operation_succeeds_and_survives_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = MatchJobManager(root / "jobs", root, ttl_seconds=3600)
            started = threading.Event()
            release = threading.Event()
            job_id = "a" * 32

            def operation(progress):
                progress("ai_match", 65, "AI matching")
                started.set()
                release.wait(timeout=2)
                return {"matches": [{"artist": "PREP"}], "download_url": f"/download/{job_id}"}

            created = manager.start(job_id, operation)
            self.assertEqual(created["job_id"], job_id)
            self.assertNotIn("result", created)
            self.assertTrue(started.wait(timeout=1))

            running = manager.get(job_id)
            self.assertEqual(running["state"], "running")
            self.assertEqual(running["stage"], "ai_match")
            self.assertEqual(running["progress"], 65)
            self.assertNotIn("result", running)

            release.set()
            succeeded = wait_for_state(manager, job_id, "succeeded")
            self.assertEqual(succeeded["progress"], 100)
            self.assertEqual(succeeded["result"]["matches"][0]["artist"], "PREP")
            self.assertFalse(manager.is_busy())

            reloaded = MatchJobManager(root / "jobs", root, ttl_seconds=3600)
            self.assertEqual(reloaded.get(job_id), succeeded)
            self.assertEqual(reloaded.latest(), succeeded)
            self.assertFalse((root / "jobs" / f"{job_id}.tmp").exists())

    def test_failure_has_error_without_result_and_allows_next_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = MatchJobManager(root / "jobs", root, ttl_seconds=3600)

            def fail(progress):
                progress("ocr", 25, "OCR")
                raise RuntimeError("provider timed out")

            manager.start("b" * 32, fail)
            failed = wait_for_state(manager, "b" * 32, "failed")
            self.assertEqual(failed["error"], "provider timed out")
            self.assertNotIn("result", failed)
            self.assertFalse(manager.is_busy())

            manager.start("c" * 32, lambda progress: {"matches": []})
            self.assertEqual(wait_for_state(manager, "c" * 32, "succeeded")["state"], "succeeded")

    def test_allows_two_jobs_but_rejects_a_third_while_both_are_running(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = MatchJobManager(root / "jobs", root, ttl_seconds=3600, max_concurrent_jobs=2)
            started = threading.Event()
            release = threading.Event()
            start_count = 0
            count_lock = threading.Lock()

            def slow(progress):
                nonlocal start_count
                with count_lock:
                    start_count += 1
                    if start_count == 2:
                        started.set()
                release.wait(timeout=2)
                return {"matches": []}

            manager.start("d" * 32, slow)
            manager.start("e" * 32, slow)
            self.assertTrue(started.wait(timeout=1))
            with self.assertRaises(JobBusyError) as context:
                manager.start("f" * 32, lambda progress: {"matches": []})
            self.assertEqual(context.exception.active_job_count, 2)
            self.assertEqual(context.exception.max_concurrent_jobs, 2)
            self.assertTrue(manager.is_busy())
            release.set()
            wait_for_state(manager, "d" * 32, "succeeded")
            wait_for_state(manager, "e" * 32, "succeeded")

    def test_cancel_hides_result_and_retains_execution_slot_until_worker_exits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = MatchJobManager(root / "jobs", root, ttl_seconds=3600, max_concurrent_jobs=2)
            first_started = threading.Event()
            second_started = threading.Event()
            continued_after_cancel = threading.Event()
            release = threading.Event()

            def slow_first(progress):
                first_started.set()
                release.wait(timeout=2)
                progress("ai_match", 65, "must not continue after cancellation")
                continued_after_cancel.set()
                return {"matches": [{"artist": "must stay hidden"}]}

            def slow_second(progress):
                second_started.set()
                release.wait(timeout=2)
                return {"matches": []}

            first_id = "a" * 32
            second_id = "b" * 32
            manager.start(first_id, slow_first)
            self.assertTrue(first_started.wait(timeout=1))

            cancelled = manager.cancel(first_id)
            self.assertEqual(cancelled["state"], "cancelled")
            self.assertNotIn("result", cancelled)

            manager.start(second_id, slow_second)
            self.assertTrue(second_started.wait(timeout=1))
            with self.assertRaises(JobBusyError):
                manager.start("c" * 32, lambda progress: {"matches": []})

            release.set()
            wait_for_state(manager, second_id, "succeeded")
            self.assertEqual(manager.get(first_id)["state"], "cancelled")
            self.assertNotIn("result", manager.get(first_id))
            self.assertFalse(continued_after_cancel.is_set())
            deadline = time.monotonic() + 1
            while manager.is_busy() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertFalse(manager.is_busy())

    def test_cleanup_removes_only_expired_hex_job_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_dir = root / "jobs"
            state_dir.mkdir()
            expired_id = "f" * 32
            current_id = "1" * 32
            invalid_name = "cache"
            old_timestamp = time.time() - 7200

            for job_id in (expired_id, current_id):
                (root / job_id).mkdir()
                state_path = state_dir / f"{job_id}.json"
                state_path.write_text(json.dumps({"job_id": job_id, "state": "succeeded"}), encoding="utf-8")
                if job_id == expired_id:
                    os.utime(state_path, (old_timestamp, old_timestamp))
            (root / invalid_name).mkdir()
            invalid_state = state_dir / f"{invalid_name}.json"
            invalid_state.write_text("{}", encoding="utf-8")
            os.utime(invalid_state, (old_timestamp, old_timestamp))

            manager = MatchJobManager(state_dir, root, ttl_seconds=3600)
            manager.cleanup_expired()

            self.assertFalse((state_dir / f"{expired_id}.json").exists())
            self.assertFalse((root / expired_id).exists())
            self.assertTrue((state_dir / f"{current_id}.json").exists())
            self.assertTrue((root / current_id).exists())
            self.assertTrue(invalid_state.exists())
            self.assertTrue((root / invalid_name).exists())

    def test_get_lazily_expires_job_without_waiting_for_another_submission(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_dir = root / "jobs"
            manager = MatchJobManager(state_dir, root, ttl_seconds=60)
            expired_id = "3" * 32
            artifact_dir = root / expired_id
            artifact_dir.mkdir()
            state_path = state_dir / f"{expired_id}.json"
            state_path.write_text(
                json.dumps({"job_id": expired_id, "state": "succeeded"}),
                encoding="utf-8",
            )
            old_timestamp = time.time() - 120
            os.utime(state_path, (old_timestamp, old_timestamp))

            self.assertIsNone(manager.get(expired_id))
            self.assertFalse(state_path.exists())
            self.assertFalse(artifact_dir.exists())

    def test_reload_marks_an_interrupted_running_job_as_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_dir = root / "jobs"
            state_dir.mkdir()
            job_id = "2" * 32
            (state_dir / f"{job_id}.json").write_text(
                json.dumps(
                    {
                        "job_id": job_id,
                        "state": "running",
                        "stage": "ai_match",
                        "progress": 65,
                        "message": "AI matching",
                        "created_at": "2026-07-14T00:00:00+00:00",
                        "updated_at": "2026-07-14T00:01:00+00:00",
                    }
                ),
                encoding="utf-8",
            )

            manager = MatchJobManager(state_dir, root, ttl_seconds=3600)
            recovered = manager.get(job_id)

            self.assertEqual(recovered["state"], "failed")
            self.assertIn("重启", recovered["error"])
            self.assertFalse(manager.is_busy())


if __name__ == "__main__":
    unittest.main()
