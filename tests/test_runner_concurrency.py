"""
Concurrency regression tests for HeteroSched benchmark runner and API.

Validates that:
1. Status queries via get_current_status() and GET /api/benchmark/status
   remain non-blocking and return is_running=True while a benchmark is in-flight.
2. Concurrent benchmark execution attempts are safely rejected with
   RuntimeError (direct runner) and HTTP 409 Conflict (Flask API).
3. Benchmark status reliably resets to is_running=False and releases locks
   after successful completion, timeouts, or binary execution failures.
"""

import unittest
from unittest.mock import patch, MagicMock
import threading
import subprocess
import time
import os
import sys

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from server.runner import (
    execute_benchmark,
    get_current_status,
    _BENCHMARK_LOCK,
)
from server.app import app


_CREATED_MOCK_FILES = []


def _create_mock_csv(output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(
            "task_id,type,priority,dim_m,dim_n,dim_k,device,"
            "t_decision_us,t_transfer_us,t_exec_us,t_sync_us,t_total_latency_us,"
            "stream_id,cpu_core\n"
            "0,SGEMM,2,256,256,256,GPU,0.5,100,500,0,600,0,0\n\n"
            "# total_time_ms,1.5\n"
        )
    _CREATED_MOCK_FILES.append(output_path)


class TestRunnerConcurrency(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        self.active_threads = []
        self.release_events = []
        # Verify initial clean state
        status = get_current_status()
        self.assertFalse(status["is_running"])
        self.assertFalse(_BENCHMARK_LOCK.locked())

    def tearDown(self):
        # Ensure any background threads from async tests are released and joined
        for event in self.release_events:
            event.set()
        for t in self.active_threads:
            if t.is_alive():
                t.join(timeout=5.0)

        # Clean up any mock run files created during tests
        global _CREATED_MOCK_FILES
        for p in _CREATED_MOCK_FILES:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
        _CREATED_MOCK_FILES.clear()

        # Assert that the runner has naturally returned to a clean state
        status = get_current_status()
        self.assertFalse(status["is_running"], "Benchmark status remained is_running=True after test")
        self.assertIsNone(status["current_trace"], "Benchmark trace remained set after test")
        self.assertFalse(_BENCHMARK_LOCK.locked(), "Benchmark lock remained locked after test")

    def test_direct_runner_non_blocking_status_and_rejection(self):
        """Verify direct runner reports is_running=True promptly and rejects concurrent runs."""
        started_event = threading.Event()
        release_event = threading.Event()
        self.release_events.append(release_event)

        def mock_run(cmd, *args, **kwargs):
            started_event.set()
            release_event.wait(timeout=5.0)
            _create_mock_csv(cmd[4])
            mock_proc = MagicMock()
            mock_proc.returncode = 0
            mock_proc.stdout = "Mock benchmark complete"
            mock_proc.stderr = ""
            return mock_proc

        worker_result = {}
        worker_error = []

        def worker():
            try:
                worker_result["meta"] = execute_benchmark("mixed")
            except Exception as e:
                worker_error.append(e)

        with patch("server.runner.subprocess.run", side_effect=mock_run):
            t = threading.Thread(target=worker)
            self.active_threads.append(t)
            t.start()

            try:
                # Wait until runner has entered execution and acquired lock
                self.assertTrue(started_event.wait(timeout=3.0), "Runner did not start")

                # 1. Status read must return immediately (< 0.2s) and report is_running=True
                t0 = time.monotonic()
                status = get_current_status()
                elapsed = time.monotonic() - t0

                self.assertLess(elapsed, 0.2, f"get_current_status blocked for {elapsed:.3f}s")
                self.assertTrue(status["is_running"])
                self.assertEqual(status["current_trace"], "mixed")
                self.assertIsNotNone(status["start_time"])

                # 2. Second concurrent launch attempt must be rejected immediately
                with self.assertRaises(RuntimeError) as ctx:
                    execute_benchmark("bursty")
                self.assertIn("in progress", str(ctx.exception).lower())
            finally:
                # Always release simulated execution and join thread cleanly
                release_event.set()
                t.join(timeout=5.0)

            self.assertFalse(t.is_alive())
            self.assertEqual(len(worker_error), 0)

        # 3. Status must reset to is_running=False
        final_status = get_current_status()
        self.assertFalse(final_status["is_running"])
        self.assertIsNone(final_status["current_trace"])
        self.assertFalse(_BENCHMARK_LOCK.locked())

    def test_direct_runner_resets_status_on_failure(self):
        """Verify status resets to False and lock is released if binary exits non-zero."""
        def mock_failed_run(cmd, *args, **kwargs):
            mock_proc = MagicMock()
            mock_proc.returncode = 1
            mock_proc.stdout = ""
            mock_proc.stderr = "CUDA error: out of memory"
            return mock_proc

        with patch("server.runner.subprocess.run", side_effect=mock_failed_run):
            with self.assertRaises(RuntimeError) as ctx:
                execute_benchmark("mixed")
            self.assertIn("failed", str(ctx.exception).lower())

        status = get_current_status()
        self.assertFalse(status["is_running"])
        self.assertIsNone(status["current_trace"])
        self.assertFalse(_BENCHMARK_LOCK.locked())

    def test_direct_runner_resets_status_on_timeout(self):
        """Verify status resets to False and lock is released if benchmark execution times out."""
        def mock_timeout_run(cmd, *args, **kwargs):
            raise subprocess.TimeoutExpired(cmd, timeout=kwargs.get("timeout", 60.0))

        with patch("server.runner.subprocess.run", side_effect=mock_timeout_run):
            with self.assertRaises(TimeoutError) as ctx:
                execute_benchmark("mixed", timeout_seconds=1.0)
            self.assertIn("exceeded", str(ctx.exception).lower())

        status = get_current_status()
        self.assertFalse(status["is_running"])
        self.assertIsNone(status["current_trace"])
        self.assertFalse(_BENCHMARK_LOCK.locked())

    def test_flask_client_status_and_concurrent_rejection(self):
        """Verify Flask test client reports is_running=True and 409 Conflict without live server."""
        started_event = threading.Event()
        release_event = threading.Event()
        self.release_events.append(release_event)

        def mock_run(cmd, *args, **kwargs):
            started_event.set()
            release_event.wait(timeout=5.0)
            _create_mock_csv(cmd[4])
            mock_proc = MagicMock()
            mock_proc.returncode = 0
            mock_proc.stdout = "Mock OK"
            mock_proc.stderr = ""
            return mock_proc

        api_response = {}

        def background_request():
            with app.test_client() as c:
                res = c.post("/api/benchmark/run", json={"trace": "mixed"})
                api_response["status_code"] = res.status_code
                api_response["data"] = res.get_json()

        with patch("server.runner.subprocess.run", side_effect=mock_run):
            t = threading.Thread(target=background_request)
            self.active_threads.append(t)
            t.start()

            try:
                self.assertTrue(started_event.wait(timeout=3.0), "Background run did not trigger")

                # Query status via test client while run is in-flight
                t0 = time.monotonic()
                status_res = self.client.get("/api/benchmark/status")
                dt = time.monotonic() - t0

                self.assertLess(dt, 0.2, f"GET /api/benchmark/status took too long: {dt:.3f}s")
                self.assertEqual(status_res.status_code, 200)
                status_data = status_res.get_json()
                self.assertTrue(status_data["is_running"])
                self.assertEqual(status_data["current_trace"], "mixed")

                # Second concurrent launch must return 409 Conflict
                conflict_res = self.client.post("/api/benchmark/run", json={"trace": "bursty"})
                self.assertEqual(conflict_res.status_code, 409)
                conflict_data = conflict_res.get_json()
                self.assertIn("progress", conflict_data.get("error", "").lower())
            finally:
                release_event.set()
                t.join(timeout=5.0)

            self.assertFalse(t.is_alive())
            self.assertEqual(api_response.get("status_code"), 200)

        # Status after run must return is_running=False
        final_res = self.client.get("/api/benchmark/status")
        self.assertEqual(final_res.status_code, 200)
        self.assertFalse(final_res.get_json()["is_running"])


    def test_benchmark_run_does_not_overwrite_or_touch_root_results_csv(self):
        """Verify benchmark execution saves to server/runs/ and does NOT touch repository root results.csv."""
        root_csv_path = os.path.join(REPO_ROOT, "results.csv")
        root_csv_exists = os.path.exists(root_csv_path)
        initial_content = None
        initial_mtime = None
        if root_csv_exists:
            with open(root_csv_path, "r", encoding="utf-8") as f:
                initial_content = f.read()
            initial_mtime = os.path.getmtime(root_csv_path)

        def mock_run(cmd, *args, **kwargs):
            _create_mock_csv(cmd[4])
            mock_proc = MagicMock()
            mock_proc.returncode = 0
            mock_proc.stdout = "Mock benchmark completed successfully"
            mock_proc.stderr = ""
            return mock_proc

        with patch("server.runner.subprocess.run", side_effect=mock_run):
            res = self.client.post("/api/benchmark/run", json={"trace": "mixed"})
            self.assertEqual(res.status_code, 200)
            data = res.get_json()
            self.assertTrue(data.get("success"))

            # Verify that run output file exists in server/runs/
            run_output_path = data["run"]["output_csv_path"]
            self.assertTrue(os.path.exists(run_output_path))
            self.assertIn("server/runs", run_output_path.replace("\\", "/"))

            # Verify root results.csv was NOT touched or overwritten
            if root_csv_exists:
                self.assertTrue(os.path.exists(root_csv_path), "Root results.csv was unexpectedly removed")
                with open(root_csv_path, "r", encoding="utf-8") as f:
                    current_content = f.read()
                self.assertEqual(current_content, initial_content, "Root results.csv was modified or overwritten by benchmark run!")
                self.assertEqual(os.path.getmtime(root_csv_path), initial_mtime, "Root results.csv mtime changed!")

            # Verify GET /api/benchmark/latest points to run file, not root results.csv
            latest_res = self.client.get("/api/benchmark/latest")
            self.assertEqual(latest_res.status_code, 200)
            latest_data = latest_res.get_json()
            self.assertEqual(latest_data["output_csv_path"], run_output_path)
            self.assertTrue(latest_data["is_fresh"])

            # Verify GET /api/benchmark/download downloads the run data successfully
            dl_res = self.client.get("/api/benchmark/download?mode=raw")
            self.assertEqual(dl_res.status_code, 200)
            self.assertIn("task_id,type,priority", dl_res.get_data(as_text=True))

    def test_failed_benchmark_run_does_not_touch_root_results_csv_and_restores_state(self):
        """Verify benchmark execution failure does not touch root results.csv and restores locks."""
        root_csv_path = os.path.join(REPO_ROOT, "results.csv")
        root_csv_exists = os.path.exists(root_csv_path)
        initial_content = None
        if root_csv_exists:
            with open(root_csv_path, "r", encoding="utf-8") as f:
                initial_content = f.read()

        def mock_failed_run(cmd, *args, **kwargs):
            mock_proc = MagicMock()
            mock_proc.returncode = 1
            mock_proc.stdout = ""
            mock_proc.stderr = "Fatal GPU simulation fault"
            return mock_proc

        with patch("server.runner.subprocess.run", side_effect=mock_failed_run):
            res = self.client.post("/api/benchmark/run", json={"trace": "mixed"})
            self.assertEqual(res.status_code, 500)

            # Root results.csv must remain untouched
            if root_csv_exists:
                with open(root_csv_path, "r", encoding="utf-8") as f:
                    current_content = f.read()
                self.assertEqual(current_content, initial_content, "Root results.csv modified during failure path!")

            # Status and lock must be restored
            status = get_current_status()
            self.assertFalse(status["is_running"])
            self.assertIsNone(status["current_trace"])
            self.assertFalse(_BENCHMARK_LOCK.locked())


if __name__ == "__main__":
    unittest.main()
