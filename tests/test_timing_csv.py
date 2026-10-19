import unittest
import subprocess
import os
import sys
import json

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, WORKSPACE_ROOT)
from server.parser import parse_benchmark_csv, CSV_COLUMNS

BINARY = os.path.join(WORKSPACE_ROOT, "build", "heterosched_bench")
TEST_TRACE = "/tmp/test_timing_trace.json"
TEST_OUTPUT = "/tmp/test_timing_output.csv"


class TestTimingAuditAndCSV(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a deterministic 6-task test trace covering CPU, GPU, and different kernels
        tasks = [
            # 2 Quicksort (strictly CPU)
            {"id": 0, "type": "QUICKSORT", "priority": 0, "dim_m": 500, "dim_n": 1, "dim_k": 1,
             "bytes_in": 2000, "bytes_out": 2000, "estimated_flops": 4500},
            {"id": 1, "type": "QUICKSORT", "priority": 1, "dim_m": 1000, "dim_n": 1, "dim_k": 1,
             "bytes_in": 4000, "bytes_out": 4000, "estimated_flops": 10000},
            # 2 Small SPMV (<100k FLOPs -> Tier 1 CPU rule)
            {"id": 2, "type": "SPMV", "priority": 0, "dim_m": 1000, "dim_n": 6, "dim_k": 1,
             "bytes_in": 4000, "bytes_out": 4000, "estimated_flops": 12000},
            # 2 Large SGEMM (>1e9 FLOPs -> Tier 1 GPU rule if CUDA available)
            {"id": 3, "type": "SGEMM", "priority": 2, "dim_m": 512, "dim_n": 512, "dim_k": 512,
             "bytes_in": 512 * 512 * 8, "bytes_out": 512 * 512 * 4, "estimated_flops": 268435456},
            {"id": 4, "type": "STRING_TOKEN", "priority": 0, "dim_m": 2000, "dim_n": 1, "dim_k": 1,
             "bytes_in": 2000, "bytes_out": 4, "estimated_flops": 2000},
            {"id": 5, "type": "CONV2D", "priority": 1, "dim_m": 128, "dim_n": 128, "dim_k": 3,
             "bytes_in": 128 * 128 * 4, "bytes_out": 128 * 128 * 4, "estimated_flops": 294912},
        ]
        with open(TEST_TRACE, "w") as f:
            json.dump(tasks, f, indent=2)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(TEST_TRACE):
            os.remove(TEST_TRACE)

    def tearDown(self):
        if os.path.exists(TEST_OUTPUT):
            os.remove(TEST_OUTPUT)

    def test_cpu_completion_latency_does_not_stop_at_submission(self):
        # Force CPU mode: all tasks must run on CPU
        cmd = [BINARY, "--trace", TEST_TRACE, "--output", TEST_OUTPUT, "--force-cpu"]
        res = subprocess.run(cmd, cwd=WORKSPACE_ROOT, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"Benchmark execution failed: {res.stderr}")

        tasks, summary = parse_benchmark_csv(TEST_OUTPUT)
        self.assertEqual(len(tasks), 6)
        self.assertEqual(summary["cpu_tasks"], 6)

        for t in tasks:
            # Latency MUST be >= exec_us because completion is waited synchronously
            self.assertGreater(t["t_exec_us"], 0.0, f"Task {t['task_id']} exec time must be positive")
            self.assertGreaterEqual(
                t["t_total_latency_us"],
                t["t_exec_us"],
                f"Task {t['task_id']}: completion latency ({t['t_total_latency_us']} us) "
                f"must be >= exec time ({t['t_exec_us']} us), not stop at enqueue!"
            )
            # Decision overhead is non-negative
            self.assertGreaterEqual(t["t_decision_us"], 0.0)

    def test_inapplicable_metric_labeling_in_summary(self):
        cmd = [BINARY, "--trace", TEST_TRACE, "--output", TEST_OUTPUT, "--force-cpu"]
        res = subprocess.run(cmd, cwd=WORKSPACE_ROOT, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        # Verify inapplicable GPU transfer time is clearly labeled rather than reporting misleading 0
        self.assertIn("Avg transfer time (GPU tasks): N/A", res.stdout)
        self.assertIn("Avg total completion latency:", res.stdout)

    def test_csv_schema_and_parser_compatibility(self):
        cmd = [BINARY, "--trace", TEST_TRACE, "--output", TEST_OUTPUT]
        res = subprocess.run(cmd, cwd=WORKSPACE_ROOT, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)

        # 1. Verify CSV column headers exactly match expected schema
        with open(TEST_OUTPUT, "r", encoding="utf-8") as f:
            header_line = f.readline().strip()
        actual_columns = header_line.split(",")
        self.assertEqual(actual_columns, CSV_COLUMNS, "CSV header columns must match expected schema")

        # 2. Verify server.parser parses cleanly
        tasks, summary = parse_benchmark_csv(TEST_OUTPUT)
        self.assertEqual(len(tasks), 6)
        self.assertGreater(summary["total_wall_time_ms"], 0.0)
        self.assertGreater(summary["throughput_tasks_per_sec"], 0.0)

    def test_gpu_timing_instrumentation_when_available(self):
        cmd = [BINARY, "--trace", TEST_TRACE, "--output", TEST_OUTPUT, "--force-gpu"]
        res = subprocess.run(cmd, cwd=WORKSPACE_ROOT, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)

        tasks, summary = parse_benchmark_csv(TEST_OUTPUT)
        gpu_tasks = [t for t in tasks if t["device"] == "GPU"]
        if gpu_tasks:
            for t in gpu_tasks:
                self.assertGreater(t["t_exec_us"], 0.0, "GPU kernel time must be positive")
                self.assertGreater(t["t_transfer_us"], 0.0, "GPU transfer time must be positive")
                self.assertGreaterEqual(t["t_sync_us"], 0.0, "GPU sync time must be non-negative")
                self.assertGreaterEqual(
                    t["t_total_latency_us"],
                    t["t_exec_us"],
                    "GPU completion latency must be >= kernel exec time"
                )


if __name__ == "__main__":
    unittest.main()
