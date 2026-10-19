import unittest
import subprocess
import os
import sys
import json

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, WORKSPACE_ROOT)
from server.parser import parse_benchmark_csv

BINARY = os.path.join(WORKSPACE_ROOT, "build", "heterosched_bench")
FAST_TRACE = "/tmp/test_fast_trace.json"
TMP_OUTPUT = "/tmp/test_cli_output.csv"


class TestAblationCLI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a fast, lightweight 10-task test trace for instantaneous CLI tests
        fast_tasks = [
            # 2 Quicksort (strictly CPU)
            {"id": 0, "type": "QUICKSORT", "priority": 0, "dim_m": 500, "dim_n": 1, "dim_k": 1,
             "bytes_in": 2000, "bytes_out": 2000, "estimated_flops": 4500},
            {"id": 1, "type": "QUICKSORT", "priority": 1, "dim_m": 1000, "dim_n": 1, "dim_k": 1,
             "bytes_in": 4000, "bytes_out": 4000, "estimated_flops": 10000},

            # 2 Small SPMV (<100k FLOPs -> Tier 1 CPU rule)
            {"id": 2, "type": "SPMV", "priority": 0, "dim_m": 1000, "dim_n": 6, "dim_k": 1,
             "bytes_in": 4000, "bytes_out": 4000, "estimated_flops": 12000},
            {"id": 3, "type": "SPMV", "priority": 1, "dim_m": 2000, "dim_n": 6, "dim_k": 1,
             "bytes_in": 8000, "bytes_out": 8000, "estimated_flops": 24000},

            # 2 Large SGEMM (>512 dim, >1e9 FLOPs -> Tier 1 GPU rule)
            {"id": 4, "type": "SGEMM", "priority": 2, "dim_m": 1024, "dim_n": 1024, "dim_k": 1024,
             "bytes_in": 1024*1024*8, "bytes_out": 1024*1024*4, "estimated_flops": 2147483648},
            {"id": 5, "type": "SGEMM", "priority": 0, "dim_m": 1024, "dim_n": 1024, "dim_k": 1024,
             "bytes_in": 1024*1024*8, "bytes_out": 1024*1024*4, "estimated_flops": 2147483648},

            # 2 Medium SGEMM (dim <= 512, intermediate FLOPs -> fallthrough to Tier 2 ML)
            {"id": 6, "type": "SGEMM", "priority": 0, "dim_m": 256, "dim_n": 256, "dim_k": 256,
             "bytes_in": 256*256*8, "bytes_out": 256*256*4, "estimated_flops": 33554432},
            {"id": 7, "type": "SGEMM", "priority": 1, "dim_m": 256, "dim_n": 256, "dim_k": 256,
             "bytes_in": 256*256*8, "bytes_out": 256*256*4, "estimated_flops": 33554432},

            # 2 Medium CONV2D (dim <= 512, intermediate FLOPs -> fallthrough to Tier 2 ML)
            {"id": 8, "type": "CONV2D", "priority": 0, "dim_m": 256, "dim_n": 256, "dim_k": 3,
             "bytes_in": 256*256*4, "bytes_out": 256*256*4, "estimated_flops": 1179648},
            {"id": 9, "type": "CONV2D", "priority": 2, "dim_m": 256, "dim_n": 256, "dim_k": 3,
             "bytes_in": 256*256*4, "bytes_out": 256*256*4, "estimated_flops": 1179648},
        ]
        with open(FAST_TRACE, "w") as f:
            json.dump(fast_tasks, f, indent=2)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(FAST_TRACE):
            os.remove(FAST_TRACE)

    def tearDown(self):
        if os.path.exists(TMP_OUTPUT):
            os.remove(TMP_OUTPUT)

    def run_bench(self, extra_args):
        cmd = [BINARY, "--trace", FAST_TRACE, "--output", TMP_OUTPUT] + extra_args
        res = subprocess.run(cmd, cwd=WORKSPACE_ROOT, capture_output=True, text=True)
        return res

    def test_unknown_flag_rejection(self):
        res = self.run_bench(["--unknown-flag-123"])
        self.assertNotEqual(res.returncode, 0, "Unknown flag must cause non-zero exit code")
        self.assertIn("Unknown argument", res.stderr + res.stdout)

    def test_conflicting_force_flags_rejection(self):
        res = self.run_bench(["--force-cpu", "--force-gpu"])
        self.assertNotEqual(res.returncode, 0, "Conflicting --force-cpu and --force-gpu must fail")
        self.assertIn("Conflicting", res.stderr + res.stdout)

    def test_conflicting_force_and_static_rule_rejection(self):
        res = self.run_bench(["--force-cpu", "--static-rule"])
        self.assertNotEqual(res.returncode, 0, "Conflicting --force-cpu and --static-rule must fail")
        self.assertIn("Conflicting", res.stderr + res.stdout)

    def test_conflicting_force_and_ablation_rejection(self):
        res = self.run_bench(["--force-cpu", "--tier1"])
        self.assertNotEqual(res.returncode, 0, "Forced mode combined with ablation flag must fail")
        self.assertIn("cannot be combined", res.stderr + res.stdout)

    def test_invalid_dim_threshold_rejection(self):
        res = self.run_bench(["--static-rule", "--dim-threshold", "-10"])
        self.assertNotEqual(res.returncode, 0, "Negative --dim-threshold must fail")

        res2 = self.run_bench(["--static-rule", "--dim-threshold", "not_a_number"])
        self.assertNotEqual(res2.returncode, 0, "Non-numeric --dim-threshold must fail")

    def test_default_behavior_preservation(self):
        # When no optional flags are passed, normal full pipeline must execute
        res = self.run_bench([])
        self.assertEqual(res.returncode, 0, f"Default run failed: {res.stderr}")
        self.assertIn("Mode: ADAPTIVE[tier1,tier2,dynamic_penalties,cross_layer]", res.stdout)
        tasks, summary = parse_benchmark_csv(TMP_OUTPUT)
        self.assertEqual(len(tasks), 10)
        self.assertEqual(summary["total_tasks"], 10)
        # 2 Quicksort (CPU) + 2 Small SPMV (Tier 1 CPU) = 4 CPU. 6 Large/Medium SGEMM/CONV2D go to GPU.
        self.assertEqual(summary["cpu_tasks"], 4)
        self.assertEqual(summary["gpu_tasks"], 6)

    def test_force_cpu_mode(self):
        res = self.run_bench(["--force-cpu"])
        self.assertEqual(res.returncode, 0, f"--force-cpu failed: {res.stderr}")
        self.assertIn("Mode: CPU_ONLY", res.stdout)
        tasks, summary = parse_benchmark_csv(TMP_OUTPUT)
        self.assertEqual(len(tasks), 10)
        self.assertEqual(summary["cpu_tasks"], 10)
        self.assertEqual(summary["gpu_tasks"], 0)

    def test_force_gpu_mode_with_safe_unsupported_routing(self):
        res = self.run_bench(["--force-gpu"])
        self.assertEqual(res.returncode, 0, f"--force-gpu failed: {res.stderr}")
        self.assertIn("Mode: GPU_ONLY", res.stdout)
        tasks, summary = parse_benchmark_csv(TMP_OUTPUT)
        self.assertEqual(len(tasks), 10)
        # 2 Quicksort (unsupported on GPU) must safely execute on CPU; 8 GPU-capable tasks run on GPU.
        self.assertEqual(summary["cpu_tasks"], 2)
        self.assertEqual(summary["gpu_tasks"], 8)
        for t in tasks:
            if t["type"] == "QUICKSORT":
                self.assertEqual(t["device"], "CPU", "QUICKSORT must strictly be routed to CPU")

    def test_static_rule_mode(self):
        res = self.run_bench(["--static-rule", "--dim-threshold", "512"])
        self.assertEqual(res.returncode, 0, f"--static-rule failed: {res.stderr}")
        self.assertIn("Mode: STATIC_RULE(threshold=512)", res.stdout)
        tasks, summary = parse_benchmark_csv(TMP_OUTPUT)
        self.assertEqual(len(tasks), 10)
        # Tasks 2, 3, 4, 5 have dim_m > 512 -> GPU (4 tasks).
        # Tasks 6, 7, 8, 9 have dim_m <= 512 (4 tasks) + 2 Quicksort -> CPU (6 tasks).
        self.assertEqual(summary["cpu_tasks"], 6)
        self.assertEqual(summary["gpu_tasks"], 4)

    def test_baseline_no_scheduling(self):
        res = self.run_bench(["--baseline"])
        self.assertEqual(res.returncode, 0, f"--baseline failed: {res.stderr}")
        self.assertIn("Mode: ADAPTIVE[baseline]", res.stdout)
        tasks, summary = parse_benchmark_csv(TMP_OUTPUT)
        self.assertEqual(len(tasks), 10)
        # 8 supported tasks alternate 4 CPU, 4 GPU + 2 Quicksort to CPU = 6 CPU, 4 GPU
        self.assertEqual(summary["cpu_tasks"], 6)
        self.assertEqual(summary["gpu_tasks"], 4)

    def test_tier1_only_ablation(self):
        res = self.run_bench(["--tier1"])
        self.assertEqual(res.returncode, 0, f"--tier1 failed: {res.stderr}")
        self.assertIn("Mode: ADAPTIVE[tier1]", res.stdout)
        tasks, summary = parse_benchmark_csv(TMP_OUTPUT)
        self.assertEqual(len(tasks), 10)
        # Tier 1 rules:
        # - 2 Quicksort: CPU
        # - 2 Small SPMV: CPU (rule: flops < 100k)
        # - 2 Large SGEMM: GPU (rule: flops > 1e9)
        # - 4 Medium tasks: Fallback to CPU when Tier 2 is disabled
        # Total: 8 CPU, 2 GPU
        self.assertEqual(summary["cpu_tasks"], 8)
        self.assertEqual(summary["gpu_tasks"], 2)

    def test_tier1_plus_tier2_ablation(self):
        res = self.run_bench(["--tier1", "--tier2"])
        self.assertEqual(res.returncode, 0, f"--tier1 --tier2 failed: {res.stderr}")
        self.assertIn("Mode: ADAPTIVE[tier1,tier2]", res.stdout)
        tasks, summary = parse_benchmark_csv(TMP_OUTPUT)
        self.assertEqual(len(tasks), 10)
        # 4 CPU, 6 GPU
        self.assertEqual(summary["cpu_tasks"], 4)
        self.assertEqual(summary["gpu_tasks"], 6)


if __name__ == "__main__":
    unittest.main()
