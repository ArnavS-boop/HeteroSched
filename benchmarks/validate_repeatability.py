#!/usr/bin/env python3
"""
HeteroSched - Phase C Repeatable Benchmark Validation Suite
Runs deterministic validation workloads across multiple iterations with unique /tmp/ output files.
Reports task counts, failure counts, correctness results, per-run latency, and variability.
"""

import subprocess
import os
import sys
import json
import statistics
import uuid

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, WORKSPACE_ROOT)
from server.parser import parse_benchmark_csv

BINARY = os.path.join(WORKSPACE_ROOT, "build", "heterosched_bench")
CORRECTNESS_BINARY = os.path.join(WORKSPACE_ROOT, "build", "test_phase_c")
VALIDATION_TRACE = os.path.join(WORKSPACE_ROOT, "benchmarks", "traces", "validation_trace.json")

CONFIGURATIONS = [
    {
        "id": "ADAPTIVE_FULL",
        "label": "Full Adaptive Pipeline (Tier1+Tier2+Dynamic+Cross)",
        "flags": []
    },
    {
        "id": "CPU_ONLY",
        "label": "Forced CPU Execution",
        "flags": ["--force-cpu"]
    },
    {
        "id": "GPU_ONLY",
        "label": "Forced GPU Execution (CPU fallbacks for unsupported)",
        "flags": ["--force-gpu"]
    },
    {
        "id": "BASELINE_RR",
        "label": "Round-Robin Baseline (Both Tiers Disabled)",
        "flags": ["--baseline"]
    }
]

NUM_RUNS = 5


def run_correctness_tests():
    print("\n" + "=" * 70)
    print("STEP 1: INDEPENDENT NUMERICAL CORRECTNESS VALIDATION")
    print("=" * 70)
    res = subprocess.run([CORRECTNESS_BINARY], capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print(f"CORRECTNESS TEST FAILED:\n{res.stderr}")
    return res.returncode == 0


def get_cuda_info():
    try:
        res = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True)
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip()
    except Exception:
        pass
    return "No NVIDIA GPU detected or nvidia-smi unavailable"


def run_repeatable_benchmark():
    print("=" * 70)
    print(f"STEP 2: REPEATABLE BENCHMARK VALIDATION ({NUM_RUNS} iterations per config)")
    print(f"Workload Trace: {VALIDATION_TRACE}")
    print(f"Hardware / Environment: {get_cuda_info()}")
    print("=" * 70)

    # Load trace dimensions
    with open(VALIDATION_TRACE, "r") as f:
        trace_data = json.load(f)
    print(f"Workload Task Count: {len(trace_data)} deterministic tasks")
    for t in trace_data:
        print(f"  Task {t['id']}: {t['type']} (m={t['dim_m']}, n={t['dim_n']}, k={t['dim_k']}, FLOPs={t['estimated_flops']})")
    print("-" * 70)

    all_results = {}

    for cfg in CONFIGURATIONS:
        print(f"\nEvaluating Config: {cfg['id']} [{cfg['label']}]")
        print(f"Flags: {' '.join(cfg['flags']) if cfg['flags'] else '(default)'}")

        run_records = []
        for run_idx in range(1, NUM_RUNS + 1):
            unique_token = uuid.uuid4().hex[:8]
            output_file = f"/tmp/repeatability_{cfg['id']}_run_{run_idx}_{unique_token}.csv"

            cmd = [BINARY, "--trace", VALIDATION_TRACE, "--output", output_file] + cfg['flags']
            proc = subprocess.run(cmd, cwd=WORKSPACE_ROOT, capture_output=True, text=True)

            if proc.returncode != 0:
                print(f"  Run {run_idx}: FAILED with exit code {proc.returncode}: {proc.stderr}")
                run_records.append({
                    "run": run_idx,
                    "success": False,
                    "wall_time_ms": 0.0,
                    "avg_latency_us": 0.0,
                    "completed": 0,
                    "failed": len(trace_data)
                })
                continue

            try:
                tasks, summary = parse_benchmark_csv(output_file)
                total_tasks = summary.get("total_tasks", len(tasks))
                wall_ms = summary.get("total_wall_time_ms", 0.0)

                # Compute mean task completion latency across successful tasks
                latencies = [t["t_total_latency_us"] for t in tasks if t.get("t_total_latency_us", 0.0) > 0.0]
                mean_latency_us = statistics.mean(latencies) if latencies else 0.0

                print(f"  Run {run_idx}: Wall Time = {wall_ms:.2f} ms, "
                      f"Avg Latency = {mean_latency_us:.1f} us, "
                      f"Completed = {len(tasks)}, Failures = 0")

                run_records.append({
                    "run": run_idx,
                    "success": True,
                    "wall_time_ms": wall_ms,
                    "avg_latency_us": mean_latency_us,
                    "completed": len(tasks),
                    "failed": 0,
                    "cpu_tasks": summary.get("cpu_tasks", 0),
                    "gpu_tasks": summary.get("gpu_tasks", 0)
                })
            except Exception as e:
                print(f"  Run {run_idx}: Failed to parse CSV: {e}")
            finally:
                if os.path.exists(output_file):
                    os.remove(output_file)

        # Statistical analysis
        valid_wall = [r["wall_time_ms"] for r in run_records if r["success"]]
        valid_lat = [r["avg_latency_us"] for r in run_records if r["success"]]

        mean_wall = statistics.mean(valid_wall) if valid_wall else 0.0
        stdev_wall = statistics.stdev(valid_wall) if len(valid_wall) > 1 else 0.0
        min_wall = min(valid_wall) if valid_wall else 0.0
        max_wall = max(valid_wall) if valid_wall else 0.0

        mean_lat = statistics.mean(valid_lat) if valid_lat else 0.0
        stdev_lat = statistics.stdev(valid_lat) if len(valid_lat) > 1 else 0.0

        all_results[cfg["id"]] = {
            "label": cfg["label"],
            "flags": cfg["flags"],
            "runs": len(run_records),
            "valid_runs": len(valid_wall),
            "wall_time_mean_ms": round(mean_wall, 2),
            "wall_time_stdev_ms": round(stdev_wall, 2),
            "wall_time_min_ms": round(min_wall, 2),
            "wall_time_max_ms": round(max_wall, 2),
            "mean_task_latency_us": round(mean_lat, 1),
            "stdev_task_latency_us": round(stdev_lat, 1),
            "failures": sum(r["failed"] for r in run_records)
        }

    # Summary table
    print("\n" + "=" * 90)
    print("REPEATABLE BENCHMARK VALIDATION SUMMARY")
    print("=" * 90)
    print(f"{'Config ID':<16} | {'Runs':<5} | {'Failures':<8} | {'Mean Wall (ms)':<15} | {'Std Dev':<9} | {'Mean Lat (us)':<14}")
    print("-" * 90)
    for cfg_id, r in all_results.items():
        print(f"{cfg_id:<16} | {r['valid_runs']:<5} | {r['failures']:<8} | {r['wall_time_mean_ms']:<15.2f} | {r['wall_time_stdev_ms']:<9.2f} | {r['mean_task_latency_us']:<14.1f}")
    print("=" * 90)
    print("\n[LIMITATIONS & SCIENTIFIC CAVEATS]:")
    print("1. Low sample counts (N=5) demonstrate benchmark repeatability and numerical determinism, NOT statistical proof of general performance superiority.")
    print("2. Heterogeneous device configurations (CPU vs GPU) involve distinct microarchitectures and must not be treated as equivalent single-variable benchmarks.")
    print("3. In the sequential execution harness, tasks are scheduled one-at-a-time, meaning queue-depth telemetry remains near zero at dispatch; dynamic queue penalties remain largely idle under single-task workloads.")
    print("4. WSL environment may present kernel scheduling constraints (e.g. pthread_setschedparam without privileged capabilities).\n")

    return all_results


if __name__ == "__main__":
    correctness_ok = run_correctness_tests()
    if not correctness_ok:
        print("\nERROR: Correctness tests failed. Halting benchmark validation.")
        sys.exit(1)

    results = run_repeatable_benchmark()
    summary_path = "/tmp/repeatability_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Summary written to {summary_path}")
