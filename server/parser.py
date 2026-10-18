"""
Safe parser for HeteroSched benchmark CSV results.
Handles trailing comment lines (e.g. '# total_time_ms,802.499'), validates fields,
computes reliable aggregate metrics, and exports clean CSVs.

Note on timing fields:
- In HeteroSched (main.cpp), 't_total_latency_us' records the elapsed time
  during runtime.SubmitTask(&task). Because task dispatch is asynchronous,
  this metric represents 'Task Submission Latency' (enqueue duration), NOT
  the end-to-end task completion latency. We clearly expose this as
  't_submission_latency_us' while preserving the original column name.
"""

import csv
import io
import os
from typing import Dict, Any, List, Optional, Tuple


CSV_COLUMNS = [
    "task_id",
    "type",
    "priority",
    "dim_m",
    "dim_n",
    "dim_k",
    "device",
    "t_decision_us",
    "t_transfer_us",
    "t_exec_us",
    "t_sync_us",
    "t_total_latency_us",
    "stream_id",
    "cpu_core",
]


def parse_benchmark_csv(filepath: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Safely reads a benchmark results CSV file.
    Returns:
        (tasks, summary_dict)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Results file not found: {filepath}")

    total_time_ms = 0.0
    tasks: List[Dict[str, Any]] = []

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    csv_lines: List[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            # Check for trailing summary metadata: # total_time_ms,802.499
            parts = stripped.lstrip("#").strip().split(",")
            if len(parts) >= 2 and parts[0].strip() == "total_time_ms":
                try:
                    total_time_ms = float(parts[1].strip())
                except ValueError:
                    pass
            continue
        csv_lines.append(line)

    if not csv_lines:
        return [], {"error": "Empty results file"}

    reader = csv.DictReader(csv_lines)
    for row in reader:
        # Extra safety check against comment or malformed rows
        task_id_raw = row.get("task_id", "").strip()
        if not task_id_raw or task_id_raw.startswith("#"):
            continue

        try:
            task_id = int(task_id_raw)
            task_type = str(row.get("type", "UNKNOWN")).strip()
            priority = int(row.get("priority", 0))
            dim_m = int(row.get("dim_m", 0))
            dim_n = int(row.get("dim_n", 0))
            dim_k = int(row.get("dim_k", 0))
            device = str(row.get("device", "UNKNOWN")).strip().upper()
            t_decision_us = float(row.get("t_decision_us", 0.0))
            t_transfer_us = float(row.get("t_transfer_us", 0.0))
            t_exec_us = float(row.get("t_exec_us", 0.0))
            t_sync_us = float(row.get("t_sync_us", 0.0))
            t_raw_latency = float(row.get("t_total_latency_us", 0.0))
            stream_id = int(row.get("stream_id", 0))
            cpu_core = int(row.get("cpu_core", 0))

            task_dict = {
                "task_id": task_id,
                "type": task_type,
                "priority": priority,
                "dim_m": dim_m,
                "dim_n": dim_n,
                "dim_k": dim_k,
                "device": device,
                "t_decision_us": t_decision_us,
                "t_transfer_us": t_transfer_us,
                "t_exec_us": t_exec_us,
                "t_sync_us": t_sync_us,
                # Explicitly documented submission latency vs raw name
                "t_submission_latency_us": t_raw_latency,
                "t_total_latency_us": t_raw_latency,
                "stream_id": stream_id,
                "cpu_core": cpu_core,
            }
            tasks.append(task_dict)
        except (ValueError, KeyError):
            continue

    summary = compute_summary(tasks, total_time_ms)
    return tasks, summary


def compute_summary(tasks: List[Dict[str, Any]], total_time_ms: float) -> Dict[str, Any]:
    """
    Computes statistical and scheduling metrics strictly based on parsed task data.
    """
    n = len(tasks)
    if n == 0:
        return {
            "total_tasks": 0,
            "cpu_tasks": 0,
            "gpu_tasks": 0,
            "cpu_ratio": 0.0,
            "gpu_ratio": 0.0,
            "total_wall_time_ms": total_time_ms,
            "throughput_tasks_per_sec": 0.0,
            "avg_decision_us": 0.0,
            "avg_exec_us": 0.0,
            "avg_transfer_us_gpu": 0.0,
            "avg_submission_latency_us": 0.0,
            "task_type_stats": {},
            "stream_distribution": {},
            "cpu_core_distribution": {},
        }

    cpu_tasks = [t for t in tasks if t["device"] == "CPU"]
    gpu_tasks = [t for t in tasks if t["device"] == "GPU"]

    cpu_count = len(cpu_tasks)
    gpu_count = len(gpu_tasks)

    total_decision = sum(t["t_decision_us"] for t in tasks)
    total_exec = sum(t["t_exec_us"] for t in tasks)
    total_gpu_transfer = sum(t["t_transfer_us"] for t in gpu_tasks)
    total_submission = sum(t["t_submission_latency_us"] for t in tasks)

    avg_decision_us = total_decision / n
    avg_exec_us = total_exec / n
    avg_transfer_us_gpu = (total_gpu_transfer / gpu_count) if gpu_count > 0 else 0.0
    avg_submission_latency_us = total_submission / n

    # Percentiles for execution time
    exec_sorted = sorted([t["t_exec_us"] for t in tasks])
    p50_exec = exec_sorted[int(n * 0.50)]
    p95_exec = exec_sorted[min(n - 1, int(n * 0.95))]
    p99_exec = exec_sorted[min(n - 1, int(n * 0.99))]

    # Throughput
    throughput = (n / (total_time_ms / 1000.0)) if total_time_ms > 0 else 0.0

    # Kernel breakdown
    task_types = sorted(list(set(t["type"] for t in tasks)))
    type_stats = {}
    for tt in task_types:
        subset = [t for t in tasks if t["type"] == tt]
        c_sub = sum(1 for t in subset if t["device"] == "CPU")
        g_sub = sum(1 for t in subset if t["device"] == "GPU")
        mean_exec = sum(t["t_exec_us"] for t in subset) / len(subset)
        type_stats[tt] = {
            "total": len(subset),
            "cpu": c_sub,
            "gpu": g_sub,
            "avg_exec_us": round(mean_exec, 2),
        }

    # Stream distribution (GPU)
    stream_counts = {}
    for t in gpu_tasks:
        s_id = str(t["stream_id"])
        stream_counts[s_id] = stream_counts.get(s_id, 0) + 1

    # CPU core distribution (CPU)
    core_counts = {}
    for t in cpu_tasks:
        c_id = str(t["cpu_core"])
        core_counts[c_id] = core_counts.get(c_id, 0) + 1

    return {
        "total_tasks": n,
        "cpu_tasks": cpu_count,
        "gpu_tasks": gpu_count,
        "cpu_ratio": round((cpu_count / n) * 100, 1),
        "gpu_ratio": round((gpu_count / n) * 100, 1),
        "total_wall_time_ms": round(total_time_ms, 3),
        "throughput_tasks_per_sec": round(throughput, 2),
        "avg_decision_us": round(avg_decision_us, 4),
        "avg_exec_us": round(avg_exec_us, 2),
        "avg_transfer_us_gpu": round(avg_transfer_us_gpu, 2),
        "avg_submission_latency_us": round(avg_submission_latency_us, 2),
        "p50_exec_us": round(p50_exec, 2),
        "p95_exec_us": round(p95_exec, 2),
        "p99_exec_us": round(p99_exec, 2),
        "task_type_stats": type_stats,
        "stream_distribution": stream_counts,
        "cpu_core_distribution": core_counts,
    }


def generate_clean_csv(tasks: List[Dict[str, Any]]) -> str:
    """
    Generates standard, cleaned CSV formatted text with the exact columns,
    excluding any comment or summary trailing lines for pure tabular processing.
    """
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=CSV_COLUMNS)
    writer.writeheader()
    for t in tasks:
        row = {
            "task_id": t["task_id"],
            "type": t["type"],
            "priority": t["priority"],
            "dim_m": t["dim_m"],
            "dim_n": t["dim_n"],
            "dim_k": t["dim_k"],
            "device": t["device"],
            "t_decision_us": t["t_decision_us"],
            "t_transfer_us": t["t_transfer_us"],
            "t_exec_us": t["t_exec_us"],
            "t_sync_us": t["t_sync_us"],
            "t_total_latency_us": t["t_total_latency_us"],
            "stream_id": t["stream_id"],
            "cpu_core": t["cpu_core"],
        }
        writer.writerow(row)
    return output.getvalue()
