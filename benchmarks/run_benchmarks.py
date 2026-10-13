import subprocess
import json
import csv
import os
import sys
import argparse


def run_benchmark(binary, trace_file, output_file):
    cmd = [binary, "--trace", trace_file, "--output", output_file]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"Error: {result.stderr}", file=sys.stderr)
    return result.returncode == 0


def parse_results(csv_path):
    if not os.path.exists(csv_path):
        return None

    tasks = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("task_id", "").startswith("#"):
                continue
            try:
                tasks.append({
                    "task_id": int(row["task_id"]),
                    "type": row["type"],
                    "device": row["device"],
                    "t_decision_us": float(row["t_decision_us"]),
                    "t_exec_us": float(row["t_exec_us"]),
                    "t_transfer_us": float(row["t_transfer_us"]),
                    "t_total_latency_us": float(row["t_total_latency_us"])
                })
            except (ValueError, KeyError):
                continue
    return tasks


def summarize(tasks):
    if not tasks:
        return {}

    latencies = [t["t_total_latency_us"] for t in tasks]
    latencies.sort()
    n = len(latencies)

    cpu_tasks = [t for t in tasks if t["device"] == "CPU"]
    gpu_tasks = [t for t in tasks if t["device"] == "GPU"]

    return {
        "total_tasks": n,
        "cpu_tasks": len(cpu_tasks),
        "gpu_tasks": len(gpu_tasks),
        "mean_latency_us": sum(latencies) / n,
        "p95_latency_us": latencies[int(n * 0.95)],
        "p99_latency_us": latencies[int(n * 0.99)],
        "mean_decision_us": sum(t["t_decision_us"] for t in tasks) / n,
        "mean_exec_us": sum(t["t_exec_us"] for t in tasks) / n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", default="build/heterosched_bench")
    parser.add_argument("--traces-dir", default="benchmarks/traces")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    traces = [f for f in os.listdir(args.traces_dir) if f.endswith(".json")]
    all_summaries = {}

    for trace in sorted(traces):
        trace_path = os.path.join(args.traces_dir, trace)
        result_path = os.path.join(args.output_dir, trace.replace(".json", "_results.csv"))

        print(f"\n{'='*60}")
        print(f"Benchmark: {trace}")
        print(f"{'='*60}")

        success = run_benchmark(args.binary, trace_path, result_path)
        if success:
            tasks = parse_results(result_path)
            summary = summarize(tasks)
            all_summaries[trace] = summary
            print(f"  Tasks: {summary.get('total_tasks', 0)}")
            print(f"  CPU/GPU split: {summary.get('cpu_tasks', 0)}/{summary.get('gpu_tasks', 0)}")
            print(f"  Mean latency: {summary.get('mean_latency_us', 0):.1f} us")
            print(f"  P95 latency: {summary.get('p95_latency_us', 0):.1f} us")

    summary_path = os.path.join(args.output_dir, "benchmark_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
