import subprocess
import json
import os
import sys
import argparse

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, WORKSPACE_ROOT)
from server.parser import parse_benchmark_csv


ABLATION_CONFIGS = [
    {
        "id": "ABLATION_0",
        "label": "Baseline (Round-Robin CPU/GPU)",
        "flags": ["--baseline"]
    },
    {
        "id": "ABLATION_1",
        "label": "Tier-1 Heuristic Only",
        "flags": ["--tier1"]
    },
    {
        "id": "ABLATION_2",
        "label": "Tier-1 + Tier-2 ML",
        "flags": ["--tier1", "--tier2"]
    },
    {
        "id": "ABLATION_3",
        "label": "Tier-1 + Tier-2 + Dynamic Penalties",
        "flags": ["--tier1", "--tier2", "--dynamic-penalties"]
    },
    {
        "id": "HETEROSCHED_FULL",
        "label": "Full HeteroSched Pipeline",
        "flags": ["--tier1", "--tier2", "--dynamic-penalties", "--cross-layer"]
    }
]

BASELINE_CONFIGS = [
    {
        "id": "CPU_ONLY",
        "label": "CPU-Only Static",
        "flags": ["--force-cpu"]
    },
    {
        "id": "GPU_ONLY",
        "label": "GPU-Only Static",
        "flags": ["--force-gpu"]
    },
    {
        "id": "STATIC_RULE",
        "label": "Static Rule-Based (dim > 512)",
        "flags": ["--static-rule", "--dim-threshold", "512"]
    }
]


def run_config(binary, trace, output, extra_flags):
    cmd = [binary, "--trace", trace, "--output", output] + extra_flags
    print(f"  Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout.strip():
        for line in result.stdout.strip().split("\n"):
            print(f"    {line}")
    if result.returncode != 0:
        print(f"    ERROR: {result.stderr.strip()}")
    return result.returncode == 0


def main():
    parser_arg = argparse.ArgumentParser()
    parser_arg.add_argument("--binary", default="build/heterosched_bench")
    parser_arg.add_argument("--trace", default="benchmarks/traces/mixed_trace.json")
    parser_arg.add_argument("--output-dir", default="/tmp/ablation_results")
    args = parser_arg.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_configs = BASELINE_CONFIGS + ABLATION_CONFIGS
    results = {}
    table_rows = []

    for config in all_configs:
        print(f"\n{'='*60}")
        print(f"Config: {config['id']} — {config['label']}")
        print(f"{'='*60}")

        output_file = os.path.join(args.output_dir, f"{config['id']}_results.csv")
        success = run_config(args.binary, args.trace, output_file, config["flags"])

        config_summary = {
            "label": config["label"],
            "success": success,
            "output_file": output_file
        }

        if success and os.path.exists(output_file):
            try:
                tasks, summary = parse_benchmark_csv(output_file)
                config_summary.update({
                    "total_tasks": summary.get("total_tasks", len(tasks)),
                    "cpu_tasks": summary.get("cpu_tasks", 0),
                    "gpu_tasks": summary.get("gpu_tasks", 0),
                    "total_wall_time_ms": summary.get("total_wall_time_ms", 0.0),
                    "throughput_tasks_per_sec": summary.get("throughput_tasks_per_sec", 0.0),
                    "avg_decision_us": summary.get("avg_decision_us", 0.0),
                    "avg_exec_us": summary.get("avg_exec_us", 0.0)
                })
                table_rows.append({
                    "id": config["id"],
                    "flags": " ".join(config["flags"]),
                    "total": summary.get("total_tasks", len(tasks)),
                    "cpu": summary.get("cpu_tasks", 0),
                    "gpu": summary.get("gpu_tasks", 0),
                    "time_ms": summary.get("total_wall_time_ms", 0.0),
                    "throughput": summary.get("throughput_tasks_per_sec", 0.0)
                })
            except Exception as e:
                print(f"    Failed to parse summary: {e}")

        results[config["id"]] = config_summary

    summary_path = os.path.join(args.output_dir, "ablation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAblation summary written to {summary_path}")

    print("\n" + "=" * 92)
    print(f"{'Config ID':<22} | {'Flags':<30} | {'Total':<6} | {'CPU':<5} | {'GPU':<5} | {'Time (ms)':<10}")
    print("=" * 92)
    for r in table_rows:
        print(f"{r['id']:<22} | {r['flags']:<30} | {r['total']:<6} | {r['cpu']:<5} | {r['gpu']:<5} | {r['time_ms']:<10.2f}")
    print("=" * 92 + "\n")


if __name__ == "__main__":
    main()
