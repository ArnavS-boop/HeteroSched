import subprocess
import json
import os
import sys
import argparse


ABLATION_CONFIGS = [
    {
        "id": "ABLATION_0",
        "label": "Baseline (No Scheduling)",
        "flags": []
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
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", default="build/heterosched_bench")
    parser.add_argument("--trace", default="benchmarks/traces/mixed_trace.json")
    parser.add_argument("--output-dir", default="results/ablations")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_configs = BASELINE_CONFIGS + ABLATION_CONFIGS
    results = {}

    for config in all_configs:
        print(f"\n{'='*60}")
        print(f"Config: {config['id']} — {config['label']}")
        print(f"{'='*60}")

        output_file = os.path.join(args.output_dir, f"{config['id']}_results.csv")
        success = run_config(args.binary, args.trace, output_file, config["flags"])

        results[config["id"]] = {
            "label": config["label"],
            "success": success,
            "output_file": output_file
        }

    summary_path = os.path.join(args.output_dir, "ablation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAblation summary written to {summary_path}")


if __name__ == "__main__":
    main()
