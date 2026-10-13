import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os


def load_results(csv_path):
    rows = []
    with open(csv_path) as f:
        lines = f.readlines()

    header = None
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if header is None:
            header = line.split(",")
            continue
        parts = line.split(",")
        if len(parts) == len(header):
            rows.append(dict(zip(header, parts)))

    df = pd.DataFrame(rows)
    numeric_cols = ["t_decision_us", "t_transfer_us", "t_exec_us",
                    "t_sync_us", "t_total_latency_us"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def plot_latency_distribution(df, output_dir):
    fig, ax = plt.subplots(figsize=(10, 6))

    cpu_tasks = df[df["device"] == "CPU"]["t_exec_us"].dropna()
    gpu_tasks = df[df["device"] == "GPU"]["t_exec_us"].dropna()

    bins = np.logspace(0, 8, 50)
    if len(cpu_tasks) > 0:
        ax.hist(cpu_tasks, bins=bins, alpha=0.7, label=f"CPU ({len(cpu_tasks)} tasks)", color="#2196F3")
    if len(gpu_tasks) > 0:
        ax.hist(gpu_tasks, bins=bins, alpha=0.7, label=f"GPU ({len(gpu_tasks)} tasks)", color="#FF5722")

    ax.set_xscale("log")
    ax.set_xlabel("Execution Time (µs)")
    ax.set_ylabel("Task Count")
    ax.set_title("Execution Latency Distribution by Device")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "latency_distribution.png"), dpi=150)
    plt.close()


def plot_device_breakdown(df, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    device_counts = df["device"].value_counts()
    colors = ["#2196F3", "#FF5722"]
    axes[0].pie(device_counts.values, labels=device_counts.index,
                autopct="%1.1f%%", colors=colors[:len(device_counts)])
    axes[0].set_title("Device Placement Distribution")

    if "type" in df.columns:
        task_device = df.groupby(["type", "device"]).size().unstack(fill_value=0)
        task_device.plot(kind="bar", ax=axes[1], color=colors[:task_device.shape[1]])
        axes[1].set_title("Device Assignment by Task Type")
        axes[1].set_ylabel("Count")
        axes[1].set_xlabel("Task Type")
        axes[1].legend(title="Device")
        axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "device_breakdown.png"), dpi=150)
    plt.close()


def plot_overhead_breakdown(df, output_dir):
    fig, ax = plt.subplots(figsize=(10, 6))

    categories = ["t_decision_us", "t_transfer_us", "t_exec_us"]
    labels = ["Decision", "Transfer", "Execution"]
    means = [df[c].mean() for c in categories if c in df.columns]
    labels = [l for l, c in zip(labels, categories) if c in df.columns]

    bars = ax.bar(labels, means, color=["#4CAF50", "#FFC107", "#9C27B0"])
    ax.set_ylabel("Mean Time (µs)")
    ax.set_title("Overhead Breakdown (Mean per Task)")
    ax.grid(True, axis="y", alpha=0.3)

    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.1f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "overhead_breakdown.png"), dpi=150)
    plt.close()


def plot_decision_overhead_timeline(df, output_dir):
    if "t_decision_us" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(12, 4))
    decision_times = df["t_decision_us"].dropna().values

    ax.plot(range(len(decision_times)), decision_times, linewidth=0.5, alpha=0.8, color="#009688")
    ax.axhline(y=5.0, color="red", linestyle="--", linewidth=1, label="Tier-1 target (5 µs)")
    ax.axhline(y=100.0, color="orange", linestyle="--", linewidth=1, label="Tier-2 target (100 µs)")
    ax.set_xlabel("Task Index")
    ax.set_ylabel("Decision Overhead (µs)")
    ax.set_title("Scheduling Decision Overhead per Task")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "decision_timeline.png"), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results.csv")
    parser.add_argument("--output-dir", default="artifacts")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = load_results(args.input)
    if df.empty:
        print("No data loaded. Check input file.")
        return

    print(f"Loaded {len(df)} task results from {args.input}")

    plot_latency_distribution(df, args.output_dir)
    plot_device_breakdown(df, args.output_dir)
    plot_overhead_breakdown(df, args.output_dir)
    plot_decision_overhead_timeline(df, args.output_dir)

    print(f"Plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
