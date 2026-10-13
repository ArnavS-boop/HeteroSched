import json
import random
import os


def generate_mixed_trace(output_path="benchmarks/traces/mixed_trace.json", count=200):
    random.seed(42)
    workloads = []

    for i in range(count):
        w_type = random.choice(["SGEMM", "QUICKSORT", "SPMV", "CONV2D"])

        if w_type == "SGEMM":
            dim = random.choice([256, 512, 1024, 2048])
            workloads.append({
                "id": i,
                "type": w_type,
                "priority": random.choice([0, 1, 2]),
                "dim_m": dim,
                "dim_n": dim,
                "dim_k": dim,
                "bytes_in": 2 * dim * dim * 4,
                "bytes_out": dim * dim * 4,
                "estimated_flops": 2 * (dim ** 3)
            })
        elif w_type == "QUICKSORT":
            n = random.randint(10000, 500000)
            workloads.append({
                "id": i,
                "type": w_type,
                "priority": random.choice([0, 1]),
                "dim_m": n,
                "dim_n": 1,
                "dim_k": 1,
                "bytes_in": n * 4,
                "bytes_out": n * 4,
                "estimated_flops": int(n * 19)
            })
        elif w_type == "CONV2D":
            dim = random.choice([256, 512, 1024])
            ksize = random.choice([3, 5, 7])
            workloads.append({
                "id": i,
                "type": w_type,
                "priority": random.choice([0, 1, 2]),
                "dim_m": dim,
                "dim_n": dim,
                "dim_k": ksize,
                "bytes_in": dim * dim * 4,
                "bytes_out": dim * dim * 4,
                "estimated_flops": dim * dim * ksize * ksize * 2
            })
        elif w_type == "SPMV":
            n = random.randint(1000, 50000)
            nnz = random.randint(5, 10)
            workloads.append({
                "id": i,
                "type": w_type,
                "priority": random.choice([0, 1]),
                "dim_m": n,
                "dim_n": nnz,
                "dim_k": 1,
                "bytes_in": n * 4,
                "bytes_out": n * 4,
                "estimated_flops": n * nnz * 2
            })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(workloads, f, indent=2)
    print(f"Saved {output_path} with {len(workloads)} tasks")


def generate_bursty_trace(output_path="benchmarks/traces/bursty_trace.json", count=150):
    random.seed(123)
    workloads = []

    for i in range(count):
        burst = (i % 30) < 10
        if burst:
            dim = random.choice([1024, 2048, 4096])
            workloads.append({
                "id": i,
                "type": "SGEMM",
                "priority": 2,
                "dim_m": dim, "dim_n": dim, "dim_k": dim,
                "bytes_in": 2 * dim * dim * 4,
                "bytes_out": dim * dim * 4,
                "estimated_flops": 2 * (dim ** 3)
            })
        else:
            n = random.randint(50000, 200000)
            workloads.append({
                "id": i,
                "type": "QUICKSORT",
                "priority": 0,
                "dim_m": n, "dim_n": 1, "dim_k": 1,
                "bytes_in": n * 4, "bytes_out": n * 4,
                "estimated_flops": int(n * 19)
            })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(workloads, f, indent=2)
    print(f"Saved {output_path} with {len(workloads)} tasks")


def generate_pcie_heavy_trace(output_path="benchmarks/traces/pcie_heavy_trace.json", count=100):
    random.seed(456)
    workloads = []

    for i in range(count):
        dim = random.choice([512, 1024, 2048, 4096])
        workloads.append({
            "id": i,
            "type": "SGEMM",
            "priority": random.choice([1, 2]),
            "dim_m": dim, "dim_n": dim, "dim_k": dim,
            "bytes_in": 2 * dim * dim * 4,
            "bytes_out": dim * dim * 4,
            "estimated_flops": 2 * (dim ** 3)
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(workloads, f, indent=2)
    print(f"Saved {output_path} with {len(workloads)} tasks")


if __name__ == "__main__":
    generate_mixed_trace()
    generate_bursty_trace()
    generate_pcie_heavy_trace()
