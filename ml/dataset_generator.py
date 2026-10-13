import numpy as np
import pandas as pd


def generate_training_data(output_file="ml/dataset.csv", samples=10000):
    np.random.seed(42)
    data = []

    for i in range(samples):
        task_type = np.random.choice(["SGEMM", "CONV2D", "SPMV", "QUICKSORT"])
        if task_type == "SGEMM":
            dim = int(np.random.choice([64, 128, 256, 512, 1024, 2048, 4096]))
            flops = 2 * (dim ** 3)
            bytes_in = 2 * (dim * dim * 4)
            bytes_out = dim * dim * 4
            cpu_time = (flops / 5e9) * 1e6 * np.random.uniform(0.9, 1.1)
            gpu_time = (flops / 14e12) * 1e6 * np.random.uniform(0.9, 1.1)
        elif task_type == "CONV2D":
            dim = int(np.random.choice([128, 256, 512, 1024, 2048]))
            ksize = int(np.random.choice([3, 5, 7]))
            flops = dim * dim * ksize * ksize * 2
            bytes_in = dim * dim * 4
            bytes_out = dim * dim * 4
            cpu_time = (flops / 3e9) * 1e6 * np.random.uniform(0.9, 1.1)
            gpu_time = (flops / 12e12) * 1e6 * np.random.uniform(0.9, 1.1)
        elif task_type == "QUICKSORT":
            n = int(np.random.uniform(1e3, 5e6))
            flops = int(n * np.log2(n))
            bytes_in = n * 4
            bytes_out = n * 4
            cpu_time = (n * 0.04) * np.random.uniform(0.9, 1.1)
            gpu_time = (n * 0.25 + 500) * np.random.uniform(0.9, 1.1)
        else:
            dim = int(np.random.uniform(100, 2000))
            flops = int(dim ** 2 * 10)
            bytes_in = dim * dim * 4
            bytes_out = dim * 4
            cpu_time = (flops / 3e9) * 1e6
            gpu_time = (flops / 8e12) * 1e6

        transfer_time = ((bytes_in + bytes_out) / 12e9) * 1e6
        pcie_load = np.random.uniform(0.0, 1.0)
        gpu_queue_depth = np.random.randint(0, 16)

        alpha = 1.0 + 2.5 * max(0.0, pcie_load - 0.70)
        effective_gpu_cost = gpu_time + (alpha * transfer_time) + (gpu_queue_depth * 150)
        effective_cpu_cost = cpu_time

        optimal_device = 1 if effective_gpu_cost < effective_cpu_cost else 0

        data.append({
            "task_type": task_type,
            "flops": flops,
            "bytes_in": bytes_in,
            "bytes_out": bytes_out,
            "pcie_load": pcie_load,
            "gpu_queue_depth": gpu_queue_depth,
            "cpu_time_us": cpu_time,
            "gpu_time_us": gpu_time,
            "transfer_time_us": transfer_time,
            "optimal_device": optimal_device
        })

    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Dataset generated at {output_file} with {samples} entries.")


if __name__ == "__main__":
    generate_training_data()
