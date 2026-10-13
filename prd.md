# Product Requirements Document (PRD): HeteroSched Engine

## 1. Executive Summary & Problem Formulation
Heterogeneous computing architectures combining multicore CPUs and discrete GPUs suffer from severe performance degradation under concurrent, mixed-workload conditions due to uncoordinated scheduling tiers[cite: 2, 8]. Existing solutions optimize either CPU thread assignment, static offloading placement, or GPU stream queues independently[cite: 3, 8]. This decoupling induces:
1. **Host-Accelerator Starvation:** Asynchronous CUDA execution stalls when the host CPU dispatcher thread is preempted or deprioritized by OS-level scheduling[cite: 3, 4].
2. **Interconnect & Queue Congestion Blindness:** Static placement offloads parallel tasks to the GPU without accounting for PCIe interconnect saturation or CUDA queueing delay, frequently causing data-transfer and queue times to exceed execution time gains[cite: 2, 3, 4].
3. **Inference Latency Overhead:** Complex deep learning and reinforcement learning schedulers impose decision latencies ($T_{decision} \sim 5\text{--}50\text{ ms}$) that negate performance improvements for fine-grained compute kernels[cite: 7].

**HeteroSched** solves these challenges by providing an adaptive, user-space, cross-layer co-scheduling runtime[cite: 8, 9]. The system integrates a two-tier gated decision controller (Tier-1 heuristic gate + Tier-2 lightweight ML predictor), a dynamic interconnect- and queue-penalized cost model, and synchronized host-thread-to-CUDA-stream priority mapping on a single heterogeneous node[cite: 3, 8, 10].

---

## 2. System Architecture & Operational Topology

```text
                                 +--------------------------------------------+
                                 |          Synthetic / Trace Workload        |
                                 |         Generator & Dispatch Engine        |
                                 +--------------------------------------------+
                                                       |
                                               Task Ingestion (JSON/Struct)
                                                       v
+------------------------------------------------------------------------------------------------------------------+
| HeteroSched Runtime Core (C++17 / Python Orchestrator)                                                          |
|                                                                                                                  |
|  +-------------------------+       +------------------------------------+       +-----------------------------+  |
|  |   Workload Monitor      | ----> |       Hierarchical Controller      | <---  |    Runtime Telemetry        |  |
|  | - Task Dimension & Type |       | Tier 1: Fast-Path Filter (<5 us)   |       | - Active PCIe Bandwidth     |  |
|  | - Data Buffer Size      |       | Tier 2: Micro-GBDT Engine (<80 us) |       | - CUDA Stream Queue Depth   |  |
|  | - Operational Intensity |       +------------------------------------+       | - CPU Runqueue Load (CFS)   |  |
|  +-------------------------+                         |                          +-----------------------------+  |
|                                                      | Placement & Priority                                      |
|                                                      v                                                           |
|                        +-------------------------------------------------------------+                           |
|                        |             Cross-Layer Dispatch Coordinator                |                           |
|                        +-------------------------------------------------------------+                           |
|                                      /                                 \                                         |
|                                     /                                   \                                        |
|                                    v                                     v                                       |
|  +---------------------------------------------------+  +-----------------------------------------------------+  |
|  |               CPU Worker Subsystem                |  |                GPU Execution Subsystem              |  |
|  |  - Policy Engine: FIFO, RR, SJF, MLFQ             |  |  - Multi-Stream Pool (High/Med/Low Priority Streams)|  |
|  |  - Dynamic Thread Affinity (pthread/hwloc)       |  |  - Async Memcpy Pipelines (cudaMemcpyAsync)         |  |
|  |  - Elevated Host Dispatcher Priority              |  |  - CUDA Events Timing Instrumentation               |  |
|  +---------------------------------------------------+  +-----------------------------------------------------+  |
+------------------------------------------------------------------------------------------------------------------+
                                                       |
                                                       v
                                 +--------------------------------------------+
                                 |        Telemetry, Evaluation Harness,      |
                                 |            & Ablation Benchmarker          |
                                 +--------------------------------------------+
```

---

## 3. Mathematical & Algorithmic Specifications

### 3.1 Contention-Aware Cost Formulation
For any incoming task $\tau$, the execution cost on candidate device $d \in \{\text{CPU}, \text{GPU}\}$ is formulated as[cite: 10]:
$$C(d) = \widehat{T}_{exec}(d) + \alpha(t) \cdot \widehat{T}_{transfer}(d) + \beta(t) \cdot T_{queue}(d) + T_{sync}(d) + T_{decision}$$

Where:
* $\widehat{T}_{exec}(d)$ is the estimated kernel/function compute duration on device $d$[cite: 10].
* $\widehat{T}_{transfer}(d)$ is the estimated data movement cost ($0$ for CPU host-allocated memory; $\frac{\text{Bytes}_{in} + \text{Bytes}_{out}}{BW_{nominal}}$ for discrete GPU)[cite: 10].
* $T_{queue}(d)$ is the accumulated waiting time of pending tasks in the target execution queue[cite: 10].
* $T_{sync}(d)$ is the blocking synchronization latency introduced by CPU-GPU fence boundaries or callback events[cite: 10].
* $T_{decision}$ is the controller profiling and inference latency[cite: 10].

#### Dynamic Penalty Modulation Functions
* **Interconnect Penalty ($\alpha(t)$):**
  $$\alpha(t) = 1.0 + \gamma_{pcie} \cdot \max\left(0, \frac{BW_{active}(t)}{BW_{peak}} - \theta_{pcie}\right)$$
  Where $BW_{active}(t)$ is the estimated instantaneous PCIe bandwidth utilization, $BW_{peak}$ is maximum bus saturation, $\theta_{pcie} = 0.70$ (saturation threshold), and $\gamma_{pcie} = 2.5$ (scaling factor).

* **Stream Queue Penalty ($\beta(t)$):**
  $$\beta(t) = 1.0 + \gamma_{queue} \cdot \left(\frac{Q_{depth}(t)}{Q_{max}}\right)^2$$
  Where $Q_{depth}(t)$ is the number of active, incomplete asynchronous CUDA stream events, and $Q_{max}$ is the queue ceiling before head-of-line blocking occurs.

### 3.2 Two-Tier Gated Controller Decision Engine
1. **Tier 1 (Zero-Overhead Fast-Path Filter, target $<5\,\mu\text{s}$):**
   * Compute Operational Intensity:
     $$I_{op} = \frac{\text{Estimated FLOPs}}{\text{Bytes}_{in} + \text{Bytes}_{out}}$$
   * Rule A (Trivial Sequential): If $\text{Estimated FLOPs} < 10^5 \implies \text{Route to CPU}$ immediately.
   * Rule B (Memory-Bound Transfer Dominated): If $I_{op} < 0.5 \text{ FLOPs/Byte}$ and $\text{Bytes}_{transfer} > 10\text{ MB} \implies \text{Route to CPU}$.
   * Rule C (Massive Compute-Bound): If $\text{Estimated FLOPs} > 10^9$ and $\alpha(t) < 1.5 \implies \text{Route to GPU High-Priority Stream}$.
2. **Tier 2 (Lightweight Predictor, target $<80\,\mu\text{s}$):**
   * Invoked only for ambiguous tasks failing Tier-1 filters.
   * Runs an optimized gradient-boosted decision tree (or compiled lookup forest) predicting $\widehat{T}_{exec}(\text{CPU})$ and $\widehat{T}_{exec}(\text{GPU})$.
   * Selects $d = \arg\min_{d} C(d)$.

### 3.3 Cross-Layer Priority Co-Scheduling
To prevent host-side CPU starvation during CUDA dispatching:
* If task $\tau$ is assigned to the GPU with high priority, its host feeder thread is pinned to an isolated core using `pthread_setaffinity_np` and assigned a higher scheduling class/priority (`sched_setscheduler` with `SCHED_RR` or reduced `nice` value of -10).
* The task is submitted to an explicit CUDA stream created via `cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, -1)` (high priority).
* Medium and low priority tasks map respectively to priority 0 streams and default CPU thread niceness.

---

## 4. Workload Definitions & Synthetic Benchmark Suite

The system must implement native, reproducible kernels spanning three distinct computational regimes[cite: 9]:

| Workload Class | Kernel Implementation | Arithmetic Intensity (FLOPs/Byte) | Optimal Target |
|---|---|---|---|
| CPU-Oriented | Recursive Quicksort, String Tokenizer | Low (< 0.2) Memory/branch bound | Multicore CPU |
| GPU-Oriented | SGEMM (Dense Matrix Mult), 2D Convolution | High (> 30.0) Parallel compute bound | CUDA Stream |
| Ambiguous / Mixed | Sparse Matrix-Vector (SpMV), Vector AlphaAXPY | Medium (0.5 - 5.0) Memory bandwidth constrained | Contention-dependent (Dynamic ML routing) |

Each workload must expose a uniform C-compatible invocation interface:
```cpp
struct TaskPayload {
    uint64_t task_id;
    TaskType type;              // SGEMM, CONV2D, SPMV, QUICKSORT, STRING_TOKEN
    uint32_t priority;          // 0 = Low, 1 = Med, 2 = High
    size_t data_bytes_in;
    size_t data_bytes_out;
    uint64_t estimated_flops;
    void* h_input;
    void* h_output;
    size_t dim_m;
    size_t dim_n;
    size_t dim_k;
};
```

---

## 5. Baselines & Experimental Comparison Grid

The framework must evaluate the following execution modes under identical workload queues[cite: 10]:
1. **CPU-Only Static:** All tasks executed across host worker threads (using FIFO and Round Robin)[cite: 10].
2. **GPU-Only Static:** All tasks offloaded to GPU default stream (blocking memory copies, single CUDA stream)[cite: 10].
3. **Static Rule-Based Placement:** Offloads to GPU if $\text{dim} > 512$, otherwise CPU[cite: 10].
4. **Decoupled ML Scheduler:** Standalone GBDT placement without dynamic transfer/queue penalties ($\alpha=1, \beta=1$) and without host thread priority synchronization[cite: 4, 10].
5. **HeteroSched Full (Proposed):** Gated Tier-1/Tier-2 controller + contention penalties $\alpha(t), \beta(t)$ + cross-layer host-accelerator co-prioritization[cite: 10].

### Evaluation Metrics[cite: 10, 11]
* End-to-End Task Latency (Mean, 95th percentile, 99th percentile)[cite: 10].
* System Throughput (Tasks per second / GFLOPs per second)[cite: 10].
* Scheduling and Inference Overhead ($T_{decision}$ breakdown in $\mu\text{s}$)[cite: 10, 11].
* Host-to-Device / Device-to-Host Transfer Overhead ($T_{transfer}$)[cite: 10, 11].
* Host CPU Thread Starvation / Wait Time (microseconds spent blocking before submission)[cite: 10, 11].
* Device Utilization (CPU core % active, GPU compute/memory active via `nvml` or `cudaEventElapsedTime`)[cite: 10, 11].

---

## 6. Directory Layout & File Manifest

```text
heterosched/
├── CMakeLists.txt
├── Makefile
├── requirements.txt
├── prd.md
├── configs/
│   ├── default_config.json
│   └── ablation_config.json
├── include/
│   ├── common.h
│   ├── workload_types.h
│   ├── cpu_scheduler.h
│   ├── gpu_executor.h
│   ├── telemetry_monitor.h
│   ├── gated_controller.h
│   └── heterosched_api.h
├── src/
│   ├── cpu/
│   │   ├── cpu_scheduler.cpp
│   │   └── cpu_kernels.cpp
│   ├── gpu/
│   │   ├── gpu_executor.cu
│   │   └── gpu_kernels.cu
│   ├── controller/
│   │   ├── gated_controller.cpp
│   │   ├── decision_forest_model.cpp
│   │   └── telemetry_monitor.cpp
│   └── main.cpp
├── ml/
│   ├── dataset_generator.py
│   ├── train_model.py
│   ├── export_to_cpp.py
│   └── models/
│       └── gbm_model.json
├── benchmarks/
│   ├── trace_generator.py
│   ├── run_benchmarks.py
│   ├── run_ablations.py
│   └── traces/
│       ├── bursty_trace.json
│       ├── mixed_trace.json
│       └── pcie_heavy_trace.json
└── scripts/
    ├── setup_env.sh
    ├── build.sh
    └── plot_results.py
```

---

## 7. Concrete Code Architecture & Module Implementations

### 7.1 Common Data Structures (`include/common.h`)

```cpp
#ifndef COMMON_H
#define COMMON_H

#include <cstdint>
#include <cstddef>
#include <chrono>

enum class TaskType {
    SGEMM = 0,
    CONV2D,
    SPMV,
    QUICKSORT,
    STRING_TOKEN
};

enum class ExecutionDevice {
    DEVICE_CPU = 0,
    DEVICE_GPU = 1
};

struct TaskExecutionMetrics {
    uint64_t task_id;
    ExecutionDevice assigned_device;
    double t_decision_us;
    double t_transfer_us;
    double t_exec_us;
    double t_sync_us;
    double t_total_latency_us;
    uint32_t stream_id_used;
    int cpu_core_used;
};

struct TaskDescriptor {
    uint64_t task_id;
    TaskType type;
    uint32_t priority; // 0: Low, 1: Normal, 2: High
    size_t dim_m;
    size_t dim_n;
    size_t dim_k;
    size_t bytes_in;
    size_t bytes_out;
    uint64_t estimated_flops;
    void* host_input;
    void* host_output;
    TaskExecutionMetrics metrics;
};

struct RuntimeState {
    double active_pcie_utilization; // 0.0 to 1.0
    uint32_t active_cuda_streams;
    uint32_t pending_gpu_tasks;
    uint32_t pending_cpu_tasks;
    double cpu_utilization;         // 0.0 to 1.0
};

#endif // COMMON_H
```

### 7.2 CPU Scheduler & Worker Pool (`src/cpu/cpu_scheduler.cpp`)

```cpp
#include "cpu_scheduler.h"
#include <pthread.h>
#include <sched.h>
#include <iostream>
#include <chrono>
#include <algorithm>

CPUScheduler::CPUScheduler(size_t num_workers) : stop_requested_(false) {
    for (size_t i = 0; i < num_workers; ++i) {
        workers_.emplace_back(&CPUScheduler::WorkerLoop, this, i);
    }
}

CPUScheduler::~CPUScheduler() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        stop_requested_ = true;
    }
    cv_.notify_all();
    for (auto& worker : workers_) {
        if (worker.joinable()) worker.join();
    }
}

void CPUScheduler::Enqueue(TaskDescriptor* task) {
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        task_queue_.push(task);
    }
    cv_.notify_one();
}

void CPUScheduler::WorkerLoop(size_t worker_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(worker_id % std::thread::hardware_concurrency(), &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

    while (true) {
        TaskDescriptor* task = nullptr;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            cv_.wait(lock, [this] { return stop_requested_ || !task_queue_.empty(); });
            if (stop_requested_ && task_queue_.empty()) return;
            task = task_queue_.front();
            task_queue_.pop();
        }

        if (task) {
            auto start_exec = std::chrono::high_resolution_clock::now();
            ExecuteCPUTask(task, worker_id);
            auto end_exec = std::chrono::high_resolution_clock::now();
            task->metrics.t_exec_us = std::chrono::duration<double, std::micro>(end_exec - start_exec).count();
            task->metrics.cpu_core_used = worker_id;
        }
    }
}

void CPUScheduler::ExecuteCPUTask(TaskDescriptor* task, size_t core_id) {
    switch (task->type) {
        case TaskType::SGEMM: {
            float* A = reinterpret_cast<float*>(task->host_input);
            float* B = A + (task->dim_m * task->dim_k);
            float* C = reinterpret_cast<float*>(task->host_output);
            for (size_t i = 0; i < task->dim_m; ++i) {
                for (size_t j = 0; j < task->dim_n; ++j) {
                    float sum = 0.0f;
                    for (size_t k = 0; k < task->dim_k; ++k) {
                        sum += A[i * task->dim_k + k] * B[k * task->dim_n + j];
                    }
                    C[i * task->dim_n + j] = sum;
                }
            }
            break;
        }
        case TaskType::QUICKSORT: {
            int* arr = reinterpret_cast<int*>(task->host_input);
            size_t n = task->dim_m;
            std::sort(arr, arr + n);
            break;
        }
        default:
            break;
    }
}
```

### 7.3 CUDA Execution Manager (`src/gpu/gpu_executor.cu`)[cite: 3, 9]

```cuda
#include "gpu_executor.h"
#include <cuda_runtime.h>
#include <iostream>

__global__ void sgemm_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float val = 0.0f;
        for (int k = 0; k < K; ++k) {
            val += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = val;
    }
}

GPUExecutor::GPUExecutor(size_t stream_pool_size) : pool_size_(stream_pool_size) {
    streams_.resize(pool_size_);
    int low_prio, high_prio;
    cudaDeviceGetStreamPriorityRange(&low_prio, &high_prio);
    for (size_t i = 0; i < pool_size_; ++i) {
        int prio = (i == 0) ? high_prio : low_prio;
        cudaStreamCreateWithPriority(&streams_[i], cudaStreamNonBlocking, prio);
    }
}

GPUExecutor::~GPUExecutor() {
    for (auto& s : streams_) {
        cudaStreamSynchronize(s);
        cudaStreamDestroy(s);
    }
}

void GPUExecutor::ExecuteAsync(TaskDescriptor* task, size_t stream_idx) {
    cudaStream_t stream = streams_[stream_idx % pool_size_];
    task->metrics.stream_id_used = stream_idx % pool_size_;

    cudaEvent_t start_transfer, start_kernel, end_kernel, end_transfer;
    cudaEventCreate(&start_transfer);
    cudaEventCreate(&start_kernel);
    cudaEventCreate(&end_kernel);
    cudaEventCreate(&end_transfer);

    void *d_in = nullptr, *d_out = nullptr;
    cudaMallocAsync(&d_in, task->bytes_in, stream);
    cudaMallocAsync(&d_out, task->bytes_out, stream);

    cudaEventRecord(start_transfer, stream);
    cudaMemcpyAsync(d_in, task->host_input, task->bytes_in, cudaMemcpyHostToDevice, stream);
    cudaEventRecord(start_kernel, stream);

    if (task->type == TaskType::SGEMM) {
        dim3 threads(16, 16);
        dim3 blocks((task->dim_n + 15) / 16, (task->dim_m + 15) / 16);
        const float* d_A = reinterpret_cast<const float*>(d_in);
        const float* d_B = d_A + (task->dim_m * task->dim_k);
        float* d_C = reinterpret_cast<float*>(d_out);
        sgemm_kernel<<<blocks, threads, 0, stream>>>(d_A, d_B, d_C, task->dim_m, task->dim_n, task->dim_k);
    }

    cudaEventRecord(end_kernel, stream);
    cudaMemcpyAsync(task->host_output, d_out, task->bytes_out, cudaMemcpyDeviceToHost, stream);
    cudaEventRecord(end_transfer, stream);

    cudaFreeAsync(d_in, stream);
    cudaFreeAsync(d_out, stream);

    cudaStreamSynchronize(stream);

    float t_transfer_in = 0.0f, t_kernel = 0.0f, t_transfer_out = 0.0f;
    cudaEventElapsedTime(&t_transfer_in, start_transfer, start_kernel);
    cudaEventElapsedTime(&t_kernel, start_kernel, end_kernel);
    cudaEventElapsedTime(&t_transfer_out, end_kernel, end_transfer);

    task->metrics.t_transfer_us = (t_transfer_in + t_transfer_out) * 1000.0;
    task->metrics.t_exec_us = t_kernel * 1000.0;

    cudaEventDestroy(start_transfer);
    cudaEventDestroy(start_kernel);
    cudaEventDestroy(end_kernel);
    cudaEventDestroy(end_transfer);
}
```

### 7.4 Two-Tier Gated Controller (`src/controller/gated_controller.cpp`)[cite: 10]

```cpp
#include "gated_controller.h"
#include <chrono>
#include <algorithm>

GatedController::GatedController(std::unique_ptr<DecisionForestModel> model)
    : model_(std::move(model)) {}

ExecutionDevice GatedController::ScheduleTask(TaskDescriptor* task, const RuntimeState& state) {
    auto start_time = std::chrono::high_resolution_clock::now();

    // TIER 1: Fast-Path Heuristic Gate (< 5 us)
    double operational_intensity = static_cast<double>(task->estimated_flops) / 
                                   (task->bytes_in + task->bytes_out + 1.0);

    if (task->estimated_flops < 100000ULL) {
        task->metrics.assigned_device = ExecutionDevice::DEVICE_CPU;
        record_decision_overhead(task, start_time);
        return ExecutionDevice::DEVICE_CPU;
    }

    if (operational_intensity < 0.5 && (task->bytes_in + task->bytes_out) > 10 * 1024 * 1024) {
        task->metrics.assigned_device = ExecutionDevice::DEVICE_CPU;
        record_decision_overhead(task, start_time);
        return ExecutionDevice::DEVICE_CPU;
    }

    if (task->estimated_flops > 1000000000ULL && state.active_pcie_utilization < 0.70) {
        task->metrics.assigned_device = ExecutionDevice::DEVICE_GPU;
        record_decision_overhead(task, start_time);
        return ExecutionDevice::DEVICE_GPU;
    }

    // TIER 2: Predictive Decision Cost Model
    double alpha = 1.0 + 2.5 * std::max(0.0, state.active_pcie_utilization - 0.70);
    double beta = 1.0 + 1.5 * (static_cast<double>(state.pending_gpu_tasks) / 32.0);

    double pred_cpu_exec = model_->PredictCPU(task);
    double pred_gpu_exec = model_->PredictGPU(task);
    double pred_transfer = static_cast<double>(task->bytes_in + task->bytes_out) / (12.0 * 1024 * 1024 * 1024 / 1e6);

    double cost_cpu = pred_cpu_exec + (state.pending_cpu_tasks * 200.0);
    double cost_gpu = pred_gpu_exec + (alpha * pred_transfer) + (beta * (state.pending_gpu_tasks * 150.0));

    ExecutionDevice decision = (cost_cpu < cost_gpu) ? ExecutionDevice::DEVICE_CPU : ExecutionDevice::DEVICE_GPU;
    task->metrics.assigned_device = decision;

    record_decision_overhead(task, start_time);
    return decision;
}

void GatedController::record_decision_overhead(TaskDescriptor* task, 
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time) {
    auto end_time = std::chrono::high_resolution_clock::now();
    task->metrics.t_decision_us = std::chrono::duration<double, std::micro>(end_time - start_time).count();
}
```

### 7.5 High-Speed Embedded C++ Decision Tree (`src/controller/decision_forest_model.cpp`)

```cpp
#include "decision_forest_model.h"

double DecisionForestModel::PredictCPU(const TaskDescriptor* task) {
    if (task->type == TaskType::QUICKSORT) {
        return static_cast<double>(task->dim_m) * 0.05;
    }
    if (task->type == TaskType::SGEMM) {
        double n = static_cast<double>(task->dim_m);
        return (2.0 * n * n * n) / (3.5e9) * 1e6;
    }
    return 1000.0;
}

double DecisionForestModel::PredictGPU(const TaskDescriptor* task) {
    if (task->type == TaskType::QUICKSORT) {
        return 50000.0;
    }
    if (task->type == TaskType::SGEMM) {
        double n = static_cast<double>(task->dim_m);
        return (2.0 * n * n * n) / (15.0e12) * 1e6;
    }
    return 500.0;
}
```

---

## 8. Machine Learning Pipeline & Training Workflow

### 8.1 Dataset Generation (`ml/dataset_generator.py`)[cite: 9, 12]

```python
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
```

### 8.2 Training & Serializer (`ml/train_model.py`)[cite: 9, 10]

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

def train_and_evaluate():
    df = pd.read_csv("ml/dataset.csv")
    df = pd.get_dummies(df, columns=["task_type"])
    
    features = [c for c in df.columns if c not in ["optimal_device", "cpu_time_us", "gpu_time_us"]]
    X = df[features]
    y = df["optimal_device"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=20, max_depth=6, random_state=42)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    print(f"Classifier Accuracy: {accuracy_score(y_test, preds):.4f}")
    print(classification_report(y_test, preds))

    joblib.dump(clf, "ml/models/rf_scheduler.pkl")
    print("Model serialized to ml/models/rf_scheduler.pkl")

if __name__ == "__main__":
    train_and_evaluate()
```

---

## 9. Comprehensive Benchmark Runner & Ablation Suite

### 9.1 Synthetic Trace Scenarios (`benchmarks/trace_generator.py`)[cite: 9, 11]

```python
import json
import random

def generate_traces():
    workloads = []
    for i in range(200):
        w_type = random.choice(["SGEMM", "QUICKSORT", "SPMV"])
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
    with open("benchmarks/traces/mixed_trace.json", "w") as f:
        json.dump(workloads, f, indent=2)
    print("Saved benchmarks/traces/mixed_trace.json")

if __name__ == "__main__":
    generate_traces()
```

### 9.2 Complete Ablation Matrix[cite: 9, 11]

| Config ID | Heuristic Gate (Tier 1) | ML Predictor (Tier 2) | Dynamic PCIe Penalty $\alpha(t)$ | Dynamic Queue Penalty $\beta(t)$ | Cross-Layer Host Priority Sync |
|---|---|---|---|---|---|
| `ABLATION_0` (Baseline) | Disabled | Disabled | Disabled | Disabled | Disabled |
| `ABLATION_1` | Enabled | Disabled | Disabled | Disabled | Disabled |
| `ABLATION_2` | Enabled | Enabled | Disabled | Disabled | Disabled |
| `ABLATION_3` | Enabled | Enabled | Enabled | Enabled | Disabled |
| `HETEROSCHED_FULL` | **Enabled** | **Enabled** | **Enabled** | **Enabled** | **Enabled** |

---

## 10. Automated Build & Execution Scripts

### 10.1 CMake Build Definition (`CMakeLists.txt`)

```cmake
cmake_minimum_required(VERSION 3.18)
project(HeteroSched LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDA REQUIRED)
find_package(Threads REQUIRED)

include_directories(
    include
    ${CUDA_INCLUDE_DIRS}
)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O3 -arch=sm_75 --extended-lambda)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -pthread -Wall")

add_library(heterosched_core STATIC
    src/cpu/cpu_scheduler.cpp
    src/gpu/gpu_executor.cu
    src/controller/gated_controller.cpp
    src/controller/decision_forest_model.cpp
    src/controller/telemetry_monitor.cpp
)

add_executable(heterosched_bench src/main.cpp)
target_link_libraries(heterosched_bench 
    PRIVATE 
    heterosched_core 
    ${CUDA_LIBRARIES} 
    Threads::Threads
)
```

### 10.2 End-to-End Orchestrator Script (`scripts/run_all.sh`)

```bash
#!/usr/bin/env bash
set -euo pipefail

echo "=============================================="
echo " Starting HeteroSched Automated Workflow"
echo "=============================================="

# 1. Setup Environment
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 2. Train Models
echo "[1/4] Generating ML dataset and training models..."
python3 ml/dataset_generator.py
python3 ml/train_model.py

# 3. Generate Benchmark Traces
echo "[2/4] Generating benchmark trace definitions..."
mkdir -p benchmarks/traces
python3 benchmarks/trace_generator.py

# 4. Build Engine
echo "[3/4] Compiling native HeteroSched runtime engine..."
mkdir -p build
cd build
cmake ..
make -j"$(nproc)"
cd ..

# 5. Run Experiments
echo "[4/4] Executing comparative benchmarks..."
./build/heterosched_bench --trace benchmarks/traces/mixed_trace.json --output results.csv
python3 scripts/plot_results.py --input results.csv --output-dir artifacts/

echo "=============================================="
echo " HeteroSched Execution Complete. Results in artifacts/"
echo "=============================================="
```

---

## 11. Verification, Validation & Acceptance Criteria

1. **Scheduling Latency Gate:** Tier-1 decisions must complete in $\le 5\,\mu\text{s}$, and Tier-2 decisions must complete in $\le 100\,\mu\text{s}$[cite: 10].
2. **Numerical Correctness:** Result matrices and sorted vectors must match single-threaded CPU ground truth within an epsilon of $10^{-5}$ for floating-point operations.
3. **Starvation Invariance:** High-priority CUDA streams must show under $10\%$ coefficient of variance in GPU launch delays under high CPU thread saturation.
4. **Dynamic Fallback:** Exceeding $70\%$ PCIe saturation must divert low-operational-intensity tasks to multicore CPU execution without deadlocking.