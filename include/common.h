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
    uint32_t priority;
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
    double active_pcie_utilization;
    uint32_t active_cuda_streams;
    uint32_t pending_gpu_tasks;
    uint32_t pending_cpu_tasks;
    double cpu_utilization;
};

#endif
