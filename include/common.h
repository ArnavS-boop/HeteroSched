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
    uint64_t task_id = 0;
    ExecutionDevice assigned_device = ExecutionDevice::DEVICE_CPU;
    double t_decision_us = 0.0;
    double t_transfer_us = 0.0;
    double t_exec_us = 0.0;
    double t_sync_us = 0.0;
    double t_total_latency_us = 0.0;
    uint32_t stream_id_used = 0;
    int cpu_core_used = -1;
    bool success = false;
};

struct TaskDescriptor {
    uint64_t task_id = 0;
    TaskType type = TaskType::SGEMM;
    uint32_t priority = 0;
    size_t dim_m = 0;
    size_t dim_n = 0;
    size_t dim_k = 0;
    size_t bytes_in = 0;
    size_t bytes_out = 0;
    uint64_t estimated_flops = 0;
    void* host_input = nullptr;
    void* host_output = nullptr;
    TaskExecutionMetrics metrics{};
};

struct RuntimeState {
    double active_pcie_utilization;
    uint32_t active_cuda_streams;
    uint32_t pending_gpu_tasks;
    uint32_t pending_cpu_tasks;
    double cpu_utilization;
};

#endif
