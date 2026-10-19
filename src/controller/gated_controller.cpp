#include <atomic>
#include "gated_controller.h"
#include "decision_forest_model.h"
#include <algorithm>

GatedController::GatedController(std::unique_ptr<DecisionForestModel> model)
    : model_(std::move(model)) {}

ExecutionDevice GatedController::ScheduleTask(TaskDescriptor* task, const RuntimeState& state) {
    auto start_time = std::chrono::high_resolution_clock::now();

    if (task->type == TaskType::QUICKSORT || task->type == TaskType::STRING_TOKEN) {
        task->metrics.assigned_device = ExecutionDevice::DEVICE_CPU;
        record_decision_overhead(task, start_time);
        return ExecutionDevice::DEVICE_CPU;
    }

    if (tier1_enabled_) {
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
    }

    if (tier2_enabled_ && model_) {
        double alpha = 1.0;
        double beta = 1.0;

        if (dynamic_penalties_enabled_) {
            alpha = 1.0 + 2.5 * std::max(0.0, state.active_pcie_utilization - 0.70);
            beta = 1.0 + 1.5 * (static_cast<double>(state.pending_gpu_tasks) / 32.0)
                       * (static_cast<double>(state.pending_gpu_tasks) / 32.0);
        }

        double pred_cpu_exec = model_->PredictCPU(task);
        double pred_gpu_exec = model_->PredictGPU(task);
        double pred_transfer = static_cast<double>(task->bytes_in + task->bytes_out) /
                               (12.0 * 1024 * 1024 * 1024 / 1e6);

        double cost_cpu = pred_cpu_exec + (state.pending_cpu_tasks * 200.0);
        double cost_gpu = pred_gpu_exec + (alpha * pred_transfer) +
                          (beta * (state.pending_gpu_tasks * 150.0));

        ExecutionDevice decision = (cost_cpu < cost_gpu) ?
            ExecutionDevice::DEVICE_CPU : ExecutionDevice::DEVICE_GPU;
        task->metrics.assigned_device = decision;
        record_decision_overhead(task, start_time);
        return decision;
    }

    if (tier1_enabled_ && !tier2_enabled_) {
        task->metrics.assigned_device = ExecutionDevice::DEVICE_CPU;
        record_decision_overhead(task, start_time);
        return ExecutionDevice::DEVICE_CPU;
    }

    // Baseline (both Tier 1 and Tier 2 disabled): Round-Robin between CPU and GPU
    static std::atomic<size_t> round_robin_counter{0};
    ExecutionDevice decision = (round_robin_counter.fetch_add(1) % 2 == 0)
        ? ExecutionDevice::DEVICE_CPU
        : ExecutionDevice::DEVICE_GPU;
    task->metrics.assigned_device = decision;
    record_decision_overhead(task, start_time);
    return decision;
}

void GatedController::record_decision_overhead(TaskDescriptor* task,
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time) {
    auto end_time = std::chrono::high_resolution_clock::now();
    task->metrics.t_decision_us = std::chrono::duration<double, std::micro>(end_time - start_time).count();
}
