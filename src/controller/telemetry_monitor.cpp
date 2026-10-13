#include "telemetry_monitor.h"

TelemetryMonitor::TelemetryMonitor()
    : active_transfer_bytes_(0),
      pending_gpu_tasks_(0),
      pending_cpu_tasks_(0),
      pcie_peak_bandwidth_(12.0 * 1024 * 1024 * 1024) {}

void TelemetryMonitor::RecordTransferStart(size_t bytes) {
    active_transfer_bytes_.fetch_add(bytes, std::memory_order_relaxed);
}

void TelemetryMonitor::RecordTransferEnd(size_t bytes) {
    active_transfer_bytes_.fetch_sub(bytes, std::memory_order_relaxed);
}

void TelemetryMonitor::IncrementGPUQueue() {
    pending_gpu_tasks_.fetch_add(1, std::memory_order_relaxed);
}

void TelemetryMonitor::DecrementGPUQueue() {
    pending_gpu_tasks_.fetch_sub(1, std::memory_order_relaxed);
}

void TelemetryMonitor::IncrementCPUQueue() {
    pending_cpu_tasks_.fetch_add(1, std::memory_order_relaxed);
}

void TelemetryMonitor::DecrementCPUQueue() {
    pending_cpu_tasks_.fetch_sub(1, std::memory_order_relaxed);
}

RuntimeState TelemetryMonitor::Snapshot() const {
    RuntimeState state;
    uint64_t active_bytes = active_transfer_bytes_.load(std::memory_order_relaxed);
    state.active_pcie_utilization = static_cast<double>(active_bytes) / pcie_peak_bandwidth_;
    state.active_cuda_streams = pending_gpu_tasks_.load(std::memory_order_relaxed);
    state.pending_gpu_tasks = pending_gpu_tasks_.load(std::memory_order_relaxed);
    state.pending_cpu_tasks = pending_cpu_tasks_.load(std::memory_order_relaxed);
    state.cpu_utilization = 0.0;
    return state;
}
