#ifndef TELEMETRY_MONITOR_H
#define TELEMETRY_MONITOR_H

#include "common.h"
#include <mutex>
#include <atomic>

class TelemetryMonitor {
public:
    TelemetryMonitor();

    void RecordTransferStart(size_t bytes);
    void RecordTransferEnd(size_t bytes);
    void IncrementGPUQueue();
    void DecrementGPUQueue();
    void IncrementCPUQueue();
    void DecrementCPUQueue();

    RuntimeState Snapshot() const;

private:
    std::atomic<uint64_t> active_transfer_bytes_;
    std::atomic<uint32_t> pending_gpu_tasks_;
    std::atomic<uint32_t> pending_cpu_tasks_;
    double pcie_peak_bandwidth_;
};

#endif
