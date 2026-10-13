#ifndef HETEROSCHED_API_H
#define HETEROSCHED_API_H

#include "common.h"
#include "cpu_scheduler.h"
#include "gpu_executor.h"
#include "gated_controller.h"
#include "telemetry_monitor.h"
#include "decision_forest_model.h"
#include <memory>
#include <vector>

struct HeteroSchedConfig {
    size_t cpu_workers = 4;
    size_t gpu_stream_pool = 4;
    bool enable_tier1 = true;
    bool enable_tier2 = true;
    bool enable_dynamic_penalties = true;
    bool enable_cross_layer_priority = true;
};

class HeteroSchedRuntime {
public:
    explicit HeteroSchedRuntime(const HeteroSchedConfig& config);
    ~HeteroSchedRuntime();

    void SubmitTask(TaskDescriptor* task);
    void WaitAll();

    const TelemetryMonitor& GetTelemetry() const { return telemetry_; }

private:
    void ApplyCrossLayerPriority(TaskDescriptor* task, ExecutionDevice device);

    HeteroSchedConfig config_;
    CPUScheduler cpu_scheduler_;
    GPUExecutor gpu_executor_;
    GatedController controller_;
    TelemetryMonitor telemetry_;

    std::vector<std::future<void>> pending_futures_;
    std::mutex submit_mutex_;
    size_t gpu_round_robin_;
};

#endif
