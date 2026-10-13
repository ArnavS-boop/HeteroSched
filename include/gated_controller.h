#ifndef GATED_CONTROLLER_H
#define GATED_CONTROLLER_H

#include "common.h"
#include <memory>
#include <chrono>

class DecisionForestModel;

class GatedController {
public:
    explicit GatedController(std::unique_ptr<DecisionForestModel> model);

    ExecutionDevice ScheduleTask(TaskDescriptor* task, const RuntimeState& state);

    void SetTier1Enabled(bool enabled) { tier1_enabled_ = enabled; }
    void SetTier2Enabled(bool enabled) { tier2_enabled_ = enabled; }
    void SetDynamicPenaltiesEnabled(bool enabled) { dynamic_penalties_enabled_ = enabled; }

private:
    void record_decision_overhead(TaskDescriptor* task,
        std::chrono::time_point<std::chrono::high_resolution_clock> start_time);

    std::unique_ptr<DecisionForestModel> model_;
    bool tier1_enabled_ = true;
    bool tier2_enabled_ = true;
    bool dynamic_penalties_enabled_ = true;
};

#endif
