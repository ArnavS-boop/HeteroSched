#ifdef NDEBUG
#undef NDEBUG
#endif

#include "common.h"
#include "gated_controller.h"
#include "decision_forest_model.h"
#include "heterosched_api.h"

#include <iostream>
#include <vector>
#include <cassert>
#include <cstdlib>

#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "\n[TEST ASSERTION FAILED] " << msg << "\n" \
                      << "  Condition: " << #cond << "\n" \
                      << "  File: " << __FILE__ << ", Line: " << __LINE__ << std::endl; \
            std::abort(); \
        } \
    } while (0)

void test_controller_tier1_only() {
    std::cout << "[Test 1] Controller with Tier-1 Heuristic Only... " << std::flush;
    GatedController controller(std::make_unique<DecisionForestModel>());
    controller.SetTier1Enabled(true);
    controller.SetTier2Enabled(false);
    controller.SetDynamicPenaltiesEnabled(false);

    RuntimeState state{};

    // 1. Trivial FLOPs (< 100k) -> CPU
    TaskDescriptor small_task{};
    small_task.type = TaskType::SPMV;
    small_task.dim_m = 1000;
    small_task.bytes_in = 4000;
    small_task.bytes_out = 4000;
    small_task.estimated_flops = 50000;
    TEST_ASSERT(controller.ScheduleTask(&small_task, state) == ExecutionDevice::DEVICE_CPU,
                "Tier 1 must route trivial tasks (<100k FLOPs) to CPU");

    // 2. Massive FLOPs (> 1e9) -> GPU
    TaskDescriptor massive_task{};
    massive_task.type = TaskType::SGEMM;
    massive_task.dim_m = 2048;
    massive_task.dim_n = 2048;
    massive_task.dim_k = 2048;
    massive_task.bytes_in = 2048 * 2048 * 8;
    massive_task.bytes_out = 2048 * 2048 * 4;
    massive_task.estimated_flops = 17000000000ULL;
    TEST_ASSERT(controller.ScheduleTask(&massive_task, state) == ExecutionDevice::DEVICE_GPU,
                "Tier 1 must route massive compute tasks (>1e9 FLOPs) to GPU");

    // 3. Intermediate task falls back to CPU when Tier 2 is disabled
    TaskDescriptor inter_task{};
    inter_task.type = TaskType::SGEMM;
    inter_task.dim_m = 256;
    inter_task.dim_n = 256;
    inter_task.dim_k = 256;
    inter_task.bytes_in = 256 * 256 * 8;
    inter_task.bytes_out = 256 * 256 * 4;
    inter_task.estimated_flops = 33554432ULL;
    TEST_ASSERT(controller.ScheduleTask(&inter_task, state) == ExecutionDevice::DEVICE_CPU,
                "Non-massive tasks must fall back to CPU when Tier 2 is disabled");

    // 4. Unsupported tasks always CPU
    TaskDescriptor qs_task{};
    qs_task.type = TaskType::QUICKSORT;
    qs_task.estimated_flops = 50000000000ULL;
    TEST_ASSERT(controller.ScheduleTask(&qs_task, state) == ExecutionDevice::DEVICE_CPU,
                "QUICKSORT must strictly route to CPU");

    std::cout << "PASSED" << std::endl;
}

void test_controller_baseline_round_robin() {
    std::cout << "[Test 2] Controller Baseline (No Scheduling / Both Tiers Disabled)... " << std::flush;
    GatedController controller(std::make_unique<DecisionForestModel>());
    controller.SetTier1Enabled(false);
    controller.SetTier2Enabled(false);
    controller.SetDynamicPenaltiesEnabled(false);

    RuntimeState state{};

    // Supported tasks must alternate between CPU and GPU
    TaskDescriptor task1{}; task1.type = TaskType::SGEMM;
    TaskDescriptor task2{}; task2.type = TaskType::SGEMM;
    ExecutionDevice d1 = controller.ScheduleTask(&task1, state);
    ExecutionDevice d2 = controller.ScheduleTask(&task2, state);

    TEST_ASSERT(d1 != d2, "Supported tasks in baseline mode must alternate (Round-Robin)");

    // Unsupported tasks must still strictly be pinned to CPU
    TaskDescriptor qs_task{}; qs_task.type = TaskType::QUICKSORT;
    TaskDescriptor st_task{}; st_task.type = TaskType::STRING_TOKEN;
    TEST_ASSERT(controller.ScheduleTask(&qs_task, state) == ExecutionDevice::DEVICE_CPU,
                "QUICKSORT must remain on CPU even in baseline mode");
    TEST_ASSERT(controller.ScheduleTask(&st_task, state) == ExecutionDevice::DEVICE_CPU,
                "STRING_TOKEN must remain on CPU even in baseline mode");

    std::cout << "PASSED" << std::endl;
}

void test_forced_modes_and_static_rule() {
    std::cout << "[Test 3] Forced Modes & Static Rule routing... " << std::flush;

    // Config: force_cpu
    HeteroSchedConfig cpu_cfg;
    cpu_cfg.force_cpu = true;
    TEST_ASSERT(GetConfigDescription(cpu_cfg) == "CPU_ONLY", "Config description must match CPU_ONLY");

    // Config: force_gpu
    HeteroSchedConfig gpu_cfg;
    gpu_cfg.force_gpu = true;
    TEST_ASSERT(GetConfigDescription(gpu_cfg) == "GPU_ONLY", "Config description must match GPU_ONLY");

    // Config: static_rule
    HeteroSchedConfig rule_cfg;
    rule_cfg.static_rule = true;
    rule_cfg.dim_threshold = 512;
    TEST_ASSERT(GetConfigDescription(rule_cfg) == "STATIC_RULE(threshold=512)",
                "Config description must match STATIC_RULE");

    std::cout << "PASSED" << std::endl;
}

int main() {
    std::cout << "\n=== HeteroSched Phase B Validation Suite ===" << std::endl;
    test_controller_tier1_only();
    test_controller_baseline_round_robin();
    test_forced_modes_and_static_rule();
    std::cout << "=== ALL PHASE B TESTS PASSED ===\n" << std::endl;
    return 0;
}
