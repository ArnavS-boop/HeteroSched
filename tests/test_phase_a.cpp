#ifdef NDEBUG
#undef NDEBUG
#endif

#include "common.h"
#include "cpu_scheduler.h"
#include "gpu_executor.h"
#include "gated_controller.h"
#include "decision_forest_model.h"
#include "telemetry_monitor.h"

#include <iostream>
#include <vector>
#include <cstring>
#include <algorithm>
#include <cstdlib>
#include <future>

bool allocate_task_buffers(TaskDescriptor* task);
void free_task_buffers(TaskDescriptor* task);

#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "\n[TEST ASSERTION FAILED] " << msg << "\n" \
                      << "  Condition: " << #cond << "\n" \
                      << "  File: " << __FILE__ << ", Line: " << __LINE__ << std::endl; \
            std::abort(); \
        } \
    } while (0)

void test_string_token_real_cpu_execution() {
    std::cout << "[Test 1] Real CPU execution of STRING_TOKEN via CPUScheduler worker... " << std::flush;

    CPUScheduler scheduler(2);

    const char* sample_text = "antigravity heterosched multi-device scheduler end-to-end token validation";
    // 7 tokens: "antigravity", "heterosched", "multi-device", "scheduler", "end-to-end", "token", "validation"
    size_t len = strlen(sample_text);

    TaskDescriptor task{};
    task.task_id = 101;
    task.type = TaskType::STRING_TOKEN;
    task.priority = 1;
    task.dim_m = len;

    bool alloc_ok = allocate_task_buffers(&task);
    TEST_ASSERT(alloc_ok, "allocate_task_buffers must return true for STRING_TOKEN");
    TEST_ASSERT(task.host_input != nullptr, "host_input must be allocated");
    TEST_ASSERT(task.host_output != nullptr, "host_output must be allocated");

    // Copy known test string into allocated buffer
    memcpy(task.host_input, sample_text, len);

    // Enqueue to real CPUScheduler and wait for worker thread execution
    auto future = scheduler.EnqueueWithFuture(&task);
    future.get();

    TEST_ASSERT(task.metrics.success == true,
                "Task must be marked as successfully completed by CPU worker");

    uint32_t reported_tokens = *reinterpret_cast<uint32_t*>(task.host_output);
    TEST_ASSERT(reported_tokens == 7, "Reported tokens must equal exactly 7");

    free_task_buffers(&task);
    TEST_ASSERT(task.host_input == nullptr && task.host_output == nullptr,
                "free_task_buffers must null out input and output pointers");

    std::cout << "PASSED (7 tokens counted by real worker thread)" << std::endl;
}

void test_quicksort_real_cpu_execution() {
    std::cout << "[Test 2] Real CPU execution of QUICKSORT via CPUScheduler worker... " << std::flush;

    CPUScheduler scheduler(2);

    const int original_array[] = {84, -12, 45, 0, 999, -500, 23, 7, 18, 3};
    const size_t n = sizeof(original_array) / sizeof(original_array[0]);

    TaskDescriptor task{};
    task.task_id = 202;
    task.type = TaskType::QUICKSORT;
    task.priority = 0;
    task.dim_m = n;

    bool alloc_ok = allocate_task_buffers(&task);
    TEST_ASSERT(alloc_ok, "allocate_task_buffers must succeed for QUICKSORT");
    TEST_ASSERT(task.host_input != nullptr, "host_input must be allocated");

    memcpy(task.host_input, original_array, n * sizeof(int));

    // Submit to real CPUScheduler and wait for worker thread execution
    auto future = scheduler.EnqueueWithFuture(&task);
    future.get();

    TEST_ASSERT(task.metrics.success == true,
                "QUICKSORT must be marked as successfully completed");

    int* result_array = reinterpret_cast<int*>(task.host_input);
    for (size_t i = 1; i < n; ++i) {
        TEST_ASSERT(result_array[i] >= result_array[i - 1],
                    "Array elements must be in non-decreasing order after sort");
    }
    TEST_ASSERT(result_array[0] == -500 && result_array[n - 1] == 999,
                "Min and max sorted bounds must match expected values");

    free_task_buffers(&task);
    std::cout << "PASSED (array strictly sorted by real worker thread)" << std::endl;
}

void test_controller_guards_unsupported_tasks() {
    std::cout << "[Test 3] Controller guard against unsupported GPU workloads... " << std::flush;
    GatedController controller(std::make_unique<DecisionForestModel>());
    controller.SetTier1Enabled(true);
    controller.SetTier2Enabled(true);
    controller.SetDynamicPenaltiesEnabled(true);

    RuntimeState state{};
    state.active_pcie_utilization = 0.05;

    // Quicksort with massive FLOPs (>1e9 would trigger Tier 1 rule 3 without guard)
    TaskDescriptor q_task{};
    q_task.type = TaskType::QUICKSORT;
    q_task.dim_m = 50000000;
    q_task.bytes_in = 200000000;
    q_task.bytes_out = 200000000;
    q_task.estimated_flops = 5000000000ULL;

    ExecutionDevice q_dev = controller.ScheduleTask(&q_task, state);
    TEST_ASSERT(q_dev == ExecutionDevice::DEVICE_CPU, "QUICKSORT must be forced to CPU");

    // String token with massive FLOPs
    TaskDescriptor s_task{};
    s_task.type = TaskType::STRING_TOKEN;
    s_task.dim_m = 1000000;
    s_task.bytes_in = 1000000;
    s_task.bytes_out = 4;
    s_task.estimated_flops = 2000000000ULL;

    ExecutionDevice s_dev = controller.ScheduleTask(&s_task, state);
    TEST_ASSERT(s_dev == ExecutionDevice::DEVICE_CPU, "STRING_TOKEN must be forced to CPU");

    std::cout << "PASSED (controller strictly forces unsupported types to CPU)" << std::endl;
}

void test_gpu_executor_rejection_and_safety() {
    std::cout << "[Test 4] GPUExecutor defensive rejection of unsupported tasks... " << std::flush;
    GPUExecutor executor(2);

    int test_data[4] = {40, 30, 20, 10};
    int backup_data[4] = {40, 30, 20, 10};

    TaskDescriptor task{};
    task.task_id = 999;
    task.type = TaskType::QUICKSORT;
    task.dim_m = 4;
    task.bytes_in = 4 * sizeof(int);
    task.bytes_out = 4 * sizeof(int);
    task.host_input = test_data;
    task.host_output = test_data;

    // Directly attempt GPU execution of QUICKSORT
    executor.ExecuteAsync(&task, 0);

    TEST_ASSERT(task.metrics.success == false,
                "GPUExecutor must record failure (metrics.success = false) for rejected task");
    for (int i = 0; i < 4; ++i) {
        TEST_ASSERT(test_data[i] == backup_data[i],
                    "Host memory must not be corrupted by uninitialized VRAM copy");
    }

    std::cout << "PASSED (rejection recorded failure, memory preserved)" << std::endl;
}

void test_task_failure_propagation() {
    std::cout << "[Test 5] Task failure propagation and unallocated buffer handling... " << std::flush;

    GPUExecutor executor(2);

    // 1. Unallocated / null buffer task passed to executor
    TaskDescriptor invalid_task{};
    invalid_task.task_id = 505;
    invalid_task.type = TaskType::SGEMM;
    invalid_task.host_input = nullptr;
    invalid_task.host_output = nullptr;

    executor.ExecuteAsync(&invalid_task, 0);

    TEST_ASSERT(invalid_task.metrics.success == false,
                "Executor must record failure when unallocated task is dispatched");

    // 2. Null task descriptor safety
    bool null_alloc = allocate_task_buffers(nullptr);
    TEST_ASSERT(null_alloc == false, "allocate_task_buffers(nullptr) must return false");
    free_task_buffers(nullptr); // Must safely no-op

    std::cout << "PASSED (unallocated/rejected tasks correctly record failure)" << std::endl;
}

void test_unsupported_cpu_task_type() {
    std::cout << "[Test 6] CPUScheduler handling of unsupported/unknown task type... " << std::flush;
    CPUScheduler scheduler(2);

    int dummy_data[4] = {1, 2, 3, 4};
    TaskDescriptor task{};
    task.task_id = 888;
    task.type = static_cast<TaskType>(9999);
    task.host_input = dummy_data;
    task.host_output = dummy_data;

    auto future = scheduler.EnqueueWithFuture(&task);
    future.get();

    TEST_ASSERT(task.metrics.success == false,
                "CPUScheduler must set metrics.success = false for unsupported task type");

    TaskDescriptor task2{};
    task2.type = static_cast<TaskType>(9999);
    bool alloc_ok = allocate_task_buffers(&task2);
    TEST_ASSERT(alloc_ok == false,
                "allocate_task_buffers must return false for unsupported task type");
    TEST_ASSERT(task2.metrics.success == false,
                "metrics.success must be false for unsupported task type buffer allocation");
    TEST_ASSERT(task2.host_input == nullptr && task2.host_output == nullptr,
                "host pointers must be nullptr when allocation fails");

    std::cout << "PASSED (unsupported CPU type rejected with metrics.success = false)" << std::endl;
}

void test_stale_and_initial_metrics_status() {
    std::cout << "[Test 7] Fresh TaskDescriptor metrics initialization & allocation status... " << std::flush;
    TaskDescriptor task{};
    TEST_ASSERT(task.metrics.success == false,
                "TaskDescriptor must initialize metrics.success to false by default");
    TEST_ASSERT(task.host_input == nullptr, "host_input must default to nullptr");
    TEST_ASSERT(task.host_output == nullptr, "host_output must default to nullptr");

    task.type = TaskType::STRING_TOKEN;
    task.dim_m = 32;
    bool alloc_ok = allocate_task_buffers(&task);
    TEST_ASSERT(alloc_ok == true, "allocate_task_buffers should succeed for valid task");
    TEST_ASSERT(task.metrics.success == false,
                "metrics.success must remain false after buffer allocation prior to execution");

    free_task_buffers(&task);
    std::cout << "PASSED (default initializers prevent uninitialized / stale success state)" << std::endl;
}

int main() {
    std::cout << "\n=== HeteroSched Phase A Validation Review Suite ===" << std::endl;
    test_string_token_real_cpu_execution();
    test_quicksort_real_cpu_execution();
    test_controller_guards_unsupported_tasks();
    test_gpu_executor_rejection_and_safety();
    test_task_failure_propagation();
    test_unsupported_cpu_task_type();
    test_stale_and_initial_metrics_status();
    std::cout << "=== ALL PHASE A VALIDATION TESTS PASSED ===\n" << std::endl;

    return 0;
}
