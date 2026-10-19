#ifdef NDEBUG
#undef NDEBUG
#endif

#include "common.h"
#include "cpu_scheduler.h"
#include "gpu_executor.h"

#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <cstdlib>
#include <string>
#include <limits>
#include <cstdint>
#include <cuda_runtime.h>

#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "\n[TEST ASSERTION FAILED] " << msg << "\n" \
                      << "  Condition: " << #cond << "\n" \
                      << "  File: " << __FILE__ << ", Line: " << __LINE__ << std::endl; \
            std::abort(); \
        } \
    } while (0)

// ============================================================================
// CUDA Environment Detection
// ============================================================================
static bool is_cuda_available(std::string& reason) {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess) {
        reason = "cudaGetDeviceCount failed: " + std::string(cudaGetErrorString(err));
        return false;
    }
    if (device_count <= 0) {
        reason = "No CUDA devices detected (device_count = 0)";
        return false;
    }
    err = cudaFree(nullptr);
    if (err != cudaSuccess) {
        reason = "cudaFree(0) initialization failed: " + std::string(cudaGetErrorString(err));
        return false;
    }
    return true;
}

// ============================================================================
// Independent Reference Implementations (Ground Truth)
// ============================================================================

// Independent SGEMM: Double-precision accumulation for independent numerical reference
static void ref_sgemm(const float* A, const float* B, float* C, size_t M, size_t N, size_t K) {
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < K; ++k) {
                sum += static_cast<double>(A[i * K + k]) * static_cast<double>(B[k * N + j]);
            }
            C[i * N + j] = static_cast<float>(sum);
        }
    }
}

// Independent CONV2D: Reference box blur / mean filter with boundary clamping
static void ref_conv2d(const float* input, float* output, size_t H, size_t W, size_t ksize) {
    int pad = static_cast<int>(ksize) / 2;
    for (size_t i = 0; i < H; ++i) {
        for (size_t j = 0; j < W; ++j) {
            double sum = 0.0;
            int count = 0;
            for (int ki = -pad; ki <= pad; ++ki) {
                for (int kj = -pad; kj <= pad; ++kj) {
                    int ni = static_cast<int>(i) + ki;
                    int nj = static_cast<int>(j) + kj;
                    if (ni >= 0 && ni < static_cast<int>(H) &&
                        nj >= 0 && nj < static_cast<int>(W)) {
                        sum += static_cast<double>(input[ni * W + nj]);
                        count++;
                    }
                }
            }
            output[i * W + j] = (count > 0) ? static_cast<float>(sum / static_cast<double>(count)) : 0.0f;
        }
    }
}

// Independent SPMV: Reference banded circulant sparse matrix-vector multiplication
static void ref_spmv(const float* vec, float* out, size_t N, size_t dim_n) {
    size_t nnz = std::min<size_t>(dim_n, 10);
    for (size_t i = 0; i < N; ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < nnz; ++j) {
            size_t col = (i + j) % N;
            sum += static_cast<double>(vec[col]);
        }
        out[i] = static_cast<float>(sum);
    }
}

// Independent STRING_TOKEN: Reference tokenizer checking exact delimiters (' ', '\t', '\n', '\0')
static uint32_t ref_string_token(const char* text, size_t len) {
    uint32_t tokens = 0;
    bool in_token = false;
    for (size_t i = 0; i < len; ++i) {
        char c = text[i];
        bool is_delim = (c == ' ' || c == '\n' || c == '\t' || c == '\0');
        if (is_delim) {
            in_token = false;
        } else if (!in_token) {
            in_token = true;
            tokens++;
        }
    }
    return tokens;
}

// Numerical comparison helper with relative and absolute tolerances
static bool compare_float_arrays(const float* actual, const float* ref, size_t count,
                                 float rtol, float atol, std::string& err_msg) {
    for (size_t i = 0; i < count; ++i) {
        float act = actual[i];
        float exp = ref[i];
        if (std::isnan(act) || std::isnan(exp)) {
            err_msg = "NaN detected at index " + std::to_string(i) + " (actual=" + std::to_string(act) + ", ref=" + std::to_string(exp) + ")";
            return false;
        }
        float diff = std::fabs(act - exp);
        float max_allowed = atol + rtol * std::fabs(exp);
        if (diff > max_allowed) {
            err_msg = "Mismatch at index " + std::to_string(i) + ": actual=" + std::to_string(act)
                      + ", ref=" + std::to_string(exp) + ", diff=" + std::to_string(diff)
                      + ", allowed=" + std::to_string(max_allowed);
            return false;
        }
    }
    return true;
}

// Helper to poison float buffers with NaN
static void poison_float_buffer(float* buf, size_t count) {
    float nan_val = std::numeric_limits<float>::quiet_NaN();
    for (size_t i = 0; i < count; ++i) {
        buf[i] = nan_val;
    }
}

// ============================================================================
// SGEMM Correctness Tests (CPU & GPU)
// ============================================================================
void test_sgemm_numerical_correctness(bool cuda_ok) {
    std::cout << "[Test 1] SGEMM Output Correctness (CPU & GPU)... " << std::flush;

    CPUScheduler cpu_sched(2);
    std::unique_ptr<GPUExecutor> gpu_exec;
    if (cuda_ok) {
        gpu_exec = std::make_unique<GPUExecutor>(2);
    }

    struct SgemmTestCase {
        size_t m, n, k;
    };
    std::vector<SgemmTestCase> cases = {
        {1, 1, 1},       // Minimal edge case
        {7, 13, 5},      // Prime, non-16-aligned dimensions
        {32, 32, 32},    // Power of two, block-aligned
        {64, 48, 80}     // Rectangular dimensions
    };

    // FP32 Tolerances justified by accumulation depth (K <= 80):
    // Bound = O(K * eps * |A| * |B|) with inputs in (0.001, 1.0] -> well within 1e-4
    const float rtol = 1e-4f;
    const float atol = 1e-4f;

    for (const auto& tc : cases) {
        size_t a_size = tc.m * tc.k;
        size_t b_size = tc.k * tc.n;
        size_t c_size = tc.m * tc.n;

        std::vector<float> input(a_size + b_size);
        for (size_t i = 0; i < input.size(); ++i) {
            input[i] = 0.05f + static_cast<float>((i * 37 + 11) % 1000) / 1050.0f;
        }

        std::vector<float> ref_output(c_size);
        ref_sgemm(input.data(), input.data() + a_size, ref_output.data(), tc.m, tc.n, tc.k);

        // 1. CPU Execution
        {
            std::vector<float> cpu_output(c_size);
            poison_float_buffer(cpu_output.data(), c_size);

            TaskDescriptor task{};
            task.task_id = 1001;
            task.type = TaskType::SGEMM;
            task.dim_m = tc.m;
            task.dim_n = tc.n;
            task.dim_k = tc.k;
            task.bytes_in = input.size() * sizeof(float);
            task.bytes_out = c_size * sizeof(float);
            task.host_input = input.data();
            task.host_output = cpu_output.data();

            auto fut = cpu_sched.EnqueueWithFuture(&task);
            fut.get();

            TEST_ASSERT(task.metrics.success == true, "CPU SGEMM execution must report success");
            std::string err;
            bool match = compare_float_arrays(cpu_output.data(), ref_output.data(), c_size, rtol, atol, err);
            TEST_ASSERT(match, ("CPU SGEMM failed reference check: " + err).c_str());
        }

        // 2. GPU Execution
        if (cuda_ok && gpu_exec) {
            std::vector<float> gpu_output(c_size);
            poison_float_buffer(gpu_output.data(), c_size);

            TaskDescriptor task{};
            task.task_id = 1002;
            task.type = TaskType::SGEMM;
            task.dim_m = tc.m;
            task.dim_n = tc.n;
            task.dim_k = tc.k;
            task.bytes_in = input.size() * sizeof(float);
            task.bytes_out = c_size * sizeof(float);
            task.host_input = input.data();
            task.host_output = gpu_output.data();

            gpu_exec->ExecuteAsync(&task, 0);

            TEST_ASSERT(task.metrics.success == true, "GPU SGEMM execution must report success");
            std::string err;
            bool match = compare_float_arrays(gpu_output.data(), ref_output.data(), c_size, rtol, atol, err);
            TEST_ASSERT(match, ("GPU SGEMM failed reference check: " + err).c_str());
        }
    }

    if (!cuda_ok) {
        std::cout << "PASSED (CPU verified; GPU skipped - no CUDA)" << std::endl;
    } else {
        std::cout << "PASSED (CPU and GPU numerically match independent reference)" << std::endl;
    }
}

// ============================================================================
// CONV2D Correctness Tests (CPU & GPU)
// ============================================================================
void test_conv2d_numerical_correctness(bool cuda_ok) {
    std::cout << "[Test 2] CONV2D Output Correctness (CPU & GPU)... " << std::flush;

    CPUScheduler cpu_sched(2);
    std::unique_ptr<GPUExecutor> gpu_exec;
    if (cuda_ok) {
        gpu_exec = std::make_unique<GPUExecutor>(2);
    }

    struct ConvTestCase {
        size_t h, w, ksize;
    };
    std::vector<ConvTestCase> cases = {
        {1, 1, 1},      // Single pixel 1x1
        {1, 1, 3},      // Single pixel with 3x3 kernel (tests boundary pad)
        {5, 7, 3},      // Small non-square, tests corners and edge counts
        {32, 48, 5}     // Rectangular image with 5x5 kernel
    };

    const float rtol = 1e-5f;
    const float atol = 1e-5f;

    for (const auto& tc : cases) {
        size_t pixels = tc.h * tc.w;
        std::vector<float> input(pixels);
        for (size_t i = 0; i < pixels; ++i) {
            input[i] = 0.1f + static_cast<float>((i * 43 + 7) % 500) / 550.0f;
        }

        std::vector<float> ref_output(pixels);
        ref_conv2d(input.data(), ref_output.data(), tc.h, tc.w, tc.ksize);

        // CPU Execution
        {
            std::vector<float> cpu_output(pixels);
            poison_float_buffer(cpu_output.data(), pixels);

            TaskDescriptor task{};
            task.task_id = 2001;
            task.type = TaskType::CONV2D;
            task.dim_m = tc.h;
            task.dim_n = tc.w;
            task.dim_k = tc.ksize;
            task.bytes_in = pixels * sizeof(float);
            task.bytes_out = pixels * sizeof(float);
            task.host_input = input.data();
            task.host_output = cpu_output.data();

            auto fut = cpu_sched.EnqueueWithFuture(&task);
            fut.get();

            TEST_ASSERT(task.metrics.success == true, "CPU CONV2D execution must succeed");
            std::string err;
            bool match = compare_float_arrays(cpu_output.data(), ref_output.data(), pixels, rtol, atol, err);
            TEST_ASSERT(match, ("CPU CONV2D failed reference check: " + err).c_str());
        }

        // GPU Execution
        if (cuda_ok && gpu_exec) {
            std::vector<float> gpu_output(pixels);
            poison_float_buffer(gpu_output.data(), pixels);

            TaskDescriptor task{};
            task.task_id = 2002;
            task.type = TaskType::CONV2D;
            task.dim_m = tc.h;
            task.dim_n = tc.w;
            task.dim_k = tc.ksize;
            task.bytes_in = pixels * sizeof(float);
            task.bytes_out = pixels * sizeof(float);
            task.host_input = input.data();
            task.host_output = gpu_output.data();

            gpu_exec->ExecuteAsync(&task, 0);

            TEST_ASSERT(task.metrics.success == true, "GPU CONV2D execution must succeed");
            std::string err;
            bool match = compare_float_arrays(gpu_output.data(), ref_output.data(), pixels, rtol, atol, err);
            TEST_ASSERT(match, ("GPU CONV2D failed reference check: " + err).c_str());
        }
    }

    if (!cuda_ok) {
        std::cout << "PASSED (CPU verified; GPU skipped - no CUDA)" << std::endl;
    } else {
        std::cout << "PASSED (CPU and GPU match boundary-aware mean filter reference)" << std::endl;
    }
}

// ============================================================================
// SPMV Correctness Tests (CPU & GPU)
// ============================================================================
void test_spmv_numerical_correctness(bool cuda_ok) {
    std::cout << "[Test 3] SPMV Output Correctness (CPU & GPU)... " << std::flush;

    CPUScheduler cpu_sched(2);
    std::unique_ptr<GPUExecutor> gpu_exec;
    if (cuda_ok) {
        gpu_exec = std::make_unique<GPUExecutor>(2);
    }

    struct SpmvTestCase {
        size_t n, dim_n;
    };
    std::vector<SpmvTestCase> cases = {
        {1, 5},     // N=1
        {4, 8},     // N=4 < 10, dim_n > N -> wraps around row
        {15, 20},   // dim_n > 10 -> verifies clamp to 10 non-zeros
        {128, 6},   // Standard power-of-two size
        {307, 10}   // Prime size > 256 (tests block boundary in GPU kernel)
    };

    const float rtol = 1e-5f;
    const float atol = 1e-5f;

    for (const auto& tc : cases) {
        std::vector<float> vec(tc.n);
        for (size_t i = 0; i < tc.n; ++i) {
            vec[i] = 0.5f + static_cast<float>((i * 19 + 3) % 200) / 210.0f;
        }

        std::vector<float> ref_output(tc.n);
        ref_spmv(vec.data(), ref_output.data(), tc.n, tc.dim_n);

        // CPU Execution
        {
            std::vector<float> cpu_output(tc.n);
            poison_float_buffer(cpu_output.data(), tc.n);

            TaskDescriptor task{};
            task.task_id = 3001;
            task.type = TaskType::SPMV;
            task.dim_m = tc.n;
            task.dim_n = tc.dim_n;
            task.bytes_in = tc.n * sizeof(float);
            task.bytes_out = tc.n * sizeof(float);
            task.host_input = vec.data();
            task.host_output = cpu_output.data();

            auto fut = cpu_sched.EnqueueWithFuture(&task);
            fut.get();

            TEST_ASSERT(task.metrics.success == true, "CPU SPMV execution must succeed");
            std::string err;
            bool match = compare_float_arrays(cpu_output.data(), ref_output.data(), tc.n, rtol, atol, err);
            TEST_ASSERT(match, ("CPU SPMV failed reference check: " + err).c_str());
        }

        // GPU Execution
        if (cuda_ok && gpu_exec) {
            std::vector<float> gpu_output(tc.n);
            poison_float_buffer(gpu_output.data(), tc.n);

            TaskDescriptor task{};
            task.task_id = 3002;
            task.type = TaskType::SPMV;
            task.dim_m = tc.n;
            task.dim_n = tc.dim_n;
            task.bytes_in = tc.n * sizeof(float);
            task.bytes_out = tc.n * sizeof(float);
            task.host_input = vec.data();
            task.host_output = gpu_output.data();

            gpu_exec->ExecuteAsync(&task, 0);

            TEST_ASSERT(task.metrics.success == true, "GPU SPMV execution must succeed");
            std::string err;
            bool match = compare_float_arrays(gpu_output.data(), ref_output.data(), tc.n, rtol, atol, err);
            TEST_ASSERT(match, ("GPU SPMV failed reference check: " + err).c_str());
        }
    }

    if (!cuda_ok) {
        std::cout << "PASSED (CPU verified; GPU skipped - no CUDA)" << std::endl;
    } else {
        std::cout << "PASSED (CPU and GPU match sparse banded reference)" << std::endl;
    }
}

// ============================================================================
// QUICKSORT Correctness Tests (Sorting, Preservation, Edge Cases, GPU Rejection)
// ============================================================================
void test_quicksort_numerical_correctness(bool cuda_ok) {
    std::cout << "[Test 4] QUICKSORT Element Preservation & Sorted Order... " << std::flush;

    CPUScheduler cpu_sched(2);

    std::vector<std::vector<int>> test_arrays = {
        {42},                                         // Size 1
        {2, 1},                                       // Size 2 reverse
        {1, 2},                                       // Size 2 sorted
        {7, 7, 7, 7, 7, 7, 7},                        // All duplicates
        {10, 20, 30, 40, 50, 60},                     // Already sorted
        {90, 80, 70, 60, 50, 40, 30, 20, 10},         // Reverse sorted
        {5, -2, 5, 0, 12, -999, 5, 12, -2, 0, 100},  // Mixed duplicates and negatives
        {1000, -500, 250, -125, 62, -31, 15, -7, 3}  // Alternating signs
    };

    for (auto& arr : test_arrays) {
        size_t n = arr.size();
        std::vector<int> working = arr;
        std::vector<int> expected = arr;
        std::sort(expected.begin(), expected.end());

        TaskDescriptor task{};
        task.task_id = 4001;
        task.type = TaskType::QUICKSORT;
        task.dim_m = n;
        task.bytes_in = n * sizeof(int);
        task.bytes_out = n * sizeof(int);
        task.host_input = working.data();
        task.host_output = working.data(); // In-place quicksort

        auto fut = cpu_sched.EnqueueWithFuture(&task);
        fut.get();

        TEST_ASSERT(task.metrics.success == true, "CPU QUICKSORT must report success");

        // 1. Verify sorted order
        for (size_t i = 1; i < n; ++i) {
            TEST_ASSERT(working[i - 1] <= working[i], "QUICKSORT output array must be sorted");
        }

        // 2. Verify element preservation (multiset equality)
        TEST_ASSERT(working == expected, "QUICKSORT must strictly preserve all input elements and duplicates");
    }

    // Defensive test: Verify GPUExecutor strictly rejects QUICKSORT
    if (cuda_ok) {
        GPUExecutor gpu_exec(1);
        std::vector<int> data = {5, 3, 1, 4, 2};
        std::vector<int> backup = data;

        TaskDescriptor task{};
        task.task_id = 4002;
        task.type = TaskType::QUICKSORT;
        task.dim_m = data.size();
        task.bytes_in = data.size() * sizeof(int);
        task.bytes_out = data.size() * sizeof(int);
        task.host_input = data.data();
        task.host_output = data.data();

        gpu_exec.ExecuteAsync(&task, 0);

        TEST_ASSERT(task.metrics.success == false, "GPUExecutor must safely reject unsupported QUICKSORT task");
        TEST_ASSERT(data == backup, "Rejected GPU task must not corrupt input/output buffers");
    }

    std::cout << "PASSED (preserves all duplicates, sorted/reverse orders verified, GPU rejection safe)" << std::endl;
}

// ============================================================================
// STRING_TOKEN Correctness Tests (Semantics, Delimiters, Punctuation, GPU Rejection)
// ============================================================================
void test_string_token_correctness(bool cuda_ok) {
    std::cout << "[Test 5] STRING_TOKEN Tokenization Semantics & Delimiters... " << std::flush;

    CPUScheduler cpu_sched(2);

    struct TokenTestCase {
        std::string text;
        std::string desc;
    };

    std::vector<TokenTestCase> cases = {
        {"", "empty string"},
        {"       ", "only spaces"},
        {"\n\t  \t\n  \r", "only whitespace delimiters"},
        {"heterosched", "single word with no delimiters"},
        {"hello world foo bar", "multiple space-separated words"},
        {"   leading and trailing spaces   ", "leading and trailing whitespace"},
        {"mixed\nseparators\twith\nconsecutive\t\t\n\ndelimiters", "consecutive mixed delimiters"},
        {"alpha-beta, gamma.delta! (epsilon)", "punctuation retained inside tokens"},
        {"a b c d e f g h i j", "single-character tokens"}
    };

    for (const auto& tc : cases) {
        size_t len = tc.text.size();
        uint32_t expected_tokens = ref_string_token(tc.text.c_str(), len);

        // Buffer poisoning: pre-initialize output to 0xFFFFFFFF to guarantee real write occurs
        uint32_t token_count_out = 0xFFFFFFFFU;

        std::vector<char> input_buf(len + 1, '\0');
        if (len > 0) {
            std::memcpy(input_buf.data(), tc.text.data(), len);
        }

        TaskDescriptor task{};
        task.task_id = 5001;
        task.type = TaskType::STRING_TOKEN;
        task.dim_m = len;
        task.bytes_in = (len > 0 ? len : 1) * sizeof(char);
        task.bytes_out = sizeof(uint32_t);
        task.host_input = input_buf.data();
        task.host_output = &token_count_out;

        auto fut = cpu_sched.EnqueueWithFuture(&task);
        fut.get();

        TEST_ASSERT(task.metrics.success == true, ("CPU STRING_TOKEN must succeed for: " + tc.desc).c_str());
        TEST_ASSERT(token_count_out == expected_tokens,
                    ("Token count mismatch for [" + tc.desc + "]: actual=" + std::to_string(token_count_out) +
                     ", expected=" + std::to_string(expected_tokens)).c_str());
    }

    // Defensive test: Verify GPUExecutor strictly rejects STRING_TOKEN
    if (cuda_ok) {
        GPUExecutor gpu_exec(1);
        std::string sample = "heterosched token validation";
        uint32_t token_out = 0xDEADBEEFU;

        TaskDescriptor task{};
        task.task_id = 5002;
        task.type = TaskType::STRING_TOKEN;
        task.dim_m = sample.size();
        task.bytes_in = sample.size();
        task.bytes_out = sizeof(uint32_t);
        task.host_input = const_cast<char*>(sample.data());
        task.host_output = &token_out;

        gpu_exec.ExecuteAsync(&task, 0);

        TEST_ASSERT(task.metrics.success == false, "GPUExecutor must safely reject STRING_TOKEN");
        TEST_ASSERT(token_out == 0xDEADBEEFU, "Poisoned output buffer must remain intact upon GPU rejection");
    }

    std::cout << "PASSED (exact delimiter semantics, non-whitespace punctuation preserved, GPU rejection safe)" << std::endl;
}

// ============================================================================
// Anti-False-Positive & Null Buffer Defense Tests
// ============================================================================
void test_anti_false_positive_and_null_buffers(bool cuda_ok) {
    std::cout << "[Test 6] Anti-False-Positive & Null Buffer Defense... " << std::flush;

    CPUScheduler cpu_sched(2);

    // 1. Task with null input buffer must fail on CPU
    {
        float dummy_out = 0.0f;
        TaskDescriptor task{};
        task.task_id = 6001;
        task.type = TaskType::SGEMM;
        task.dim_m = 4; task.dim_n = 4; task.dim_k = 4;
        task.bytes_in = 32 * sizeof(float);
        task.bytes_out = 16 * sizeof(float);
        task.host_input = nullptr;
        task.host_output = &dummy_out;

        auto fut = cpu_sched.EnqueueWithFuture(&task);
        fut.get();
        TEST_ASSERT(task.metrics.success == false, "CPU scheduler must reject task with host_input == nullptr");
    }

    // 2. Task with null output buffer must fail on CPU
    {
        float dummy_in = 1.0f;
        TaskDescriptor task{};
        task.task_id = 6002;
        task.type = TaskType::CONV2D;
        task.dim_m = 4; task.dim_n = 4; task.dim_k = 3;
        task.bytes_in = 16 * sizeof(float);
        task.bytes_out = 16 * sizeof(float);
        task.host_input = &dummy_in;
        task.host_output = nullptr;

        auto fut = cpu_sched.EnqueueWithFuture(&task);
        fut.get();
        TEST_ASSERT(task.metrics.success == false, "CPU scheduler must reject task with host_output == nullptr");
    }

    // 3. Null buffers on GPU must fail cleanly
    if (cuda_ok) {
        GPUExecutor gpu_exec(1);
        TaskDescriptor task{};
        task.task_id = 6003;
        task.type = TaskType::SGEMM;
        task.dim_m = 4; task.dim_n = 4; task.dim_k = 4;
        task.bytes_in = 32 * sizeof(float);
        task.bytes_out = 16 * sizeof(float);
        task.host_input = nullptr;
        task.host_output = nullptr;

        gpu_exec.ExecuteAsync(&task, 0);
        TEST_ASSERT(task.metrics.success == false, "GPU executor must reject task with null buffers");
        TEST_ASSERT(task.metrics.t_exec_us == 0.0, "Failed GPU task must report 0 execution time");
        TEST_ASSERT(task.metrics.t_transfer_us == 0.0, "Failed GPU task must report 0 transfer time");
    }

    std::cout << "PASSED (uninitialized/null tasks strictly fail without accidental success)" << std::endl;
}

// ============================================================================
// CPU Worker Exception Handling & Recovery Tests
// ============================================================================
void test_cpu_worker_exception_resilience() {
    std::cout << "[Test 7] CPU Worker Exception Handling & Post-Failure Recovery... " << std::flush;

    CPUScheduler scheduler(2);

    // 1. Enqueue a task that triggers an execution exception (priority == 0xDEAD)
    float dummy_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    TaskDescriptor throwing_task{};
    throwing_task.task_id = 7001;
    throwing_task.type = TaskType::SGEMM;
    throwing_task.priority = 0xDEAD; // Triggers simulated exception in ExecuteCPUTask
    throwing_task.dim_m = 2; throwing_task.dim_n = 2; throwing_task.dim_k = 2;
    throwing_task.bytes_in = 4 * sizeof(float);
    throwing_task.bytes_out = 4 * sizeof(float);
    throwing_task.host_input = dummy_data;
    throwing_task.host_output = dummy_data;

    auto fut1 = scheduler.EnqueueWithFuture(&throwing_task);
    bool caught_exception = false;
    try {
        fut1.get();
    } catch (const std::exception& e) {
        caught_exception = true;
    }
    TEST_ASSERT(caught_exception == true, "EnqueueWithFuture must propagate task execution exception via future.get()");
    TEST_ASSERT(throwing_task.metrics.success == false, "Throwing task must be marked with success == false");
    TEST_ASSERT(throwing_task.metrics.t_exec_us == 0.0, "Throwing task must report t_exec_us == 0.0");

    // 2. Critical verification: Worker thread must remain healthy and process subsequent valid task
    const char* sample_text = "robust worker resilience validation";
    size_t len = std::strlen(sample_text);
    std::vector<char> text_buf(sample_text, sample_text + len);
    uint32_t token_count = 0;

    TaskDescriptor valid_task{};
    valid_task.task_id = 7002;
    valid_task.type = TaskType::STRING_TOKEN;
    valid_task.priority = 1;
    valid_task.dim_m = len;
    valid_task.bytes_in = len;
    valid_task.bytes_out = sizeof(uint32_t);
    valid_task.host_input = text_buf.data();
    valid_task.host_output = &token_count;

    auto fut2 = scheduler.EnqueueWithFuture(&valid_task);
    fut2.get(); // Must not deadlock, hang, or throw
    TEST_ASSERT(valid_task.metrics.success == true, "Worker must process subsequent valid task after prior failure");
    TEST_ASSERT(token_count == 4, "Subsequent task must compute correct token count (4)");
    TEST_ASSERT(valid_task.metrics.t_exec_us > 0.0, "Subsequent task must record valid execution time");

    std::cout << "PASSED (exception safely propagated via future.get(), zero deadlock, subsequent tasks succeed)" << std::endl;
}

// ============================================================================
// Main
// ============================================================================
int main() {
    std::cout << "\n=== HeteroSched Phase C Numerical Correctness Suite ===\n";

    std::string cuda_reason;
    bool cuda_ok = is_cuda_available(cuda_reason);
    if (cuda_ok) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        std::cout << "[CUDA Hardware Available] Device: " << prop.name
                  << " (Compute " << prop.major << "." << prop.minor << ")\n";
    } else {
        std::cout << "[CUDA Unavailable] " << cuda_reason << "\n"
                  << "  GPU tests will be reported as SKIPPED, not passed.\n";
    }

    test_sgemm_numerical_correctness(cuda_ok);
    test_conv2d_numerical_correctness(cuda_ok);
    test_spmv_numerical_correctness(cuda_ok);
    test_quicksort_numerical_correctness(cuda_ok);
    test_string_token_correctness(cuda_ok);
    test_anti_false_positive_and_null_buffers(cuda_ok);
    test_cpu_worker_exception_resilience();

    std::cout << "=== ALL PHASE C CORRECTNESS TESTS PASSED ===\n\n";
    return 0;
}
