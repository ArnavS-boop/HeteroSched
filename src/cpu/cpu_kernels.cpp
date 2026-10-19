#include "common.h"
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>

// Deterministic fast inline 32-bit Xorshift PRNG (Marsaglia, 2003)
// Seed strategy: Knuth's multiplicative golden-ratio hash of task_id + constant offset
// Guarantees independent, reproducible, non-zero initial state per task.
static inline uint32_t init_task_prng_state(uint64_t task_id) {
    uint32_t state = static_cast<uint32_t>((task_id + 1ULL) * 2654435761ULL);
    return (state == 0) ? 0x12345678U : state;
}

static inline uint32_t xorshift32(uint32_t& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

// Bounded in (0.001f, 1.0f] to guarantee valid, non-zero FP32 workload data
static inline float fast_task_rand_float(uint32_t& state) {
    uint32_t val = xorshift32(state);
    return 0.001f + static_cast<float>(val & 0x00FFFFFF) * (0.999f / 16777216.0f);
}

// Positive non-zero 31-bit integer for Quicksort
static inline int fast_task_rand_int(uint32_t& state) {
    uint32_t val = xorshift32(state);
    return static_cast<int>(val & 0x7FFFFFFF);
}

void allocate_sgemm_buffers(TaskDescriptor* task) {
    size_t a_size = task->dim_m * task->dim_k;
    size_t b_size = task->dim_k * task->dim_n;
    size_t c_size = task->dim_m * task->dim_n;
    task->bytes_in = (a_size + b_size) * sizeof(float);
    task->bytes_out = c_size * sizeof(float);
    task->estimated_flops = 2ULL * task->dim_m * task->dim_n * task->dim_k;

    float* input = static_cast<float*>(malloc(task->bytes_in));
    float* output = static_cast<float*>(calloc(c_size, sizeof(float)));
    if (!input || !output) {
        if (input) free(input);
        if (output) free(output);
        task->host_input = nullptr;
        task->host_output = nullptr;
        return;
    }
    uint32_t state = init_task_prng_state(task->task_id);
    for (size_t i = 0; i < a_size + b_size; ++i) {
        input[i] = fast_task_rand_float(state);
    }
    task->host_input = input;
    task->host_output = output;
}

void allocate_quicksort_buffers(TaskDescriptor* task) {
    size_t n = task->dim_m;
    task->bytes_in = n * sizeof(int);
    task->bytes_out = n * sizeof(int);
    task->estimated_flops = static_cast<uint64_t>(n * log2(static_cast<double>(n)));

    int* data = static_cast<int*>(malloc(task->bytes_in));
    if (!data) {
        task->host_input = nullptr;
        task->host_output = nullptr;
        return;
    }
    uint32_t state = init_task_prng_state(task->task_id);
    for (size_t i = 0; i < n; ++i) {
        data[i] = fast_task_rand_int(state);
    }
    task->host_input = data;
    task->host_output = data;
}

void allocate_conv2d_buffers(TaskDescriptor* task) {
    size_t pixels = task->dim_m * task->dim_n;
    size_t ksize = task->dim_k;
    task->bytes_in = pixels * sizeof(float);
    task->bytes_out = pixels * sizeof(float);
    task->estimated_flops = pixels * ksize * ksize * 2;

    float* input = static_cast<float*>(malloc(task->bytes_in));
    float* output = static_cast<float*>(calloc(pixels, sizeof(float)));
    if (!input || !output) {
        if (input) free(input);
        if (output) free(output);
        task->host_input = nullptr;
        task->host_output = nullptr;
        return;
    }
    uint32_t state = init_task_prng_state(task->task_id);
    for (size_t i = 0; i < pixels; ++i) {
        input[i] = fast_task_rand_float(state);
    }
    task->host_input = input;
    task->host_output = output;
}

void allocate_spmv_buffers(TaskDescriptor* task) {
    size_t n = task->dim_m;
    task->bytes_in = n * sizeof(float);
    task->bytes_out = n * sizeof(float);
    task->estimated_flops = n * task->dim_n * 2;

    float* vec = static_cast<float*>(malloc(task->bytes_in));
    float* out = static_cast<float*>(calloc(n, sizeof(float)));
    if (!vec || !out) {
        if (vec) free(vec);
        if (out) free(out);
        task->host_input = nullptr;
        task->host_output = nullptr;
        return;
    }
    uint32_t state = init_task_prng_state(task->task_id);
    for (size_t i = 0; i < n; ++i) {
        vec[i] = fast_task_rand_float(state);
    }
    task->host_input = vec;
    task->host_output = out;
}

void allocate_string_token_buffers(TaskDescriptor* task) {
    size_t len = task->dim_m;
    if (len == 0) len = 1;
    task->bytes_in = len * sizeof(char);
    task->bytes_out = sizeof(uint32_t);
    task->estimated_flops = len;

    char* text = static_cast<char*>(malloc(task->bytes_in));
    uint32_t* token_out = static_cast<uint32_t*>(calloc(1, sizeof(uint32_t)));
    if (!text || !token_out) {
        if (text) free(text);
        if (token_out) free(token_out);
        task->host_input = nullptr;
        task->host_output = nullptr;
        return;
    }

    uint32_t state = init_task_prng_state(task->task_id);
    for (size_t i = 0; i < len; ++i) {
        uint32_t val = xorshift32(state);
        uint32_t mod = val % 10;
        if (mod == 0) {
            text[i] = ' ';
        } else if (mod == 1 && (val & 1)) {
            text[i] = '\n';
        } else if (mod == 1) {
            text[i] = '\t';
        } else {
            text[i] = static_cast<char>('a' + (val % 26));
        }
    }

    task->host_input = text;
    task->host_output = token_out;
}

bool allocate_task_buffers(TaskDescriptor* task) {
    if (!task) return false;
    task->metrics.success = false;
    switch (task->type) {
        case TaskType::SGEMM:
            allocate_sgemm_buffers(task);
            break;
        case TaskType::QUICKSORT:
            allocate_quicksort_buffers(task);
            break;
        case TaskType::CONV2D:
            allocate_conv2d_buffers(task);
            break;
        case TaskType::SPMV:
            allocate_spmv_buffers(task);
            break;
        case TaskType::STRING_TOKEN:
            allocate_string_token_buffers(task);
            break;
        default:
            task->host_input = nullptr;
            task->host_output = nullptr;
            break;
    }
    bool ok = (task->host_input != nullptr && task->host_output != nullptr);
    task->metrics.success = false;
    return ok;
}

void free_task_buffers(TaskDescriptor* task) {
    if (!task) return;
    if (task->type == TaskType::QUICKSORT) {
        free(task->host_input);
        task->host_input = nullptr;
        task->host_output = nullptr;
        return;
    }
    if (task->host_input) {
        free(task->host_input);
        task->host_input = nullptr;
    }
    if (task->host_output) {
        free(task->host_output);
        task->host_output = nullptr;
    }
}
