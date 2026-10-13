#include "common.h"
#include <cstdlib>
#include <cstring>
#include <cmath>

void allocate_sgemm_buffers(TaskDescriptor* task) {
    size_t a_size = task->dim_m * task->dim_k;
    size_t b_size = task->dim_k * task->dim_n;
    size_t c_size = task->dim_m * task->dim_n;
    task->bytes_in = (a_size + b_size) * sizeof(float);
    task->bytes_out = c_size * sizeof(float);
    task->estimated_flops = 2ULL * task->dim_m * task->dim_n * task->dim_k;

    float* input = static_cast<float*>(malloc(task->bytes_in));
    float* output = static_cast<float*>(calloc(c_size, sizeof(float)));
    for (size_t i = 0; i < a_size + b_size; ++i) {
        input[i] = static_cast<float>(rand()) / RAND_MAX;
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
    for (size_t i = 0; i < n; ++i) {
        data[i] = rand();
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
    for (size_t i = 0; i < pixels; ++i) {
        input[i] = static_cast<float>(rand()) / RAND_MAX;
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
    for (size_t i = 0; i < n; ++i) {
        vec[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    task->host_input = vec;
    task->host_output = out;
}

void allocate_task_buffers(TaskDescriptor* task) {
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
        default:
            break;
    }
}

void free_task_buffers(TaskDescriptor* task) {
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
