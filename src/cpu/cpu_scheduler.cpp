#include "cpu_scheduler.h"
#include <pthread.h>
#include <sched.h>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstring>

CPUScheduler::CPUScheduler(size_t num_workers) : stop_requested_(false) {
    for (size_t i = 0; i < num_workers; ++i) {
        workers_.emplace_back(&CPUScheduler::WorkerLoop, this, i);
    }
}

CPUScheduler::~CPUScheduler() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        stop_requested_ = true;
    }
    cv_.notify_all();
    for (auto& worker : workers_) {
        if (worker.joinable()) worker.join();
    }
}

void CPUScheduler::Enqueue(TaskDescriptor* task) {
    auto* p = new std::promise<void>();
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        task_queue_.push({task, p});
    }
    cv_.notify_one();
}

std::future<void> CPUScheduler::EnqueueWithFuture(TaskDescriptor* task) {
    auto* p = new std::promise<void>();
    auto f = p->get_future();
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        task_queue_.push({task, p});
    }
    cv_.notify_one();
    return f;
}

size_t CPUScheduler::PendingTasks() {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    return task_queue_.size();
}

void CPUScheduler::WorkerLoop(size_t worker_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(worker_id % std::thread::hardware_concurrency(), &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

    while (true) {
        TaskDescriptor* task = nullptr;
        std::promise<void>* promise = nullptr;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            cv_.wait(lock, [this] { return stop_requested_ || !task_queue_.empty(); });
            if (stop_requested_ && task_queue_.empty()) return;
            auto& front = task_queue_.front();
            task = front.first;
            promise = front.second;
            task_queue_.pop();
        }

        if (task) {
            auto start_exec = std::chrono::high_resolution_clock::now();
            ExecuteCPUTask(task, worker_id);
            auto end_exec = std::chrono::high_resolution_clock::now();
            task->metrics.t_exec_us = std::chrono::duration<double, std::micro>(end_exec - start_exec).count();
            task->metrics.cpu_core_used = static_cast<int>(worker_id);
        }

        if (promise) {
            promise->set_value();
            delete promise;
        }
    }
}

void CPUScheduler::ExecuteCPUTask(TaskDescriptor* task, size_t /*core_id*/) {
    switch (task->type) {
        case TaskType::SGEMM: {
            float* A = reinterpret_cast<float*>(task->host_input);
            float* B = A + (task->dim_m * task->dim_k);
            float* C = reinterpret_cast<float*>(task->host_output);
            for (size_t i = 0; i < task->dim_m; ++i) {
                for (size_t j = 0; j < task->dim_n; ++j) {
                    float sum = 0.0f;
                    for (size_t k = 0; k < task->dim_k; ++k) {
                        sum += A[i * task->dim_k + k] * B[k * task->dim_n + j];
                    }
                    C[i * task->dim_n + j] = sum;
                }
            }
            break;
        }
        case TaskType::CONV2D: {
            float* input = reinterpret_cast<float*>(task->host_input);
            float* output = reinterpret_cast<float*>(task->host_output);
            size_t h = task->dim_m;
            size_t w = task->dim_n;
            int ksize = static_cast<int>(task->dim_k);
            int pad = ksize / 2;
            for (size_t i = 0; i < h; ++i) {
                for (size_t j = 0; j < w; ++j) {
                    float sum = 0.0f;
                    int count = 0;
                    for (int ki = -pad; ki <= pad; ++ki) {
                        for (int kj = -pad; kj <= pad; ++kj) {
                            int ni = static_cast<int>(i) + ki;
                            int nj = static_cast<int>(j) + kj;
                            if (ni >= 0 && ni < static_cast<int>(h) &&
                                nj >= 0 && nj < static_cast<int>(w)) {
                                sum += input[ni * w + nj];
                                count++;
                            }
                        }
                    }
                    output[i * w + j] = sum / static_cast<float>(count);
                }
            }
            break;
        }
        case TaskType::SPMV: {
            float* vec = reinterpret_cast<float*>(task->host_input);
            float* out = reinterpret_cast<float*>(task->host_output);
            size_t n = task->dim_m;
            for (size_t i = 0; i < n; ++i) {
                float sum = 0.0f;
                size_t nnz_per_row = std::min<size_t>(task->dim_n, 10);
                for (size_t j = 0; j < nnz_per_row; ++j) {
                    size_t col = (i + j) % n;
                    sum += vec[col] * 1.0f;
                }
                out[i] = sum;
            }
            break;
        }
        case TaskType::QUICKSORT: {
            int* arr = reinterpret_cast<int*>(task->host_input);
            size_t n = task->dim_m;
            std::sort(arr, arr + n);
            if (task->host_output && task->host_output != task->host_input) {
                std::memcpy(task->host_output, task->host_input, n * sizeof(int));
            }
            break;
        }
        case TaskType::STRING_TOKEN: {
            char* text = reinterpret_cast<char*>(task->host_input);
            uint32_t* token_count = reinterpret_cast<uint32_t*>(task->host_output);
            size_t len = task->dim_m;
            uint32_t tokens = 0;
            bool in_token = false;
            for (size_t i = 0; i < len; ++i) {
                if (text[i] == ' ' || text[i] == '\n' || text[i] == '\t' || text[i] == '\0') {
                    in_token = false;
                } else if (!in_token) {
                    in_token = true;
                    tokens++;
                }
            }
            *token_count = tokens;
            break;
        }
    }
}
