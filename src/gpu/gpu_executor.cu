#include "gpu_executor.h"
#include <cstdio>

__global__ void sgemm_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float val = 0.0f;
        for (int k = 0; k < K; ++k) {
            val += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = val;
    }
}

__global__ void conv2d_kernel(const float* input, float* output, int H, int W, int ksize) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < H && col < W) {
        float sum = 0.0f;
        int count = 0;
        int pad = ksize / 2;
        for (int ki = -pad; ki <= pad; ++ki) {
            for (int kj = -pad; kj <= pad; ++kj) {
                int ni = row + ki;
                int nj = col + kj;
                if (ni >= 0 && ni < H && nj >= 0 && nj < W) {
                    sum += input[ni * W + nj];
                    count++;
                }
            }
        }
        output[row * W + col] = sum / static_cast<float>(count);
    }
}

__global__ void spmv_kernel(const float* vec, float* out, int n, int nnz_per_row) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float sum = 0.0f;
        for (int j = 0; j < nnz_per_row; ++j) {
            int col = (i + j) % n;
            sum += vec[col] * 1.0f;
        }
        out[i] = sum;
    }
}

GPUExecutor::GPUExecutor(size_t stream_pool_size) : pool_size_(stream_pool_size) {
    streams_.resize(pool_size_);
    int low_prio, high_prio;
    cudaDeviceGetStreamPriorityRange(&low_prio, &high_prio);
    for (size_t i = 0; i < pool_size_; ++i) {
        int prio = (i == 0) ? high_prio : low_prio;
        cudaStreamCreateWithPriority(&streams_[i], cudaStreamNonBlocking, prio);
    }
}

GPUExecutor::~GPUExecutor() {
    for (auto& s : streams_) {
        cudaStreamSynchronize(s);
        cudaStreamDestroy(s);
    }
}

void GPUExecutor::ExecuteAsync(TaskDescriptor* task, size_t stream_idx) {
    cudaStream_t stream = streams_[stream_idx % pool_size_];
    task->metrics.stream_id_used = static_cast<uint32_t>(stream_idx % pool_size_);

    cudaEvent_t start_transfer, start_kernel, end_kernel, end_transfer;
    cudaEventCreate(&start_transfer);
    cudaEventCreate(&start_kernel);
    cudaEventCreate(&end_kernel);
    cudaEventCreate(&end_transfer);

    void *d_in = nullptr, *d_out = nullptr;
    cudaMallocAsync(&d_in, task->bytes_in, stream);
    cudaMallocAsync(&d_out, task->bytes_out, stream);

    cudaEventRecord(start_transfer, stream);
    cudaMemcpyAsync(d_in, task->host_input, task->bytes_in, cudaMemcpyHostToDevice, stream);
    cudaEventRecord(start_kernel, stream);

    switch (task->type) {
        case TaskType::SGEMM: {
            dim3 threads(16, 16);
            dim3 blocks((task->dim_n + 15) / 16, (task->dim_m + 15) / 16);
            const float* d_A = reinterpret_cast<const float*>(d_in);
            const float* d_B = d_A + (task->dim_m * task->dim_k);
            float* d_C = reinterpret_cast<float*>(d_out);
            sgemm_kernel<<<blocks, threads, 0, stream>>>(d_A, d_B, d_C,
                static_cast<int>(task->dim_m), static_cast<int>(task->dim_n),
                static_cast<int>(task->dim_k));
            break;
        }
        case TaskType::CONV2D: {
            dim3 threads(16, 16);
            dim3 blocks((task->dim_n + 15) / 16, (task->dim_m + 15) / 16);
            conv2d_kernel<<<blocks, threads, 0, stream>>>(
                reinterpret_cast<const float*>(d_in),
                reinterpret_cast<float*>(d_out),
                static_cast<int>(task->dim_m), static_cast<int>(task->dim_n),
                static_cast<int>(task->dim_k));
            break;
        }
        case TaskType::SPMV: {
            int n = static_cast<int>(task->dim_m);
            int nnz = static_cast<int>(std::min<size_t>(task->dim_n, 10));
            int block_size = 256;
            int grid_size = (n + block_size - 1) / block_size;
            spmv_kernel<<<grid_size, block_size, 0, stream>>>(
                reinterpret_cast<const float*>(d_in),
                reinterpret_cast<float*>(d_out),
                n, nnz);
            break;
        }
        default:
            break;
    }

    cudaEventRecord(end_kernel, stream);
    cudaMemcpyAsync(task->host_output, d_out, task->bytes_out, cudaMemcpyDeviceToHost, stream);
    cudaEventRecord(end_transfer, stream);

    cudaFreeAsync(d_in, stream);
    cudaFreeAsync(d_out, stream);

    cudaStreamSynchronize(stream);

    float t_transfer_in = 0.0f, t_kernel = 0.0f, t_transfer_out = 0.0f;
    cudaEventElapsedTime(&t_transfer_in, start_transfer, start_kernel);
    cudaEventElapsedTime(&t_kernel, start_kernel, end_kernel);
    cudaEventElapsedTime(&t_transfer_out, end_kernel, end_transfer);

    task->metrics.t_transfer_us = (t_transfer_in + t_transfer_out) * 1000.0;
    task->metrics.t_exec_us = t_kernel * 1000.0;

    cudaEventDestroy(start_transfer);
    cudaEventDestroy(start_kernel);
    cudaEventDestroy(end_kernel);
    cudaEventDestroy(end_transfer);
}
