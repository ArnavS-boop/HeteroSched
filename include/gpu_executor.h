#ifndef GPU_EXECUTOR_H
#define GPU_EXECUTOR_H

#include "common.h"
#include <vector>
#include <cuda_runtime.h>

class GPUExecutor {
public:
    explicit GPUExecutor(size_t stream_pool_size);
    ~GPUExecutor();

    void ExecuteAsync(TaskDescriptor* task, size_t stream_idx);
    size_t PoolSize() const { return pool_size_; }

private:
    std::vector<cudaStream_t> streams_;
    size_t pool_size_;
};

#endif
