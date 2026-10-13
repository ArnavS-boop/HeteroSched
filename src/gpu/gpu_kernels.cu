#include <cuda_runtime.h>
#include <cstdio>

void gpu_warmup() {
    void* d_tmp = nullptr;
    cudaMalloc(&d_tmp, 1024);
    cudaFree(d_tmp);
    cudaDeviceSynchronize();
}

void print_gpu_info() {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        fprintf(stderr, "No CUDA-capable devices found.\n");
        return;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("  SM count: %d\n", prop.multiProcessorCount);
    printf("  Global memory: %.1f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("  Memory bus width: %d-bit\n", prop.memoryBusWidth);

    int low_prio, high_prio;
    cudaDeviceGetStreamPriorityRange(&low_prio, &high_prio);
    printf("  Stream priority range: [%d, %d]\n", high_prio, low_prio);
}
