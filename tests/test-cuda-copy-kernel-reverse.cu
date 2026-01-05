#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <cstring>

#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(-1); \
    }

// Copy Kernel: Reads from Host (src), Writes to Device (dst)
// src must be Pinned Memory (Zero-Copy)
__global__ void copy_kernel_h2d(const float* __restrict__ src, float* __restrict__ dst, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dst[idx] = src[idx];
    }
}

int main() {
    int priority_low, priority_high;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high));
    
    cudaStream_t stream_large, stream_small;
    // Large transfer uses Low Priority
    CUDA_CHECK(cudaStreamCreateWithPriority(&stream_large, cudaStreamNonBlocking, priority_low));
    // Small transfer uses High Priority
    CUDA_CHECK(cudaStreamCreateWithPriority(&stream_small, cudaStreamNonBlocking, priority_high));

    // Large: 500MB
    size_t size_large_bytes = 500 * 1024 * 1024;
    int elems_large = size_large_bytes / sizeof(float);
    // Small: 1MB
    size_t size_small_bytes = 1 * 1024 * 1024;

    float *h_large, *d_large;
    void *h_small, *d_small;

    std::cout << "Allocating Pinned Memory..." << std::endl;
    // For Large Zero-Copy access by kernel
    CUDA_CHECK(cudaHostAlloc(&h_large, size_large_bytes, cudaHostAllocMapped));
    CUDA_CHECK(cudaMalloc(&d_large, size_large_bytes));
    
    // For Small Memcpy
    CUDA_CHECK(cudaMallocHost(&h_small, size_small_bytes));
    CUDA_CHECK(cudaMalloc(&d_small, size_small_bytes));

    // Initialize
    for(int i=0; i<elems_large; ++i) h_large[i] = 1.0f;
    memset(h_small, 2, size_small_bytes);

    // Get Device Pointer for Host Memory (Required for Kernel Access)
    float *d_h_large_mapped;
    CUDA_CHECK(cudaHostGetDevicePointer(&d_h_large_mapped, h_large, 0));

    // Warmup
    copy_kernel_h2d<<<1024, 256, 0, stream_large>>>(d_h_large_mapped, d_large, 1024*256);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "Test: Copy Kernel (Large) vs Memcpy (Small)" << std::endl;

    // 1. Start Large Transfer via Kernel (Low Priority)
    // Using enough blocks to cover the data
    int threads = 256;
    int blocks = (elems_large + threads - 1) / threads;
    
    // Launch Kernel
    copy_kernel_h2d<<<blocks, threads, 0, stream_large>>>(d_h_large_mapped, d_large, elems_large);

    // 2. Start Small Transfer via Memcpy (High Priority)
    auto start_cpu = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpyAsync(d_small, h_small, size_small_bytes, cudaMemcpyHostToDevice, stream_small));
    CUDA_CHECK(cudaStreamSynchronize(stream_small));
    auto end_cpu = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed_small = end_cpu - start_cpu;
    std::cout << "Small Memcpy finished in " << elapsed_small.count() << " ms" << std::endl;

    // Check if Large is still running
    cudaError_t status = cudaStreamQuery(stream_large);
    if (status == cudaErrorNotReady) {
        std::cout << "SUCCESS: Large Kernel is still running! Memcpy bypassed the Compute Engine." << std::endl;
    } else {
        std::cout << "WARNING: Large Kernel finished. It might have been too fast." << std::endl;
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    // Cleanup
    cudaFreeHost(h_large);
    cudaFreeHost(h_small);
    cudaFree(d_large);
    cudaFree(d_small);
    cudaStreamDestroy(stream_large);
    cudaStreamDestroy(stream_small);

    return 0;
}
