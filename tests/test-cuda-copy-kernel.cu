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
    // Small: 1MB
    size_t size_small_bytes = 1 * 1024 * 1024;
    int elems_small = size_small_bytes / sizeof(float);

    void *h_large, *d_large;
    float *h_small, *d_small; // Use float for kernel convenience

    std::cout << "Allocating Pinned Memory..." << std::endl;
    CUDA_CHECK(cudaMallocHost(&h_large, size_large_bytes));
    CUDA_CHECK(cudaMalloc(&d_large, size_large_bytes));
    
    // For Zero-Copy access by kernel, we strictly need cudaHostAllocMapped
    // But cudaMallocHost usually works on modern systems (UVA). 
    // To be safe and explicit for Zero-Copy:
    CUDA_CHECK(cudaHostAlloc(&h_small, size_small_bytes, cudaHostAllocMapped));
    CUDA_CHECK(cudaMalloc(&d_small, size_small_bytes));

    memset(h_large, 1, size_large_bytes);
    // Initialize small buffer
    for(int i=0; i<elems_small; ++i) h_small[i] = (float)i;

    // Get Device Pointer for Host Memory (Required for Kernel Access)
    float *d_h_small_mapped;
    CUDA_CHECK(cudaHostGetDevicePointer(&d_h_small_mapped, h_small, 0));

    // Warmup
    copy_kernel_h2d<<<elems_small/256 + 1, 256, 0, stream_small>>>(d_h_small_mapped, d_small, elems_small);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "Test 1: Standard Memcpy vs Memcpy (The Blocking Case)" << std::endl;
    
    CUDA_CHECK(cudaMemcpyAsync(d_large, h_large, size_large_bytes, cudaMemcpyHostToDevice, stream_large));
    
    auto start_cpu = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpyAsync(d_small, h_small, size_small_bytes, cudaMemcpyHostToDevice, stream_small));
    CUDA_CHECK(cudaStreamSynchronize(stream_small));
    auto end_cpu = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double, std::milli> elapsed_memcpy = end_cpu - start_cpu;
    std::cout << "Small Memcpy finished in " << elapsed_memcpy.count() << " ms" << std::endl;
    CUDA_CHECK(cudaDeviceSynchronize()); // Wait for large to finish

    std::cout << "\nTest 2: Memcpy (Large) vs Copy Kernel (Small) (The Proposed Solution)" << std::endl;

    // Start Large Transfer again
    CUDA_CHECK(cudaMemcpyAsync(d_large, h_large, size_large_bytes, cudaMemcpyHostToDevice, stream_large));

    start_cpu = std::chrono::high_resolution_clock::now();
    
    // Launch Copy Kernel for Small Transfer
    // Note: We pass d_h_small_mapped (the device pointer to host memory) as source
    copy_kernel_h2d<<<elems_small/256 + 1, 256, 0, stream_small>>>(d_h_small_mapped, d_small, elems_small);
    
    CUDA_CHECK(cudaStreamSynchronize(stream_small));
    end_cpu = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed_kernel = end_cpu - start_cpu;
    std::cout << "Small Copy-Kernel finished in " << elapsed_kernel.count() << " ms" << std::endl;

    // Check if Large is still running
    cudaError_t status = cudaStreamQuery(stream_large);
    if (status == cudaErrorNotReady) {
        std::cout << "SUCCESS: Large Transfer is still running! Kernel bypassed the Copy Engine queue." << std::endl;
    } else {
        std::cout << "WARNING: Large Transfer finished. It might have been too fast." << std::endl;
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    // Verify correctness of kernel copy
    std::vector<float> verify(elems_small);
    CUDA_CHECK(cudaMemcpy(verify.data(), d_small, size_small_bytes, cudaMemcpyDeviceToHost));
    if (verify[0] != 0.0f || verify[elems_small-1] != (float)(elems_small-1)) {
        std::cerr << "Verification FAILED!" << std::endl;
    } else {
        std::cout << "Verification PASSED." << std::endl;
    }

    // Cleanup
    cudaFreeHost(h_large);
    cudaFreeHost(h_small);
    cudaFree(d_large);
    cudaFree(d_small);
    cudaStreamDestroy(stream_large);
    cudaStreamDestroy(stream_small);

    return 0;
}
