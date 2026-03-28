#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cstring>

#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(-1); \
    }

// Copy Kernel: Reads from Host (src), Writes to Device (dst)
__global__ void copy_kernel_float4(const float4* __restrict__ src, float4* __restrict__ dst, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < size; i += stride) {
        dst[i] = src[i];
    }
}

int main() {
    int priority_low, priority_high;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high));
    
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    
    std::cout << "Device: " << prop.name << std::endl;
    std::cout << "asyncEngineCount: " << prop.asyncEngineCount << std::endl;
    std::cout << "Total SMs: " << prop.multiProcessorCount << std::endl;
    std::cout << "Priority Range: Low=" << priority_low << ", High=" << priority_high << std::endl;

    // Setup sizes
    size_t size_large = 500 * 1024 * 1024;
    size_t size_small = 100 * 1024 * 1024; // Make it substantial to measure bandwidth

    // Pointers
    float4 *h_large, *d_large;
    float4 *h_small, *d_small;

    // Allocation (Mapped Pinned for both)
    CUDA_CHECK(cudaHostAlloc(&h_large, size_large, cudaHostAllocMapped));
    CUDA_CHECK(cudaMalloc(&d_large, size_large));
    
    CUDA_CHECK(cudaHostAlloc(&h_small, size_small, cudaHostAllocMapped));
    CUDA_CHECK(cudaMalloc(&d_small, size_small));

    // Initialize
    memset(h_large, 1, size_large);
    memset(h_small, 2, size_small);

    // Get Device Pointers
    float4 *d_src_large, *d_src_small;
    CUDA_CHECK(cudaHostGetDevicePointer(&d_src_large, h_large, 0));
    CUDA_CHECK(cudaHostGetDevicePointer(&d_src_small, h_small, 0));

    // Streams with Priorities
    cudaStream_t stream_low, stream_high;
    CUDA_CHECK(cudaStreamCreateWithPriority(&stream_low, cudaStreamNonBlocking, priority_low));
    CUDA_CHECK(cudaStreamCreateWithPriority(&stream_high, cudaStreamNonBlocking, priority_high));

    // Events
    cudaEvent_t start_small, stop_small;
    CUDA_CHECK(cudaEventCreate(&start_small));
    CUDA_CHECK(cudaEventCreate(&stop_small));

    int threads = 256;
    int blocks = prop.multiProcessorCount * 32; // Ensure we have enough blocks to saturate all SMs multiple times

    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "Kernel Config: " << blocks << " blocks, " << threads << " threads/block" << std::endl;
    std::cout << "Theoretical Occupancy: " << (float)blocks / prop.multiProcessorCount << " blocks per SM" << std::endl;
    std::cout << "Test: Large Kernel (Low Prio) vs Small Kernel (High Prio)" << std::endl;

    // 1. Baseline: Small Kernel Alone
    std::cout << "Measuring Baseline Small Kernel..." << std::endl;
    CUDA_CHECK(cudaEventRecord(start_small, stream_high));
    copy_kernel_float4<<<blocks, threads, 0, stream_high>>>(d_src_small, (float4*)d_small, size_small / sizeof(float4));
    CUDA_CHECK(cudaEventRecord(stop_small, stream_high));
    CUDA_CHECK(cudaStreamSynchronize(stream_high));
    
    float ms_base = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms_base, start_small, stop_small));
    double bw_base = (size_small / 1e9) / (ms_base / 1000.0);
    std::cout << "  Baseline Time: " << ms_base << " ms | Bandwidth: " << bw_base << " GB/s" << std::endl;

    // 2. Concurrent: Large (Low) + Small (High)
    std::cout << "Measuring Concurrent (Small interrupting Large)..." << std::endl;
    
    // Launch Large (Low Prio)
    copy_kernel_float4<<<blocks, threads, 0, stream_low>>>(d_src_large, (float4*)d_large, size_large / sizeof(float4));
    
    // Launch Small (High Prio) immediately after
    CUDA_CHECK(cudaEventRecord(start_small, stream_high));
    copy_kernel_float4<<<blocks, threads, 0, stream_high>>>(d_src_small, (float4*)d_small, size_small / sizeof(float4));
    CUDA_CHECK(cudaEventRecord(stop_small, stream_high));

    // Wait for Small
    CUDA_CHECK(cudaStreamSynchronize(stream_high));
    
    float ms_conc = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms_conc, start_small, stop_small));
    double bw_conc = (size_small / 1e9) / (ms_conc / 1000.0);
    
    std::cout << "  Concurrent Time: " << ms_conc << " ms | Bandwidth: " << bw_conc << " GB/s" << std::endl;
    
    double slowdown = (ms_conc - ms_base) / ms_base * 100.0;
    std::cout << "  Slowdown: " << slowdown << "%" << std::endl;

    // Cleanup
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaFreeHost(h_large);
    cudaFreeHost(h_small);
    cudaFree(d_large);
    cudaFree(d_small);
    cudaStreamDestroy(stream_low);
    cudaStreamDestroy(stream_high);

    return 0;
}
