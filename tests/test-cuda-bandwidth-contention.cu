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
    // Setup sizes
    // DMA: 300 MB
    size_t size_dma = 300 * 1024 * 1024;
    // SM: 100 MB (approx 1/3 of DMA)
    size_t size_sm = 100 * 1024 * 1024;

    // Pointers
    void *h_dma, *d_dma;
    float4 *h_sm, *d_sm; // float4 for kernel

    // Allocation
    CUDA_CHECK(cudaMallocHost(&h_dma, size_dma)); // Standard Pinned for DMA
    CUDA_CHECK(cudaMalloc(&d_dma, size_dma));

    CUDA_CHECK(cudaHostAlloc(&h_sm, size_sm, cudaHostAllocMapped)); // Mapped Pinned for SM
    CUDA_CHECK(cudaMalloc(&d_sm, size_sm));

    // Initialize
    memset(h_dma, 1, size_dma);
    memset(h_sm, 2, size_sm);

    // Get Device Pointer for SM source
    float4 *d_src_mapped;
    CUDA_CHECK(cudaHostGetDevicePointer(&d_src_mapped, h_sm, 0));

    // Streams
    cudaStream_t stream_dma, stream_sm;
    CUDA_CHECK(cudaStreamCreate(&stream_dma));
    CUDA_CHECK(cudaStreamCreate(&stream_sm));

    // Events for timing
    cudaEvent_t start_dma, stop_dma;
    cudaEvent_t start_sm, stop_sm;
    CUDA_CHECK(cudaEventCreate(&start_dma));
    CUDA_CHECK(cudaEventCreate(&stop_dma));
    CUDA_CHECK(cudaEventCreate(&start_sm));
    CUDA_CHECK(cudaEventCreate(&stop_sm));

    std::cout << "DMA Transfer Size: " << size_dma / 1024 / 1024 << " MB" << std::endl;
    std::cout << "SM  Transfer Size: " << size_sm / 1024 / 1024 << " MB" << std::endl;
    std::cout << "------------------------------------------------" << std::endl;

    // --- Baseline 1: DMA Only ---
    std::cout << "1. Baseline: DMA Only (300 MB)..." << std::endl;
    CUDA_CHECK(cudaEventRecord(start_dma, stream_dma));
    CUDA_CHECK(cudaMemcpyAsync(d_dma, h_dma, size_dma, cudaMemcpyHostToDevice, stream_dma));
    CUDA_CHECK(cudaEventRecord(stop_dma, stream_dma));
    CUDA_CHECK(cudaStreamSynchronize(stream_dma));
    
    float ms_dma_base = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms_dma_base, start_dma, stop_dma));
    double bw_dma_base = (size_dma / 1e9) / (ms_dma_base / 1000.0);
    std::cout << "   Time: " << ms_dma_base << " ms | Bandwidth: " << bw_dma_base << " GB/s" << std::endl;

    // --- Baseline 2: SM Only ---
    std::cout << "2. Baseline: SM Only (100 MB)..." << std::endl;
    int threads = 256;
    int blocks = 1024;
    int num_float4 = size_sm / sizeof(float4);

    CUDA_CHECK(cudaEventRecord(start_sm, stream_sm));
    copy_kernel_float4<<<blocks, threads, 0, stream_sm>>>(d_src_mapped, (float4*)d_sm, num_float4);
    CUDA_CHECK(cudaEventRecord(stop_sm, stream_sm));
    CUDA_CHECK(cudaStreamSynchronize(stream_sm));

    float ms_sm_base = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms_sm_base, start_sm, stop_sm));
    double bw_sm_base = (size_sm / 1e9) / (ms_sm_base / 1000.0);
    std::cout << "   Time: " << ms_sm_base << " ms | Bandwidth: " << bw_sm_base << " GB/s" << std::endl;

    // --- Test 3: Concurrent ---
    std::cout << "3. Concurrent: DMA (300 MB) + SM (100 MB)..." << std::endl;
    
    // Synchronize before starting
    CUDA_CHECK(cudaDeviceSynchronize());

    // Launch both
    // Note: Launch order matters slightly, but we want them to overlap.
    // Launching the larger one first usually ensures overlap.
    CUDA_CHECK(cudaEventRecord(start_dma, stream_dma));
    CUDA_CHECK(cudaMemcpyAsync(d_dma, h_dma, size_dma, cudaMemcpyHostToDevice, stream_dma));
    CUDA_CHECK(cudaEventRecord(stop_dma, stream_dma));

    CUDA_CHECK(cudaEventRecord(start_sm, stream_sm));
    copy_kernel_float4<<<blocks, threads, 0, stream_sm>>>(d_src_mapped, (float4*)d_sm, num_float4);
    CUDA_CHECK(cudaEventRecord(stop_sm, stream_sm));

    // Wait for both
    CUDA_CHECK(cudaDeviceSynchronize());

    float ms_dma_conc = 0, ms_sm_conc = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms_dma_conc, start_dma, stop_dma));
    CUDA_CHECK(cudaEventElapsedTime(&ms_sm_conc, start_sm, stop_sm));

    double bw_dma_conc = (size_dma / 1e9) / (ms_dma_conc / 1000.0);
    double bw_sm_conc = (size_sm / 1e9) / (ms_sm_conc / 1000.0);

    std::cout << "   [DMA] Time: " << ms_dma_conc << " ms (vs " << ms_dma_base << " ms) | Bandwidth: " << bw_dma_conc << " GB/s" << std::endl;
    std::cout << "   [SM ] Time: " << ms_sm_conc << " ms (vs " << ms_sm_base << " ms) | Bandwidth: " << bw_sm_conc << " GB/s" << std::endl;

    double slowdown_dma = (ms_dma_conc - ms_dma_base) / ms_dma_base * 100.0;
    double slowdown_sm = (ms_sm_conc - ms_sm_base) / ms_sm_base * 100.0;

    std::cout << "   Slowdown -> DMA: " << slowdown_dma << "% | SM: " << slowdown_sm << "%" << std::endl;

    // Cleanup
    cudaFreeHost(h_dma);
    cudaFreeHost(h_sm);
    cudaFree(d_dma);
    cudaFree(d_sm);
    cudaStreamDestroy(stream_dma);
    cudaStreamDestroy(stream_sm);
    
    return 0;
}
