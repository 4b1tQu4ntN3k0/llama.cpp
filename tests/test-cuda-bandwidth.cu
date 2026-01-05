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

// Use float4 for better bandwidth utilization (16 bytes per thread)
__global__ void copy_kernel_float4(const float4* __restrict__ src, float4* __restrict__ dst, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < size; i += stride) {
        dst[i] = src[i];
    }
}

int main() {
    size_t size_bytes = 500 * 1024 * 1024; // 500 MB
    size_t num_float4 = size_bytes / sizeof(float4);

    float4 *h_pinned, *d_data;
    
    // Allocate Pinned Memory (Mapped)
    // This is critical for the Kernel to access host memory directly
    CUDA_CHECK(cudaHostAlloc(&h_pinned, size_bytes, cudaHostAllocMapped));
    CUDA_CHECK(cudaMalloc(&d_data, size_bytes));

    // Initialize
    memset(h_pinned, 0, size_bytes);

    float4 *d_src_mapped;
    CUDA_CHECK(cudaHostGetDevicePointer(&d_src_mapped, h_pinned, 0));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    std::cout << "Transfer Size: " << size_bytes / 1024 / 1024 << " MB" << std::endl;

    // --- Test 1: Standard cudaMemcpyAsync (Copy Engine) ---
    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "Testing Copy Engine (cudaMemcpyAsync)..." << std::endl;
    
    // Warmup
    CUDA_CHECK(cudaMemcpyAsync(d_data, h_pinned, size_bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto start = std::chrono::high_resolution_clock::now();
    int iterations = 20;
    for(int i=0; i<iterations; ++i) {
        CUDA_CHECK(cudaMemcpyAsync(d_data, h_pinned, size_bytes, cudaMemcpyHostToDevice, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto end = std::chrono::high_resolution_clock::now();
    
    double seconds = std::chrono::duration<double>(end - start).count() / iterations;
    double gb_s = (size_bytes / 1e9) / seconds;
    std::cout << "  Avg Time: " << seconds * 1000 << " ms" << std::endl;
    std::cout << "  Bandwidth: " << gb_s << " GB/s" << std::endl;


    // --- Test 2: Compute Engine (Kernel Reading Mapped Memory) ---
    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "Testing Compute Engine (Kernel Reading Mapped Memory)..." << std::endl;

    int threads = 256;
    // Use enough blocks to saturate the memory bus, but not too many to cause scheduling overhead
    int blocks = 2048; 

    // Warmup
    copy_kernel_float4<<<blocks, threads, 0, stream>>>(d_src_mapped, d_data, num_float4);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    start = std::chrono::high_resolution_clock::now();
    for(int i=0; i<iterations; ++i) {
        copy_kernel_float4<<<blocks, threads, 0, stream>>>(d_src_mapped, d_data, num_float4);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    end = std::chrono::high_resolution_clock::now();

    seconds = std::chrono::duration<double>(end - start).count() / iterations;
    gb_s = (size_bytes / 1e9) / seconds;
    std::cout << "  Avg Time: " << seconds * 1000 << " ms" << std::endl;
    std::cout << "  Bandwidth: " << gb_s << " GB/s" << std::endl;

    // Cleanup
    CUDA_CHECK(cudaFreeHost(h_pinned));
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaStreamDestroy(stream));

    return 0;
}
