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
    std::cout << std::endl;

    // Setup sizes - use 50MB for each transfer to make it measurable
    size_t size = 500 * 1024 * 1024;
    size_t count = size / sizeof(float);

    // Pointers for two concurrent H2D transfers
    float *h_src1 = nullptr, *d_dst1 = nullptr;
    float *h_src2 = nullptr, *d_dst2 = nullptr;

    // Allocate pinned host memory (required for async concurrent transfers)
    CUDA_CHECK(cudaHostAlloc(&h_src1, size, cudaHostAllocMapped));
    CUDA_CHECK(cudaMalloc(&d_dst1, size));
    
    CUDA_CHECK(cudaHostAlloc(&h_src2, size, cudaHostAllocMapped));
    CUDA_CHECK(cudaMalloc(&d_dst2, size));

    // Initialize host memory
    std::cout << "Initializing host memory..." << std::endl;
    for (size_t i = 0; i < count; i++) {
        h_src1[i] = 1.0f;
        h_src2[i] = 2.0f;
    }
    std::cout << "Host memory initialized." << std::endl;

    // Get device pointers for mapped memory
    float *d_src1, *d_src2;
    CUDA_CHECK(cudaHostGetDevicePointer(&d_src1, h_src1, 0));
    CUDA_CHECK(cudaHostGetDevicePointer(&d_src2, h_src2, 0));

    // Create two streams with different priorities
    cudaStream_t stream1, stream2;
    CUDA_CHECK(cudaStreamCreateWithPriority(&stream1, cudaStreamNonBlocking, priority_low));
    CUDA_CHECK(cudaStreamCreateWithPriority(&stream2, cudaStreamNonBlocking, priority_high));

    // Events for timing
    cudaEvent_t start1, stop1, start2, stop2;
    CUDA_CHECK(cudaEventCreate(&start1));
    CUDA_CHECK(cudaEventCreate(&stop1));
    CUDA_CHECK(cudaEventCreate(&start2));
    CUDA_CHECK(cudaEventCreate(&stop2));
    
    // Outer timer for concurrent test
    cudaEvent_t concurrent_start, concurrent_stop;
    CUDA_CHECK(cudaEventCreate(&concurrent_start));
    CUDA_CHECK(cudaEventCreate(&concurrent_stop));

    std::cout << "Transfer size per stream: " << size / (1024 * 1024) << " MB" << std::endl;
    std::cout << "================================================" << std::endl;

    // Test 1: Sequential transfers (baseline) - truly sequential
    std::cout << "Test 1: Sequential H2D transfers..." << std::endl;
    
    // First transfer - complete it fully before starting the second
    CUDA_CHECK(cudaEventRecord(start1, stream1));
    CUDA_CHECK(cudaMemcpyAsync(d_dst1, h_src1, size, cudaMemcpyHostToDevice, stream1));
    CUDA_CHECK(cudaEventRecord(stop1, stream1));
    CUDA_CHECK(cudaStreamSynchronize(stream1));  // Wait for first to complete
    
    // Second transfer - only starts after first is done
    CUDA_CHECK(cudaEventRecord(start2, stream2));
    CUDA_CHECK(cudaMemcpyAsync(d_dst2, h_src2, size, cudaMemcpyHostToDevice, stream2));
    CUDA_CHECK(cudaEventRecord(stop2, stream2));
    CUDA_CHECK(cudaStreamSynchronize(stream2));  // Wait for second to complete
    
    float ms_seq1 = 0, ms_seq2 = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms_seq1, start1, stop1));
    CUDA_CHECK(cudaEventElapsedTime(&ms_seq2, start2, stop2));
    
    double bw_seq1 = (size / 1e9) / (ms_seq1 / 1000.0);
    double bw_seq2 = (size / 1e9) / (ms_seq2 / 1000.0);
    
    std::cout << "  Stream 1: " << ms_seq1 << " ms | " << bw_seq1 << " GB/s" << std::endl;
    std::cout << "  Stream 2: " << ms_seq2 << " ms | " << bw_seq2 << " GB/s" << std::endl;
    std::cout << "  Total sequential time: " << (ms_seq1 + ms_seq2) << " ms" << std::endl;
    std::cout << std::endl;

    // Test 2: Concurrent transfers - truly simultaneous
    std::cout << "Test 2: Concurrent H2D transfers..." << std::endl;
    
    // Reset events
    CUDA_CHECK(cudaEventDestroy(start1));
    CUDA_CHECK(cudaEventDestroy(stop1));
    CUDA_CHECK(cudaEventDestroy(start2));
    CUDA_CHECK(cudaEventDestroy(stop2));
    CUDA_CHECK(cudaEventCreate(&start1));
    CUDA_CHECK(cudaEventCreate(&stop1));
    CUDA_CHECK(cudaEventCreate(&start2));
    CUDA_CHECK(cudaEventCreate(&stop2));
    
    // Use CPU timer to measure overall concurrent execution time
    auto cpu_start = std::chrono::high_resolution_clock::now();
    
    // Launch both transfers at the same time WITHOUT synchronization in between
    CUDA_CHECK(cudaEventRecord(start1, stream1));
    CUDA_CHECK(cudaMemcpyAsync(d_dst1, h_src1, size, cudaMemcpyHostToDevice, stream1));
    CUDA_CHECK(cudaEventRecord(stop1, stream1));
    
    CUDA_CHECK(cudaEventRecord(start2, stream2));
    CUDA_CHECK(cudaMemcpyAsync(h_src2, d_dst2, size, cudaMemcpyDeviceToHost, stream2));
    CUDA_CHECK(cudaEventRecord(stop2, stream2));
    
    // Wait for both to complete
    CUDA_CHECK(cudaStreamSynchronize(stream1));
    CUDA_CHECK(cudaStreamSynchronize(stream2));
    
    auto cpu_stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = cpu_stop - cpu_start;
    double ms_conc_total = cpu_duration.count();
    
    float ms_conc1 = 0, ms_conc2 = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms_conc1, start1, stop1));
    CUDA_CHECK(cudaEventElapsedTime(&ms_conc2, start2, stop2));
    
    double bw_conc1 = (size / 1e9) / (ms_conc1 / 1000.0);
    double bw_conc2 = (size / 1e9) / (ms_conc2 / 1000.0);
    
    std::cout << "  Stream 1: " << ms_conc1 << " ms | " << bw_conc1 << " GB/s" << std::endl;
    std::cout << "  Stream 2: " << ms_conc2 << " ms | " << bw_conc2 << " GB/s" << std::endl;
    std::cout << "  Overall concurrent time (outer timer): " << ms_conc_total << " ms" << std::endl;
    std::cout << std::endl;

    // Analysis
    std::cout << "================================================" << std::endl;
    std::cout << "Analysis:" << std::endl;
    
    double total_seq = ms_seq1 + ms_seq2;
    double total_conc = ms_conc_total;  // Use outer timer for accurate measurement
    double speedup = total_seq / total_conc;
    double overlap = (1.0 - total_conc / total_seq) * 100.0;
    
    std::cout << "  Sequential total: " << total_seq << " ms" << std::endl;
    std::cout << "  Concurrent total: " << total_conc << " ms" << std::endl;
    std::cout << "  Speedup: " << speedup << "x" << std::endl;
    std::cout << "  Overlap: " << overlap << "%" << std::endl;
    std::cout << std::endl;
    
    if (speedup > 1.5) {
        std::cout << "✓ EXCELLENT: Strong concurrent H2D transfer support!" << std::endl;
        std::cout << "  The GPU has multiple copy engines working in parallel." << std::endl;
    } else if (speedup > 1.2) {
        std::cout << "✓ GOOD: Moderate concurrent H2D transfer support." << std::endl;
        std::cout << "  Some overlap detected, but not full parallelism." << std::endl;
    } else if (speedup > 1.0) {
        std::cout << "△ LIMITED: Weak concurrent H2D transfer support." << std::endl;
        std::cout << "  Minimal overlap between transfers." << std::endl;
    } else {
        std::cout << "✗ NONE: No concurrent H2D transfer support detected." << std::endl;
        std::cout << "  Transfers appear to be serialized despite using multiple streams." << std::endl;
    }

    // Cleanup
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaFreeHost(h_src1);
    cudaFreeHost(h_src2);
    cudaFree(d_dst1);
    cudaFree(d_dst2);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaEventDestroy(start1);
    cudaEventDestroy(stop1);
    cudaEventDestroy(start2);
    cudaEventDestroy(stop2);
    cudaEventDestroy(concurrent_start);
    cudaEventDestroy(concurrent_stop);

    return 0;
}
