#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <chrono>
#include <cstring>
#include <iomanip>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        return 1; \
    } \
}

void print_bandwidth(const char* label, size_t bytes, double milliseconds) {
    double gb = bytes / (1024.0 * 1024.0 * 1024.0);
    double sec = milliseconds / 1000.0;
    std::cout << std::left << std::setw(35) << label 
              << ": " << std::fixed << std::setprecision(2) << (gb / sec) << " GB/s" 
              << " (" << std::setprecision(2) << milliseconds << " ms)" << std::endl;
}

int main() {
    size_t size = 512 * 1024 * 1024; // 512 MB
    std::cout << "Transfer Size: " << size / (1024*1024) << " MB" << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;

    // 1. Allocate Memory
    void* h_pageable = malloc(size);
    void* h_pinned;
    CHECK_CUDA(cudaMallocHost(&h_pinned, size));
    void* d_device;
    CHECK_CUDA(cudaMalloc(&d_device, size));

    if (!h_pageable || !h_pinned || !d_device) {
        std::cerr << "Memory allocation failed" << std::endl;
        return 1;
    }

    // Initialize data to trigger standard OS page faulting behaviors
    memset(h_pageable, 1, size);
    memset(h_pinned, 1, size);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    float elapsed_gpu_time = 0;

    int n_iter = 10;
    int n_warmup = 3;

    // --- Warmup ---
    for(int i=0; i<n_warmup; i++) {
        memcpy(h_pinned, h_pageable, size);
        cudaMemcpy(d_device, h_pageable, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_device, h_pinned, size, cudaMemcpyHostToDevice);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // --- Test 1: CPU Pageable -> CPU Pinned (memcpy) ---
    // Using std::chrono because this is a CPU operation
    auto t1 = std::chrono::high_resolution_clock::now();
    for(int i=0; i<n_iter; i++) {
        memcpy(h_pinned, h_pageable, size);
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() / (double)n_iter;
    print_bandwidth("CPU(Pageable) -> CPU(Pinned)", size, cpu_ms);


    // --- Test 2: CPU (Pageable) -> GPU ---
    CHECK_CUDA(cudaEventRecord(start));
    for(int i=0; i<n_iter; i++) {
        CHECK_CUDA(cudaMemcpy(d_device, h_pageable, size, cudaMemcpyHostToDevice));
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_gpu_time, start, stop));
    print_bandwidth("CPU(Pageable) -> GPU", size, elapsed_gpu_time / n_iter);

    // --- Test 3: CPU (Pinned) -> GPU ---
    CHECK_CUDA(cudaEventRecord(start));
    for(int i=0; i<n_iter; i++) {
        CHECK_CUDA(cudaMemcpy(d_device, h_pinned, size, cudaMemcpyHostToDevice));
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_gpu_time, start, stop));
    print_bandwidth("CPU(Pinned)   -> GPU", size, elapsed_gpu_time / n_iter);

    // --- Test 4: CPU Calculation Performance ---
    std::cout << "------------------------------------------------------------" << std::endl;
    std::cout << "CPU Calculation Performance (Read-Modify-Write of float array)" << std::endl;
    
    // Cast to float for calculation test
    float* f_pageable = (float*)h_pageable;
    float* f_pinned = (float*)h_pinned;
    size_t num_elements = size / sizeof(float);

    // Initialize with valid float data to avoid denormal penalties
    for(size_t i=0; i<num_elements; i++) {
        f_pageable[i] = 1.0f;
        f_pinned[i] = 1.0f;
    }

    int n_calc_iter = 20;

    // Test Pageable
    auto t_calc1 = std::chrono::high_resolution_clock::now();
    for(int k=0; k<n_calc_iter; k++) {
        for(size_t i=0; i<num_elements; i++) {
            f_pageable[i] += 1.01f;
        }
    }
    auto t_calc2 = std::chrono::high_resolution_clock::now();
    double pageable_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_calc2 - t_calc1).count() / (double)n_calc_iter;
    // 2x size because we read and write each element
    print_bandwidth("Calc on Pageable Mem", size * 2, pageable_ms);

    // Test Pinned
    t_calc1 = std::chrono::high_resolution_clock::now();
    for(int k=0; k<n_calc_iter; k++) {
        for(size_t i=0; i<num_elements; i++) {
            f_pinned[i] += 1.01f;
        }
    }
    t_calc2 = std::chrono::high_resolution_clock::now();
    double pinned_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_calc2 - t_calc1).count() / (double)n_calc_iter;
    print_bandwidth("Calc on Pinned Mem", size * 2, pinned_ms);

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_device);
    cudaFreeHost(h_pinned);
    free(h_pageable);

    return 0;
}
