#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <cstring>
#include <cmath>

#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(-1); \
    }

// A kernel that simulates computation load
__global__ void compute_kernel(float* data, int size, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = data[idx];
        for (int i = 0; i < iterations; ++i) {
            val = sinf(val) * cosf(val);
        }
        data[idx] = val;
    }
}

void measure_baseline(size_t size_transfer, void* h_transfer, void* d_transfer, 
                     size_t compute_elems, float* d_compute, int iterations,
                     cudaStream_t stream_transfer, cudaStream_t stream_compute) {
    
    std::cout << "\n=== Baseline Measurements ===" << std::endl;
    
    // Measure Transfer Alone
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start, stream_transfer));
    CUDA_CHECK(cudaMemcpyAsync(d_transfer, h_transfer, size_transfer, cudaMemcpyHostToDevice, stream_transfer));
    CUDA_CHECK(cudaEventRecord(stop, stream_transfer));
    CUDA_CHECK(cudaStreamSynchronize(stream_transfer));
    
    float ms_transfer = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms_transfer, start, stop));
    std::cout << "Baseline Transfer Duration: " << ms_transfer << " ms" << std::endl;

    // Measure Compute Alone
    CUDA_CHECK(cudaEventRecord(start, stream_compute));
    compute_kernel<<<compute_elems/256, 256, 0, stream_compute>>>(d_compute, compute_elems, iterations);
    CUDA_CHECK(cudaEventRecord(stop, stream_compute));
    CUDA_CHECK(cudaStreamSynchronize(stream_compute));
    
    float ms_compute = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms_compute, start, stop));
    std::cout << "Baseline Compute Duration:  " << ms_compute << " ms" << std::endl;
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

int main() {
    int priority_low, priority_high;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high));
    std::cout << "Priority Range: Low Value=" << priority_low << ", High Value=" << priority_high << std::endl;

    cudaStream_t stream_transfer, stream_compute;
    // Transfer gets Low Priority
    CUDA_CHECK(cudaStreamCreateWithPriority(&stream_transfer, cudaStreamNonBlocking, priority_low));
    // Compute gets High Priority
    CUDA_CHECK(cudaStreamCreateWithPriority(&stream_compute, cudaStreamNonBlocking, priority_high));

    // Large Transfer: 1GB to ensure it takes long enough (~80ms on PCIe 4.0 x16)
    size_t size_transfer = 1024 * 1024 * 1024;
    // Compute Data
    size_t size_compute = 1024 * 1024; 
    int compute_elems = size_compute / sizeof(float);

    void *h_transfer, *d_transfer;
    float *d_compute;

    std::cout << "Allocating Pinned Memory..." << std::endl;
    CUDA_CHECK(cudaMallocHost(&h_transfer, size_transfer));
    CUDA_CHECK(cudaMalloc(&d_transfer, size_transfer));
    CUDA_CHECK(cudaMalloc(&d_compute, size_compute));

    memset(h_transfer, 1, size_transfer);

    // Warmup
    compute_kernel<<<compute_elems/256, 256, 0, stream_compute>>>(d_compute, compute_elems, 100);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Adjust iterations to be faster than transfer
    // Target: ~20ms compute vs ~80ms transfer
    int iterations = 10000; 

    measure_baseline(size_transfer, h_transfer, d_transfer, compute_elems, d_compute, iterations, stream_transfer, stream_compute);

    std::cout << "\n=== Overlap Test ===" << std::endl;
    std::cout << "Starting Large Transfer (Low Priority) of " << size_transfer / 1024 / 1024 << " MB..." << std::endl;

    cudaEvent_t start_transfer, stop_transfer;
    cudaEvent_t start_compute, stop_compute;
    CUDA_CHECK(cudaEventCreate(&start_transfer));
    CUDA_CHECK(cudaEventCreate(&stop_transfer));
    CUDA_CHECK(cudaEventCreate(&start_compute));
    CUDA_CHECK(cudaEventCreate(&stop_compute));

    // 1. Start Large Transfer
    CUDA_CHECK(cudaEventRecord(start_transfer, stream_transfer));
    CUDA_CHECK(cudaMemcpyAsync(d_transfer, h_transfer, size_transfer, cudaMemcpyHostToDevice, stream_transfer));
    CUDA_CHECK(cudaEventRecord(stop_transfer, stream_transfer));

    // Small delay to ensure transfer has started
    // std::this_thread::sleep_for(std::chrono::microseconds(100));

    std::cout << "Starting Compute Kernel (High Priority)..." << std::endl;

    // 2. Start Compute Kernel
    CUDA_CHECK(cudaEventRecord(start_compute, stream_compute));
    compute_kernel<<<compute_elems/256, 256, 0, stream_compute>>>(d_compute, compute_elems, iterations);
    CUDA_CHECK(cudaEventRecord(stop_compute, stream_compute));

    // 3. Wait for Compute
    CUDA_CHECK(cudaStreamSynchronize(stream_compute));
    std::cout << "Compute Kernel finished." << std::endl;

    // 4. Check Transfer status
    cudaError_t status = cudaStreamQuery(stream_transfer);
    if (status == cudaErrorNotReady) {
        std::cout << "SUCCESS: Transfer is still running! Compute finished BEFORE Transfer." << std::endl;
    } else if (status == cudaSuccess) {
        std::cout << "WARNING: Transfer already finished. Compute was too slow or Transfer too fast." << std::endl;
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    float ms_transfer, ms_compute;
    CUDA_CHECK(cudaEventElapsedTime(&ms_transfer, start_transfer, stop_transfer));
    CUDA_CHECK(cudaEventElapsedTime(&ms_compute, start_compute, stop_compute));

    std::cout << "\n[Timing Analysis]" << std::endl;
    std::cout << "Transfer Duration: " << ms_transfer << " ms" << std::endl;
    std::cout << "Compute Duration:  " << ms_compute << " ms" << std::endl;

    // Cleanup
    cudaFreeHost(h_transfer);
    cudaFree(d_transfer);
    cudaFree(d_compute);
    cudaStreamDestroy(stream_transfer);
    cudaStreamDestroy(stream_compute);
    
    return 0;
}
