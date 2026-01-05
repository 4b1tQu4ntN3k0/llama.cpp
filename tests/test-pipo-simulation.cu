#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cstring>
#include <thread>

#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(-1); \
    }

// -------------------------------------------------------------------------
// 1. Optimized Copy Kernel (SM) for Small Data
// Uses float4 for max bandwidth, Grid-Stride Loop for flexibility
// -------------------------------------------------------------------------
__global__ void copy_kernel_optimized(const float4* __restrict__ src, float4* __restrict__ dst, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    // Grid-Stride Loop: Allows the kernel to handle any size and saturate bus
    for (int i = idx; i < size; i += stride) {
        dst[i] = src[i];
    }
}

// -------------------------------------------------------------------------
// 2. Compute Kernel (SM) - Simulating Heavy Inference (GEMM)
// -------------------------------------------------------------------------
__global__ void compute_kernel_simulation(float* data, int size, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < size; i += stride) {
        float val = data[i];
        // Heavy math simulation
        #pragma unroll
        for(int k=0; k<iterations; ++k) {
            val = fmaf(val, val, 0.0001f); 
        }
        data[i] = val;
    }
}

int main() {
    // 1. Setup Device & Priorities
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    
    int priority_low, priority_high;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high));
    
    std::cout << "Device: " << prop.name << " (SMs: " << prop.multiProcessorCount << ")" << std::endl;
    std::cout << "Priority Range: Low=" << priority_low << ", High=" << priority_high << std::endl;

    // 2. Configuration
    // Background Task: Weight Loading (Large, DMA)
    // Increased to 500MB to verify concurrency
    size_t size_weights = 500 * 1024 * 1024; 
    
    // Foreground Task: Input Tensor (Small, SM Copy)
    size_t size_input = 20 * 1024 * 1024;    // 20 MB
    
    // Compute Task: Simulated Workload
    // We want compute to take some time, e.g., 20-30ms
    int compute_iters = 1000; 

    // 3. Memory Allocation
    void *h_weights, *d_weights;
    float4 *h_input, *d_input;
    float *d_compute_buffer; // Buffer for computation

    // Weights: Standard Pinned (for DMA)
    CUDA_CHECK(cudaMallocHost(&h_weights, size_weights));
    CUDA_CHECK(cudaMalloc(&d_weights, size_weights));

    // Input: Mapped Pinned (for SM Copy) - CRITICAL
    CUDA_CHECK(cudaHostAlloc(&h_input, size_input, cudaHostAllocMapped));
    CUDA_CHECK(cudaMalloc(&d_input, size_input));
    
    // Compute Buffer
    CUDA_CHECK(cudaMalloc(&d_compute_buffer, size_input)); // Reuse size for simplicity

    // Initialize
    memset(h_weights, 1, size_weights);
    memset(h_input, 2, size_input);

    // Get Device Pointer for Input (Zero-Copy)
    float4 *d_src_input;
    CUDA_CHECK(cudaHostGetDevicePointer(&d_src_input, h_input, 0));

    // 4. Streams
    cudaStream_t stream_bg_dma;   // Low Priority
    cudaStream_t stream_fg_compute; // High Priority
    
    CUDA_CHECK(cudaStreamCreateWithPriority(&stream_bg_dma, cudaStreamNonBlocking, priority_low));
    CUDA_CHECK(cudaStreamCreateWithPriority(&stream_fg_compute, cudaStreamNonBlocking, priority_high));

    // 5. Kernel Config
    // Optimize for Occupancy
    int threads = 256;
    // Use enough blocks to saturate the GPU memory bus. 
    // Since compute and copy are serial in the same stream, we can use full GPU for copy.
    int blocks = prop.multiProcessorCount * 8; 

    // Events for timing
    cudaEvent_t start_input, stop_input;
    cudaEvent_t start_compute, stop_compute;
    CUDA_CHECK(cudaEventCreate(&start_input));
    CUDA_CHECK(cudaEventCreate(&stop_input));
    CUDA_CHECK(cudaEventCreate(&start_compute));
    CUDA_CHECK(cudaEventCreate(&stop_compute));

    // Warmup
    copy_kernel_optimized<<<blocks, threads, 0, stream_fg_compute>>>(d_src_input, (float4*)d_input, size_input / sizeof(float4));
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- Baseline Measurements ---
    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "Baseline Measurements (Independent Execution)" << std::endl;

    // 1. Baseline DMA Weight Load
    float ms_base_dma = 0;
    CUDA_CHECK(cudaEventRecord(start_input, stream_bg_dma));
    CUDA_CHECK(cudaMemcpyAsync(d_weights, h_weights, size_weights, cudaMemcpyHostToDevice, stream_bg_dma));
    CUDA_CHECK(cudaEventRecord(stop_input, stream_bg_dma));
    CUDA_CHECK(cudaStreamSynchronize(stream_bg_dma));
    CUDA_CHECK(cudaEventElapsedTime(&ms_base_dma, start_input, stop_input));
    std::cout << "  Baseline DMA Weight Load: " << ms_base_dma << " ms" << std::endl;

    // 2. Baseline SM Input Load
    float ms_base_input = 0;
    CUDA_CHECK(cudaEventRecord(start_input, stream_fg_compute));
    copy_kernel_optimized<<<blocks, threads, 0, stream_fg_compute>>>(d_src_input, (float4*)d_input, size_input / sizeof(float4));
    CUDA_CHECK(cudaEventRecord(stop_input, stream_fg_compute));
    CUDA_CHECK(cudaStreamSynchronize(stream_fg_compute));
    CUDA_CHECK(cudaEventElapsedTime(&ms_base_input, start_input, stop_input));
    std::cout << "  Baseline SM Input Load:   " << ms_base_input << " ms" << std::endl;

    // 3. Baseline SM Computation
    float ms_base_compute = 0;
    CUDA_CHECK(cudaEventRecord(start_compute, stream_fg_compute));
    compute_kernel_simulation<<<blocks, threads, 0, stream_fg_compute>>>(d_compute_buffer, size_input / sizeof(float), compute_iters);
    CUDA_CHECK(cudaEventRecord(stop_compute, stream_fg_compute));
    CUDA_CHECK(cudaStreamSynchronize(stream_fg_compute));
    CUDA_CHECK(cudaEventElapsedTime(&ms_base_compute, start_compute, stop_compute));
    std::cout << "  Baseline SM Computation:  " << ms_base_compute << " ms" << std::endl;

    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "PIPO Simulation Started" << std::endl;
    std::cout << "Scenario:" << std::endl;
    std::cout << "  [Background] DMA Weight Load: " << size_weights / 1024 / 1024 << " MB (Low Prio)" << std::endl;
    std::cout << "  [Foreground] SM Input Load:   " << size_input / 1024 / 1024 << " MB (High Prio)" << std::endl;
    std::cout << "  [Foreground] SM Computation:  Simulated GEMM (High Prio)" << std::endl;
    std::cout << "------------------------------------------------" << std::endl;

    // --- Execution ---

    // Step A: Start Background DMA (Weights)
    // This will clog the Copy Engine for a while (~60-70ms)
    CUDA_CHECK(cudaMemcpyAsync(d_weights, h_weights, size_weights, cudaMemcpyHostToDevice, stream_bg_dma));

    // Step B: Immediately Launch Foreground Pipeline (Input -> Compute)
    // Because it's High Priority, the SM Copy Kernel should bypass the DMA queue.
    
    // 1. Input Transfer (SM Kernel)
    CUDA_CHECK(cudaEventRecord(start_input, stream_fg_compute));
    copy_kernel_optimized<<<blocks, threads, 0, stream_fg_compute>>>(
        d_src_input, (float4*)d_input, size_input / sizeof(float4)
    );
    CUDA_CHECK(cudaEventRecord(stop_input, stream_fg_compute));

    // 2. Computation (SM Kernel) - Serial in same stream
    CUDA_CHECK(cudaEventRecord(start_compute, stream_fg_compute));
    compute_kernel_simulation<<<blocks, threads, 0, stream_fg_compute>>>(
        d_compute_buffer, size_input / sizeof(float), compute_iters
    );
    CUDA_CHECK(cudaEventRecord(stop_compute, stream_fg_compute));

    // Wait for Foreground to finish
    CUDA_CHECK(cudaStreamSynchronize(stream_fg_compute));

    // Check Background status
    cudaError_t bg_status = cudaStreamQuery(stream_bg_dma);
    
    // --- Reporting ---
    float ms_input = 0, ms_compute = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms_input, start_input, stop_input));
    CUDA_CHECK(cudaEventElapsedTime(&ms_compute, start_compute, stop_compute));

    double bw_input = (size_input / 1e9) / (ms_input / 1000.0);

    std::cout << "\n[Results]" << std::endl;
    std::cout << "Foreground Input Transfer Time: " << ms_input << " ms" << std::endl;
    std::cout << "Foreground Input Bandwidth:     " << bw_input << " GB/s" << std::endl;
    std::cout << "Foreground Compute Time:        " << ms_compute << " ms" << std::endl;
    
    if (bg_status == cudaErrorNotReady) {
        std::cout << "\n[Verification] SUCCESS: Background DMA was still running during Foreground tasks." << std::endl;
        std::cout << "This confirms the Input Transfer successfully bypassed the DMA queue!" << std::endl;
    } else {
        std::cout << "\n[Verification] WARNING: Background DMA finished too early. Increase weight size." << std::endl;
    }

    // Cleanup
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaFreeHost(h_weights);
    cudaFreeHost(h_input);
    cudaFree(d_weights);
    cudaFree(d_input);
    cudaFree(d_compute_buffer);
    cudaStreamDestroy(stream_bg_dma);
    cudaStreamDestroy(stream_fg_compute);

    return 0;
}
