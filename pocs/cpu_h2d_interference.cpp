#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

// CPU workload: Matrix Multiplication (Compute Bound)
// A: N x N, B: N x N -> C: N x N
void cpu_matmul(int N, const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C) {
    // Simple O(N^3) matmul
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// CPU workload: Vector Addition (Memory Bound)
// A: Size, B: Size -> C: Size
// Repeat 'iters' times to last long enough
void cpu_vecadd(size_t size, int iters, const float* A, const float* B, float* C) {
    for (int k = 0; k < iters; ++k) {
        #pragma omp parallel for
        for (size_t i = 0; i < size; ++i) {
            C[i] = A[i] + B[i];
        }
        // Prevent optimization
        if (C[0] > 1e9) printf("dummy\n"); 
    }
}

// H2D workload: Continuously copy data to GPU
void h2d_workload(bool* stop, size_t size, void* h_data, void* d_data, cudaStream_t stream) {
    while (!*stop) {
        cudaMemcpyAsync(d_data, h_data, size, cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);
    }
}

void benchmark_memory_types(size_t size, int iters) {
    std::cout << "\n--- Memory Type Benchmark (Vector Add) ---" << std::endl;
    std::cout << "Vector Size: " << size << " elements (" << size * 4 / (1024*1024) << " MB)" << std::endl;

    auto run_test = [&](const char* name, float* A, float* B, float* C) {
        // Initialize
        #pragma omp parallel for
        for(size_t i=0; i<size; ++i) { A[i] = 1.0f; B[i] = 2.0f; }

        auto start = std::chrono::high_resolution_clock::now();
        cpu_vecadd(size, iters, A, B, C);
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << name << ": " << time_ms << " ms" << std::endl;
    };

    // 1. Pageable (malloc)
    float *A_pg = (float*)malloc(size * sizeof(float));
    float *B_pg = (float*)malloc(size * sizeof(float));
    float *C_pg = (float*)malloc(size * sizeof(float));
    run_test("Pageable (malloc)", A_pg, B_pg, C_pg);
    free(A_pg); free(B_pg); free(C_pg);

    // 2. Pinned (cudaMallocHost)
    float *A_pin, *B_pin, *C_pin;
    cudaMallocHost(&A_pin, size * sizeof(float));
    cudaMallocHost(&B_pin, size * sizeof(float));
    cudaMallocHost(&C_pin, size * sizeof(float));
    run_test("Pinned (cudaMallocHost)", A_pin, B_pin, C_pin);
    cudaFreeHost(A_pin); cudaFreeHost(B_pin); cudaFreeHost(C_pin);

    // 3. Pinned Write-Combined (cudaHostAllocWriteCombined)
    float *A_wc, *B_wc, *C_wc;
    cudaHostAlloc(&A_wc, size * sizeof(float), cudaHostAllocWriteCombined);
    cudaHostAlloc(&B_wc, size * sizeof(float), cudaHostAllocWriteCombined);
    cudaHostAlloc(&C_wc, size * sizeof(float), cudaHostAllocWriteCombined);
    run_test("Pinned WC (cudaHostAllocWriteCombined)", A_wc, B_wc, C_wc);
    cudaFreeHost(A_wc); cudaFreeHost(B_wc); cudaFreeHost(C_wc);
}

int main(int argc, char** argv) {
    int mode = 0; // 0: MatMul, 1: VecAdd, 2: Memory Benchmark
    if (argc > 1) mode = std::atoi(argv[1]);

    int N = 1024; // Matrix size
    size_t vec_size = 1024 * 1024 * 128; // 128M floats = 512MB
    int vec_iters = 100;

    if (mode == 2) {
        benchmark_memory_types(vec_size, 10); // Reduced iterations for benchmark
        return 0;
    }

    if (argc > 2) N = std::atoi(argv[2]);
    
    size_t transfer_size = 1024 * 1024 * 100; // 100 MB
    if (argc > 3) transfer_size = (size_t)std::atoi(argv[3]) * 1024 * 1024;

    std::cout << "Mode: " << (mode == 0 ? "MatMul (Compute Bound)" : "VecAdd (Memory Bound)") << std::endl;
    std::cout << "H2D Transfer Size: " << transfer_size / (1024*1024) << " MB" << std::endl;

    // Setup GPU data
    void *h_data, *d_data;
    cudaError_t err = cudaMallocHost(&h_data, transfer_size);
    if (err != cudaSuccess) {
        std::cerr << "cudaMallocHost failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    err = cudaMalloc(&d_data, transfer_size);
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    memset(h_data, 1, transfer_size);
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Setup CPU data
    std::vector<float> A, B, C;
    if (mode == 0) {
        std::cout << "Matrix Size: " << N << "x" << N << std::endl;
        A.resize(N * N, 1.0f);
        B.resize(N * N, 2.0f);
        C.resize(N * N);
    } else {
        std::cout << "Vector Size: " << vec_size << " elements (" << vec_size * 4 / (1024*1024) << " MB)" << std::endl;
        A.resize(vec_size, 1.0f);
        B.resize(vec_size, 2.0f);
        C.resize(vec_size);
    }

    // --- Baseline: CPU Only ---
    std::cout << "Running CPU Baseline..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    if (mode == 0) cpu_matmul(N, A, B, C);
    else cpu_vecadd(vec_size, vec_iters, A.data(), B.data(), C.data());
    auto end = std::chrono::high_resolution_clock::now();
    double cpu_baseline = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "CPU Baseline Time: " << cpu_baseline << " ms" << std::endl;

    // --- Interference: CPU + H2D ---
    std::cout << "Running CPU + H2D Interference..." << std::endl;
    bool stop = false;
    std::thread h2d_thread(h2d_workload, &stop, transfer_size, h_data, d_data, stream);

    // Warmup H2D slightly
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    start = std::chrono::high_resolution_clock::now();
    if (mode == 0) cpu_matmul(N, A, B, C);
    else cpu_vecadd(vec_size, vec_iters, A.data(), B.data(), C.data());
    end = std::chrono::high_resolution_clock::now();
    double cpu_interference = std::chrono::duration<double, std::milli>(end - start).count();
    
    stop = true;
    h2d_thread.join();

    std::cout << "CPU + H2D Time: " << cpu_interference << " ms" << std::endl;
    std::cout << "Slowdown: " << (cpu_interference / cpu_baseline - 1.0) * 100.0 << "%" << std::endl;

    // Cleanup
    cudaFreeHost(h_data);
    cudaFree(d_data);
    cudaStreamDestroy(stream);

    return 0;
}

