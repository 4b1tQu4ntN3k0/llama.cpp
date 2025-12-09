#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "common.h"

#include <vector>
#include <thread>
#include <future>
#include <chrono>
#include <iostream>
#include <string>
#include <cstring>
#include <algorithm>

// Helper to measure time
static int64_t time_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()
    ).count();
}

// Helper to create a simple graph: C = A * B
struct SimpleGraph {
    struct ggml_context * ctx;
    struct ggml_tensor * a;
    struct ggml_tensor * b;
    struct ggml_tensor * c;
    struct ggml_cgraph * gf;
    ggml_backend_buffer_t buffer;

    SimpleGraph(ggml_backend_t backend, int rows, int cols, int inner) {
        // Init context
        struct ggml_init_params params = {
            /*.mem_size   =*/ 1024*1024*10, // 10 MB for graph overhead
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
        };
        ctx = ggml_init(params);

        // Create tensors
        a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, inner, rows);
        b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, inner, cols);
        c = ggml_mul_mat(ctx, a, b);
        
        gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, c);

        // Allocate
        buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    }

    ~SimpleGraph() {
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }

    void compute(ggml_backend_t backend) {
        ggml_backend_graph_compute(backend, gf);
    }
};

int main(int argc, char ** argv) {
    ggml_time_init();
    ggml_backend_load_all();

    // Initialize backends
    ggml_backend_t backend_cpu = ggml_backend_init_by_name("cpu", NULL);
    
    // Find GPU backend
    ggml_backend_t backend_gpu_compute = NULL;
    ggml_backend_t backend_gpu_transfer = NULL;

    size_t dev_count = ggml_backend_dev_count();
    printf("Available devices: %zu\n", dev_count);
    for (size_t i = 0; i < dev_count; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const char * name = ggml_backend_dev_name(dev);
        enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
        printf("  Device %zu: %s (Type: %d)\n", i, name, type);
        
        if (type == GGML_BACKEND_DEVICE_TYPE_GPU) {
            if (!backend_gpu_compute) {
                backend_gpu_compute = ggml_backend_dev_init(dev, NULL);
                // Try to init a second one for transfer
                backend_gpu_transfer = ggml_backend_dev_init(dev, NULL);
            }
        }
    }

    if (!backend_gpu_compute) {
        fprintf(stderr, "GPU backend not found!\n");
        return 1;
    }
    if (!backend_gpu_transfer) {
        backend_gpu_transfer = backend_gpu_compute; // Fallback
    }

    // Parameters
    // Adjust these to get significant duration (~50-100ms)
    int matrix_size_cpu = 2048; 
    int matrix_size_gpu = 7000; 
    size_t transfer_size = 1024 * 1024 * 512; // 512 MB

    printf("Initializing tasks...\n");
    printf("  CPU Matrix Size: %d x %d\n", matrix_size_cpu, matrix_size_cpu);
    printf("  GPU Matrix Size: %d x %d\n", matrix_size_gpu, matrix_size_gpu);
    printf("  Transfer Size: %zu MB\n", transfer_size / (1024*1024));

    // 1. Setup CPU Task
    SimpleGraph task_cpu(backend_cpu, matrix_size_cpu, matrix_size_cpu, matrix_size_cpu);

    // 2. Setup GPU Task
    SimpleGraph task_gpu(backend_gpu_compute, matrix_size_gpu, matrix_size_gpu, matrix_size_gpu);

    // 3. Setup Transfer Task
    struct ggml_init_params params = { 1024, NULL, true };
    struct ggml_context * ctx_transfer = ggml_init(params);
    struct ggml_tensor * tensor_transfer = ggml_new_tensor_1d(ctx_transfer, GGML_TYPE_F32, transfer_size / sizeof(float));
    
    // Allocate on the transfer backend
    ggml_backend_buffer_t buffer_transfer = ggml_backend_alloc_ctx_tensors(ctx_transfer, backend_gpu_transfer);
    
    // Host data
    std::vector<uint8_t> host_data(transfer_size, 1);

    // Warmup
    printf("Warmup...\n");
    task_cpu.compute(backend_cpu);
    task_gpu.compute(backend_gpu_compute);
    ggml_backend_tensor_set(tensor_transfer, host_data.data(), 0, transfer_size);
    ggml_backend_synchronize(backend_gpu_compute);
    ggml_backend_synchronize(backend_gpu_transfer);

    // Measure Individual
    printf("Benchmarking Individual Steps...\n");
    
    int64_t t0 = time_us();
    task_cpu.compute(backend_cpu);
    int64_t t_cpu = time_us() - t0;

    t0 = time_us();
    task_gpu.compute(backend_gpu_compute);
    ggml_backend_synchronize(backend_gpu_compute);
    int64_t t_gpu = time_us() - t0;

    t0 = time_us();
    ggml_backend_tensor_set(tensor_transfer, host_data.data(), 0, transfer_size);
    ggml_backend_synchronize(backend_gpu_transfer);
    int64_t t_transfer = time_us() - t0;

    printf("Baseline Results:\n");
    printf("  CPU Compute:  %8.2f ms\n", t_cpu / 1000.0);
    printf("  GPU Compute:  %8.2f ms\n", t_gpu / 1000.0);
    printf("  H2D Transfer: %8.2f ms\n", t_transfer / 1000.0);
    printf("  Sum (Serial): %8.2f ms\n", (t_cpu + t_gpu + t_transfer) / 1000.0);

    // Measure Concurrent
    printf("Benchmarking Concurrent Execution...\n");
    
    ggml_backend_synchronize(backend_gpu_compute);
    ggml_backend_synchronize(backend_gpu_transfer);
    
    t0 = time_us();

    // Launch CPU
    auto future_cpu = std::async(std::launch::async, [&]() {
        task_cpu.compute(backend_cpu);
    });

    // Launch GPU Compute
    auto future_gpu = std::async(std::launch::async, [&]() {
        task_gpu.compute(backend_gpu_compute);
        ggml_backend_synchronize(backend_gpu_compute);
    });

    // Launch Transfer
    auto future_transfer = std::async(std::launch::async, [&]() {
        ggml_backend_tensor_set(tensor_transfer, host_data.data(), 0, transfer_size);
        ggml_backend_synchronize(backend_gpu_transfer);
    });

    future_cpu.wait();
    future_gpu.wait();
    future_transfer.wait();

    int64_t t_concurrent = time_us() - t0;

    printf("Concurrent Results:\n");
    printf("  Total Time:   %8.2f ms\n", t_concurrent / 1000.0);
    
    double ideal = std::max({(double)t_cpu, (double)t_gpu, (double)t_transfer});
    printf("  Ideal Time:   %8.2f ms (max of single tasks)\n", ideal / 1000.0);
    printf("  Overhead:     %8.2f ms\n", (t_concurrent - ideal) / 1000.0);
    
    if (t_concurrent < (t_cpu + t_gpu + t_transfer) * 0.8) {
        printf("SUCCESS: Significant concurrency detected!\n");
    } else {
        printf("WARNING: Execution appears serialized.\n");
    }

    // Cleanup
    ggml_free(ctx_transfer);
    ggml_backend_buffer_free(buffer_transfer);
    ggml_backend_free(backend_cpu);
    ggml_backend_free(backend_gpu_compute);
    if (backend_gpu_transfer != backend_gpu_compute) {
        ggml_backend_free(backend_gpu_transfer);
    }

    return 0;
}
