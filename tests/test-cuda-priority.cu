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

int main() {
    int priority_low, priority_high;
    // 获取当前设备支持的优先级范围
    // 注意：数值越小，优先级越高（例如 high=-1, low=0）
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high));
    std::cout << "Priority Range: Low Value=" << priority_low << ", High Value=" << priority_high << std::endl;

    cudaStream_t stream_low, stream_high;
    CUDA_CHECK(cudaStreamCreateWithPriority(&stream_low, cudaStreamNonBlocking, priority_low));
    CUDA_CHECK(cudaStreamCreateWithPriority(&stream_high, cudaStreamNonBlocking, priority_high));

    // 大块数据: 500MB
    size_t size_large = 500 * 1024 * 1024;
    // 小块数据: 1MB
    size_t size_small = 1 * 1024 * 1024;

    void *h_large, *d_large;
    void *h_small, *d_small;

    // 关键：必须使用 Pinned Memory (cudaMallocHost) 才能实现真正的异步传输
    std::cout << "Allocating Pinned Memory..." << std::endl;
    CUDA_CHECK(cudaMallocHost(&h_large, size_large));
    CUDA_CHECK(cudaMallocHost(&h_small, size_small));
    CUDA_CHECK(cudaMalloc(&d_large, size_large));
    CUDA_CHECK(cudaMalloc(&d_small, size_small));

    // 初始化数据，强制物理内存分配
    memset(h_large, 1, size_large);
    memset(h_small, 2, size_small);

    // Warmup (预热)
    cudaMemcpyAsync(d_small, h_small, size_small, cudaMemcpyHostToDevice, stream_high);
    cudaDeviceSynchronize();

    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "Starting Large Transfer (Low Priority) of " << size_large / 1024 / 1024 << " MB..." << std::endl;
    
    // 1. 启动低优先级的大任务 (切分为小块以允许抢占)
    size_t chunk_size = 10 * 1024 * 1024; // 10MB chunks
    size_t num_chunks = size_large / chunk_size;
    
    // 用于记录每个 chunk 的时间
    std::vector<cudaEvent_t> start_events(num_chunks);
    std::vector<cudaEvent_t> stop_events(num_chunks);
    for (size_t i = 0; i < num_chunks; ++i) {
        CUDA_CHECK(cudaEventCreate(&start_events[i]));
        CUDA_CHECK(cudaEventCreate(&stop_events[i]));
    }

    // 记录小任务的时间
    cudaEvent_t small_start, small_stop;
    CUDA_CHECK(cudaEventCreate(&small_start));
    CUDA_CHECK(cudaEventCreate(&small_stop));

    for (size_t i = 0; i < num_chunks; ++i) {
        size_t offset = i * chunk_size;
        CUDA_CHECK(cudaEventRecord(start_events[i], stream_low));
        CUDA_CHECK(cudaMemcpyAsync((char*)d_large + offset, (char*)h_large + offset, chunk_size, cudaMemcpyHostToDevice, stream_low));
        CUDA_CHECK(cudaEventRecord(stop_events[i], stream_low));
    }
    if (size_large % chunk_size != 0) {
        size_t offset = num_chunks * chunk_size;
        CUDA_CHECK(cudaMemcpyAsync((char*)d_large + offset, (char*)h_large + offset, size_large % chunk_size, cudaMemcpyHostToDevice, stream_low));
    }

    // 稍微延迟一点点，确保大任务已经提交给 GPU Copy Engine
    // std::this_thread::sleep_for(std::chrono::microseconds(100));

    std::cout << "Starting Small Transfer (High Priority) of " << size_small / 1024 / 1024 << " MB..." << std::endl;
    
    auto start_cpu = std::chrono::high_resolution_clock::now();
    
    // 2. 启动高优先级的小任务
    CUDA_CHECK(cudaEventRecord(small_start, stream_high));
    CUDA_CHECK(cudaMemcpyAsync(d_small, h_small, size_small, cudaMemcpyHostToDevice, stream_high));
    CUDA_CHECK(cudaEventRecord(small_stop, stream_high));
    
    // 3. 只等待高优先级任务完成
    CUDA_CHECK(cudaStreamSynchronize(stream_high));
    
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end_cpu - start_cpu;

    std::cout << "Small Transfer finished in " << elapsed.count() << " ms." << std::endl;

    // 4. 检查此时大任务的状态
    cudaError_t status = cudaStreamQuery(stream_low);
    if (status == cudaErrorNotReady) {
        std::cout << "SUCCESS: Large Transfer is still running! Small task finished first." << std::endl;
    } else if (status == cudaSuccess) {
        std::cout << "WARNING: Large Transfer already finished. It might have blocked the Small task." << std::endl;
    } else {
        std::cerr << "Error querying low priority stream." << std::endl;
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    // 打印详细时间线
    float small_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&small_ms, small_start, small_stop));
    std::cout << "\n[Timeline Analysis]" << std::endl;
    std::cout << "Small Task: Duration = " << small_ms << " ms" << std::endl;

    // 以第一个 chunk 的开始时间为基准
    std::cout << "Large Chunks Timeline:" << std::endl;
    for (size_t i = 0; i < num_chunks; ++i) {
        float start_ms = 0, duration_ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&start_ms, start_events[0], start_events[i]));
        CUDA_CHECK(cudaEventElapsedTime(&duration_ms, start_events[i], stop_events[i]));
        
        std::cout << "  Chunk " << i << ": Start = " << start_ms << " ms, Duration = " << duration_ms << " ms, End = " << start_ms + duration_ms << " ms" << std::endl;
    }

    // 尝试估算小任务在时间轴上的位置
    float small_start_offset = 0;
    cudaError_t err = cudaEventElapsedTime(&small_start_offset, start_events[0], small_start);
    if (err == cudaSuccess) {
        std::cout << "Small Task Start Offset (relative to Chunk 0 Start): " << small_start_offset << " ms" << std::endl;
        std::cout << "Small Task End Offset: " << small_start_offset + small_ms << " ms" << std::endl;
    } else {
        std::cout << "Cannot calculate cross-stream timing (device might not support it)." << std::endl;
    }

    // Cleanup
    for (size_t i = 0; i < num_chunks; ++i) {
        cudaEventDestroy(start_events[i]);
        cudaEventDestroy(stop_events[i]);
    }
    cudaEventDestroy(small_start);
    cudaEventDestroy(small_stop);

    return 0;
}
