#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <vector>
#include <string>
#include <chrono>
#include <random>
#include <cstring>
#include <map>

// 定义算子配置
struct OpConfig {
    std::string name;
    std::vector<int64_t> shape; // [ne0, ne1]. For mul_mat weight, ne0 is input_dim, ne1 is output_dim?
                                // ggml_mul_mat(A, B): result has rows = A->ne[1].
                                // If A is [5120, 1024], result has 1024 rows.
    ggml_type type;
};

// User provided:
// blk.0.attn_k.weight [5120, 1024] Q4_K
// blk.0.attn_q.weight [5120, 5120] Q4_K
// ...
const std::vector<OpConfig> ops = {
    {"attn_k",      {5120, 1024},  GGML_TYPE_Q4_K},
    {"attn_q",      {5120, 5120},  GGML_TYPE_Q4_K},
    {"attn_v",      {5120, 1024},  GGML_TYPE_Q6_K},
    {"ffn_down",    {17408, 5120}, GGML_TYPE_Q6_K}, 
    {"ffn_gate",    {5120, 17408}, GGML_TYPE_Q4_K}, 
    {"ffn_up",      {5120, 17408}, GGML_TYPE_Q4_K}, 
    {"attn_output", {5120, 5120},  GGML_TYPE_Q4_K},
};

// 辅助函数：获取类型大小
size_t get_tensor_size(ggml_type type, int64_t ne0, int64_t ne1) {
    return (ne0 * ne1 * ggml_type_size(type)) / ggml_blck_size(type);
}

void benchmark_operator(ggml_backend_t backend, const OpConfig& op, int batch_size) {
    std::string type_name = ggml_type_name(op.type);

    // 1. 初始化 Context
    size_t ctx_size = 1024 * 1024 * 64; // 足以容纳图节点
    struct ggml_init_params init_params = { ctx_size, NULL, true };
    struct ggml_context * ctx = ggml_init(init_params);

    // 2. 创建 Tensor
    // 输入 Tensor 'x': [ne0_weight, batch_size]
    // 注意：ggml_mul_mat(W, X) 要求 W->ne[0] == X->ne[0]
    struct ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, op.shape[0], batch_size);
    // 权重 Tensor 'w': [ne0, ne1]
    struct ggml_tensor * weight = ggml_new_tensor_2d(ctx, op.type, op.shape[0], op.shape[1]);
    
    // 3. 构建计算图 (Mul Mat)
    struct ggml_tensor * result = ggml_mul_mat(ctx, weight, input);
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, result);

    // 4. 后端分配
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        fprintf(stderr, "Failed to allocate buffer for %s\n", op.name.c_str());
        ggml_free(ctx);
        return;
    }

    // 5. 准备随机数据
    size_t weight_size = get_tensor_size(op.type, op.shape[0], op.shape[1]);
    size_t input_size = get_tensor_size(GGML_TYPE_F32, op.shape[0], batch_size);
    
    std::vector<uint8_t> host_weight(weight_size);
    std::vector<uint8_t> host_input(input_size);
    
    // 简单填充 (Dummy Data)
    for(size_t i=0; i<weight_size; i++) host_weight[i] = (uint8_t)(i % 255);
    for(size_t i=0; i<input_size; i++) host_input[i] = (uint8_t)(i % 255);

    // === 测试过程 ===

    // A. 传输时间测试 (Transfer Time)
    // 我们测量将数据从 Host 内存 set 到 Backend Tensor 的时间
    // 这包括了权重的传输和输入的传输
    int64_t t_transfer_start = ggml_time_us();
    
    ggml_backend_tensor_set(weight, host_weight.data(), 0, weight_size);
    ggml_backend_tensor_set(input, host_input.data(), 0, input_size);
    
    // 必须同步以确保传输完成
    ggml_backend_synchronize(backend);
    int64_t t_transfer_end = ggml_time_us();
    double transfer_ms = (t_transfer_end - t_transfer_start) / 1000.0;

    // B. 计算时间测试 (Compute Time)
    // 预热
    ggml_backend_graph_compute(backend, gf);
    ggml_backend_synchronize(backend);

    int n_iter = 10;
    int64_t t_compute_start = ggml_time_us();
    for (int i = 0; i < n_iter; i++) {
        ggml_backend_graph_compute(backend, gf);
    }
    ggml_backend_synchronize(backend);
    int64_t t_compute_end = ggml_time_us();
    double compute_ms = (t_compute_end - t_compute_start) / 1000.0 / n_iter;

    printf("| %-15s | %-6s | %5ldx%-5ld | %8.3f ms | %8.3f ms |\n", 
           op.name.c_str(), type_name.c_str(), op.shape[0], op.shape[1], 
           transfer_ms, compute_ms);

    // 清理
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

int main(int argc, char ** argv) {
    ggml_time_init();
    ggml_backend_load_all();

    int batch_size = 1; // 默认 batch size 1
    if (argc > 1) batch_size = atoi(argv[1]);

    printf("Benchmarking Operator Performance (Batch Size: %d)\n", batch_size);
    printf("Transfer: Time to move Weights + Input from CPU RAM to Backend VRAM/RAM\n");
    printf("Compute:  Time to execute ggml_mul_mat\n\n");

    // 1. CPU 测试
    ggml_backend_t cpu_backend = ggml_backend_init_by_name("cpu", NULL);
    if (cpu_backend) {
        printf("=== CPU Backend ===\n");
        printf("| %-15s | %-6s | %-11s | %-11s | %-11s |\n", "Op Name", "Type", "Shape", "Transfer", "Compute");
        printf("|%s|\n", std::string(66, '-').c_str());
        
        for (const auto& op : ops) {
            benchmark_operator(cpu_backend, op, batch_size);
        }
        printf("\n");
        ggml_backend_free(cpu_backend);
    }

    // 2. GPU 测试
    ggml_backend_dev_t gpu_dev = NULL;
    size_t dev_count = ggml_backend_dev_count();
    for (size_t i = 0; i < dev_count; ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            gpu_dev = d;
            break;
        }
    }

    if (gpu_dev) {
        ggml_backend_t gpu_backend = ggml_backend_dev_init(gpu_dev, NULL);
        if (gpu_backend) {
            printf("=== GPU Backend (%s) ===\n", ggml_backend_dev_name(gpu_dev));
            printf("| %-15s | %-6s | %-11s | %-11s | %-11s |\n", "Op Name", "Type", "Shape", "Transfer", "Compute");
            printf("|%s|\n", std::string(66, '-').c_str());

            for (const auto& op : ops) {
                benchmark_operator(gpu_backend, op, batch_size);
            }
            ggml_backend_free(gpu_backend);
        }
    } else {
        printf("No GPU backend found.\n");
    }

    return 0;
}
