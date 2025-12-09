#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <vector>
#include <string>
#include <chrono>
#include <random>
#include <cstring>

// Model parameters
struct ModelParams {
    const char* name;
    int n_embd;
    int n_head;
    int n_kv_head;
    int head_dim;
    int n_ff;
};

const ModelParams models[] = {
    {"llama-3-8b",    4096, 32, 8, 128, 14336},
    {"llama-3.1-70b", 8192, 64, 8, 128, 28672},
    {"qwen-2.5-14b",  5120, 40, 8, 128, 13824},
    {"qwen-2.5-32b",  5120, 40, 8, 128, 27648},
};

const int n_seq = 1;

// Helper to generate random data
static std::vector<uint8_t> generate_random_data(enum ggml_type type, int64_t ne0, int64_t ne1) {
    int64_t nelements = ne0 * ne1;
    int64_t nbytes = nelements * ggml_type_size(type) / ggml_blck_size(type);
    std::vector<uint8_t> data(nbytes);
    std::vector<float> float_data(nelements);
    
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    for (auto & x : float_data) x = dist(rng);
    
    if (type == GGML_TYPE_F32) {
        memcpy(data.data(), float_data.data(), data.size());
    } else if (type == GGML_TYPE_F16) {
        ggml_fp32_to_fp16_row(float_data.data(), (ggml_fp16_t*)data.data(), float_data.size());
    }
    return data;
}

struct LayerWeights {
    std::vector<uint8_t> wq;
    std::vector<uint8_t> wk;
    std::vector<uint8_t> wv;
    std::vector<uint8_t> wo;
    std::vector<uint8_t> w1;
    std::vector<uint8_t> w2;
    std::vector<uint8_t> w3;
};

void benchmark_backend(ggml_backend_t backend, const char* name, const LayerWeights& weights, const ModelParams& params, int batch_size) {
    printf("\n=== Benchmarking %s on %s (Batch Size: %d) ===\n", params.name, name, batch_size);
    
    // Create Context
    size_t ctx_size = 1024 * 1024 * 512; 
    struct ggml_init_params init_params = { ctx_size, NULL, true };
    struct ggml_context * ctx = ggml_init(init_params);

    // Define Tensors
    struct ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, params.n_embd, batch_size);
    struct ggml_tensor * wq = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, params.n_embd, params.n_embd);
    struct ggml_tensor * wk = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, params.n_embd, params.n_kv_head * params.head_dim);
    struct ggml_tensor * wv = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, params.n_embd, params.n_kv_head * params.head_dim);
    struct ggml_tensor * wo = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, params.n_embd, params.n_embd);
    struct ggml_tensor * w1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, params.n_embd, params.n_ff);
    struct ggml_tensor * w2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, params.n_ff, params.n_embd);
    struct ggml_tensor * w3 = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, params.n_embd, params.n_ff);

    // Build Graph
    struct ggml_tensor * cur = input;
    struct ggml_tensor * Q = ggml_mul_mat(ctx, wq, cur);
    struct ggml_tensor * K = ggml_mul_mat(ctx, wk, cur);
    struct ggml_tensor * V = ggml_mul_mat(ctx, wv, cur);
    
    // Dummy ops to consume K and V
    struct ggml_tensor * KQ = ggml_mul(ctx, Q, ggml_sum(ctx, K));
    struct ggml_tensor * KQV = ggml_mul(ctx, KQ, ggml_sum(ctx, V));
    
    struct ggml_tensor * attn_out = ggml_mul_mat(ctx, wo, Q);
    attn_out = ggml_add(ctx, attn_out, KQV);
    cur = ggml_add(ctx, cur, attn_out);
    
    struct ggml_tensor * ffn_gate = ggml_mul_mat(ctx, w1, cur);
    struct ggml_tensor * ffn_up = ggml_mul_mat(ctx, w3, cur);
    struct ggml_tensor * ffn_act = ggml_silu(ctx, ffn_gate);
    struct ggml_tensor * ffn_inter = ggml_mul(ctx, ffn_act, ffn_up);
    struct ggml_tensor * ffn_out = ggml_mul_mat(ctx, w2, ffn_inter);
    struct ggml_tensor * output = ggml_add(ctx, cur, ffn_out);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, output);

    // Allocate
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        fprintf(stderr, "Failed to allocate buffer\n");
        ggml_free(ctx);
        return;
    }

    // Generate Input Data
    std::vector<uint8_t> input_data = generate_random_data(GGML_TYPE_F32, params.n_embd, batch_size);

    // Transfer
    printf("Transferring weights...\n");
    int64_t t_transfer_start = ggml_time_us();
    ggml_backend_tensor_set(wq, weights.wq.data(), 0, weights.wq.size());
    ggml_backend_tensor_set(wk, weights.wk.data(), 0, weights.wk.size());
    ggml_backend_tensor_set(wv, weights.wv.data(), 0, weights.wv.size());
    ggml_backend_tensor_set(wo, weights.wo.data(), 0, weights.wo.size());
    ggml_backend_tensor_set(w1, weights.w1.data(), 0, weights.w1.size());
    ggml_backend_tensor_set(w2, weights.w2.data(), 0, weights.w2.size());
    ggml_backend_tensor_set(w3, weights.w3.data(), 0, weights.w3.size());
    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size());
    ggml_backend_synchronize(backend);
    int64_t t_transfer_end = ggml_time_us();

    double transfer_time_ms = (t_transfer_end - t_transfer_start) / 1000.0;
    size_t total_bytes = weights.wq.size() + weights.wk.size() + weights.wv.size() + weights.wo.size() +
                         weights.w1.size() + weights.w2.size() + weights.w3.size();
    printf("Layer Size: %.2f MB\n", total_bytes / (1024.0 * 1024.0));
    printf("Transfer Time: %.2f ms\n", transfer_time_ms);
    printf("Bandwidth: %.2f GB/s\n", (total_bytes / (1024.0 * 1024.0 * 1024.0)) / (transfer_time_ms / 1000.0));

    // Compute
    printf("Warming up...\n");
    ggml_backend_graph_compute(backend, gf);
    ggml_backend_synchronize(backend);

    printf("Benchmarking compute...\n");
    int64_t t_compute_start = ggml_time_us();
    for (int i = 0; i < 10; i++) {
        ggml_backend_graph_compute(backend, gf);
    }
    ggml_backend_synchronize(backend);
    int64_t t_compute_end = ggml_time_us();

    double compute_time_ms = (t_compute_end - t_compute_start) / 1000.0 / 10.0;
    printf("Average Compute Time: %.2f ms\n", compute_time_ms);

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

int main(int argc, char ** argv) {
    ggml_time_init();
    ggml_backend_load_all();

    const char* model_name = "llama-3-8b";
    if (argc > 1) {
        model_name = argv[1];
    }

    const ModelParams* params = NULL;
    for (const auto& m : models) {
        if (strcmp(m.name, model_name) == 0) {
            params = &m;
            break;
        }
    }

    if (!params) {
        fprintf(stderr, "Unknown model: %s\n", model_name);
        fprintf(stderr, "Available models:\n");
        for (const auto& m : models) {
            fprintf(stderr, "  %s\n", m.name);
        }
        return 1;
    }

    // Generate Weights
    printf("Generating random weights for %s...\n", params->name);
    LayerWeights weights;
    weights.wq = generate_random_data(GGML_TYPE_F16, params->n_embd, params->n_embd);
    weights.wk = generate_random_data(GGML_TYPE_F16, params->n_embd, params->n_kv_head * params->head_dim);
    weights.wv = generate_random_data(GGML_TYPE_F16, params->n_embd, params->n_kv_head * params->head_dim);
    weights.wo = generate_random_data(GGML_TYPE_F16, params->n_embd, params->n_embd);
    weights.w1 = generate_random_data(GGML_TYPE_F16, params->n_embd, params->n_ff);
    weights.w2 = generate_random_data(GGML_TYPE_F16, params->n_ff, params->n_embd);
    weights.w3 = generate_random_data(GGML_TYPE_F16, params->n_embd, params->n_ff);

    
    std::vector<int> batch_sizes = {1, 2, 4, 8, 16, 32};
    // std::vector<int> batch_sizes = {8, 8};
    if (argc > 2) {
        batch_sizes = {atoi(argv[2])};
    }

    // CPU Benchmark
    ggml_backend_t cpu_backend = ggml_backend_init_by_name("cpu", NULL);
    if (cpu_backend) {
        for (int bs : batch_sizes) {
            benchmark_backend(cpu_backend, "CPU", weights, *params, bs);
        }
        ggml_backend_free(cpu_backend);
    }

    // GPU Benchmark
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
            for (int bs : batch_sizes) {
                benchmark_backend(gpu_backend, ggml_backend_dev_name(gpu_dev), weights, *params, bs);
            }
            ggml_backend_free(gpu_backend);
        }
    } else {
        printf("\nNo GPU found.\n");
    }

    return 0;
}
