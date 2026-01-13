#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <vector>
#include <string>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <algorithm>

// Qwen3 14B Operator Definitions
struct OpConfig {
    std::string name;
    int64_t input_dim;  // ne0 of weight (columns)
    int64_t output_dim; // ne1 of weight (rows)
    ggml_type type;
};

// Based on provided weights info
const std::vector<OpConfig> qwen3_ops = {
    {"attn_q",      5120,  5120,  GGML_TYPE_Q4_K},
    {"attn_k",      5120,  1024,  GGML_TYPE_Q4_K},
    {"attn_v",      5120,  1024,  GGML_TYPE_Q6_K},
    {"attn_o",      5120,  5120,  GGML_TYPE_Q4_K},
    {"ffn_gate",    5120, 17408,  GGML_TYPE_Q4_K},
    {"ffn_up",      5120, 17408,  GGML_TYPE_Q4_K},
    {"ffn_down",   17408,  5120,  GGML_TYPE_Q6_K} 
};

// Utils
size_t get_tensor_size(ggml_type type, int64_t ne0, int64_t ne1) {
    return (ne0 * ne1 * ggml_type_size(type)) / ggml_blck_size(type);
}

struct PerfResult {
    double t_cpu;      // ms
    double t_transfer; // ms
    double t_gpu;      // ms
};

// Helper to build graph
struct TestGraph {
    struct ggml_context* ctx;
    struct ggml_cgraph* gf;
    struct ggml_tensor* input;
    struct ggml_tensor* weight;
    struct ggml_tensor* output;
};

TestGraph build_graph(const OpConfig& op, int batch_size) {
    size_t ctx_size = 1024 * 1024 * 4;
    struct ggml_init_params init_params = { ctx_size, NULL, true };
    struct ggml_context* ctx = ggml_init(init_params);

    struct ggml_tensor* weight = ggml_new_tensor_2d(ctx, op.type, op.input_dim, op.output_dim);
    struct ggml_tensor* input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, op.input_dim, batch_size);
    struct ggml_tensor* output = ggml_mul_mat(ctx, weight, input);
    
    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, output);
    
    return {ctx, gf, input, weight, output};
}

// Benchmark a single operator
PerfResult benchmark_op(ggml_backend_t cpu_backend, ggml_backend_t gpu_backend, const OpConfig& op, int batch_size) {
    PerfResult res = {0, 0, 0};
    
    size_t w_sz = get_tensor_size(op.type, op.input_dim, op.output_dim);
    size_t i_sz = get_tensor_size(GGML_TYPE_F32, op.input_dim, batch_size);
    
    std::vector<uint8_t> host_w(w_sz);
    std::vector<uint8_t> host_i(i_sz);
    memset(host_w.data(), 1, w_sz);
    memset(host_i.data(), 1, i_sz);

    // --- 1. Measure CPU Compute ---
    if (cpu_backend) {
        TestGraph g = build_graph(op, batch_size);
        ggml_backend_buffer_t cpu_buf = ggml_backend_alloc_ctx_tensors(g.ctx, cpu_backend);
        
        if (cpu_buf) {
            ggml_backend_tensor_set(g.weight, host_w.data(), 0, w_sz);
            ggml_backend_tensor_set(g.input, host_i.data(), 0, i_sz);
            
            // Warmup
            ggml_backend_graph_compute(cpu_backend, g.gf);
            
            int64_t t0 = ggml_time_us();
            ggml_backend_graph_compute(cpu_backend, g.gf);
            res.t_cpu = (ggml_time_us() - t0) / 1000.0;
            
            ggml_backend_buffer_free(cpu_buf);
        }
        ggml_free(g.ctx);
    }

    // --- 2. Measure GPU Performance ---
    if (gpu_backend) {
        TestGraph g = build_graph(op, batch_size);
        ggml_backend_buffer_t gpu_buf = ggml_backend_alloc_ctx_tensors(g.ctx, gpu_backend);
        
        if (gpu_buf) {
            // A. Transfer Time
            ggml_backend_synchronize(gpu_backend);
            int64_t t_trans_start = ggml_time_us();
            
            ggml_backend_tensor_set(g.weight, host_w.data(), 0, w_sz);
            ggml_backend_tensor_set(g.input, host_i.data(), 0, i_sz);
            
            ggml_backend_synchronize(gpu_backend);
            res.t_transfer = (ggml_time_us() - t_trans_start) / 1000.0;
            
            // B. Compute Time
            ggml_backend_graph_compute(gpu_backend, g.gf); // Warmup
            ggml_backend_synchronize(gpu_backend);
            
            int64_t t_comp_start = ggml_time_us();
            int n_iter = 5;
            for(int i=0; i<n_iter; i++) {
                ggml_backend_graph_compute(gpu_backend, g.gf);
            }
            ggml_backend_synchronize(gpu_backend);
            res.t_gpu = (ggml_time_us() - t_comp_start) / 1000.0 / n_iter;

            ggml_backend_buffer_free(gpu_buf);
        } else {
            fprintf(stderr, "Failed to alloc GPU buffer for %s\n", op.name.c_str());
        }
        ggml_free(g.ctx);
    }

    return res;
}

// VRAM Planner
struct TensorPlan {
    std::string name;
    double size_mb;
    double transfer_ms;
    double compute_ms;
    double score; // Priority/ROI
};

void recommend_vram_strategy(const std::vector<OpConfig>& ops, ggml_backend_t backend, size_t vram_limit_mb) {
    const int N_LAYERS = 39; // User specified
    printf("\n=== VRAM Residency Strategy Planner (Limit: %zu MB, Layers: %d) ===\n", vram_limit_mb, N_LAYERS);
    printf("Strategy: Prioritize tensors that are Transfer-Bound (Transfer Time > Compute Budget)\n");
    
    // 1. Profile all ops first (Assume Batch Size = 1 for decoding latency optimization)
    std::vector<TensorPlan> suggestions;
    double single_layer_size = 0;

    for (const auto& op : ops) {
        // Run a quick benchmark (BS=1)
        PerfResult res = benchmark_op(NULL, backend, op, 1);
        
        size_t bytes = get_tensor_size(op.type, op.input_dim, op.output_dim);
        double size_mb = bytes / (1024.0 * 1024.0);
        single_layer_size += size_mb;

        // Score = Saving / Size
        double stall_time = std::max(0.0, res.t_transfer - res.t_gpu);
        double score = stall_time / size_mb;

        suggestions.push_back({op.name, size_mb, res.t_transfer, res.t_gpu, score});
    }

    // 2. Sort by Score (Desc)
    std::sort(suggestions.begin(), suggestions.end(), [](const TensorPlan& a, const TensorPlan& b) {
        return a.score > b.score;
    });

    // 3. Fill VRAM
    double current_vram = 0;
    double estimated_stall_saved = 0;
    
    printf("\n%-15s | %-8s | %-8s | %-8s | %-8s | %-13s | %-10s\n", 
           "Op Name", "Size(MB)", "Trans(ms)", "Gpu(ms)", "Score", "Resident/Tot", "Saved(ms)");
    printf("--------------------------------------------------------------------------------------------\n");

    for (const auto& plan : suggestions) {
        // Calculate how many layers of this op we can fit
        double remaining = vram_limit_mb - current_vram;
        int count = 0;
        if (remaining > 0) {
            count = (int)(remaining / plan.size_mb);
            if (count > N_LAYERS) count = N_LAYERS;
        }

        current_vram += count * plan.size_mb;
        double saved_per_layer = std::max(0.0, plan.transfer_ms - plan.compute_ms);
        double total_saved = count * saved_per_layer;
        estimated_stall_saved += total_saved;

        printf("%-15s | %8.2f | %8.3f | %8.3f | %8.5f | %3d / %-3d     | %8.2f\n", 
            plan.name.c_str(), plan.size_mb, plan.transfer_ms, plan.compute_ms, 
            plan.score,
            count, N_LAYERS, total_saved);
    }

    double total_model_weights = single_layer_size * N_LAYERS;
    printf("\nTotal Model Weights (39 Layers): %.2f MB\n", total_model_weights);
    printf("VRAM Used:                       %.2f MB (%.1f%%)\n", current_vram, (current_vram/vram_limit_mb)*100);
    printf("Est. Latency Reduction per Tok:  %.2f ms\n", estimated_stall_saved);
}

int main(int argc, char ** argv) {
    ggml_time_init();
    ggml_backend_load_all();

    ggml_backend_t gpu = NULL;
    size_t dev_count = ggml_backend_dev_count();
    for (size_t i=0; i<dev_count; i++) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            gpu = ggml_backend_dev_init(d, NULL);
            break;
        }
    }

    if (gpu) {
        // Ask for VRAM limit
        size_t vram_limit = 4096; // Default 4GB
        if (argc > 1) vram_limit = atoi(argv[1]);

        recommend_vram_strategy(qwen3_ops, gpu, vram_limit);
        
        ggml_backend_free(gpu);
    } else {
        printf("No GPU found.\n");
    }
    return 0;
}
