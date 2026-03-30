#include "models.h"
#include <regex>

static bool need_offload(const std::vector<std::regex> & regex, std::string name) {
    for (const auto & pattern : regex) {
        if (std::regex_search(name, pattern)) {
            return true;
        }
    }
    return false;
}

static void init_regex(std::vector<std::regex> & regex, const std::vector<const char *> & patterns) {
    regex.clear();
    for (const auto & p : patterns) {
        if (p) {
            regex.emplace_back(p);
        }
    }
}

llm_build_glm4_pipo::llm_build_glm4_pipo(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v;
    const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    bool use_mrope = hparams.use_mrope();
    if (ubatch.embd && !use_mrope) {
        // unfortunately, we need to forcefully stop here, to avoid users complaining about wrong results
        GGML_ABORT("This GGUF does not support multimodal. Please reconvert it.");
    }

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    ggml_tensor * attn_norm;
    ggml_tensor * attn_post_norm;
    ggml_tensor * wq;
    ggml_tensor * bq;
    ggml_tensor * wk;
    ggml_tensor * bk;
    ggml_tensor * wv;
    ggml_tensor * bv;
    ggml_tensor * wqkv;
    ggml_tensor * bqkv;
    ggml_tensor * wo;
    ggml_tensor * ffn_norm;
    ggml_tensor * ffn_up;
    ggml_tensor * ffn_down;
    ggml_tensor * ffn_post_norm;
    std::vector<std::regex> regex;
    if (params.gtype == LLM_GRAPH_TYPE_DEFAULT_PREFILL) {
        init_regex(regex, model.p_offload_weights);
    } else {
        GGML_ASSERT(params.gtype == LLM_GRAPH_TYPE_DEFAULT_DECODE);
        init_regex(regex, model.d_offload_weights);
    }

    // Only process up to last layer (skip final NextN layer)
    // Final layer tensors are loaded but not processed in forward pass
    const int n_transformer_layers = n_layer - hparams.nextn_predict_layers;
    for (int il = 0; il < n_transformer_layers; ++il) {
        auto get_offloaded = [&](ggml_tensor * t) {
            if (t && need_offload(regex, std::string(t->name))) {
                struct ggml_tensor * dynamic_tensor = ggml_backend_sched_get_pipo_tensor(params.sched, model.name_weight_map.at(std::string(t->name)));
                res->dynamic_src_tensor_list[dynamic_tensor->name].push_back(t);
                res->dynamic_dst_tensor_list[dynamic_tensor->name].push_back(dynamic_tensor);
                return dynamic_tensor;
            }
            return t;
        };

        const llama_layer & layer = model.layers[il];
#define ASSIGN_PIPO_TENSOR(name) name = get_offloaded(layer.name);
        ASSIGN_PIPO_TENSOR(attn_norm);
        ASSIGN_PIPO_TENSOR(attn_post_norm);
        ASSIGN_PIPO_TENSOR(wq);
        ASSIGN_PIPO_TENSOR(bq);
        ASSIGN_PIPO_TENSOR(wk);
        ASSIGN_PIPO_TENSOR(bk);
        ASSIGN_PIPO_TENSOR(wv);
        ASSIGN_PIPO_TENSOR(bv);
        ASSIGN_PIPO_TENSOR(wqkv);
        ASSIGN_PIPO_TENSOR(bqkv);
        ASSIGN_PIPO_TENSOR(wo);
        ASSIGN_PIPO_TENSOR(ffn_norm);
        ASSIGN_PIPO_TENSOR(ffn_up);
        ASSIGN_PIPO_TENSOR(ffn_down);
        ASSIGN_PIPO_TENSOR(ffn_post_norm);
#undef ASSIGN_PIPO_TENSOR

        ggml_tensor * inpSA = inpL;

        // Pre-attention norm
        cur = build_norm(inpL, attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            ggml_tensor * Qcur = nullptr;
            ggml_tensor * Kcur = nullptr;
            ggml_tensor * Vcur = nullptr;

            if (wqkv == nullptr) {
                Qcur = build_lora_mm(wq, cur);
                if (bq) {
                    Qcur = ggml_add(ctx0, Qcur, bq);
                }
                Kcur = build_lora_mm(wk, cur);
                if (bk) {
                    Kcur = ggml_add(ctx0, Kcur, bk);
                }
                Vcur = build_lora_mm(wv, cur);
                if (bv) {
                    Vcur = ggml_add(ctx0, Vcur, bv);
                }
                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
                Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);
            } else {
                cur = build_lora_mm(wqkv, cur);
                cb(cur, "wqkv", il);
                if (bqkv) {
                    cur = ggml_add(ctx0, cur, bqkv);
                    cb(cur, "bqkv", il);
                }
                Qcur = ggml_view_3d(ctx0, cur, n_embd_head, n_head, n_tokens, n_embd_head * sizeof(float), cur->nb[1],
                                    0 * sizeof(float) * (n_embd));
                Kcur = ggml_view_3d(ctx0, cur, n_embd_head, n_head_kv, n_tokens, n_embd_head * sizeof(float),
                                    cur->nb[1], 1 * sizeof(float) * (n_embd));
                Vcur = ggml_view_3d(ctx0, cur, n_embd_head, n_head_kv, n_tokens, n_embd_head * sizeof(float),
                                    cur->nb[1], 1 * sizeof(float) * (n_embd + n_embd_gqa));
            }

            if (use_mrope) {
                Qcur = ggml_rope_multi(ctx0, Qcur, inp_pos, nullptr,
                            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
                            ext_factor, attn_factor, beta_fast, beta_slow);

                Kcur = ggml_rope_multi(ctx0, Kcur, inp_pos, nullptr,
                            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
                            ext_factor, attn_factor, beta_fast, beta_slow);
            } else {
                // Normal RoPE
                Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, n_rot,
                                    rope_type, n_ctx_orig, freq_base, freq_scale,
                                    ext_factor, attn_factor, beta_fast, beta_slow);

                Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, n_rot,
                                    rope_type, n_ctx_orig, freq_base, freq_scale,
                                    ext_factor, attn_factor, beta_fast, beta_slow);
            }

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(inp_attn,
            wo, NULL,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f / sqrtf(float(n_embd_head)), il);
        }
        if (il == n_transformer_layers - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }
        // Post-attention norm (new!)
        cur = build_norm(cur, attn_post_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "post_attn_norm", il);

        // Add the input (residual connection after post-attention norm)
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // FF
        {
            // Pre-MLP norm
            cur = build_norm(ffn_inp, ffn_norm, NULL, LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            // MLP
            cur = build_ffn(cur,
                ffn_up, NULL, NULL,
                    NULL, NULL, NULL,
                ffn_down, NULL, NULL,
                    NULL, LLM_FFN_SWIGLU, LLM_FFN_SEQ, il);
            cb(cur, "ffn_out", il);

            // Post-MLP norm
            cur = build_norm(cur, ffn_post_norm, NULL, LLM_NORM_RMS, il);
            cb(cur, "post_mlp_norm", il);
        }
        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }
    // Final norm
    cur = build_norm(inpL, model.output_norm, NULL, LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // Output projection
    cur = build_lora_mm(model.output, cur);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
