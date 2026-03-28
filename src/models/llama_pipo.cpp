#include "models.h"
#include <regex>

static bool need_offload(const std::vector<std::regex>& regex, std::string name){
    for(const auto & pattern: regex){
        if (std::regex_search(name, pattern)){
            return true;
        }
    }
    return false;
}

static void init_regex(std::vector<std::regex>& regex, const std::vector<const char*>& patterns){
    regex.clear();
    for(const auto &p:patterns){
        if (p) {
            regex.emplace_back(p);
        }
    }
}

template <bool embed>
llm_build_llama_pipo<embed>::llm_build_llama_pipo(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v;

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
    GGML_ASSERT(n_embd_head == hparams.n_rot);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    using inp_attn_type = std::conditional_t<embed, llm_graph_input_attn_no_cache, llm_graph_input_attn_kv>;

    inp_attn_type * inp_attn = nullptr;
    if constexpr (embed) {
        inp_attn = build_attn_inp_no_cache();
    } else {
        inp_attn = build_attn_inp_kv();
    }

    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f/sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    std::vector<std::regex> regex;
    if(params.gtype == LLM_GRAPH_TYPE_DEFAULT_PREFILL){
        init_regex(regex, model.p_offload_weights);
    }
    else{
        GGML_ASSERT(params.gtype == LLM_GRAPH_TYPE_DEFAULT_DECODE);
        init_regex(regex, model.d_offload_weights);
    }
    ggml_tensor* attn_norm;
    ggml_tensor* wq;
    ggml_tensor* bq;
    ggml_tensor* wk;
    ggml_tensor* bk;
    ggml_tensor* wv; 
    ggml_tensor* bv;
    ggml_tensor* wo;
    ggml_tensor* bo;
    ggml_tensor* ffn_norm;
    ggml_tensor* ffn_up;
    ggml_tensor* ffn_gate;
    ggml_tensor* ffn_down;
    ggml_tensor* ffn_up_b;
    ggml_tensor* ffn_gate_b;
    ggml_tensor* ffn_down_b;
    ggml_tensor* ffn_gate_inp;
    ggml_tensor* ffn_up_exps;
    ggml_tensor* ffn_gate_exps;
    ggml_tensor* ffn_down_exps;

   
    for (int il = 0; il < n_layer; ++il) {
        auto get_offloaded = [&](ggml_tensor * t) {
            if (t && need_offload(regex, std::string(t->name))) {
                struct ggml_tensor * dynamic_tensor = ggml_backend_sched_get_pipo_tensor(params.sched, model.name_weight_map.at(std::string(t->name)));
                res->dynamic_src_tensor_list[dynamic_tensor->name].push_back(t);
                res->dynamic_dst_tensor_list[dynamic_tensor->name].push_back(dynamic_tensor);
                return dynamic_tensor;
            }
            return t;
        };

        const llama_layer& layer = model.layers[il];
#define ASSIGN_PIPO_TENSOR(name) name = get_offloaded(layer.name);
        ASSIGN_PIPO_TENSOR(attn_norm);
        ASSIGN_PIPO_TENSOR(wq);
        ASSIGN_PIPO_TENSOR(bq);
        ASSIGN_PIPO_TENSOR(wk);
        ASSIGN_PIPO_TENSOR(bk);
        ASSIGN_PIPO_TENSOR(wv); 
        ASSIGN_PIPO_TENSOR(bv);
        ASSIGN_PIPO_TENSOR(wo);
        ASSIGN_PIPO_TENSOR(bo);
        ASSIGN_PIPO_TENSOR(ffn_norm);
        ASSIGN_PIPO_TENSOR(ffn_up);
        ASSIGN_PIPO_TENSOR(ffn_gate);
        ASSIGN_PIPO_TENSOR(ffn_down);
        ASSIGN_PIPO_TENSOR(ffn_up_b);
        ASSIGN_PIPO_TENSOR(ffn_gate_b);
        ASSIGN_PIPO_TENSOR(ffn_down_b);
        ASSIGN_PIPO_TENSOR(ffn_gate_inp);
        ASSIGN_PIPO_TENSOR(ffn_up_exps);
        ASSIGN_PIPO_TENSOR(ffn_gate_exps);
        ASSIGN_PIPO_TENSOR(ffn_down_exps);
#undef ASSIGN_PIPO_TENSOR

 
        ggml_tensor * inpSA = inpL;

        // norm
        cur = build_norm(inpL,
                attn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            // rope freq factors for llama3; may return nullptr for llama2 and other models
            ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);

            // compute Q and K and RoPE them
            ggml_tensor * Qcur = build_lora_mm(wq, cur);
            cb(Qcur, "Qcur", il);
            if (bq) {
                Qcur = ggml_add(ctx0, Qcur, bq);
                cb(Qcur, "Qcur", il);
            }
            ggml_tensor * Kcur = build_lora_mm(wk, cur);
            cb(Kcur, "Kcur", il);
            if (bk) {
                Kcur = ggml_add(ctx0, Kcur, bk);
                cb(Kcur, "Kcur", il);
            }
            ggml_tensor * Vcur = build_lora_mm(wv, cur);
            cb(Vcur, "Vcur", il);
            if (bv) {
                Vcur = ggml_add(ctx0, Vcur, bv);
                cb(Vcur, "Vcur", il);
            }
            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            if (hparams.use_kq_norm) {
                // Llama4TextL2Norm
                Qcur = ggml_rms_norm(ctx0, Qcur, hparams.f_norm_rms_eps);
                Kcur = ggml_rms_norm(ctx0, Kcur, hparams.f_norm_rms_eps);
                cb(Qcur, "Qcur_normed", il);
                cb(Kcur, "Kcur_normed", il);
            }
            cur = build_attn(inp_attn,
                    wo, bo,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
            cb(cur, "attn_out", il);
        }
        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // feed-forward network (non-MoE)
        if (ffn_gate_inp == nullptr) {

            cur = build_norm(ffn_inp,
                    ffn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            cur = build_ffn(cur,
                    ffn_up,   ffn_up_b,   NULL,
                    ffn_gate, ffn_gate_b, NULL,
                    ffn_down, ffn_down_b, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        } else {
            // MoE branch
            cur = build_norm(ffn_inp,
                    ffn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            cur = build_moe_ffn(cur,
                    ffn_gate_inp,
                    ffn_up_exps,
                    ffn_gate_exps,
                    ffn_down_exps,
                    nullptr,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, true,
                    false, 0.0,
                    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                    il);
            cb(cur, "ffn_moe_out", il);
        }
        cur = ggml_add(ctx0, cur, ffn_inp);
        cb(cur, "ffn_out", il);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }
    cur = inpL;

    cur = build_norm(cur,
            model.output_norm, NULL,
            LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    if constexpr (!embed) {
        // lm_head
        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res->t_logits = cur;
    }

    ggml_build_forward_expand(gf, cur);
}

template struct llm_build_llama_pipo<false>;
template struct llm_build_llama_pipo<true>;
