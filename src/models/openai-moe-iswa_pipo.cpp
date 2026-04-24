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

llm_build_openai_moe_iswa_pipo::llm_build_openai_moe_iswa_pipo(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv_iswa();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    ggml_tensor * attn_norm;
    ggml_tensor * wq;
    ggml_tensor * bq;
    ggml_tensor * wk;
    ggml_tensor * bk;
    ggml_tensor * wv;
    ggml_tensor * bv;
    ggml_tensor * wo;
    ggml_tensor * bo;
    ggml_tensor * attn_sinks;
    ggml_tensor * attn_post_norm;
    ggml_tensor * ffn_gate_inp;
    ggml_tensor * ffn_gate_inp_b;
    ggml_tensor * ffn_up_exps;
    ggml_tensor * ffn_up_exps_b;
    ggml_tensor * ffn_gate_exps;
    ggml_tensor * ffn_gate_exps_b;
    ggml_tensor * ffn_down_exps;
    ggml_tensor * ffn_down_exps_b;

    std::vector<std::regex> regex;
    if (params.gtype == LLM_GRAPH_TYPE_DEFAULT_PREFILL) {
        init_regex(regex, model.p_offload_weights);
    } else {
        GGML_ASSERT(params.gtype == LLM_GRAPH_TYPE_DEFAULT_DECODE);
        init_regex(regex, model.d_offload_weights);
    }

    for (int il = 0; il < n_layer; ++il) {
        const float freq_base_l  = model.get_rope_freq_base(cparams, il);
        const float freq_scale_l = model.get_rope_freq_scale(cparams, il);

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
        attn_norm       = get_offloaded(layer.attn_norm);
        wq              = get_offloaded(layer.wq);
        bq              = get_offloaded(layer.bq);
        wk              = get_offloaded(layer.wk);
        bk              = get_offloaded(layer.bk);
        wv              = get_offloaded(layer.wv);
        bv              = get_offloaded(layer.bv);
        wo              = get_offloaded(layer.wo);
        bo              = get_offloaded(layer.bo);
        attn_sinks      = get_offloaded(layer.attn_sinks);
        attn_post_norm  = get_offloaded(layer.attn_post_norm);
        ffn_gate_inp    = get_offloaded(layer.ffn_gate_inp);
        ffn_gate_inp_b  = get_offloaded(layer.ffn_gate_inp_b);
        ffn_up_exps     = get_offloaded(layer.ffn_up_exps);
        ffn_up_exps_b   = get_offloaded(layer.ffn_up_exps_b);
        ffn_gate_exps   = get_offloaded(layer.ffn_gate_exps);
        ffn_gate_exps_b = get_offloaded(layer.ffn_gate_exps_b);
        ffn_down_exps   = get_offloaded(layer.ffn_down_exps);
        ffn_down_exps_b = get_offloaded(layer.ffn_down_exps_b);

        ggml_tensor * inpSA = inpL;

        cur = build_norm(inpL,
                attn_norm, nullptr,
                LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        {
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
            Qcur = ggml_reshape_3d(ctx0, Qcur, n_rot, n_head,    n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_rot, n_head_kv, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_rot, n_head_kv, n_tokens);

            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(inp_attn,
                    wo, bo,
                    Qcur, Kcur, Vcur, nullptr, attn_sinks, nullptr, 1.0f/sqrtf(float(n_rot)), il);

            cb(cur, "attn_out", il);
        }
        if (il == n_layer - 1) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        cur = ffn_inp;
        cur = build_norm(cur,
                attn_post_norm, nullptr,
                LLM_NORM_RMS, il);
        cb(cur, "attn_post_norm", il);

        cur = build_moe_ffn(cur,
                ffn_gate_inp,  ffn_gate_inp_b,
                ffn_up_exps,   ffn_up_exps_b,
                ffn_gate_exps, ffn_gate_exps_b,
                ffn_down_exps, ffn_down_exps_b,
                nullptr,
                n_expert, n_expert_used,
                LLM_FFN_SWIGLU_OAI_MOE, false,
                false, 0.0,
                LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT,
                il);
        cb(cur, "ffn_moe_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }
    cur = inpL;

    cur = build_norm(cur,
            model.output_norm, NULL,
            LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
