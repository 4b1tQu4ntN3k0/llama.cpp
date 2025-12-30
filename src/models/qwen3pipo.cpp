#include "models.h"

llm_build_qwen3_pipo::llm_build_qwen3_pipo(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v;
    const int n_gpu_layers = 10;
    const int n_cpu_layers_per_split = 3;

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
    GGML_ASSERT(n_embd_head == hparams.n_rot);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    ggml_tensor * attn_norm;
    ggml_tensor * wq;
    ggml_tensor * wk;
    ggml_tensor * wv;
    ggml_tensor * attn_q_norm;
    ggml_tensor * attn_k_norm;
    ggml_tensor * wo;
    ggml_tensor * bo;
    ggml_tensor * ffn_norm;
    ggml_tensor * ffn_up;
    ggml_tensor * ffn_gate;
    ggml_tensor * ffn_down;

    for (int il = 0; il < n_layer; ++il) {
        bool is_dynamic_layer = il>n_gpu_layers && (il-n_gpu_layers+1)%(n_cpu_layers_per_split+1)==0;
        if(is_dynamic_layer){
            const llama_layer& layer = model.layers[il];
            attn_norm = model.name_weight_map.at(std::string(layer.attn_norm->name));
            wq = model.name_weight_map.at(std::string(layer.wq->name));
            wk = model.name_weight_map.at(std::string(layer.wk->name));
            wv = model.name_weight_map.at(std::string(layer.wv->name));
            attn_q_norm = model.name_weight_map.at(std::string(layer.attn_q_norm->name));
            attn_k_norm = model.name_weight_map.at(std::string(layer.attn_k_norm->name));
            wo = model.name_weight_map.at(std::string(layer.wo->name));
            bo = nullptr;
            ffn_norm = model.name_weight_map.at(std::string(layer.ffn_norm->name));
            ffn_up = model.name_weight_map.at(std::string(layer.ffn_up->name));
            ffn_gate = model.name_weight_map.at(std::string(layer.ffn_gate->name));
            ffn_down = model.name_weight_map.at(std::string(layer.ffn_down->name));
            res->src_tensors.push_back({
                model.layers[il].attn_norm,
                model.layers[il].wq,
                model.layers[il].wk,
                model.layers[il].wv,
                model.layers[il].attn_q_norm,
                model.layers[il].attn_k_norm,
                model.layers[il].wo,
                model.layers[il].ffn_norm,
                model.layers[il].ffn_up,
                model.layers[il].ffn_gate,
                model.layers[il].ffn_down,
                get_kv_tensor(inp_attn, il).first,
                get_kv_tensor(inp_attn, il).second,
            });
            res->dst_tensors.push_back({
                attn_norm,
                wq,
                wk,
                wv,
                attn_q_norm,
                attn_k_norm,
                wo,
                ffn_norm,
                ffn_up,
                ffn_gate,
                ffn_down,
                get_kv_tensor(inp_attn, -1).first,
                get_kv_tensor(inp_attn, -1).second,
            });
        }
        else{
            attn_norm = model.layers[il].attn_norm;
            wq = model.layers[il].wq;
            wk = model.layers[il].wk;
            wv = model.layers[il].wv;
            attn_q_norm = model.layers[il].attn_q_norm;
            attn_k_norm = model.layers[il].attn_k_norm;
            wo = model.layers[il].wo;
            bo = model.layers[il].bo;
            ffn_norm = model.layers[il].ffn_norm;
            ffn_up = model.layers[il].ffn_up;
            ffn_gate = model.layers[il].ffn_gate;
            ffn_down = model.layers[il].ffn_down;
        }
        ggml_tensor * inpSA = inpL;

        // norm
        cur = build_norm(inpL,
                attn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            // compute Q and K and RoPE them
            ggml_tensor * Qcur = build_lora_mm(wq, cur);
            cb(Qcur, "Qcur", il);

            ggml_tensor * Kcur = build_lora_mm(wk, cur);
            cb(Kcur, "Kcur", il);

            ggml_tensor * Vcur = build_lora_mm(wv, cur);
            cb(Vcur, "Vcur", il);

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

            Qcur = build_norm(Qcur, attn_q_norm, NULL, LLM_NORM_RMS, il);
            cb(Qcur, "Qcur_normed", il);

            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            Kcur = build_norm(Kcur, attn_k_norm, NULL, LLM_NORM_RMS, il);
            cb(Kcur, "Kcur_normed", il);

            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(inp_attn,
                    wo, bo,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
        }
        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // feed-forward network
        cur = build_norm(ffn_inp,
                ffn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur,
                ffn_up,   NULL, NULL,
                ffn_gate, NULL, NULL,
                ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);

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

    // lm_head
    cur = build_lora_mm(model.output, cur);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
