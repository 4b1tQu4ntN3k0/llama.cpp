#include "models.h"

llm_build_qwen3_layer::llm_build_qwen3_layer(const llama_model & model, const llm_graph_params & params, int layer_id) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v;

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
    GGML_ASSERT(n_embd_head == hparams.n_rot);

    ggml_tensor * cur;
	ggml_tensor * inpL_base;

    inpL_base = build_inp_embd(model.tok_embd);
	// inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * inp_out_ids = build_inp_out_ids();
    
	ggml_tensor * inpL = ggml_dup_tensor(ctx0, inpL_base);
	ggml_tensor * inpSA = inpL;
	ggml_set_name(inpSA, "layer_input");
	
	const llama_layer& layer = model.layers[layer_id];

	// norm
	cur = build_norm(inpL,
			model.name_weight_map.at(std::string(layer.attn_norm->name)), NULL,
			LLM_NORM_RMS, 0);
	cb(cur, "attn_norm", 0);

	// self-attention
	{
		// compute Q and K and RoPE them
		ggml_tensor * Qcur = build_lora_mm(model.name_weight_map.at(std::string(layer.wq->name)), cur);
		cb(Qcur, "Qcur", 0);

		ggml_tensor * Kcur = build_lora_mm(model.name_weight_map.at(std::string(layer.wk->name)), cur);
		cb(Kcur, "Kcur", 0);

		ggml_tensor * Vcur = build_lora_mm(model.name_weight_map.at(std::string(layer.wv->name)), cur);
		cb(Vcur, "Vcur", 0);

		Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
		Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
		Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

		Qcur = build_norm(Qcur, model.name_weight_map.at(std::string(layer.attn_q_norm->name)), NULL, LLM_NORM_RMS, 0);
		cb(Qcur, "Qcur_normed", 0);

		Qcur = ggml_rope_ext(
				ctx0, Qcur, inp_pos, nullptr,
				n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
				ext_factor, attn_factor, beta_fast, beta_slow
				);

		Kcur = build_norm(Kcur, model.name_weight_map.at(std::string(layer.attn_k_norm->name)), NULL, LLM_NORM_RMS, 0);
		cb(Kcur, "Kcur_normed", 0);

		Kcur = ggml_rope_ext(
				ctx0, Kcur, inp_pos, nullptr,
				n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
				ext_factor, attn_factor, beta_fast, beta_slow
				);

		cb(Qcur, "Qcur", 0);
		cb(Kcur, "Kcur", 0);
		cb(Vcur, "Vcur", 0);

		cur = build_attn(inp_attn,
				model.name_weight_map.at(std::string(layer.wo->name)), nullptr,
				Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), 0);
	}
	if (0 == n_layer - 1 && inp_out_ids) {
			cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
			inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
	}
	ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
	cb(ffn_inp, "ffn_inp", 0);

	// feed-forward network
	cur = build_norm(ffn_inp,
			model.name_weight_map.at(std::string(layer.ffn_norm->name)), NULL,
			LLM_NORM_RMS, 0);
	cb(cur, "ffn_norm", 0);

	cur = build_ffn(cur,
			model.name_weight_map.at(std::string(layer.ffn_up->name)),   NULL, NULL,
			model.name_weight_map.at(std::string(layer.ffn_gate->name)), NULL, NULL,
			model.name_weight_map.at(std::string(layer.ffn_down->name)), NULL, NULL,
			NULL,
			LLM_FFN_SILU, LLM_FFN_PAR, 0);
	cb(cur, "ffn_out", 0);

	cur = ggml_add(ctx0, cur, ffn_inp);

	cur = build_cvec(cur, 0);
	cb(cur, "l_out", 0);
    
    // cur = inpL;

    // cur = build_norm(cur,
    //         model.output_norm, NULL,
    //         LLM_NORM_RMS, -1);

    // cb(cur, "result_norm", -1);
    // res->t_embd = cur;

    // // lm_head
    // cur = build_lora_mm(model.output, cur);

    // cb(cur, "result_output", -1);
    // res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
