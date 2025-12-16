#include <stddef.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "model.h"
#include "model_defs.h"
#include "engine.h"
#include "vision.h"
#include "tokenize.h"
#include "threadpool.h"
#include "tools.h"
#include "math_dispatch.h"
#include "main.h"

int transformer_layer_granite(struct TIEContext *ctx, int layer_idx, int batch_len);
int build_prompt_granite(struct TIEContext *ctx, int *prompt_tokens, int *user_text_tokens, int user_text_token_count,
			 bool has_image);

static const TensorDef GRANITE_GLOBAL_TENSORS[] = {
	DECLARE_GLOBAL_TENSORS_BASE_DEFS,
};

static const TensorDef GRANITE_LAYER_TENSORS[] = {
	{"blk.%u.attn_k.weight", offsetof(LayerWeights, attn_k), FLAG_NONE},
	{"blk.%u.attn_norm.weight", offsetof(LayerWeights, attn_norm), FLAG_NONE},
	{"blk.%u.attn_q.weight", offsetof(LayerWeights, attn_q), FLAG_NONE},
	{"blk.%u.attn_v.weight", offsetof(LayerWeights, attn_v), FLAG_NONE},
	{"blk.%u.ffn_norm.weight", offsetof(LayerWeights, ffn_norm), FLAG_NONE},
	{"blk.%u.ffn_gate.weight", offsetof(LayerWeights, ffn_gate), FLAG_NONE},
	{"blk.%u.ffn_up.weight", offsetof(LayerWeights, ffn_up), FLAG_NONE},
	{"blk.%u.ffn_down.weight", offsetof(LayerWeights, ffn_down), FLAG_NONE},
	{"blk.%u.attn_output.weight", offsetof(LayerWeights, attn_out), FLAG_OPTIONAL},
};

static const MetadataDef GRANITE_METADATA_DEFS[] = {
	{"%s.context_length", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, seq_length), false, false},
	{"%s.embedding_length", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, embed_dim), false, false},
	{"%s.block_count", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, num_layers), false, false},
	{"%s.feed_forward_length", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, ffn_dim), false, false},
	{"%s.attention.head_count", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, num_heads), false, false},
	{"%s.attention.layer_norm_rms_epsilon", GGUF_METADATA_VALUE_TYPE_FLOAT32, offsetof(Model, norm_eps), false,
	 false},
	{"%s.rope.freq_base", GGUF_METADATA_VALUE_TYPE_FLOAT32, offsetof(Model, rope_freq_base), false, false}, // (MoE)
	{"%s.expert_count", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, expert_count), false, false},
	{"%s.expert_used_count", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, expert_used_count), false, false},
	{"%s.expert_shared_feed_forward_length", GGUF_METADATA_VALUE_TYPE_UINT32,
	 offsetof(Model, expert_shared_ffn_dim), false, false},
	{"%s.attention.scale", GGUF_METADATA_VALUE_TYPE_FLOAT32, offsetof(Model, attn_scale), false, false},
	{"%s.attention.head_count_kv", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, attn_head_count_kv), true,
	 false},
	{"%s.rope.dimension_count", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, rope_dimension_count), false,
	 false},
	{"%s.embedding_scale", GGUF_METADATA_VALUE_TYPE_FLOAT32, offsetof(Model, embedding_scale), false, false},
	{"%s.residual_scale", GGUF_METADATA_VALUE_TYPE_FLOAT32, offsetof(Model, residual_scale), false, false},
	{"%s.logit_scale", GGUF_METADATA_VALUE_TYPE_FLOAT32, offsetof(Model, logit_scale), false, false},
	{"%s.ssm.conv_kernel", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, ssm_conv_kernel), false, false},
	{"%s.ssm.state_size", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, ssm_state_size), false, false},
	{"%s.ssm.group_count", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, ssm_group_count), false, false},
	{"%s.ssm.inner_size", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, ssm_inner_size), false, false},
	{"%s.ssm.time_step_rank", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, ssm_time_step_rank), false, false},
	DECLARE_TOKENIZER_BASE_METADATA_DEFS,
};

static const BufferDef GRANITE_BUFFERS[] = {
	{offsetof(MemLayout, hidden_state), SIZE_EMBED_DIM, INTERNAL_MEMORY_TYPE, FLAG_NONE},
	{offsetof(MemLayout, normed_qkv_input), SIZE_EMBED_DIM, INTERNAL_MEMORY_TYPE, FLAG_NONE},
	{offsetof(MemLayout, Q), SIZE_Q_DIM, INTERNAL_MEMORY_TYPE, FLAG_NONE},
	{offsetof(MemLayout, K), SIZE_KV_DIM, INTERNAL_MEMORY_TYPE, FLAG_NONE},
	{offsetof(MemLayout, V), SIZE_KV_DIM, INTERNAL_MEMORY_TYPE, FLAG_NONE},
	{offsetof(MemLayout, attn_output), SIZE_EMBED_DIM, INTERNAL_MEMORY_TYPE, FLAG_NONE},
	{offsetof(MemLayout, attn_proj_output), SIZE_EMBED_DIM, INTERNAL_MEMORY_TYPE, FLAG_NONE},
	{offsetof(MemLayout, normed_ffn_input), SIZE_EMBED_DIM, INTERNAL_MEMORY_TYPE, FLAG_NONE},
	{offsetof(MemLayout, gate_proj_output), SIZE_FFN_DIM, INTERNAL_MEMORY_TYPE, FLAG_NONE},
	{offsetof(MemLayout, up_proj_output), SIZE_FFN_DIM, INTERNAL_MEMORY_TYPE, FLAG_NONE},
	{offsetof(MemLayout, ffn_down_output), SIZE_EMBED_DIM, INTERNAL_MEMORY_TYPE, FLAG_NONE},
	{offsetof(MemLayout, logits), SIZE_VOCAB_SIZE, GGML_TYPE_F32, FLAG_NONE},
};

static const TokenizeDef GRANITE_TOKENIZE_DEF = {
	.token_detect_specials = 1,
	.token_load_merges = 1,
	.token_load_scores = 0,
};

ModelDef GRANITE_DEF = {.arch = ARCH_GRANITE,
			.name = "Granite",
			.params =
				{
					.is_moe = 0,
					.is_hybrid = 0,
					.sot_token_id = 100264,
					.eot_token_id = 100265,
					.eos_token_id = 100257,
					.newline_token_id = 198,
					.role_user_token_id = 882,
					.role_model_token_id = 78191,
					.vision_start_token_id = -1,
					.vision_end_token_id = -1,
					.vision_embed_token_id = -1,

				},
			.interface =
				{
					.tokenize_encode = tokenize_bpe,
					.tokenize_decode = decode_token_bpe,
					.build_prompt = build_prompt_granite,

					.embedding_scale = embedding_scale_granite,
					.transformer_layer = transformer_layer_granite,
					.build_rope_cache = build_rope_cache_global,
				},
			DECLARE_LANGUAGE_MODEL_DEF(GRANITE, GRANITE)};

/*
 * -------------------------------------------------------------------------
 * GRANITE SPECIFIC IMPLEMENTATION
 * -------------------------------------------------------------------------
 */

void embedding_scale_granite(struct TIEContext *ctx, MemType *hidden_state_slice)
{
	float scale = ctx->model->embedding_scale; // Loaded from metadata

	float *data = (float *)hidden_state_slice->data;
	for (int i = 0; i < ctx->model->embed_dim; i++) {
		data[i] *= scale;
	}
}

int transformer_layer_granite(struct TIEContext *ctx, int layer_idx, int batch_len)
{
	LayerWeights *l = &ctx->model->layers[layer_idx];
	//	int layer_type = ctx->model->layer_types[layer_idx];
	float res_scale = ctx->model->residual_scale;
	int sink_len = 4;
	//	float tolerance = 1e-5;


	// Input Layernorm
	for (int i = 0; i < batch_len; i++) {
		size_t offset = (size_t)i * ctx->model->embed_dim;
		MemType hidden_slice = mem_slice(&ctx->mem.hidden_state, offset);
		MemType normed_slice = mem_slice(&ctx->mem.normed_qkv_input, offset);

		// RMSNorm
		dispatch_rms_norm(&hidden_slice, &l->attn_norm, &normed_slice, ctx->model->embed_dim,
				  ctx->model->norm_eps);
	}

	// ATTENTION
	int kv_dim = ctx->model->num_kv_heads * ctx->model->head_dim;
	int q_dim = ctx->model->num_heads * ctx->model->head_dim;
	int start_pos = ctx->kv_pos;

	// Q/K/V Projections
	dispatch_mat_mat(ctx, &ctx->mem.normed_qkv_input, &l->attn_q, &ctx->mem.Q, batch_len, ctx->model->embed_dim,
			 q_dim, true);
	dispatch_mat_mat(ctx, &ctx->mem.normed_qkv_input, &l->attn_k, &ctx->mem.K, batch_len, ctx->model->embed_dim,
			 kv_dim, true);
	dispatch_mat_mat(ctx, &ctx->mem.normed_qkv_input, &l->attn_v, &ctx->mem.V, batch_len, ctx->model->embed_dim,
			 kv_dim, true);

	// RoPE
	RopeCacheType *rope = ctx->model->rope_cache_global;

	for (int i = 0; i < batch_len; i++) {
		int absolute_pos = start_pos + i;

		// Apply to Q
		for (int h = 0; h < ctx->model->num_heads; h++) {
			MemType q_slice = mem_slice(&ctx->mem.Q, (size_t)i * q_dim + h * ctx->model->head_dim);
			dispatch_apply_rope_cache_interleaved(rope, &q_slice, absolute_pos, ctx->model->head_dim);
		}

		// Apply to K
		for (int h = 0; h < ctx->model->num_kv_heads; h++) {
			MemType k_slice = mem_slice(&ctx->mem.K, (size_t)i * kv_dim + h * ctx->model->head_dim);
			dispatch_apply_rope_cache_interleaved(rope, &k_slice, absolute_pos, ctx->model->head_dim);
		}
	}

	// KV Cache Store
	dispatch_store_KV_cache(ctx, layer_idx, start_pos, batch_len, sink_len);

	// Attention
	attention(ctx, batch_len, layer_idx, layer_idx, start_pos, ATTN_TYPE_GLOBAL, attention_worker, sink_len);

	// Output Projection
	dispatch_mat_mat(ctx, &ctx->mem.attn_output, &l->attn_out, &ctx->mem.attn_proj_output, batch_len, q_dim,
			 ctx->model->embed_dim, true);

	// RESIDUAL 1: hidden = hidden + attn_out * res_scale
	// Granite applies scale to the residual branch
	dispatch_apply_residual_scaled(&ctx->mem.hidden_state, &ctx->mem.attn_proj_output,
				       batch_len * ctx->model->embed_dim, res_scale);

	// POST-NORM
	for (int i = 0; i < batch_len; i++) {
		size_t offset = (size_t)i * ctx->model->embed_dim;
		MemType hidden_slice = mem_slice(&ctx->mem.hidden_state, offset);
		MemType normed_slice = mem_slice(&ctx->mem.normed_ffn_input, offset); // Reuse buffer
		dispatch_rms_norm(&hidden_slice, &l->ffn_norm, &normed_slice, ctx->model->embed_dim,
				  ctx->model->norm_eps);
	}

	// Gate + Up projections
	dispatch_mat_mat(ctx, &ctx->mem.normed_ffn_input, &l->ffn_gate, &ctx->mem.gate_proj_output, batch_len,
			 ctx->model->embed_dim, ctx->model->ffn_dim, true);
	dispatch_mat_mat(ctx, &ctx->mem.normed_ffn_input, &l->ffn_up, &ctx->mem.up_proj_output, batch_len,
			 ctx->model->embed_dim, ctx->model->ffn_dim, true);

	/* Call the interface activation function */
	dispatch_swiglu_activation(&ctx->mem.gate_proj_output, &ctx->mem.up_proj_output,
				   batch_len * ctx->model->ffn_dim);

	// Down projection
	dispatch_mat_mat(ctx, &ctx->mem.gate_proj_output, &l->ffn_down, &ctx->mem.ffn_down_output, batch_len,
			 ctx->model->ffn_dim, ctx->model->embed_dim, true);

	// Second Residual Add
	dispatch_apply_residual_scaled(&ctx->mem.hidden_state, &ctx->mem.ffn_down_output,
				       batch_len * ctx->model->embed_dim, res_scale);
	return 0;
}

int build_prompt_granite(struct TIEContext *ctx, int *prompt_tokens, int *user_text_tokens, int user_text_token_count,
			 bool has_image)
{
	ModelDef *def = ctx->model->def;
	int prompt_len = 0;

	// Look up Special Tokens
	int start_role = get_special_token_id(ctx, "<|start_of_role|>", 100264);
	int end_role = get_special_token_id(ctx, "<|end_of_role|>", 100265);
	int eos = get_special_token_id(ctx, "<|end_of_text|>", 100257);
	int newline = (def->params.newline_token_id != -1) ? def->params.newline_token_id : 198;

	// User Turn (Standard)
	// <|start_of_role|>user<|end_of_role|>
	prompt_tokens[prompt_len++] = start_role;
	prompt_tokens[prompt_len++] = def->params.role_user_token_id;
	prompt_tokens[prompt_len++] = end_role;

	// User Content
	memcpy(&prompt_tokens[prompt_len], user_text_tokens, user_text_token_count * sizeof(int));
	prompt_len += user_text_token_count;

	// End User Turn: <|end_of_text|>\n
	prompt_tokens[prompt_len++] = eos;
	prompt_tokens[prompt_len++] = newline;

	// Assistant Header
	// <|start_of_role|>assistant<|end_of_role|>
	prompt_tokens[prompt_len++] = start_role;
	prompt_tokens[prompt_len++] = def->params.role_model_token_id;
	prompt_tokens[prompt_len++] = end_role;

	return prompt_len;
}
