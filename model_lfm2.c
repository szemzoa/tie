#include <stddef.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "model.h"
#include "model_defs.h"
#include "engine.h"
#include "vision.h"
#include "tokenize.h"
#include "threadpool.h"
#include "tools.h"
#include "math_dispatch.h"
#include "main.h"

#ifdef CONFIG_ENABLE_AVX2
#include <immintrin.h>
#endif

int transformer_layer_lfm2(struct TIEContext *ctx, int layer_idx, int batch_len);
int build_prompt_lfm2(struct TIEContext *ctx, int *prompt_tokens, int *user_text_tokens, int user_text_token_count,
		      bool has_image);

static const TensorDef LFM2_GLOBAL_TENSORS[] = {
	{"token_embd.weight", offsetof(Model, token_embd), FLAG_NONE},
	{"token_embd_norm.weight", offsetof(Model, output_norm), FLAG_NONE},
};

static const TensorDef LFM2_LAYER_TENSORS[] = {
	{"blk.%u.attn_norm.weight", offsetof(LayerWeights, attn_norm), FLAG_NONE},
	{"blk.%u.ffn_down.weight", offsetof(LayerWeights, ffn_down), FLAG_NONE},
	{"blk.%u.ffn_gate.weight", offsetof(LayerWeights, ffn_gate), FLAG_NONE},
	{"blk.%u.ffn_norm.weight", offsetof(LayerWeights, ffn_norm), FLAG_NONE},
	{"blk.%u.ffn_up.weight", offsetof(LayerWeights, ffn_up), FLAG_NONE},
	{"blk.%u.attn_k.weight", offsetof(LayerWeights, attn_k), FLAG_OPTIONAL},
	{"blk.%u.attn_k_norm.weight", offsetof(LayerWeights, attn_k_norm), FLAG_OPTIONAL},
	{"blk.%u.attn_q.weight", offsetof(LayerWeights, attn_q), FLAG_OPTIONAL},
	{"blk.%u.attn_q_norm.weight", offsetof(LayerWeights, attn_q_norm), FLAG_OPTIONAL},
	{"blk.%u.attn_v.weight", offsetof(LayerWeights, attn_v), FLAG_OPTIONAL},
	{"blk.%u.attn_output.weight", offsetof(LayerWeights, attn_out), FLAG_OPTIONAL},
	{"blk.%u.shortconv.conv.weight", offsetof(LayerWeights, sconv_conv), FLAG_OPTIONAL},
	{"blk.%u.shortconv.conv.bias", offsetof(LayerWeights, sconv_bias), FLAG_OPTIONAL},
	{"blk.%u.shortconv.in_proj.weight", offsetof(LayerWeights, sconv_in_proj), FLAG_OPTIONAL},
	{"blk.%u.shortconv.out_proj.weight", offsetof(LayerWeights, sconv_out_proj), FLAG_OPTIONAL},
};

static const MetadataDef LFM2_METADATA_DEFS[] = {
	{"%s.context_length", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, seq_length), false, false},
	{"%s.embedding_length", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, embed_dim), false, false},
	{"%s.block_count", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, num_layers), false, false},
	{"%s.feed_forward_length", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, ffn_dim), false, false},
	{"%s.attention.head_count", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, num_heads), false, false},
	{"%s.attention.layer_norm_rms_epsilon", GGUF_METADATA_VALUE_TYPE_FLOAT32, offsetof(Model, norm_eps), false,
	 false},
	{"%s.rope.freq_base", GGUF_METADATA_VALUE_TYPE_FLOAT32, offsetof(Model, rope_freq_base), false, false}, // (MoE)
	{"%s.attention.head_count_kv", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, attn_head_count_kv), true,
	 false},
	{"%s.shortconv.l_cache", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, conv_kernel_size), false, false},
	/* MoE */
	{"%s.expert_count", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, expert_count), false, true},
	{"%s.expert_used_count", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, expert_used_count), false, true},
	{"%s.expert_feed_forward_length", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, expert_ffn_dim), false,
	 true},
	{"%s.leading_dense_block_count", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, expert_leading_dense_layers),
	 false, true},
	DECLARE_TOKENIZER_BASE_METADATA_DEFS,
};

static const BufferDef LFM2_BUFFERS[] = {
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

static const TokenizeDef LFM2_TOKENIZE_DEF = {
	.token_detect_specials = 1,
	.token_load_merges = 1,
	.token_load_scores = 0,
};

ModelDef LFM2_DEF = {
	.arch = ARCH_LFM2,
	.name = "LFM2",
	.params =
		{
			.is_moe = 0,
			.is_hybrid = 1,
			.sot_token_id = 1,
			.eot_token_id = 2,
			.eos_token_id = 7,
			.newline_token_id = 708,
			.role_user_token_id = 6423,
			.role_model_token_id = 64015,
			.vision_start_token_id = 498,
			.vision_end_token_id = 499,
			.vision_embed_token_id = 396,

		},
	.interface =
		{
			.tokenize_encode = tokenize_bpe,
			.tokenize_decode = decode_token_bpe,
			.build_prompt = build_prompt_lfm2,
			.transformer_layer = transformer_layer_lfm2,
			//				     .process_image_vision = process_image_vision_lfm2vl,
			.build_rope_cache = build_rope_cache_global,
		},
	DECLARE_LANGUAGE_MODEL_DEF(LFM2, LFM2)};


static const TensorDef LFM2_MOE_LAYER_TENSORS[] = {
	{"blk.%u.attn_norm.weight", offsetof(LayerWeights, attn_norm), FLAG_NONE},
	{"blk.%u.ffn_down.weight", offsetof(LayerWeights, ffn_down), FLAG_OPTIONAL},
	{"blk.%u.ffn_gate.weight", offsetof(LayerWeights, ffn_gate), FLAG_OPTIONAL},
	{"blk.%u.ffn_norm.weight", offsetof(LayerWeights, ffn_norm), FLAG_OPTIONAL},
	{"blk.%u.ffn_up.weight", offsetof(LayerWeights, ffn_up), FLAG_OPTIONAL},
	{"blk.%u.ffn_gate_inp.weight", offsetof(LayerWeights, ffn_gate_inp), FLAG_OPTIONAL},
	{"blk.%u.ffn_gate_exps.weight", offsetof(LayerWeights, ffn_gate_exps), FLAG_OPTIONAL},
	{"blk.%u.ffn_down_exps.weight", offsetof(LayerWeights, ffn_down_exps), FLAG_OPTIONAL},
	{"blk.%u.ffn_up_exps.weight", offsetof(LayerWeights, ffn_up_exps), FLAG_OPTIONAL},
	{"blk.%u.exp_probs_b.bias", offsetof(LayerWeights, exp_probs_b_bias), FLAG_OPTIONAL},
	{"blk.%u.attn_k.weight", offsetof(LayerWeights, attn_k), FLAG_OPTIONAL},
	{"blk.%u.attn_k_norm.weight", offsetof(LayerWeights, attn_k_norm), FLAG_OPTIONAL},
	{"blk.%u.attn_q.weight", offsetof(LayerWeights, attn_q), FLAG_OPTIONAL},
	{"blk.%u.attn_q_norm.weight", offsetof(LayerWeights, attn_q_norm), FLAG_OPTIONAL},
	{"blk.%u.attn_v.weight", offsetof(LayerWeights, attn_v), FLAG_OPTIONAL},
	{"blk.%u.attn_output.weight", offsetof(LayerWeights, attn_out), FLAG_OPTIONAL},
	{"blk.%u.shortconv.conv.weight", offsetof(LayerWeights, sconv_conv), FLAG_OPTIONAL},
	{"blk.%u.shortconv.conv.bias", offsetof(LayerWeights, sconv_bias), FLAG_OPTIONAL},
	{"blk.%u.shortconv.in_proj.weight", offsetof(LayerWeights, sconv_in_proj), FLAG_OPTIONAL},
	{"blk.%u.shortconv.out_proj.weight", offsetof(LayerWeights, sconv_out_proj), FLAG_OPTIONAL},
};

ModelDef LFM2_MOE_DEF = {.arch = ARCH_LFM2_MOE,
			 .name = "LFM2-MoE",
			 .params =
				 {
					 .is_moe = 1,
					 .is_hybrid = 1,
					 .sot_token_id = 1,
					 .eot_token_id = 2,
					 .eos_token_id = 7,
					 .newline_token_id = 708,
					 .role_user_token_id = 6423,
					 .role_model_token_id = 64015,
					 .vision_start_token_id = -1,
					 .vision_end_token_id = -1,
					 .vision_embed_token_id = 396,

				 },
			 .interface =
				 {
					 .tokenize_encode = tokenize_bpe,
					 .tokenize_decode = decode_token_bpe,
					 .build_prompt = build_prompt_lfm2,
					 .transformer_layer = transformer_layer_lfm2,
					 .build_rope_cache = build_rope_cache_global,
				 },
			 DECLARE_LANGUAGE_MODEL_DEF(LFM2, LFM2_MOE)};


/*
 * -------------------------------------------------------------------------
 * LFM2 SPECIFIC IMPLEMENTATION
 * -------------------------------------------------------------------------
 */
#ifdef CONFIG_ENABLE_AVX2
__attribute__((target("avx2"))) void lfm2_conv1d_step_avx2(float *y_out, const float *x_in, float *ring_buffer,
							   const float *weights, const float *bias, int embed_dim,
							   int kernel_size, int *pos_ptr)
{
	// LFM2 Fixed Kernel Size Optimization
	if (kernel_size != 3) {
		return;
	}

	int pos = *pos_ptr;

	// Calculate ring buffer offsets relative to base pointer (i*3)
	// p0: Current (t)   -> Corresponds to Weight index 2 (Newest)
	// p1: Prev (t-1)    -> Corresponds to Weight index 1
	// p2: Prev2 (t-2)   -> Corresponds to Weight index 0 (Oldest)
	int p1 = (pos + 2) % 3; // Equivalent to (pos - 1 + 3) % 3
	int p2 = (pos + 1) % 3; // Equivalent to (pos - 2 + 3) % 3

	// Prepare Gather Indices for strided access (Stride = 3 floats)
	// Base offsets: 0, 3, 6, 9, 12, 15, 18, 21
	__m256i v_base_idx = _mm256_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21);

	// Ring Indices
	__m256i v_idx1 = _mm256_add_epi32(v_base_idx, _mm256_set1_epi32(p1));
	__m256i v_idx2 = _mm256_add_epi32(v_base_idx, _mm256_set1_epi32(p2));

	// Weight Indices
	__m256i v_widx0 = _mm256_add_epi32(v_base_idx, _mm256_set1_epi32(2));
	__m256i v_widx1 = _mm256_add_epi32(v_base_idx, _mm256_set1_epi32(1));
	__m256i v_widx2 = _mm256_add_epi32(v_base_idx, _mm256_set1_epi32(0));

	int i = 0;
	// Process 8 channels at a time
	for (; i <= embed_dim - 8; i += 8) {
		// Load 8 Input values
		__m256 x_val = _mm256_loadu_ps(x_in + i);

		// Scalar Update Ring Buffer
		float *r_ptr = ring_buffer + i * 3 + pos;
		r_ptr[0 * 3] = x_in[i + 0];
		r_ptr[1 * 3] = x_in[i + 1];
		r_ptr[2 * 3] = x_in[i + 2];
		r_ptr[3 * 3] = x_in[i + 3];
		r_ptr[4 * 3] = x_in[i + 4];
		r_ptr[5 * 3] = x_in[i + 5];
		r_ptr[6 * 3] = x_in[i + 6];
		r_ptr[7 * 3] = x_in[i + 7];

		// Load Bias
		__m256 sum = (bias) ? _mm256_loadu_ps(bias + i) : _mm256_setzero_ps();

		// Gather Weights (Base pointer shifts by i*3)
		const float *w_base = weights + i * 3;
		__m256 w0 = _mm256_i32gather_ps(w_base, v_widx0, 4);
		__m256 w1 = _mm256_i32gather_ps(w_base, v_widx1, 4);
		__m256 w2 = _mm256_i32gather_ps(w_base, v_widx2, 4);

		// Gather History from Ring
		const float *r_base = ring_buffer + i * 3;
		// r0 is just x_val
		__m256 r1 = _mm256_i32gather_ps(r_base, v_idx1, 4);
		__m256 r2 = _mm256_i32gather_ps(r_base, v_idx2, 4);

		// Fused Multiply-Add
		sum = _mm256_fmadd_ps(x_val, w0, sum);
		sum = _mm256_fmadd_ps(r1, w1, sum);
		sum = _mm256_fmadd_ps(r2, w2, sum);

		_mm256_storeu_ps(y_out + i, sum);
	}
}
#endif

void lfm2_conv1d_step(float *y_out, const float *x_in, float *ring_buffer, const float *weights, const float *bias,
		      int embed_dim, int kernel_size, int *pos_ptr)
{
#ifdef CONFIG_ENABLE_AVX2
	// Dispatch to AVX2 if dimensions align
	if (embed_dim % 8 == 0 && kernel_size == 3) {
		lfm2_conv1d_step_avx2(y_out, x_in, ring_buffer, weights, bias, embed_dim, kernel_size, pos_ptr);
		return;
	}
#endif

	int pos = *pos_ptr;

	for (int i = 0; i < embed_dim; i++) {
		// Pointer to this channel's ring buffer history
		float *channel_ring = ring_buffer + i * kernel_size;

		// Update History (Overwrite oldest)
		channel_ring[pos] = x_in[i];

		// Convolution Dot Product
		const float *w = weights + i * kernel_size;
		float sum = (bias) ? bias[i] : 0.0f;

		for (int k = 0; k < kernel_size; k++) {
			// Read backwards from current pos
			int ring_idx = (pos - k + kernel_size) % kernel_size;

			// Match with weight (Newest data * Last weight)
			int w_idx = (kernel_size - 1) - k;
			sum += channel_ring[ring_idx] * w[w_idx];
		}

		y_out[i] = sum;
	}
}

void dispatch_lfm2_shortconv(struct TIEContext *ctx, int layer_idx, MemType *normed_input, MemType *output,
			     int batch_len)
{
	LayerWeights *l = &ctx->model->layers[layer_idx];
	LayerConvState *state = &ctx->mem.lfm2_conv_states[layer_idx];
	int embed_dim = ctx->model->embed_dim;
	int kernel_size = ctx->model->conv_kernel_size;

	// Input Projection
	dispatch_mat_mat(ctx, normed_input, &l->sconv_in_proj, &ctx->mem.sconv_in_proj_output, batch_len, embed_dim,
			 3 * embed_dim, true);

	float *in_ptr = (float *)ctx->mem.sconv_in_proj_output.data;
	float *out_ptr = (float *)ctx->mem.sconv_conv_output.data;

	// Process Tokens
	for (int t = 0; t < batch_len; t++) {
		float *token_base = in_ptr + t * (3 * embed_dim);
		float *dest = out_ptr + t * embed_dim;

		float *B = token_base;
		float *C = token_base + embed_dim;
		float *X = token_base + 2 * embed_dim;

		// B * X
		for (int i = 0; i < embed_dim; i++) {
			X[i] = B[i] * X[i];
		}

		// Conv
		lfm2_conv1d_step(dest, X, state->buffer, (float *)l->sconv_conv.mem.data,
				 (l->sconv_bias.mem.data) ? (float *)l->sconv_bias.mem.data : NULL, embed_dim,
				 kernel_size, &state->pos);

		state->pos = (state->pos + 1) % kernel_size;

		// Gate
		for (int i = 0; i < embed_dim; i++) {
			dest[i] = C[i] * dest[i];
		}
	}

	// Output Projection
	dispatch_mat_mat(ctx, &ctx->mem.sconv_conv_output, &l->sconv_out_proj, output, batch_len, embed_dim, embed_dim,
			 true);
}

int transformer_layer_lfm2(struct TIEContext *ctx, int layer_idx, int batch_len)
{
	LayerWeights *l = &ctx->model->layers[layer_idx];
	int layer_type = ctx->model->layer_types[layer_idx];
	AttentionType attn_type = ATTN_TYPE_GLOBAL;
	int sink_len = 4;


	// RMSNorm
	for (int i = 0; i < batch_len; i++) {
		size_t offset = i * ctx->model->embed_dim;
		MemType hidden = mem_slice(&ctx->mem.hidden_state, offset);
		MemType normed = mem_slice(&ctx->mem.normed_qkv_input, offset);
		dispatch_rms_norm(&hidden, &l->attn_norm, &normed, ctx->model->embed_dim, ctx->model->norm_eps);
	}

	// MIXER BLOCK (Attention or ShortConv)
	if (layer_type == LAYER_TYPE_ATTENTION) {

		// (Standard Attention Logic)
		int kv_dim = ctx->model->num_kv_heads * ctx->model->head_dim;
		int q_dim = ctx->model->num_heads * ctx->model->head_dim;
		int start_pos = ctx->kv_pos;

		dispatch_mat_mat(ctx, &ctx->mem.normed_qkv_input, &l->attn_q, &ctx->mem.Q, batch_len,
				 ctx->model->embed_dim, q_dim, true);
		dispatch_mat_mat(ctx, &ctx->mem.normed_qkv_input, &l->attn_k, &ctx->mem.K, batch_len,
				 ctx->model->embed_dim, kv_dim, true);
		dispatch_mat_mat(ctx, &ctx->mem.normed_qkv_input, &l->attn_v, &ctx->mem.V, batch_len,
				 ctx->model->embed_dim, kv_dim, true);

		// RoPE
		for (int i = 0; i < batch_len; i++) {
			int absolute_pos = start_pos + i;
			for (int h = 0; h < ctx->model->num_heads; h++) {
				MemType Q_slice = mem_slice(&ctx->mem.Q, (size_t)i * q_dim + h * ctx->model->head_dim);
				if (l->attn_q_norm.mem.data)
					dispatch_rms_norm(&Q_slice, &l->attn_q_norm, &Q_slice, ctx->model->head_dim,
							  ctx->model->norm_eps);
				dispatch_apply_rope_cache(ctx->model->rope_cache_global, &Q_slice, absolute_pos,
							  ctx->model->head_dim);
			}
			for (int h = 0; h < ctx->model->num_kv_heads; h++) {
				MemType K_slice = mem_slice(&ctx->mem.K, (size_t)i * kv_dim + h * ctx->model->head_dim);
				if (l->attn_k_norm.mem.data)
					dispatch_rms_norm(&K_slice, &l->attn_k_norm, &K_slice, ctx->model->head_dim,
							  ctx->model->norm_eps);
				dispatch_apply_rope_cache(ctx->model->rope_cache_global, &K_slice, absolute_pos,
							  ctx->model->head_dim);
			}
		}

		dispatch_store_KV_cache(ctx, layer_idx, start_pos, batch_len, sink_len);

		attention(ctx, batch_len, layer_idx, layer_idx, start_pos, attn_type, attention_worker, sink_len);

		dispatch_mat_mat(ctx, &ctx->mem.attn_output, &l->attn_out, &ctx->mem.attn_proj_output, batch_len, q_dim,
				 ctx->model->embed_dim, true);

	} else {
		// ShortConv
		dispatch_lfm2_shortconv(ctx, layer_idx, &ctx->mem.normed_qkv_input, &ctx->mem.attn_proj_output,
					batch_len);
	}

	// Residual Add 1
	dispatch_apply_residual(&ctx->mem.hidden_state, &ctx->mem.attn_proj_output, batch_len * ctx->model->embed_dim);

	// FFN BLOCK (Hybrid Dense/MoE)
	// RMSNorm
	for (int i = 0; i < batch_len; i++) {
		size_t offset = i * ctx->model->embed_dim;
		MemType hidden = mem_slice(&ctx->mem.hidden_state, offset);
		MemType normed = mem_slice(&ctx->mem.normed_ffn_input, offset);
		dispatch_rms_norm(&hidden, &l->ffn_norm, &normed, ctx->model->embed_dim, ctx->model->norm_eps);
	}

	// Only use MoE if layer_idx >= leading_dense_block_count
	bool use_moe_ffn = ctx->model->is_moe && (layer_idx >= ctx->model->expert_leading_dense_layers);

	if (use_moe_ffn) {

		// MoE FFN
		size_t block_size_bytes = ggml_block_size(l->ffn_up_exps.mem.type);
		if (block_size_bytes == 0)
			return -1;

		int num_threads = thread_pool->num_threads;
		expert_task_t tasks[num_threads];

		for (int i = 0; i < batch_len; i++) {
			MemType normed_input_for_token_i =
				mem_slice(&ctx->mem.normed_ffn_input, (size_t)i * ctx->model->embed_dim);
			MemType ffn_out_slice = mem_slice(&ctx->mem.ffn_down_output, (size_t)i * ctx->model->embed_dim);
			MemType *ffn_out_fp32_scratch = &ctx->mem.expert_out_fp32;
			float *ffn_out_fp32_token_buffer = ctx->mem.expert_out_fp32.data;

			// Router Projection (Raw Logits)
			dispatch_mat_vec(ctx, &normed_input_for_token_i, &l->ffn_gate_inp, &ctx->mem.expert_scores,
					 ctx->model->embed_dim, ctx->model->expert_count, false);

			float *scores = (float *)ctx->mem.expert_scores.data;
			float selection_scores[ctx->model->expert_count];

			// LFM2 Routing Logic: Sigmoid -> Add Bias -> TopK
			// Apply Sigmoid to logits
			for (int e = 0; e < ctx->model->expert_count; e++) {
				scores[e] = 1.0f / (1.0f + expf(-scores[e])); // Sigmoid
			}

			// Prepare Selection Scores (Sigmoid + Bias)
			if (l->exp_probs_b_bias.mem.data) {
				float *bias = (float *)l->exp_probs_b_bias.mem.data;
				for (int e = 0; e < ctx->model->expert_count; e++) {
					selection_scores[e] = scores[e] + bias[e];
				}
			} else {
				memcpy(selection_scores, scores, ctx->model->expert_count * sizeof(float));
			}

			// Select Top-K based on biased scores
			ExpertChoice top_experts[ctx->model->expert_used_count];
			find_top_k(selection_scores, ctx->model->expert_count, ctx->model->expert_used_count,
				   top_experts);

			// Gather Weights from ORIGINAL Sigmoid (unbiased)
			float gate_values[ctx->model->expert_used_count];
			float sum_weights = 0.0f;

			for (int j = 0; j < ctx->model->expert_used_count; j++) {
				int expert_idx = top_experts[j].index;
				gate_values[j] = scores[expert_idx];
				sum_weights += gate_values[j];
			}

			// L1 Normalization
			float inv_sum = 1.0f / (sum_weights + 1e-6f);
			for (int j = 0; j < ctx->model->expert_used_count; j++) {
				gate_values[j] *= inv_sum;
			}

			// Parallel Expert Execution
			for (int j = 0; j < ctx->model->expert_used_count; j++) {
				tasks[j] = (expert_task_t){
					.ctx = ctx,
					.thread_id = j,
					.layer_idx = layer_idx,
					.expert_idx = top_experts[j].index,
					.normed_input = normed_input_for_token_i,
				};
				thread_pool_submit(thread_pool, process_expert_task, &tasks[j]);
			}
			thread_pool_wait(thread_pool);

			// Accumulate
			memset(ffn_out_fp32_scratch->data, 0, ctx->model->embed_dim * sizeof(float));
			for (int j = 0; j < ctx->model->expert_used_count; j++) {
				float gate_val = gate_values[j];
				float *expert_result = ctx->mem.expert_outputs[j].data;
				for (int k = 0; k < ctx->model->embed_dim; k++) {
					ffn_out_fp32_token_buffer[k] += gate_val * expert_result[k];
				}
			}
			dispatch_convert(ffn_out_fp32_scratch, &ffn_out_slice, ctx->model->embed_dim);
		}

	} else {
		// Dense FFN (Layers 0, 1)
		dispatch_mat_mat(ctx, &ctx->mem.normed_ffn_input, &l->ffn_gate, &ctx->mem.gate_proj_output, batch_len,
				 ctx->model->embed_dim, ctx->model->ffn_dim, true);
		dispatch_mat_mat(ctx, &ctx->mem.normed_ffn_input, &l->ffn_up, &ctx->mem.up_proj_output, batch_len,
				 ctx->model->embed_dim, ctx->model->ffn_dim, true);

		dispatch_swiglu_activation(&ctx->mem.gate_proj_output, &ctx->mem.up_proj_output,
					   batch_len * ctx->model->ffn_dim);

		dispatch_mat_mat(ctx, &ctx->mem.gate_proj_output, &l->ffn_down, &ctx->mem.ffn_down_output, batch_len,
				 ctx->model->ffn_dim, ctx->model->embed_dim, true);
	}

	// Residual Add 2
	dispatch_apply_residual(&ctx->mem.hidden_state, &ctx->mem.ffn_down_output, batch_len * ctx->model->embed_dim);

	return 0;
}

int build_prompt_lfm2(struct TIEContext *ctx, int *prompt_tokens, int *user_text_tokens, int user_text_token_count,
		      bool has_image)
{
	VisionModel *vm = ctx->model_vision;
	ModelDef *def = ctx->model->def;
	ModelDef *vdef = ctx->model->def;
	int prompt_len = 0;

	prompt_tokens[prompt_len++] = def->params.sot_token_id;
	prompt_tokens[prompt_len++] = 6;
	prompt_tokens[prompt_len++] = def->params.role_user_token_id;
	prompt_tokens[prompt_len++] = def->params.newline_token_id;

	if (has_image) {

		int vision_start_token = 498;
		int vision_end_token = 499;

		// Calculate token count based on image size and unshuffle factor
		int patch_size = vm->patch_size; // 16
		int factor = vm->def->params.proj_scale_factor;

		int w = ctx->vision_mem.image_raw_width;
		int h = ctx->vision_mem.image_raw_height;

		// 256 / 16 = 16 patches
		int w_patches = w / patch_size;
		int h_patches = h / patch_size;

		// 16 / 2 = 8 tokens
		int w_tokens = w_patches / factor;
		int h_tokens = h_patches / factor;
		int total_image_tokens = w_tokens * h_tokens; // 64 for 256x256

		printf("LFM2 Vision: Generating %d tokens (%dx%d grid)\n", total_image_tokens, w_tokens, h_tokens);
		// Start Wrapper
		prompt_tokens[prompt_len++] = vision_start_token;

		// Image Tokens
		for (int i = 0; i < total_image_tokens; i++) {
			prompt_tokens[prompt_len++] = vdef->params.vision_embed_token_id;
		}

		// End Wrapper
		prompt_tokens[prompt_len++] = vision_end_token;
	}

	memcpy(&prompt_tokens[prompt_len], user_text_tokens, user_text_token_count * sizeof(int));
	prompt_len += user_text_token_count;

	prompt_tokens[prompt_len++] = 7;
	prompt_tokens[prompt_len++] = def->params.newline_token_id;
	prompt_tokens[prompt_len++] = 6;
	prompt_tokens[prompt_len++] = def->params.role_model_token_id;
	prompt_tokens[prompt_len++] = def->params.newline_token_id;

	return prompt_len;
}
