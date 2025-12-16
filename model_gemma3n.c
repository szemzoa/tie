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

int transformer_layer_gemma3n(struct TIEContext *ctx, int layer_idx, int batch_len);
int build_prompt_gemma3n(struct TIEContext *ctx, int *prompt_tokens, int *user_text_tokens, int user_text_token_count,
			 bool has_image);
static void post_generate_step_gemma3n(struct TIEContext *ctx, int step, size_t prompt_len);

static const TensorDef GEMMA3N_GLOBAL_TENSORS[] = {
	DECLARE_GLOBAL_TENSORS_BASE_DEFS,
	{"altup_proj.weight", offsetof(Model, altup_proj), FLAG_NONE},
	{"altup_unembd_proj.weight", offsetof(Model, altup_unembd_proj), FLAG_NONE},
	{"per_layer_model_proj.weight", offsetof(Model, per_layer_model_proj), FLAG_NONE},
	{"per_layer_proj_norm.weight", offsetof(Model, per_layer_proj_norm), FLAG_NONE},
	{"per_layer_token_embd.weight", offsetof(Model, per_layer_token_embd), FLAG_NONE}};

static const TensorDef GEMMA3N_LAYER_TENSORS[] = {
	{"blk.%u.attn_q.weight", offsetof(LayerWeights, attn_q), FLAG_NONE},
	{"blk.%u.attn_k.weight", offsetof(LayerWeights, attn_k), FLAG_NONE},
	{"blk.%u.attn_v.weight", offsetof(LayerWeights, attn_v), FLAG_NONE},
	{"blk.%u.attn_norm.weight", offsetof(LayerWeights, attn_norm), FLAG_NONE},
	{"blk.%u.attn_q_norm.weight", offsetof(LayerWeights, attn_q_norm), FLAG_NONE},
	{"blk.%u.attn_k_norm.weight", offsetof(LayerWeights, attn_k_norm), FLAG_NONE},
	{"blk.%u.attn_output.weight", offsetof(LayerWeights, attn_out), FLAG_NONE},
	{"blk.%u.ffn_norm.weight", offsetof(LayerWeights, ffn_norm), FLAG_NONE},
	{"blk.%u.ffn_gate.weight", offsetof(LayerWeights, ffn_gate), FLAG_NONE},
	{"blk.%u.ffn_up.weight", offsetof(LayerWeights, ffn_up), FLAG_NONE},
	{"blk.%u.ffn_down.weight", offsetof(LayerWeights, ffn_down), FLAG_NONE},
	{"blk.%u.post_attention_norm.weight", offsetof(LayerWeights, post_attn_norm), FLAG_NONE},
	{"blk.%u.post_ffw_norm.weight", offsetof(LayerWeights, post_ffw_norm), FLAG_NONE},
	{"blk.%u.altup_correct_coef.weight", offsetof(LayerWeights, altup_correct_coef), FLAG_NONE},
	{"blk.%u.altup_correct_scale.weight", offsetof(LayerWeights, altup_correct_scale), FLAG_NONE},
	{"blk.%u.altup_predict_coef.weight", offsetof(LayerWeights, altup_predict_coef), FLAG_NONE},
	{"blk.%u.altup_router.weight", offsetof(LayerWeights, altup_router), FLAG_NONE},
	{"blk.%u.altup_router_norm.weight", offsetof(LayerWeights, altup_router_norm), FLAG_NONE},
	{"blk.%u.inp_gate.weight", offsetof(LayerWeights, inp_gate), FLAG_NONE},
	{"blk.%u.laurel_l.weight", offsetof(LayerWeights, laurel_l), FLAG_NONE},
	{"blk.%u.laurel_post_norm.weight", offsetof(LayerWeights, laurel_post_norm), FLAG_NONE},
	{"blk.%u.laurel_r.weight", offsetof(LayerWeights, laurel_r), FLAG_NONE},
	{"blk.%u.proj.weight", offsetof(LayerWeights, proj), FLAG_NONE},
	{"blk.%u.post_norm.weight", offsetof(LayerWeights, post_norm), FLAG_NONE},
};

static const MetadataDef GEMMA3N_METADATA_DEFS[] = {
	{"%s.context_length", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, seq_length), false, false},
	{"%s.embedding_length", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, embed_dim), false, false},
	{"%s.block_count", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, num_layers), false, false},
	{"%s.attention.head_count", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, num_heads), false, false},
	{"%s.attention.layer_norm_rms_epsilon", GGUF_METADATA_VALUE_TYPE_FLOAT32, offsetof(Model, norm_eps), false,
	 false},
	{"%s.attention.key_length", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, head_dim), false, false},
	{"%s.rope.freq_base", GGUF_METADATA_VALUE_TYPE_FLOAT32, offsetof(Model, rope_freq_base), false, false},
	{"%s.attention.head_count_kv", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, num_kv_heads), false, false},
	{"%s.embedding_length_per_layer_input", GGUF_METADATA_VALUE_TYPE_FLOAT32, offsetof(Model, pli_dim), false,
	 false},
	{"%s.attention.sliding_window", GGUF_METADATA_VALUE_TYPE_FLOAT32, offsetof(Model, attn_sliding_window), false,
	 false},
	{"%s.altup.num_inputs", GGUF_METADATA_VALUE_TYPE_FLOAT32, offsetof(Model, altup_num_inputs), false, false},
	// This one is optional, with a fallback
	{"%s.rope.scaling.factor", GGUF_METADATA_VALUE_TYPE_FLOAT32, offsetof(Model, rope_scale_factor), false, true},
	DECLARE_TOKENIZER_BASE_METADATA_DEFS,
};

static const BufferDef GEMMA3N_BUFFERS[] = {
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
	{offsetof(MemLayout, residual_stratch), SIZE_EMBED_DIM, GGML_TYPE_F32, FLAG_NONE},
	{offsetof(MemLayout, per_layer_inputs), SIZE_NUM_LAYERS_X_PLI_DIM, GGML_TYPE_F32, FLAG_NONE},
};

static const TokenizeDef GEMMA3N_TOKENIZE_DEF = {
	.token_detect_specials = 0,
	.token_load_merges = 0,
	.token_load_scores = 1,
};

ModelDef GEMMA3N_DEF = {.arch = ARCH_GEMMA3N,
			.name = "Gemma-3n",
			.params =
				{
					.sot_token_id = 105,
					.eot_token_id = 106,
					.eos_token_id = 106,
					.newline_token_id = 107,
					.role_user_token_id = 2364,
					.role_model_token_id = 4368,
					.final_logit_softcap = 30.0,
					.vision_start_token_id = -1,
					.vision_end_token_id = -1,
					.vision_embed_token_id = -1,
				},
			.interface =
				{
					.tokenize_encode = tokenize_sp,
					.tokenize_decode = decode_token_sp,
					.build_prompt = build_prompt_gemma3n,

					.process_prompt_custom = process_prompt_gemma3n,
					.prepare_next_token_custom = prepare_next_token_gemma3n,
					.post_generate_step = post_generate_step_gemma3n,
					.embedding_scale = embedding_scale_gemma3,

					.transformer_layer = transformer_layer_gemma3n,
					.build_rope_cache = build_rope_cache_shared,
				},
			DECLARE_LANGUAGE_MODEL_DEF(GEMMA3N, GEMMA3N)};

/*
 * -------------------------------------------------------------------------
 * GEMMA-3N SPECIFIC IMPLEMENTATION
 * -------------------------------------------------------------------------
 */

static void post_generate_step_gemma3n(struct TIEContext *ctx, int step, size_t prompt_len)
{
	// For the first generation step, process the LAST token of the prompt.
	// For subsequent steps, buffer contains 1 token.
	size_t n_tokens_in_buffer = (step == 0) ? prompt_len : 1;
	post_process_altup_states(ctx, &ctx->mem.hidden_state, ctx->mem.altup_hidden_states, n_tokens_in_buffer);
}

void prepare_next_token_gemma3n(struct TIEContext *ctx, int next_token)
{
	const int embed_dim = ctx->model->embed_dim;
	const int pli_dim = ctx->model->pli_dim;
	const int num_layers = ctx->model->num_layers;

	// Create a temporary buffer for the single token's scaled embedding
	MemType single_token_embedding;
	alloc_memtype(&single_token_embedding, GGML_TYPE_F32, embed_dim);

	// Prepare the altup_hidden_states for the next step
	dispatch_embedding_row(&ctx->model->token_embd, next_token, &single_token_embedding, embed_dim);
	ctx->model->interface.embedding_scale(ctx, &single_token_embedding);

	// Copy the base embedding to the first state
	memcpy(ctx->mem.altup_hidden_states[0].data, single_token_embedding.data,
	       embed_dim * ggml_type_size(single_token_embedding.type));

	// Create the other 3 parallel states for this single token
	create_altup_parallel_states(ctx, &ctx->mem.altup_hidden_states[0], 1, &ctx->model->altup_proj,
				     ctx->mem.altup_hidden_states);

	// Prepare the per_layer_inputs for the next step
	MemType pli_from_lookup, pli_from_proj;
	alloc_memtype(&pli_from_lookup, GGML_TYPE_F32, num_layers * pli_dim);
	alloc_memtype(&pli_from_proj, GGML_TYPE_F32, num_layers * pli_dim);

	// Get the raw PLI for the single token
	calculate_and_deinterleave_pli_raw(ctx, &next_token, 1, &pli_from_lookup);

	// Project the main embedding
	dispatch_mat_mat(ctx, &single_token_embedding, &ctx->model->per_layer_model_proj, &pli_from_proj, 1, embed_dim,
			 num_layers * pli_dim, true);

	// Scale and Norm the projected component
	float scale = 1.0f / sqrtf((float)embed_dim);
	float *proj_data = (float *)pli_from_proj.data;
	for (size_t i = 0; i < (size_t)num_layers * pli_dim; i++) {
		proj_data[i] *= scale;
	}

	for (int i = 0; i < num_layers; i++) {
		MemType slice = mem_slice(&pli_from_proj, i * pli_dim);
		dispatch_rms_norm(&slice, &ctx->model->per_layer_proj_norm, &slice, pli_dim, ctx->model->norm_eps);
	}

	// Add the two sources into the final persistent buffer.
	dispatch_apply_residual_to_buffer(&pli_from_lookup, &pli_from_proj, &ctx->mem.per_layer_inputs,
					  num_layers * pli_dim);

	// Apply the final scale factor
	scale = 1.0f / sqrtf(2.0f);
	float *final_data = (float *)ctx->mem.per_layer_inputs.data;
	for (size_t i = 0; i < (size_t)num_layers * pli_dim; i++) {
		final_data[i] *= scale;
	}

	free_memtype(&single_token_embedding);
	free_memtype(&pli_from_lookup);
	free_memtype(&pli_from_proj);
}

// Process prompt tokens (Gemma-3n)
void process_prompt_gemma3n(struct TIEContext *ctx, int *prompt_tokens, size_t prompt_len)
{
	const int embed_dim = ctx->model->embed_dim;
	const int pli_dim = ctx->model->pli_dim;
	const int num_layers = ctx->model->num_layers;

	// Allocate all necessary buffers
	MemType scaled_token_embeddings;
	alloc_memtype(&scaled_token_embeddings, GGML_TYPE_F32, prompt_len * embed_dim);

	MemType pli_from_lookup_scaled;
	alloc_memtype(&pli_from_lookup_scaled, GGML_TYPE_F32, prompt_len * num_layers * pli_dim);

	MemType pli_from_projection;
	alloc_memtype(&pli_from_projection, GGML_TYPE_F32, prompt_len * num_layers * pli_dim);

	// Temporary buffer for de-interleaving the projected PLI
	MemType pli_from_projection_deinterleaved;
	alloc_memtype(&pli_from_projection_deinterleaved, GGML_TYPE_F32, prompt_len * num_layers * pli_dim);

	// Get the raw token embeddings
	for (int i = 0; i < prompt_len; i++) {
		MemType slice = mem_slice(&scaled_token_embeddings, i * embed_dim);
		dispatch_embedding_row(&ctx->model->token_embd, prompt_tokens[i], &slice, embed_dim);
	}

	// Scale the embeddings
	for (int i = 0; i < prompt_len; i++) {
		MemType slice = mem_slice(&scaled_token_embeddings, i * embed_dim);
		ctx->model->interface.embedding_scale(ctx, &slice);
	}

	// Create parallel AltUp states
	memcpy(ctx->mem.altup_hidden_states[0].data, scaled_token_embeddings.data,
	       prompt_len * embed_dim * ggml_type_size(scaled_token_embeddings.type));
	create_altup_parallel_states(ctx, &ctx->mem.altup_hidden_states[0], prompt_len, &ctx->model->altup_proj,
				     ctx->mem.altup_hidden_states);

	// Look up PLI from its dedicated table (creates [B, L, D] layout)
	calculate_and_deinterleave_pli_raw(ctx, prompt_tokens, prompt_len, &pli_from_lookup_scaled);

	// Project the main embeddings (creates flat [B, L*D] layout)
	dispatch_mat_mat(ctx, &scaled_token_embeddings, &ctx->model->per_layer_model_proj, &pli_from_projection,
			 prompt_len, embed_dim, num_layers * pli_dim, true);

	// Scale and Norm the projected embeddings (still in flat layout)
	float scale = 1.0f / sqrtf((float)embed_dim);
	float *proj_data = (float *)pli_from_projection.data;
	for (size_t i = 0; i < prompt_len * num_layers * pli_dim; i++) {
		proj_data[i] *= scale;
	}

	int num_vectors_to_norm = prompt_len * num_layers;
	for (int i = 0; i < num_vectors_to_norm; i++) {
		MemType vector_slice = mem_slice(&pli_from_projection, i * pli_dim);
		dispatch_rms_norm(&vector_slice, &ctx->model->per_layer_proj_norm, &vector_slice, pli_dim,
				  ctx->model->norm_eps);
	}

	// De-interleave the projected PLI to match the lookup PLI's layout
	for (int i = 0; i < prompt_len; i++) {
		for (int l = 0; l < num_layers; l++) {
			size_t src_offset = (i * num_layers + l) * pli_dim;
			MemType src_slice = mem_slice(&pli_from_projection, src_offset);

			size_t dest_offset = (i * num_layers + l) * pli_dim;
			MemType dest_slice = mem_slice(&pli_from_projection_deinterleaved, dest_offset);

			memcpy(dest_slice.data, src_slice.data, pli_dim * ggml_type_size(dest_slice.type));
		}
	}

	// Add the two sources together (now with matching layouts)
	dispatch_apply_residual_to_buffer(&pli_from_lookup_scaled, &pli_from_projection_deinterleaved,
					  &ctx->mem.per_layer_inputs, prompt_len * num_layers * pli_dim);

	// Apply the final scale
	scale = 1.0f / sqrtf(2.0f);
	float *final_data = (float *)ctx->mem.per_layer_inputs.data;
	for (size_t i = 0; i < prompt_len * num_layers * pli_dim; i++) {
		final_data[i] *= scale;
	}

	for (int l = 0; l < num_layers; l++) {
		ctx->model->interface.transformer_layer(ctx, l, prompt_len);
	}

	ctx->kv_pos += prompt_len;

	free_memtype(&scaled_token_embeddings);
	free_memtype(&pli_from_lookup_scaled);
	free_memtype(&pli_from_projection);
	free_memtype(&pli_from_projection_deinterleaved); // Don't forget to free the new buffer
}

// Implements the full LAUREL block logic for a given input.
void dispatch_laurel(struct TIEContext *ctx, MemType *output, const MemType *input, LayerWeights *l, int batch_len)
{
	const int embed_dim = ctx->model->embed_dim;
	const int laurel_rank = 64; // From the config

	MemType laurel_left;
	alloc_memtype(&laurel_left, GGML_TYPE_F32, batch_len * laurel_rank);

	MemType laurel_right;
	alloc_memtype(&laurel_right, GGML_TYPE_F32, batch_len * embed_dim);

	// Left projection: input @ laurel_l.weight
	dispatch_mat_mat(ctx, input, &l->laurel_l, &laurel_left, batch_len, embed_dim, laurel_rank, true);

	// Right projection: laurel_left @ laurel_r.weight
	dispatch_mat_mat(ctx, &laurel_left, &l->laurel_r, &laurel_right, batch_len, laurel_rank, embed_dim, true);

	// Post-LAUREL normalization
	for (int i = 0; i < batch_len; i++) {
		MemType slice = mem_slice(&laurel_right, i * embed_dim);
		dispatch_rms_norm(&slice, &l->laurel_post_norm, &slice, embed_dim, ctx->model->norm_eps);
	}

	// Final residual connection: output = input + normed_result
	dispatch_apply_residual_to_buffer(input, &laurel_right, output, batch_len * embed_dim);

	free_memtype(&laurel_left);
	free_memtype(&laurel_right);
}

// This can be optimized with AVX2.
void dispatch_subtract_to_buffer(const MemType *src1, const MemType *src2, MemType *dest, int size)
{
	const float *src1_data = (const float *)src1->data;
	const float *src2_data = (const float *)src2->data;
	float *dest_data = (float *)dest->data;

	// A simple loop for element-wise addition.
	for (int i = 0; i < size; i++) {
		dest_data[i] = src1_data[i] - src2_data[i];
	}
}

// This can be optimized with AVX2.
void dispatch_elementwise_mul(MemType *dest, const MemType *src1, const MemType *src2, int size)
{
	float *dest_data = (float *)dest->data;
	const float *src1_data = (const float *)src1->data;
	const float *src2_data = (const float *)src2->data;

	for (int i = 0; i < size; i++) {
		dest_data[i] = src1_data[i] * src2_data[i];
	}
}

// This can be optimized with AVX2.
void dispatch_elementwise_mul_tensor(MemType *dest, const MemType *src1, const Tensor *src2, int batch_len,
				     int embed_dim)
{
	float *dest_data = (float *)dest->data;
	const float *src1_data = (const float *)src1->data;
	const float *src2_data = (const float *)src2->mem.data;

	int size = batch_len * embed_dim;

	for (int i = 0; i < size; i++) {
		// Use the modulo operator to repeat/broadcast the smaller src2 vector
		// for each token in the batch.
		dest_data[i] = src1_data[i] * src2_data[i % embed_dim];
	}
}

void dispatch_rms_norm_weightless(MemType *tensor, int size, float eps)
{
	float *x = (float *)tensor->data;

	// Unrolled accumulation for faster reduction
	float ss0 = 0.0f, ss1 = 0.0f, ss2 = 0.0f, ss3 = 0.0f;
	int i = 0;
	for (; i <= size - 4; i += 4) {
		ss0 = fmaf(x[i + 0], x[i + 0], ss0);
		ss1 = fmaf(x[i + 1], x[i + 1], ss1);
		ss2 = fmaf(x[i + 2], x[i + 2], ss2);
		ss3 = fmaf(x[i + 3], x[i + 3], ss3);
	}
	float ss = ss0 + ss1 + ss2 + ss3;
	for (; i < size; i++) {
		ss = fmaf(x[i], x[i], ss);
	}

	float inv_rms = 1.0f / sqrtf(ss / size + eps);

	// Apply scale
	for (int i = 0; i < size; ++i) {
		x[i] *= inv_rms;
	}
}

void dispatch_gaussian_topk(MemType *gate_tensor, int size)
{
	float *data = (float *)gate_tensor->data;
	const float f_sparsity_std_mul = 1.6448533535f; // Corresponds to the 95th percentile

	if (size == 0)
		return;

	// Numerically Stable Single-Pass Algorithm (Welford's)
	float mean = 0.0;
	float m2 = 0.0;
	float delta;

	for (int i = 0; i < size; i++) {
		delta = data[i] - mean;
		mean += delta / (i + 1);
		m2 += delta * (data[i] - mean);
	}

	float variance = m2 / size; // Population variance
	float std_dev = sqrt(variance);

	// Determine the cutoff value
	float cutoff = mean + std_dev * f_sparsity_std_mul;

	// Apply the ReLU-like sparsity: output = max(0, input - cutoff)
	for (int i = 0; i < size; i++) {
		data[i] = (data[i] > cutoff) ? (data[i] - cutoff) : 0.0f;
	}
}

inline float gelu_fast(float x)
{
	return 0.5f * x * (1.0f + tanhf(0.79788456f * x * (1.0f + 0.044715f * x * x)));
}

// GELU activation function to a tensor in-place.
// This can be optimized with AVX2.
void dispatch_gelu_inplace(MemType *tensor, int size)
{
	float *data = (float *)tensor->data;
	for (int i = 0; i < size; i++) {
		data[i] = gelu_fast(data[i]);
	}
}

void dispatch_softcap_logits(MemType *logits, int size, float cap)
{
	if (cap <= 0.0f)
		return;

	float *data = (float *)logits->data;
	float inv_cap = 1.0f / cap;

	for (int i = 0; i < size; i++) {
		data[i] = tanhf(data[i] * inv_cap) * cap;
	}
}

void dispatch_apply_residual_to_buffer(const MemType *src1, const MemType *src2, MemType *dest, int size)
{
	const float *src1_data = (const float *)src1->data;
	const float *src2_data = (const float *)src2->data;
	float *dest_data = (float *)dest->data;

	for (int i = 0; i < size; i++) {
		dest_data[i] = src1_data[i] + src2_data[i];
	}
}

void dispatch_altup_predict(struct TIEContext *ctx, MemType *predicted_states, MemType *input_states, LayerWeights *l,
			    int batch_len, int active_idx)
{
	const int embed_dim = ctx->model->embed_dim;
	const int num_altup = ctx->model->altup_num_inputs;
	MemType *active_state_in = &input_states[active_idx];

	MemType normed_for_router, modalities, prediction_coefs;
	alloc_memtype(&normed_for_router, GGML_TYPE_F32, batch_len * embed_dim);
	alloc_memtype(&modalities, GGML_TYPE_F32, batch_len * num_altup);
	alloc_memtype(&prediction_coefs, GGML_TYPE_F32, batch_len * (num_altup * num_altup));


	for (int i = 0; i < batch_len; i++) {
		size_t offset = (size_t)i * embed_dim;
		MemType active_slice = mem_slice(active_state_in, offset);
		MemType normed_slice = mem_slice(&normed_for_router, offset);

		// Normalize one token at a time
		dispatch_rms_norm(&active_slice, &l->altup_router_norm, &normed_slice, embed_dim, ctx->model->norm_eps);

		// Scale one token at a time
		float scale = 1.0f / (float)embed_dim;
		float *norm_data = (float *)normed_slice.data;
		for (int d = 0; d < embed_dim; d++) {
			norm_data[d] *= scale;
		}
	}

	dispatch_mat_mat(ctx, &normed_for_router, &l->altup_router, &modalities, batch_len, embed_dim, num_altup, true);

	float *mod = (float *)modalities.data;
	for (int i = 0; i < batch_len * num_altup; i++)
		mod[i] = tanhf(mod[i]);
	dispatch_mat_mat(ctx, &modalities, &l->altup_predict_coef, &prediction_coefs, batch_len, num_altup,
			 num_altup * num_altup, true);

	// Use Coefficients to Mix States
	const float *coefs_data = (const float *)prediction_coefs.data;
	MemType mixed_states;
	alloc_memtype(&mixed_states, GGML_TYPE_F32, batch_len * num_altup * embed_dim);

	// Re-order the loops to create a [state][token][dim] memory layout.
	// This makes the data for each state contiguous across the entire batch.
	for (int i = 0; i < num_altup; i++) {	      // For each OUTPUT state
		for (int t = 0; t < batch_len; t++) { // For each token
			const float *coefs_for_token = coefs_data + t * (num_altup * num_altup);
			float *out_vec = (float *)mixed_states.data + (i * batch_len + t) * embed_dim;
			memset(out_vec, 0, embed_dim * sizeof(float));

			for (int j = 0; j < num_altup; j++) { // For each INPUT state
				const float *in_vec = (const float *)input_states[j].data + t * embed_dim;
				const float coef = coefs_for_token[i * num_altup + j];
				for (int d = 0; d < embed_dim; d++) {
					out_vec[d] += coef * in_vec[d];
				}
			}
		}
	}

	// Final Residual Connection
	for (int i = 0; i < num_altup; i++) {
		// Slice points to the start of the contiguous data for state `i`.
		MemType slice = mem_slice(&mixed_states, (size_t)i * batch_len * embed_dim);
		dispatch_apply_residual_to_buffer(&input_states[i], &slice, &predicted_states[i],
						  batch_len * embed_dim);
	}

	free_memtype(&normed_for_router);
	free_memtype(&modalities);
	free_memtype(&prediction_coefs);
	free_memtype(&mixed_states);
}

// corrected_states: Output: The final 4 corrected states for this layer
// predictions: Input: The 4 states from the 'P' step
// final_active_output: Input: The result of the 'A' step
void dispatch_altup_correct(struct TIEContext *ctx, MemType *corrected_states, const MemType *predictions,
			    MemType *final_active_output, LayerWeights *l, int batch_len, int active_idx)
{
	const int embed_dim = ctx->model->embed_dim;
	const int num_altup = ctx->model->altup_num_inputs;

	MemType modalities, correction_coefs, innovation;
	alloc_memtype(&modalities, GGML_TYPE_F32, batch_len * num_altup);
	alloc_memtype(&correction_coefs, GGML_TYPE_F32, batch_len * num_altup);
	alloc_memtype(&innovation, GGML_TYPE_F32, batch_len * embed_dim);

	// Create a single buffer for all per-token calculations
	MemType temp_token_vec;
	alloc_memtype(&temp_token_vec, GGML_TYPE_F32, embed_dim);
	float *temp_data = (float *)temp_token_vec.data;

	// Process the Corrector Path ONE TOKEN AT A TIME
	for (int i = 0; i < batch_len; i++) {
		size_t offset = (size_t)i * embed_dim;
		MemType active_slice = mem_slice(final_active_output, offset);

		// Normalize the single active token's vector
		dispatch_rms_norm(&active_slice, &l->altup_router_norm, &temp_token_vec, embed_dim,
				  ctx->model->norm_eps);

		// Scale the result
		float scale = 1.0f / (float)embed_dim;
		for (int d = 0; d < embed_dim; d++) {
			temp_data[d] *= scale;
		}

		// Compute modalities for this single token
		MemType modalities_slice = mem_slice(&modalities, i * num_altup);
		dispatch_mat_vec(ctx, &temp_token_vec, &l->altup_router, &modalities_slice, embed_dim, num_altup, true);

		// Apply tanh to this token's modalities
		float *mod_slice_data = (float *)modalities_slice.data;
		for (int d = 0; d < num_altup; d++) {
			mod_slice_data[d] = tanhf(mod_slice_data[d]);
		}
	}

	// Compute correction coefficients for the whole batch
	dispatch_mat_mat(ctx, &modalities, &l->altup_correct_coef, &correction_coefs, batch_len, num_altup, num_altup,
			 true);

	// Add 1.0 to all coefficients
	float *coef = (float *)correction_coefs.data;
	for (int i = 0; i < batch_len * num_altup; i++) {
		coef[i] += 1.0f;
	}

	// Calculate the "innovation" vector for the whole batch
	dispatch_subtract_to_buffer(final_active_output, &predictions[active_idx], &innovation, batch_len * embed_dim);

	// Apply the correction to all states
	const float *final_coefs_data = (const float *)correction_coefs.data;
	for (int i = 0; i < num_altup; i++) {
		for (int t = 0; t < batch_len; t++) {
			const float *innovation_vec = (const float *)innovation.data + t * embed_dim;
			const float *prediction_vec = (const float *)predictions[i].data + t * embed_dim;
			float *corrected_vec = (float *)corrected_states[i].data + t * embed_dim;
			const float coef_for_this_state = final_coefs_data[t * num_altup + i];
			for (int d = 0; d < embed_dim; d++) {
				corrected_vec[d] = prediction_vec[d] + innovation_vec[d] * coef_for_this_state;
			}
		}
	}

	free_memtype(&temp_token_vec);
	free_memtype(&modalities);
	free_memtype(&correction_coefs);
	free_memtype(&innovation);
}

// This function calculates the final refinement_residual
void dispatch_pli_gating(struct TIEContext *ctx,
			 MemType *refinement_residual,	 // The final output buffer
			 const MemType *active_state,	 // Input: The active state from the 'Corrected' array
			 const MemType *per_layer_input, // Input: The 256-dim PLI vector for this layer
			 LayerWeights *l, int batch_len)
{
	const int embed_dim = ctx->model->embed_dim;
	const int pli_dim = ctx->model->pli_dim;
	MemType scaled_active_state, gated_state, gelu_state, modulated_state, projected_state;

	alloc_memtype(&scaled_active_state, GGML_TYPE_F32, batch_len * embed_dim);
	alloc_memtype(&gated_state, GGML_TYPE_F32, batch_len * pli_dim);
	alloc_memtype(&gelu_state, GGML_TYPE_F32, batch_len * pli_dim);
	alloc_memtype(&modulated_state, GGML_TYPE_F32, batch_len * pli_dim);
	alloc_memtype(&projected_state, GGML_TYPE_F32, batch_len * embed_dim);

	// Scale the Active State
	// This is an element-wise multiplication by the altup_correct_scale vector
	dispatch_elementwise_mul_tensor(&scaled_active_state, active_state, &l->altup_correct_scale, batch_len,
					embed_dim);

	// Gate Projection (2048-dim -> 256-dim)
	dispatch_mat_mat(ctx, &scaled_active_state, &l->inp_gate, &gated_state, batch_len, embed_dim, pli_dim, true);

	// GeLU Activation
	// The GeLU for this gate is special; it doesn't have an 'up' projection, so we pass NULL or a dummy.
	// Let's create a temporary dummy buffer for the 'up' projection if needed.
	memcpy(gelu_state.data, gated_state.data, batch_len * pli_dim * ggml_type_size(gated_state.type));
	dispatch_gelu_inplace(&gelu_state, batch_len * pli_dim);

	// Modulate by Per-Layer Input
	// Element-wise multiplication: gelu_state * per_layer_input
	dispatch_elementwise_mul(&modulated_state, &gelu_state, per_layer_input, batch_len * pli_dim);

	// Project Back (256-dim -> 2048-dim)
	dispatch_mat_mat(ctx, &modulated_state, &l->proj, &projected_state, batch_len, pli_dim, embed_dim, true);

	// Final Norm
	// The output is the final refinement_residual
	for (int i = 0; i < batch_len; i++) {
		MemType projected_slice = mem_slice(&projected_state, i * embed_dim);
		MemType dest_slice = mem_slice(refinement_residual, i * embed_dim);

		// Apply the norm to each token's vector individually
		dispatch_rms_norm(&projected_slice, &l->post_norm, &dest_slice, embed_dim, ctx->model->norm_eps);
	}

	free_memtype(&scaled_active_state);
	free_memtype(&gated_state);
	free_memtype(&gelu_state);
	free_memtype(&modulated_state);
	free_memtype(&projected_state);
}

/* Extracts all per-layer inputs for a specific layer from the main PLI buffer.
 * The source buffer has a layout of [batch_len][num_layers][pli_dim].
 * The destination buffer will have a contiguous layout of [batch_len][pli_dim].
 */
void get_per_layer_input_for_layer(MemType *dest_buffer,    // Pre-allocated destination buffer
				   MemType *src_pli_buffer, // The main [batch_len, num_layers, pli_dim] buffer
				   int layer_idx, int batch_len, int num_layers, int pli_dim)
{
	// Loop through each token in the batch
	for (int i = 0; i < batch_len; i++) {
		// Calculate the offset to find the source data for this token at the specified layer
		size_t src_offset = (i * num_layers + layer_idx) * pli_dim;
		MemType src_slice = mem_slice(src_pli_buffer, src_offset);

		// Calculate the offset in the destination buffer
		size_t dest_offset = i * pli_dim;
		MemType dest_slice = mem_slice(dest_buffer, dest_offset);

		// Copy the pli_dim (e.g., 256) floats from the source to the destination
		memcpy(dest_slice.data, src_slice.data, pli_dim * ggml_type_size(dest_slice.type));
	}
}

// Function to calculate per-token magnitude
void calculate_per_token_magnitude(float *magnitudes, const MemType *state, size_t num_tokens, int dim)
{
	for (size_t t = 0; t < num_tokens; t++) {
		const float *token_vec = (const float *)state->data + t * dim;
		float ss = 0.0;
		for (int i = 0; i < dim; i++) {
			ss += token_vec[i] * token_vec[i];
		}

		// Divide by the dimension 'dim' to calculate the mean of squares, then take the square root.
		magnitudes[t] = sqrt(ss / dim);
	}
}

// Function to create the parallel states
void create_altup_parallel_states(struct TIEContext *ctx, MemType *base_state, size_t prompt_len,
				  Tensor *altup_proj_tensor, MemType *destination)
{
	const int embed_dim = ctx->model->embed_dim;

	// Get a pointer to the single, large altup_proj tensor
	size_t matrix_size_bytes = (size_t)embed_dim * embed_dim * ggml_type_size(altup_proj_tensor->mem.type);

	// Loop and create the other parallel states (state[1] through state[3])
	for (int i = 1; i < ctx->model->altup_num_inputs; i++) {
		MemType *dest_state = &destination[i];

		// Create a temporary "view" of the i-th matrix within the flat tensor
		Tensor altup_proj_slice = *altup_proj_tensor;
		size_t offset = (size_t)(i - 1) * matrix_size_bytes;
		altup_proj_slice.mem.data = (uint8_t *)altup_proj_tensor->mem.data + offset;

		// Project: dest_state = base_state @ altup_projection_slice
		dispatch_mat_mat(ctx, base_state, &altup_proj_slice, dest_state, prompt_len, embed_dim, embed_dim,
				 true);

		for (size_t t = 0; t < prompt_len; t++) {
			const float *base_vec = (const float *)base_state->data + t * embed_dim;
			float *dest_vec = (float *)dest_state->data + t * embed_dim;

			// Calculate sum of squares for both vectors
			float ss_base = 0.0;
			float ss_dest = 0.0;
			for (int d = 0; d < embed_dim; d++) {
				ss_base += base_vec[d] * base_vec[d];
				ss_dest += dest_vec[d] * dest_vec[d];
			}

			// Calculate the final scaling factor directly
			const float scale_factor = sqrtf(ss_base / (ss_dest + 1e-12f));

			// Apply the scaling
			for (int d = 0; d < embed_dim; d++) {
				dest_vec[d] *= scale_factor;
			}
		}
	}
}

void calculate_and_deinterleave_pli_raw(struct TIEContext *ctx, int *prompt_tokens, size_t prompt_len,
					MemType *dest_buffer)
{
	const int pli_dim = ctx->model->pli_dim;
	const int num_layers = ctx->model->num_layers;

	// Allocate a temporary buffer to hold the giant (e.g., 7680-dim) vector for ONE token.
	MemType temp_pli_vector;
	alloc_memtype(&temp_pli_vector, GGML_TYPE_F32, num_layers * pli_dim);

	// Loop through each token in the prompt.
	for (int i = 0; i < prompt_len; i++) {
		int token_id = prompt_tokens[i];

		// Look up the full (e.g., 7680-dim) vector for the current token from the PLI embedding table.
		dispatch_embedding_row(&ctx->model->per_layer_token_embd, token_id, &temp_pli_vector,
				       num_layers * pli_dim);

		// De-interleave the temporary vector into the final destination buffer.
		// This loop takes the stacked vector (e.g., [layer0_data, layer1_data, ...])
		// and distributes it into the final [token, layer, dim] layout.
		for (int l = 0; l < num_layers; l++) {
			// Source: The l-th slice of the temporary vector.
			MemType src_slice = mem_slice(&temp_pli_vector, l * pli_dim);

			// Destination: The correct spot in the final buffer for token `i` at layer `l`.
			size_t dest_offset = (i * num_layers * pli_dim) + (l * pli_dim);
			MemType dest_slice = mem_slice(dest_buffer, dest_offset);

			// Copy the 256 floats for this layer.
			memcpy(dest_slice.data, src_slice.data, pli_dim * ggml_type_size(dest_slice.type));

			float scale = sqrtf(256.0f);
			float *data = (float *)dest_slice.data;
			for (size_t d = 0; d < pli_dim; d++) {
				data[d] *= scale;
			}
		}
	}

	free_memtype(&temp_pli_vector);
}

void post_process_altup_states(struct TIEContext *ctx,
			       MemType *final_hidden_state, // Output: The single, final vector
			       MemType *final_altup_states, // Input: The array of 4 states
			       size_t n_tokens)
{
	size_t last_token_idx = n_tokens - 1;
	const int embed_dim = ctx->model->embed_dim;
	const int num_altup = ctx->model->altup_num_inputs;

	// The active index is always 0 for Gemma-3N
	const int active_idx = 0;

	// Temporary buffer to hold the final versions of each state before averaging
	MemType temp_states[num_altup];
	for (int i = 0; i < num_altup; i++) {
		alloc_memtype(&temp_states[i], GGML_TYPE_F32, embed_dim);
	}

	//  The active state (index 0) is our "base". Copy its last token's data directly.
	MemType base_state_slice = mem_slice(&final_altup_states[active_idx], last_token_idx * embed_dim);
	memcpy(temp_states[active_idx].data, base_state_slice.data, embed_dim * sizeof(float));

	// Un-Project and Rescale the INACTIVE states (1, 2, 3)
	float target_magnitude;
	calculate_per_token_magnitude(&target_magnitude, &base_state_slice, 1, embed_dim);

	size_t matrix_size_bytes =
		(size_t)embed_dim * embed_dim * ggml_type_size(ctx->model->altup_unembd_proj.mem.type);

	for (int i = 0; i < num_altup - 1; i++) {
		// The inactive states are 1, 2, 3. The projection matrices are 0, 1, 2.
		int inactive_state_idx = i + 1;
		int proj_matrix_idx = i;

		MemType *dest_state = &temp_states[inactive_state_idx];
		MemType src_slice = mem_slice(&final_altup_states[inactive_state_idx], last_token_idx * embed_dim);

		Tensor unembd_proj_slice = ctx->model->altup_unembd_proj;
		unembd_proj_slice.mem.data =
			(uint8_t *)unembd_proj_slice.mem.data + (size_t)proj_matrix_idx * matrix_size_bytes;

		dispatch_mat_vec(ctx, &src_slice, &unembd_proj_slice, dest_state, embed_dim, embed_dim, false);

		float new_magnitude;
		calculate_per_token_magnitude(&new_magnitude, dest_state, 1, embed_dim);
		float scale_factor = target_magnitude / (new_magnitude + 1e-12f);

		float *dest_data = (float *)dest_state->data;
		for (int d = 0; d < embed_dim; d++) {
			dest_data[d] *= scale_factor;
		}
	}

	// Average All States into the final_hidden_state buffer
	float inv_num_altup = 1.0f / (float)num_altup;
	float *final_data = (float *)final_hidden_state->data;
	memset(final_data, 0, embed_dim * sizeof(float));

	for (int i = 0; i < num_altup; i++) {
		const float *src_data = (const float *)temp_states[i].data;
		for (int d = 0; d < embed_dim; d++) {
			final_data[d] += src_data[d]; // Sum first...
		}
	}

	// Scale once at the end.
	for (int d = 0; d < embed_dim; d++) {
		final_data[d] *= inv_num_altup;
	}

	for (int i = 0; i < num_altup; i++) {
		free_memtype(&temp_states[i]);
	}
}

int transformer_layer_gemma3n(struct TIEContext *ctx, int layer_idx, int batch_len)
{
	LayerWeights *l = &ctx->model->layers[layer_idx];
	int kv_dim = ctx->model->num_kv_heads * ctx->model->head_dim;
	int q_dim = ctx->model->num_heads * ctx->model->head_dim;
	AttentionType attn_type = ATTN_TYPE_LOCAL;
	RopeCacheType *active_rope_cache = ctx->model->rope_cache_local;
	const int pli_dim = ctx->model->pli_dim;
	const int embed_dim = ctx->model->embed_dim;
	const int DATA_ACTIVE_IDX = 0; // The first state is used for the main computation path
	int sink_len = 4;


	// The absolute starting position for this batch
	int start_pos = ctx->kv_pos;

	// Determine the attention and rope cache type for the current layer
	if ((layer_idx + 1) % 5 == 0) {
		attn_type = ATTN_TYPE_GLOBAL;
		active_rope_cache = ctx->model->rope_cache_global; // Use the global cache
	}

	// Gemma-3n KV sharing logic
	int first_kv_shared_layer_idx = ctx->model->num_layers - ctx->model->shared_kv_layers;
	bool is_kv_shared_layer = (layer_idx >= first_kv_shared_layer_idx);

	int kv_source_layer_idx = layer_idx;
	bool store_full_length_kv = false;

	if (is_kv_shared_layer) {
		// For shared layers, find the last non-shared layer of the same type
		int my_type = attn_type;
		kv_source_layer_idx = -1;
		for (int i = first_kv_shared_layer_idx - 1; i >= 0; --i) {
			int prev_type = ((i + 1) % 5 == 0) ? ATTN_TYPE_GLOBAL : ATTN_TYPE_LOCAL;
			if (prev_type == my_type) {
				kv_source_layer_idx = i;
				break;
			}
		}
		if (kv_source_layer_idx < 0) {
			fprintf(stderr, "[ERROR] No kv_source_layer_idx found for layer %d type %d\n", layer_idx,
				my_type);
			exit(1);
		}
	} else {
		// For non-shared layers, decide if this is the last non-shared layer of its type
		int my_type = attn_type;
		store_full_length_kv = true;
		for (int i = layer_idx + 1; i < first_kv_shared_layer_idx; ++i) {
			int next_type = ((i + 1) % 5 == 0) ? ATTN_TYPE_GLOBAL : ATTN_TYPE_LOCAL;
			if (next_type == my_type) {
				store_full_length_kv = false;
				break;
			}
		}
	}

	dispatch_altup_predict(ctx, ctx->mem.altup_predicted_states, ctx->mem.altup_hidden_states, l, batch_len,
			       DATA_ACTIVE_IDX);

	// Save First Residual: residual_1 = active_prediction.
	memcpy(ctx->mem.residual_stratch.data, ctx->mem.altup_predicted_states[DATA_ACTIVE_IDX].data,
	       batch_len * ctx->model->embed_dim * sizeof(float));

	// Attention Block
	// RMSNorm
	for (int i = 0; i < batch_len; i++) {
		size_t offset = (size_t)i * ctx->model->embed_dim;

		// Create slices for the specific token being processed
		MemType input_slice = mem_slice(&ctx->mem.altup_predicted_states[DATA_ACTIVE_IDX], offset);
		MemType normed_slice = mem_slice(&ctx->mem.normed_qkv_input, offset);

		dispatch_rms_norm(&input_slice, &l->attn_norm, &normed_slice, ctx->model->embed_dim,
				  ctx->model->norm_eps);
	}

	MemType laurel_output;
	alloc_memtype(&laurel_output, GGML_TYPE_F32, batch_len * ctx->model->embed_dim);

	dispatch_laurel(ctx, &laurel_output, &ctx->mem.normed_qkv_input, l, batch_len);

	// Compute Q/K/V Matrices
	dispatch_mat_mat(ctx, &ctx->mem.normed_qkv_input, &l->attn_q, &ctx->mem.Q, batch_len, ctx->model->embed_dim,
			 q_dim, true);

	if (!is_kv_shared_layer) {
		// For non shared layers, compute K and V normally
		dispatch_mat_mat(ctx, &ctx->mem.normed_qkv_input, &l->attn_k, &ctx->mem.K, batch_len,
				 ctx->model->embed_dim, kv_dim, true);

		dispatch_mat_mat(ctx, &ctx->mem.normed_qkv_input, &l->attn_v, &ctx->mem.V, batch_len,
				 ctx->model->embed_dim, kv_dim, true);
	}

	// Apply RoPE
	for (int i = 0; i < batch_len; i++) {
		// The absolute position for the current token in the batch
		int absolute_pos = start_pos + i;

		for (int h = 0; h < ctx->model->num_heads; h++) {
			MemType Q_slice = mem_slice(&ctx->mem.Q, (size_t)i * q_dim + h * ctx->model->head_dim);

			if (l->attn_q_norm.mem.data) {
				dispatch_rms_norm(&Q_slice, &l->attn_q_norm, &Q_slice, ctx->model->head_dim,
						  ctx->model->norm_eps);
			}

			// Use the absolute position for RoPE
			dispatch_apply_rope_cache(active_rope_cache, &Q_slice, absolute_pos, ctx->model->head_dim);
		}

		if (!is_kv_shared_layer) {

			for (int h = 0; h < ctx->model->num_kv_heads; h++) {
				MemType K_slice = mem_slice(&ctx->mem.K, (size_t)i * kv_dim + h * ctx->model->head_dim);

				if (l->attn_k_norm.mem.data) {
					dispatch_rms_norm(&K_slice, &l->attn_k_norm, &K_slice, ctx->model->head_dim,
							  ctx->model->norm_eps);
				}

				// Use the absolute position for RoPE
				dispatch_apply_rope_cache(active_rope_cache, &K_slice, absolute_pos,
							  ctx->model->head_dim);
			}

			for (int h = 0; h < ctx->model->num_kv_heads; h++) {
				MemType V_slice = mem_slice(&ctx->mem.V, (size_t)i * kv_dim + h * ctx->model->head_dim);

				dispatch_rms_norm_weightless(&V_slice, ctx->model->head_dim, ctx->model->norm_eps);
			}
		}
	}

	// Store K/V to cache
	if (!is_kv_shared_layer) {
		dispatch_store_KV_cache(ctx, layer_idx, start_pos, batch_len, sink_len);
	}

	// Multi-Head Attention Calculation
	attention(ctx, batch_len, layer_idx, kv_source_layer_idx, start_pos, attn_type, attention_worker, sink_len);

	// Output projection
	dispatch_mat_mat(ctx, &ctx->mem.attn_output, &l->attn_out, &ctx->mem.attn_proj_output, batch_len, q_dim,
			 ctx->model->embed_dim, true);

	// Apply POST-ATTENTION norm
	for (int i = 0; i < batch_len; i++) {
		size_t offset = (size_t)i * ctx->model->embed_dim;

		MemType attn_proj_slice = mem_slice(&ctx->mem.attn_proj_output, offset);

		dispatch_rms_norm(&attn_proj_slice, &l->post_attn_norm, &attn_proj_slice, ctx->model->embed_dim,
				  ctx->model->norm_eps);
	}

	// Add the post-normed attention output to the original residual
	dispatch_apply_residual(&ctx->mem.residual_stratch, &ctx->mem.attn_proj_output,
				batch_len * ctx->model->embed_dim);

	// Mix LAUREL and Attention: Compute attn_laurel_mix = (attn_with_residual + laurel_out) * (1.0f / sqrtf(2.0f)).
	MemType attn_laurel_mix;
	alloc_memtype(&attn_laurel_mix, GGML_TYPE_F32, batch_len * ctx->model->embed_dim);

	float scale = 1.0f / sqrtf(2.0f);
	for (int i = 0; i < batch_len; i++) {
		size_t offset = (size_t)i * ctx->model->embed_dim;

		MemType attn_laurel_mix_slice = mem_slice(&attn_laurel_mix, offset);
		MemType residual_stratch_slice = mem_slice(&ctx->mem.residual_stratch, offset);
		MemType laurel_output_slice = mem_slice(&laurel_output, offset);

		float *attn_laurel_mix_out = (float *)attn_laurel_mix_slice.data;
		float *residual_stratch_data = (float *)residual_stratch_slice.data;
		float *laurel_output_data = (float *)laurel_output_slice.data;

		for (int d = 0; d < ctx->model->embed_dim; d++)
			attn_laurel_mix_out[d] = (residual_stratch_data[d] + laurel_output_data[d]) * scale;
	}

	// Save Second Residual: residual_2 = attn_laurel_mix.
	memcpy(ctx->mem.residual_stratch.data, attn_laurel_mix.data, batch_len * ctx->model->embed_dim * sizeof(float));

	// FFN Block
	// RMSNorm
	for (int i = 0; i < batch_len; i++) {
		size_t offset = (size_t)i * ctx->model->embed_dim;

		// Create slices for the specific token being processed
		MemType attn_laurel_mix_slice = mem_slice(&attn_laurel_mix, offset);
		MemType normed_ffn_input_slice = mem_slice(&ctx->mem.normed_ffn_input, offset);

		dispatch_rms_norm(&attn_laurel_mix_slice, &l->ffn_norm, &normed_ffn_input_slice, ctx->model->embed_dim,
				  ctx->model->norm_eps);
	}

	// Gate + Up projections
	dispatch_mat_mat(ctx, &ctx->mem.normed_ffn_input, &l->ffn_gate, &ctx->mem.gate_proj_output, batch_len,
			 ctx->model->embed_dim, ctx->model->ffn_dim, true);
	dispatch_mat_mat(ctx, &ctx->mem.normed_ffn_input, &l->ffn_up, &ctx->mem.up_proj_output, batch_len,
			 ctx->model->embed_dim, ctx->model->ffn_dim, true);

	if (layer_idx < 10) {
		for (int i = 0; i < batch_len; i++) {
			// Get a slice for the current token's gate vector
			MemType gate_slice = mem_slice(&ctx->mem.gate_proj_output, i * ctx->model->ffn_dim);

			// Apply sparsity to this token's slice only
			dispatch_gaussian_topk(&gate_slice, ctx->model->ffn_dim);
		}
	}

	/* Call the interface activation function */
	dispatch_geglu_activation(&ctx->mem.gate_proj_output, &ctx->mem.up_proj_output,
				  batch_len * ctx->model->ffn_dim);

	// Down projection
	dispatch_mat_mat(ctx, &ctx->mem.gate_proj_output, &l->ffn_down, &ctx->mem.ffn_down_output, batch_len,
			 ctx->model->ffn_dim, ctx->model->embed_dim, true);

	// Apply POST-FFN norm
	for (int i = 0; i < batch_len; i++) {
		size_t offset = (size_t)i * ctx->model->embed_dim;
		MemType ffn_down_slice = mem_slice(&ctx->mem.ffn_down_output, offset);

		dispatch_rms_norm(&ffn_down_slice, &l->post_ffw_norm, &ffn_down_slice, ctx->model->embed_dim,
				  ctx->model->norm_eps);
	}

	// Second Residual Add.
	// Add the post-normed FFN output to the residual
	dispatch_apply_residual(&ctx->mem.residual_stratch, &ctx->mem.ffn_down_output,
				batch_len * ctx->model->embed_dim);

	// Create a temporary buffer to build the final output for THIS layer
	MemType final_layer_output[ctx->model->altup_num_inputs];
	for (int i = 0; i < ctx->model->altup_num_inputs; i++) {
		alloc_memtype(&final_layer_output[i], GGML_TYPE_F32, batch_len * embed_dim);
	}

	// ctx->mem.residual_stratch is the final active output.
	dispatch_altup_correct(ctx, final_layer_output, ctx->mem.altup_predicted_states, &ctx->mem.residual_stratch, l,
			       batch_len, DATA_ACTIVE_IDX);

	// The Final Gating and Refinement
	MemType refinement_residual;
	alloc_memtype(&refinement_residual, GGML_TYPE_F32, batch_len * embed_dim);

	// Create a buffer to hold the contiguous per-layer inputs for this layer
	MemType per_layer_input_for_layer;
	alloc_memtype(&per_layer_input_for_layer, GGML_TYPE_F32, batch_len * pli_dim);

	// Gather the scattered PLI data
	get_per_layer_input_for_layer(&per_layer_input_for_layer, &ctx->mem.per_layer_inputs, layer_idx, batch_len,
				      ctx->model->num_layers, pli_dim);

	// Calculate the refinement_residual using the active slice from the corrected states
	dispatch_pli_gating(
		ctx, &refinement_residual,
		&final_layer_output[DATA_ACTIVE_IDX], // The input is ALWAYS the active data path's corrected state
		&per_layer_input_for_layer, l, batch_len);

	// Apply the refinement to the INACTIVE states
	for (int j = 0; j < ctx->model->altup_num_inputs; j++) {
		if (j == DATA_ACTIVE_IDX)
			continue;

		dispatch_apply_residual(&final_layer_output[j], &refinement_residual, batch_len * embed_dim);
	}

	// Commit the temporary buffer back to the main state
	for (int i = 0; i < ctx->model->altup_num_inputs; i++) {
		memcpy(ctx->mem.altup_hidden_states[i].data, final_layer_output[i].data,
		       batch_len * embed_dim * ggml_type_size(final_layer_output[i].type));

		free_memtype(&final_layer_output[i]);
	}

	free_memtype(&refinement_residual);
	free_memtype(&per_layer_input_for_layer);

	return 0;
};

int build_prompt_gemma3n(struct TIEContext *ctx, int *prompt_tokens, int *user_text_tokens, int user_text_token_count,
			 bool has_image)
{
	ModelDef *def = ctx->model->def;
	int prompt_len = 0;

	// BOS
	if (ctx->model->add_bos_token == 1 && ctx->model->bos_token_sent == 0) {
		prompt_tokens[prompt_len++] = ctx->model->bos_token_id;
		ctx->model->bos_token_sent = 1;
	}

	// Start of User Turn
	prompt_tokens[prompt_len++] = def->params.sot_token_id;
	prompt_tokens[prompt_len++] = def->params.role_user_token_id;
	prompt_tokens[prompt_len++] = def->params.newline_token_id;

	// User's Text Message
	memcpy(&prompt_tokens[prompt_len], user_text_tokens, user_text_token_count * sizeof(int));
	prompt_len += user_text_token_count;

	// End of User Turn & Start of Model Turn
	prompt_tokens[prompt_len++] = def->params.eot_token_id;
	prompt_tokens[prompt_len++] = def->params.newline_token_id;
	prompt_tokens[prompt_len++] = def->params.sot_token_id;
	prompt_tokens[prompt_len++] = def->params.role_model_token_id;
	prompt_tokens[prompt_len++] = def->params.newline_token_id;

	return prompt_len;
}