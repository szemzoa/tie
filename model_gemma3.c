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

int transformer_layer_gemma3(struct TIEContext *ctx, int layer_idx, int batch_len);
int build_prompt_gemma3(struct TIEContext *ctx, int *prompt_tokens, int *user_text_tokens, int user_text_token_count,
			bool has_image);

static const TensorDef GEMMA3_GLOBAL_TENSORS[] = {
	DECLARE_GLOBAL_TENSORS_BASE_DEFS,
};

static const TensorDef GEMMA3_LAYER_TENSORS[] = {
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
};

static const MetadataDef GEMMA3_METADATA_DEFS[] = {
	DECLARE_LLAMA_BASE_METADATA_DEFS,
	{"%s.attention.sliding_window", GGUF_METADATA_VALUE_TYPE_FLOAT32, offsetof(Model, attn_sliding_window), false,
	 false},
	DECLARE_TOKENIZER_BASE_METADATA_DEFS,
};

static const BufferDef GEMMA3_BUFFERS[] = {
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
};

static const TokenizeDef GEMMA3_TOKENIZE_DEF = {
	.token_detect_specials = 0,
	.token_load_merges = 0,
	.token_load_scores = 1,
};

ModelDef GEMMA3_DEF = {.arch = ARCH_GEMMA3,
		       .name = "Gemma-3",
		       .params =
			       {
				       .sot_token_id = 105,
				       .eot_token_id = 106,
				       .eos_token_id = 106,
				       .newline_token_id = 107,
				       .role_user_token_id = 2364,
				       .role_model_token_id = 4368,
				       .double_newline_token_id = 108,
				       .vision_start_token_id = 255999,
				       .vision_end_token_id = 256000,
				       .vision_embed_token_id = 262144,
			       },
		       .interface =
			       {
				       .tokenize_encode = tokenize_sp,
				       .tokenize_decode = decode_token_sp,
				       .build_prompt = build_prompt_gemma3,
				       .embedding_scale = embedding_scale_gemma3,
				       .transformer_layer = transformer_layer_gemma3,
				       .process_image_vision = process_image_vision_gemma3,
				       .build_rope_cache = build_rope_cache_shared,
			       },
		       DECLARE_LANGUAGE_MODEL_DEF(GEMMA3, GEMMA3)};

static const TensorDef GEMMA3_CLIP_GLOBAL_TENSORS[] = {
	{"mm.input_projection.weight", offsetof(VisionModel, input_projection), FLAG_NONE},
	{"mm.soft_emb_norm.weight", offsetof(VisionModel, soft_embd_norm), FLAG_NONE},
	{"v.patch_embd.bias", offsetof(VisionModel, patch_embd_bias), FLAG_NONE},
	{"v.patch_embd.weight", offsetof(VisionModel, patch_embd), FLAG_NONE},
	{"v.position_embd.weight", offsetof(VisionModel, position_embd), FLAG_NONE},
	{"v.post_ln.bias", offsetof(VisionModel, post_ln_bias), FLAG_NONE},
	{"v.post_ln.weight", offsetof(VisionModel, post_ln), FLAG_NONE},
};

static const TensorDef GEMMA3_CLIP_LAYER_TENSORS[] = {
	{"v.blk.%u.ln1.bias", offsetof(VisionLayerWeights, ln1_bias), FLAG_NONE},
	{"v.blk.%u.ln1.weight", offsetof(VisionLayerWeights, ln1), FLAG_NONE},
	{"v.blk.%u.ln2.bias", offsetof(VisionLayerWeights, ln2_bias), FLAG_NONE},
	{"v.blk.%u.ln2.weight", offsetof(VisionLayerWeights, ln2), FLAG_NONE},
	{"v.blk.%u.attn_k.bias", offsetof(VisionLayerWeights, attn_k_bias), FLAG_NONE},
	{"v.blk.%u.attn_k.weight", offsetof(VisionLayerWeights, attn_k), FLAG_NONE},
	{"v.blk.%u.attn_q.bias", offsetof(VisionLayerWeights, attn_q_bias), FLAG_NONE},
	{"v.blk.%u.attn_q.weight", offsetof(VisionLayerWeights, attn_q), FLAG_NONE},
	{"v.blk.%u.attn_v.bias", offsetof(VisionLayerWeights, attn_v_bias), FLAG_NONE},
	{"v.blk.%u.attn_v.weight", offsetof(VisionLayerWeights, attn_v), FLAG_NONE},
	{"v.blk.%u.attn_out.bias", offsetof(VisionLayerWeights, attn_out_bias), FLAG_NONE},
	{"v.blk.%u.attn_out.weight", offsetof(VisionLayerWeights, attn_out), FLAG_NONE},
	{"v.blk.%u.ffn_up.bias", offsetof(VisionLayerWeights, ffn_up_bias), FLAG_NONE},
	{"v.blk.%u.ffn_up.weight", offsetof(VisionLayerWeights, ffn_up), FLAG_NONE},
	{"v.blk.%u.ffn_down.bias", offsetof(VisionLayerWeights, ffn_down_bias), FLAG_NONE},
	{"v.blk.%u.ffn_down.weight", offsetof(VisionLayerWeights, ffn_down), FLAG_NONE},
};

static const MetadataDef GEMMA3_CLIP_METADATA_DEFS[] = {DECLARE_BASE_VISION_METADATA_DEFS};

static const BufferDef GEMMA3_CLIP_BUFFERS[] = {
	{offsetof(MemLayoutVision, image_raw), SIZE_VISION_IMAGE_RAW, GGML_TYPE_F32, FLAG_NONE},
	{offsetof(MemLayoutVision, patch_embeds), SIZE_VISION_PATCH_EMBEDS, INTERNAL_MEMORY_TYPE, FLAG_NONE},
	{offsetof(MemLayoutVision, hidden_state), SIZE_VISION_SEQ_LEN_X_EMBED_DIM, INTERNAL_MEMORY_TYPE, FLAG_NONE},
	{offsetof(MemLayoutVision, Q), SIZE_VISION_SEQ_LEN_X_EMBED_DIM, INTERNAL_MEMORY_TYPE, FLAG_NONE},
	{offsetof(MemLayoutVision, K), SIZE_VISION_SEQ_LEN_X_EMBED_DIM, INTERNAL_MEMORY_TYPE, FLAG_NONE},
	{offsetof(MemLayoutVision, V), SIZE_VISION_SEQ_LEN_X_EMBED_DIM, INTERNAL_MEMORY_TYPE, FLAG_NONE},
	{offsetof(MemLayoutVision, residual_scratch), SIZE_VISION_SEQ_LEN_X_EMBED_DIM, INTERNAL_MEMORY_TYPE, FLAG_NONE},
	{offsetof(MemLayoutVision, normed_input), SIZE_VISION_SEQ_LEN_X_EMBED_DIM, INTERNAL_MEMORY_TYPE, FLAG_NONE},
	{offsetof(MemLayoutVision, qkv_proj_output), SIZE_VISION_SEQ_LEN_X_QKV_DIM_X3, INTERNAL_MEMORY_TYPE, FLAG_NONE},
	{offsetof(MemLayoutVision, attn_output), SIZE_VISION_SEQ_LEN_X_EMBED_DIM, INTERNAL_MEMORY_TYPE, FLAG_NONE},
	{offsetof(MemLayoutVision, attn_proj_output), SIZE_VISION_SEQ_LEN_X_EMBED_DIM, INTERNAL_MEMORY_TYPE, FLAG_NONE},
	{offsetof(MemLayoutVision, ffn_up_output), SIZE_VISION_SEQ_LEN_X_FFN_DIM, INTERNAL_MEMORY_TYPE, FLAG_NONE},
	{offsetof(MemLayoutVision, ffn_down_output), SIZE_VISION_SEQ_LEN_X_EMBED_DIM, INTERNAL_MEMORY_TYPE, FLAG_NONE},
	{offsetof(MemLayoutVision, pooled_embeddings), SIZE_VISION_POOLED_EMBEDS, INTERNAL_MEMORY_TYPE, FLAG_NONE},
	{offsetof(MemLayoutVision, projected_embeddings), SIZE_VISION_PROJ_EMBEDS, INTERNAL_MEMORY_TYPE, FLAG_NONE},
};

ModelDef GEMMA3_CLIP_DEF = {.arch = ARCH_CLIP_VISION,
			    .projector = VISION_PROJECTOR_GEMMA3,
			    .name = "Gemma3-clip",
			    .params =
				    {
					    .proj_scale_factor = 4,
				    },
			    .interface = {},
			    DECLARE_VISION_MODEL_DEF(GEMMA3_CLIP)};


/*
 * -------------------------------------------------------------------------
 * GEMMA-3 SPECIFIC IMPLEMENTATION
 * -------------------------------------------------------------------------
 */

void embedding_scale_gemma3(struct TIEContext *ctx, MemType *hidden_state_slice)
{
	float *hidden_data_fp32;

	float scale = sqrtf((float)ctx->model->embed_dim);
	hidden_data_fp32 = (float *)hidden_state_slice->data;
	for (int j = 0; j < ctx->model->embed_dim; j++) {
		float val = hidden_data_fp32[j];
		val *= scale;
		hidden_data_fp32[j] = val;
	}
}

int transformer_layer_gemma3(struct TIEContext *ctx, int layer_idx, int batch_len)
{
	LayerWeights *l = &ctx->model->layers[layer_idx];
	int kv_dim = ctx->model->num_kv_heads * ctx->model->head_dim;
	int q_dim = ctx->model->num_heads * ctx->model->head_dim;
	AttentionType attn_type;
	RopeCacheType *active_rope_cache;
	int sink_len = 4;

	// The absolute starting position for this batch
	int start_pos = ctx->kv_pos;

	// Determine the attention and rope cache type for the current layer
	if ((layer_idx + 1) % 6 == 0) {
		attn_type = ATTN_TYPE_GLOBAL;
		active_rope_cache = ctx->model->rope_cache_global; // Use the global cache
	} else {
		attn_type = ATTN_TYPE_LOCAL;
		active_rope_cache = ctx->model->rope_cache_local; // Use the local cache
	}

	// Save the residual for the attention block.
	// We use a scratch buffer to hold the original hidden_state.
	memcpy(ctx->mem.residual_stratch.data, ctx->mem.hidden_state.data,
	       batch_len * ctx->model->embed_dim * sizeof(float));

	// Attention Block
	// RMSNorm
	for (int i = 0; i < batch_len; i++) {
		size_t offset = (size_t)i * ctx->model->embed_dim;

		// Create slices for the specific token being processed
		MemType hidden_state_slice = mem_slice(&ctx->mem.hidden_state, offset);
		MemType normed_input_slice = mem_slice(&ctx->mem.normed_qkv_input, offset);

		dispatch_rms_norm(&hidden_state_slice, &l->attn_norm, &normed_input_slice, ctx->model->embed_dim,
				  ctx->model->norm_eps);
	}

	// Compute Q/K/V Matrices
	dispatch_mat_mat(ctx, &ctx->mem.normed_qkv_input, &l->attn_q, &ctx->mem.Q, batch_len, ctx->model->embed_dim,
			 q_dim, true);

	dispatch_mat_mat(ctx, &ctx->mem.normed_qkv_input, &l->attn_k, &ctx->mem.K, batch_len, ctx->model->embed_dim,
			 kv_dim, true);

	dispatch_mat_mat(ctx, &ctx->mem.normed_qkv_input, &l->attn_v, &ctx->mem.V, batch_len, ctx->model->embed_dim,
			 kv_dim, true);

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

		for (int h = 0; h < ctx->model->num_kv_heads; h++) {
			MemType K_slice = mem_slice(&ctx->mem.K, (size_t)i * kv_dim + h * ctx->model->head_dim);

			if (l->attn_k_norm.mem.data) {
				dispatch_rms_norm(&K_slice, &l->attn_k_norm, &K_slice, ctx->model->head_dim,
						  ctx->model->norm_eps);
			}

			// Use the absolute position for RoPE
			dispatch_apply_rope_cache(active_rope_cache, &K_slice, absolute_pos, ctx->model->head_dim);
		}
	}

	// Store K/V to cache
	dispatch_store_KV_cache(ctx, layer_idx, start_pos, batch_len, sink_len);

	// Multi-Head Attention Calculation
	attention(ctx, batch_len, layer_idx, layer_idx, start_pos, attn_type, attention_worker, sink_len);

	// Output projection
	dispatch_mat_mat(ctx, &ctx->mem.attn_output, &l->attn_out, &ctx->mem.attn_proj_output, batch_len, q_dim,
			 ctx->model->embed_dim, true);

	// POST-ATTENTION norm
	for (int i = 0; i < batch_len; i++) {
		size_t offset = (size_t)i * ctx->model->embed_dim;

		MemType attn_proj_slice = mem_slice(&ctx->mem.attn_proj_output, offset);

		dispatch_rms_norm(&attn_proj_slice, &l->post_attn_norm, &attn_proj_slice, ctx->model->embed_dim,
				  ctx->model->norm_eps);
	}

	// Add the attention residual.
	// Add the post-normed attention output to the original residual we saved.
	// The result is stored in `hidden_state`.
	dispatch_apply_residual(&ctx->mem.residual_stratch, &ctx->mem.attn_proj_output,
				batch_len * ctx->model->embed_dim);

	memcpy(ctx->mem.hidden_state.data, ctx->mem.residual_stratch.data,
	       batch_len * ctx->model->embed_dim * sizeof(float));


	// FFN Block
	// Save the residual for the FFN block.
	memcpy(ctx->mem.residual_stratch.data, ctx->mem.hidden_state.data,
	       batch_len * ctx->model->embed_dim * sizeof(float));

	// RMSNorm
	for (int i = 0; i < batch_len; i++) {
		size_t offset = (size_t)i * ctx->model->embed_dim;

		// Create slices for the specific token being processed
		MemType hidden_state_slice = mem_slice(&ctx->mem.hidden_state, offset);
		MemType normed_ffn_input_slice = mem_slice(&ctx->mem.normed_ffn_input, offset);

		dispatch_rms_norm(&hidden_state_slice, &l->ffn_norm, &normed_ffn_input_slice, ctx->model->embed_dim,
				  ctx->model->norm_eps);
	}

	// Gate + Up projections
	dispatch_mat_mat(ctx, &ctx->mem.normed_ffn_input, &l->ffn_gate, &ctx->mem.gate_proj_output, batch_len,
			 ctx->model->embed_dim, ctx->model->ffn_dim, true);
	dispatch_mat_mat(ctx, &ctx->mem.normed_ffn_input, &l->ffn_up, &ctx->mem.up_proj_output, batch_len,
			 ctx->model->embed_dim, ctx->model->ffn_dim, true);

	/* Call the interface activation function */
	dispatch_geglu_activation(&ctx->mem.gate_proj_output, &ctx->mem.up_proj_output,
				  batch_len * ctx->model->ffn_dim);

	// Down projection
	dispatch_mat_mat(ctx, &ctx->mem.gate_proj_output, &l->ffn_down, &ctx->mem.ffn_down_output, batch_len,
			 ctx->model->ffn_dim, ctx->model->embed_dim, true);

	// POST-FFN norm.
	for (int i = 0; i < batch_len; i++) {
		size_t offset = (size_t)i * ctx->model->embed_dim;
		MemType ffn_down_slice = mem_slice(&ctx->mem.ffn_down_output, offset);

		dispatch_rms_norm(&ffn_down_slice, &l->post_ffw_norm, &ffn_down_slice, ctx->model->embed_dim,
				  ctx->model->norm_eps);
	}

	// Second Residual Add.
	// Add the post-normed FFN output to the residual we saved
	dispatch_apply_residual(&ctx->mem.residual_stratch, &ctx->mem.ffn_down_output,
				batch_len * ctx->model->embed_dim);

	// Copy the correct final result back to hidden_state for the next layer.
	memcpy(ctx->mem.hidden_state.data, ctx->mem.residual_stratch.data,
	       batch_len * ctx->model->embed_dim * sizeof(float));

	return 0;
}

int build_prompt_gemma3(struct TIEContext *ctx, int *prompt_tokens, int *user_text_tokens, int user_text_token_count,
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

	if (has_image) {

		prompt_tokens[prompt_len++] = def->params.double_newline_token_id; // DOUBLE NEWLINE
		prompt_tokens[prompt_len++] = def->params.vision_start_token_id;   // <start_of_image>

		for (int i = 0; i < 256; ++i)
			prompt_tokens[prompt_len++] = def->params.vision_embed_token_id;

		prompt_tokens[prompt_len++] = def->params.vision_end_token_id;	   // <end_of_image>
		prompt_tokens[prompt_len++] = def->params.double_newline_token_id; // DOUBLE NEWLINE
	}

	// End of User Turn & Start of Model Turn
	prompt_tokens[prompt_len++] = def->params.eot_token_id;
	prompt_tokens[prompt_len++] = def->params.newline_token_id;
	prompt_tokens[prompt_len++] = def->params.sot_token_id;
	prompt_tokens[prompt_len++] = def->params.role_model_token_id;
	prompt_tokens[prompt_len++] = def->params.newline_token_id;

	return prompt_len;
}