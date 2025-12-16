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

int transformer_layer_qwen3(struct TIEContext *ctx, int layer_idx, int batch_len);
int build_prompt_qwen3(struct TIEContext *ctx, int *prompt_tokens, int *user_text_tokens, int user_text_token_count,
		       bool has_image);


static const TensorDef QWEN3_GLOBAL_TENSORS[] = {
	DECLARE_GLOBAL_TENSORS_BASE_DEFS,
};

static const TensorDef QWEN3_LAYER_TENSORS[] = {
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
};

static const TensorDef QWEN3_MOE_LAYER_TENSORS[] = {
	{"blk.%u.attn_q.weight", offsetof(LayerWeights, attn_q), FLAG_NONE},
	{"blk.%u.attn_k.weight", offsetof(LayerWeights, attn_k), FLAG_NONE},
	{"blk.%u.attn_v.weight", offsetof(LayerWeights, attn_v), FLAG_NONE},
	{"blk.%u.attn_norm.weight", offsetof(LayerWeights, attn_norm), FLAG_NONE},
	{"blk.%u.attn_q_norm.weight", offsetof(LayerWeights, attn_q_norm), FLAG_NONE},
	{"blk.%u.attn_k_norm.weight", offsetof(LayerWeights, attn_k_norm), FLAG_NONE},
	{"blk.%u.attn_output.weight", offsetof(LayerWeights, attn_out), FLAG_NONE},
	{"blk.%u.ffn_norm.weight", offsetof(LayerWeights, ffn_norm), FLAG_NONE},
	{"blk.%u.ffn_gate_inp.weight", offsetof(LayerWeights, ffn_gate_inp), FLAG_NONE},
	{"blk.%u.ffn_gate_exps.weight", offsetof(LayerWeights, ffn_gate_exps), FLAG_NONE},
	{"blk.%u.ffn_down_exps.weight", offsetof(LayerWeights, ffn_down_exps), FLAG_NONE},
	{"blk.%u.ffn_up_exps.weight", offsetof(LayerWeights, ffn_up_exps), FLAG_NONE},
};

static const MetadataDef QWEN3_METADATA_DEFS[] = {
	DECLARE_LLAMA_BASE_METADATA_DEFS,
	// (MoE)
	{"%s.expert_count", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, expert_count), false, true},
	{"%s.expert_used_count", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, expert_used_count), false, true},
	{"%s.expert_feed_forward_length", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, expert_ffn_dim), false,
	 true},
	{"%s.rope.dimension_sections", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, mrope_sections), true, true},
	DECLARE_TOKENIZER_BASE_METADATA_DEFS,
};

static const BufferDef QWEN3_BUFFERS[] = {
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
	{offsetof(MemLayout, pos_ids), SIZE_POS_IDS, GGML_TYPE_I32, FLAG_NONE},
};

static const TokenizeDef QWEN3_TOKENIZE_DEF = {
	.token_detect_specials = 1,
	.token_load_merges = 1,
	.token_load_scores = 0,
};

ModelDef QWEN3_DEF = {.arch = ARCH_QWEN3,
		      .name = "Qwen3",
		      .params =
			      {
				      .sot_token_id = 151644,
				      .eot_token_id = 151645,
				      .eos_token_id = 151645,
				      .newline_token_id = 198,
				      .role_user_token_id = 872,
				      .role_model_token_id = 77091,
				      .vision_start_token_id = -1,
				      .vision_end_token_id = -1,
				      .vision_embed_token_id = -1,

			      },
		      .interface =
			      {
				      //			.build_system_prompt = build_system_prompt_qwen3,
				      .tokenize_encode = tokenize_bpe,
				      .tokenize_decode = decode_token_bpe,
				      .build_prompt = build_prompt_qwen3,
				      .transformer_layer = transformer_layer_qwen3,
				      .build_rope_cache = build_rope_cache_global,
			      },
		      DECLARE_LANGUAGE_MODEL_DEF(QWEN3, QWEN3)};

ModelDef DEEPSEEK_QWEN3_DEF = {
	.arch = ARCH_DEEPSEEK_QWEN3,
	.name = "Deepseek-R1-0528-Qwen3",
	.params =
		{
			.sot_token_id = -1,
			.eot_token_id = -1,
			.eos_token_id = 151645,
			.newline_token_id = -1,
			.role_user_token_id = 151669,
			.role_model_token_id = 151670,
			.vision_start_token_id = -1,
			.vision_end_token_id = -1,
			.vision_embed_token_id = -1,

		},
	.interface =
		{
			//			.build_system_prompt = build_system_prompt_qwen3,
			.tokenize_encode = tokenize_bpe,
			.tokenize_decode = decode_token_bpe,
			.build_prompt = build_prompt_qwen3,
			.transformer_layer = transformer_layer_qwen3,
			.build_rope_cache = build_rope_cache_global,
		},
	DECLARE_LANGUAGE_MODEL_DEF(QWEN3, QWEN3)};

ModelDef QWEN3_MOE_DEF = {.arch = ARCH_QWEN3_MOE,
			  .name = "Qwen3-MoE",
			  .params =
				  {
					  .is_moe = 1,
					  .sot_token_id = 151644,
					  .eot_token_id = 151645,
					  .eos_token_id = 151645,
					  .newline_token_id = 198,
					  .role_user_token_id = 872,
					  .role_model_token_id = 77091,
					  .vision_start_token_id = -1,
					  .vision_end_token_id = -1,
					  .vision_embed_token_id = -1,
				  },
			  .interface =
				  {
					  //			.build_system_prompt = build_system_prompt_qwen3,
					  .tokenize_encode = tokenize_bpe,
					  .tokenize_decode = decode_token_bpe,
					  .build_prompt = build_prompt_qwen3,
					  .transformer_layer = transformer_layer_qwen3,
					  .build_rope_cache = build_rope_cache_global,
				  },
			  DECLARE_LANGUAGE_MODEL_DEF(QWEN3, QWEN3_MOE)};

ModelDef QWEN3VL_DEF = {.arch = ARCH_QWEN3VL,
			.name = "Qwen3-VL",
			.params =
				{
					.sot_token_id = 151644,
					.eot_token_id = 151645,
					.eos_token_id = 151645,
					.newline_token_id = 198,
					.role_user_token_id = 872,
					.role_model_token_id = 77091,
					.vision_start_token_id = 151652, // <|vision_start|>
					.vision_end_token_id = 151653,	 // <|vision_end|>
					.vision_embed_token_id = 151654, // <|vision_pad|>
				},
			.interface =
				{
					//			.build_system_prompt = build_system_prompt_qwen3,
					.tokenize_encode = tokenize_bpe,
					.tokenize_decode = decode_token_bpe,
					.build_prompt = build_prompt_qwen3,
					.transformer_layer = transformer_layer_qwen3,
					.process_image_vision = process_image_vision_qwen3vl,
					.build_rope_cache = build_rope_cache_dynamic,
				},
			DECLARE_LANGUAGE_MODEL_DEF(QWEN3, QWEN3)};

ModelDef QWEN3VL_MOE_DEF = {.arch = ARCH_QWEN3VL_MOE,
			    .name = "Qwen3-VL-MoE",
			    .params =
				    {
					    .is_moe = 1,
					    .sot_token_id = 151644,
					    .eot_token_id = 151645,
					    .eos_token_id = 151645,
					    .newline_token_id = 198,
					    .role_user_token_id = 872,
					    .role_model_token_id = 77091,
					    .vision_start_token_id = 151652, // <|vision_start|>
					    .vision_end_token_id = 151653,   // <|vision_end|>
					    .vision_embed_token_id = 151654, // <|vision_pad|>
				    },
			    .interface =
				    {
					    //			.build_system_prompt = build_system_prompt_qwen3,
					    .tokenize_encode = tokenize_bpe,
					    .tokenize_decode = decode_token_bpe,
					    .build_prompt = build_prompt_qwen3,
					    .transformer_layer = transformer_layer_qwen3,
					    .process_image_vision = process_image_vision_qwen3vl,
					    .build_rope_cache = build_rope_cache_dynamic,
				    },
			    DECLARE_LANGUAGE_MODEL_DEF(QWEN3, QWEN3_MOE)};

static const TensorDef QWEN3VL_CLIP_GLOBAL_TENSORS[] = {
	{"v.patch_embd.bias", offsetof(VisionModel, patch_embd_bias), FLAG_NONE},
	{"v.patch_embd.weight", offsetof(VisionModel, patch_embd), FLAG_NONE},
	{"v.patch_embd.weight.1", offsetof(VisionModel, patch_embd_1), FLAG_NONE},
	{"v.post_ln.bias", offsetof(VisionModel, post_ln_bias), FLAG_NONE},
	{"v.post_ln.weight", offsetof(VisionModel, post_ln), FLAG_NONE},
	{"v.position_embd.weight", offsetof(VisionModel, position_embd), FLAG_NONE},

	// Main Projector (Tensors 306-309)
	{"mm.0.bias", offsetof(VisionModel, mm_0_bias), FLAG_NONE},
	{"mm.0.weight", offsetof(VisionModel, mm_0_weight), FLAG_NONE},
	{"mm.2.bias", offsetof(VisionModel, mm_2_bias), FLAG_NONE},
	{"mm.2.weight", offsetof(VisionModel, mm_2_weight), FLAG_NONE},
};

static const TensorDef QWEN3VL_CLIP_LAYER_TENSORS[] = {
	{"v.blk.%u.attn_out.bias", offsetof(VisionLayerWeights, attn_out_bias), FLAG_NONE},
	{"v.blk.%u.attn_out.weight", offsetof(VisionLayerWeights, attn_out), FLAG_NONE},
	{"v.blk.%u.ffn_up.bias", offsetof(VisionLayerWeights, ffn_up_bias), FLAG_NONE},
	{"v.blk.%u.ffn_up.weight", offsetof(VisionLayerWeights, ffn_up), FLAG_NONE},
	{"v.blk.%u.ffn_down.bias", offsetof(VisionLayerWeights, ffn_down_bias), FLAG_NONE},
	{"v.blk.%u.ffn_down.weight", offsetof(VisionLayerWeights, ffn_down), FLAG_NONE},
	{"v.blk.%u.ln1.bias", offsetof(VisionLayerWeights, ln1_bias), FLAG_NONE},
	{"v.blk.%u.ln1.weight", offsetof(VisionLayerWeights, ln1), FLAG_NONE},
	{"v.blk.%u.ln2.bias", offsetof(VisionLayerWeights, ln2_bias), FLAG_NONE},
	{"v.blk.%u.ln2.weight", offsetof(VisionLayerWeights, ln2), FLAG_NONE},
	{"v.blk.%u.attn_qkv.bias", offsetof(VisionLayerWeights, attn_qkv_bias), FLAG_NONE},
	{"v.blk.%u.attn_qkv.weight", offsetof(VisionLayerWeights, attn_qkv), FLAG_NONE},

	// DeepStack Projector
	{"v.deepstack.%u.fc1.bias", offsetof(VisionLayerWeights, ds_fc1_bias), FLAG_OPTIONAL},
	{"v.deepstack.%u.fc1.weight", offsetof(VisionLayerWeights, ds_fc1_weight), FLAG_OPTIONAL},
	{"v.deepstack.%u.fc2.bias", offsetof(VisionLayerWeights, ds_fc2_bias), FLAG_OPTIONAL},
	{"v.deepstack.%u.fc2.weight", offsetof(VisionLayerWeights, ds_fc2_weight), FLAG_OPTIONAL},
	{"v.deepstack.%u.norm.bias", offsetof(VisionLayerWeights, ds_norm_bias), FLAG_OPTIONAL},
	{"v.deepstack.%u.norm.weight", offsetof(VisionLayerWeights, ds_norm_weight), FLAG_OPTIONAL},
};

static const MetadataDef QWEN3VL_CLIP_METADATA_DEFS[] = {
	DECLARE_BASE_VISION_METADATA_DEFS,
	{"%s.vision.spatial_merge_size", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(VisionModel, spatial_merge_size),
	 false, false},
	{"%s.vision.is_deepstack_layers", GGUF_METADATA_VALUE_TYPE_BOOL, offsetof(VisionModel, is_deepstack_layers),
	 true, false},
};

static const BufferDef QWEN3VL_CLIP_BUFFERS[] = {
	{offsetof(MemLayoutVision, image_raw), SIZE_VISION_IMAGE_RAW, GGML_TYPE_F32, FLAG_NONE},
	{offsetof(MemLayoutVision, patch_embeds), SIZE_VISION_PATCH_EMBEDS, INTERNAL_MEMORY_TYPE, FLAG_NONE},
	{offsetof(MemLayoutVision, hidden_state), SIZE_VISION_SEQ_LEN_X_EMBED_DIM, INTERNAL_MEMORY_TYPE, FLAG_NONE},
	{offsetof(MemLayoutVision, QKV_fused), SIZE_VISION_SEQ_LEN_X_QKV_DIM_X3, INTERNAL_MEMORY_TYPE, FLAG_NONE},
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
	{offsetof(MemLayoutVision, merger_norm_buf), SIZE_VISION_MERGER_TEMP, GGML_TYPE_F32, FLAG_NONE},
	{offsetof(MemLayoutVision, merger_fc1_buf), SIZE_VISION_MERGER_TEMP, GGML_TYPE_F32, FLAG_NONE},
};

ModelDef QWEN3VL_CLIP_DEF = {.arch = ARCH_CLIP_VISION,
			     .projector = VISION_PROJECTOR_QWEN3VL,
			     .name = "Qwen3-VL-clip",
			     .params = {},
			     .interface = {},
			     DECLARE_VISION_MODEL_DEF(QWEN3VL_CLIP)};

/*
 * -------------------------------------------------------------------------
 * QWEN3 SPECIFIC IMPLEMENTATION
 * -------------------------------------------------------------------------
 */

int transformer_layer_qwen3(struct TIEContext *ctx, int layer_idx, int batch_len)
{
	LayerWeights *l = &ctx->model->layers[layer_idx];
	int kv_dim = ctx->model->num_kv_heads * ctx->model->head_dim;
	int q_dim = ctx->model->num_heads * ctx->model->head_dim;
	AttentionType attn_type = ATTN_TYPE_GLOBAL;
	RopeCacheType *active_rope_cache = ctx->model->rope_cache_global;
	int sink_len = 4;


	// The absolute starting position for this batch
	int start_pos = ctx->kv_pos;

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
			if (ctx->model->use_mrope == 0) {
				dispatch_apply_rope_cache(active_rope_cache, &Q_slice, absolute_pos,
							  ctx->model->head_dim);
			} else {
				dispatch_apply_mrope_cache(active_rope_cache, &Q_slice, absolute_pos,
							   ctx->model->head_dim);
			}
		}

		for (int h = 0; h < ctx->model->num_kv_heads; h++) {
			MemType K_slice = mem_slice(&ctx->mem.K, (size_t)i * kv_dim + h * ctx->model->head_dim);

			if (l->attn_k_norm.mem.data) {
				dispatch_rms_norm(&K_slice, &l->attn_k_norm, &K_slice, ctx->model->head_dim,
						  ctx->model->norm_eps);
			}

			// Use the absolute position for RoPE
			if (ctx->model->use_mrope == 0) {
				dispatch_apply_rope_cache(active_rope_cache, &K_slice, absolute_pos,
							  ctx->model->head_dim);
			} else {
				dispatch_apply_mrope_cache(active_rope_cache, &K_slice, absolute_pos,
							   ctx->model->head_dim);
			}
		}
	}

	// Store K/V to cache
	dispatch_store_KV_cache(ctx, layer_idx, start_pos, batch_len, sink_len);

	// Multi-Head Attention Calculation
	attention(ctx, batch_len, layer_idx, layer_idx, start_pos, attn_type, attention_worker, sink_len);

	// Output projection
	dispatch_mat_mat(ctx, &ctx->mem.attn_output, &l->attn_out, &ctx->mem.attn_proj_output, batch_len, q_dim,
			 ctx->model->embed_dim, true);

	// Add residual
	dispatch_apply_residual(&ctx->mem.hidden_state, &ctx->mem.attn_proj_output, batch_len * ctx->model->embed_dim);

	// FFN Block
	// RMSNorm
	for (int i = 0; i < batch_len; i++) {
		size_t offset = (size_t)i * ctx->model->embed_dim;

		// Create slices for the specific token being processed
		MemType hidden_state_slice = mem_slice(&ctx->mem.hidden_state, offset);
		MemType normed_ffn_input_slice = mem_slice(&ctx->mem.normed_ffn_input, offset);

		dispatch_rms_norm(&hidden_state_slice, &l->ffn_norm, &normed_ffn_input_slice, ctx->model->embed_dim,
				  ctx->model->norm_eps);
	}

	// MoE
	if (ctx->model->is_moe) {

		// Get the size of a single quantization block for the expert tensors
		size_t block_size_bytes = ggml_block_size(l->ffn_up_exps.mem.type);
		if (block_size_bytes == 0) {
			return -1;
		}

		int num_threads = thread_pool->num_threads;
		expert_task_t tasks[num_threads];

		for (int i = 0; i < batch_len; i++) {

			// Pointers for the current token
			MemType normed_input_for_token_i =
				mem_slice(&ctx->mem.normed_ffn_input, (size_t)i * ctx->model->embed_dim);

			// Create a slice for the destination buffer
			MemType ffn_out_slice = mem_slice(&ctx->mem.ffn_down_output, (size_t)i * ctx->model->embed_dim);

			// Use a per-thread scratch buffer for FP32 accumulation
			MemType *ffn_out_fp32_scratch = &ctx->mem.expert_out_fp32;
			float *ffn_out_fp32_token_buffer = ctx->mem.expert_out_fp32.data;

			// Route, Select, and Gate
			// The input type for the router is the intermediate type, but the output scores are always
			// FP32.
			dispatch_mat_vec(ctx, &normed_input_for_token_i, &l->ffn_gate_inp, &ctx->mem.expert_scores,
					 ctx->model->embed_dim, ctx->model->expert_count, false);

			ExpertChoice top_experts[ctx->model->expert_used_count];
			find_top_k((float *)ctx->mem.expert_scores.data, ctx->model->expert_count,
				   ctx->model->expert_used_count, top_experts);

			float gate_values[ctx->model->expert_used_count];

			for (int j = 0; j < ctx->model->expert_used_count; j++)
				gate_values[j] = top_experts[j].score;

			softmax(gate_values, ctx->model->expert_used_count);

			// Parallel Expert Processing
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

			// Accumulate results in the FP32 temporary buffer
			memset(ffn_out_fp32_scratch->data, 0, ctx->model->embed_dim * sizeof(float));

			for (int j = 0; j < ctx->model->expert_used_count; j++) {
				float gate_val = gate_values[j];
				float *expert_result = ctx->mem.expert_outputs[j].data;
				for (int k = 0; k < ctx->model->embed_dim; k++) {
					ffn_out_fp32_token_buffer[k] += gate_val * expert_result[k];
				}
			}

			// Convert the final FP32 result to the destination format
			dispatch_convert(ffn_out_fp32_scratch, &ffn_out_slice, ctx->model->embed_dim);
		}

	} else { // DENSE FFN

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
	}

	// Add residual
	dispatch_apply_residual(&ctx->mem.hidden_state, &ctx->mem.ffn_down_output, batch_len * ctx->model->embed_dim);

	return 0;
}

int build_prompt_qwen3(struct TIEContext *ctx, int *prompt_tokens, int *user_text_tokens, int user_text_token_count,
		       bool has_image)
{
	VisionModel *vm = ctx->model_vision;
	ModelDef *def = ctx->model->def;
	//	ModelDef *vdef = ctx->model->def;
	int prompt_len = 0;


	// BOS (only for the very first turn)
	if (ctx->model->add_bos_token == 1 && ctx->model->bos_token_sent == 0) {
		prompt_tokens[prompt_len++] = ctx->model->bos_token_id;
		ctx->model->bos_token_sent = 1;
	}

	// Start of User Turn
	if (def->params.sot_token_id != -1)
		prompt_tokens[prompt_len++] = def->params.sot_token_id;

	prompt_tokens[prompt_len++] = def->params.role_user_token_id;

	if (def->params.newline_token_id != -1)
		prompt_tokens[prompt_len++] = def->params.newline_token_id;

	// User's Text Message
	memcpy(&prompt_tokens[prompt_len], user_text_tokens, user_text_token_count * sizeof(int));
	prompt_len += user_text_token_count;

	if (has_image) {

		// Use Dynamic Dimensions from the loaded image
		int raw_w = ctx->vision_mem.image_raw_width;
		int raw_h = ctx->vision_mem.image_raw_height;

		// Calculate patches
		int w_patches = raw_w / vm->patch_size;
		int h_patches = raw_h / vm->patch_size;

		// Calculate merged grid size
		// Qwen3-VL reduces resolution by spatial_merge_size
		int merged_w = w_patches / vm->spatial_merge_size;
		int merged_h = h_patches / vm->spatial_merge_size;

		int num_patches = merged_w * merged_h;

		printf("Dynamic Vision Tokens: %dx%d image -> %dx%d patches -> %d tokens\n", raw_w, raw_h, merged_w,
		       merged_h, num_patches);

		// <|vision_start|>
		prompt_tokens[prompt_len++] = def->params.vision_start_token_id;

		int patch_token = def->params.vision_embed_token_id;
		for (int i = 0; i < num_patches; i++) {
			prompt_tokens[prompt_len++] = patch_token;
		}

		// <|vision_end|>
		prompt_tokens[prompt_len++] = def->params.vision_end_token_id;

		// newline
		prompt_tokens[prompt_len++] = def->params.newline_token_id;
	}

	// End of User Turn & Start of Model Turn
	if (def->params.eot_token_id != -1)
		prompt_tokens[prompt_len++] = def->params.eot_token_id;

	if (def->params.newline_token_id != -1)
		prompt_tokens[prompt_len++] = def->params.newline_token_id;

	if (def->params.sot_token_id != -1)
		prompt_tokens[prompt_len++] = def->params.sot_token_id;

	prompt_tokens[prompt_len++] = def->params.role_model_token_id;

	if (def->params.newline_token_id != -1)
		prompt_tokens[prompt_len++] = def->params.newline_token_id;

	return prompt_len;
}
