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
	// Qwen-specific (MoE)
	{"%s.expert_count", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, expert_count), false, true},
	{"%s.expert_used_count", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, expert_used_count), false, true},
	{"%s.expert_feed_forward_length", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, expert_ffn_dim), false, true},
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

ModelDef QWEN3_DEF = {
	.arch = ARCH_QWEN3,
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
		        .vision_end_token_id   = -1,
		        .vision_embed_token_id = -1,

		},
	.interface =
		{
//			.build_system_prompt = build_system_prompt_qwen3,
			.tokenize_prompt = tokenize_bpe,
			.decode_token = decode_token_bpe,
			.prepare_next_token = prepare_next_token_standard,
			.embedding_scale = NULL,
			.transformer_layer = transformer_layer_qwen3,
			.build_rope_cache = build_rope_cache_global,
		},
	DECLARE_LANGUAGE_MODEL_DEF(QWEN3, QWEN3)
};

ModelDef QWEN3_MOE_DEF = {
	.arch = ARCH_QWEN3_MOE,
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
		        .vision_end_token_id   = -1,
		        .vision_embed_token_id = -1,
		},
	.interface =
		{
//			.build_system_prompt = build_system_prompt_qwen3,
			.tokenize_prompt = tokenize_bpe,
			.decode_token = decode_token_bpe,
			.prepare_next_token = prepare_next_token_standard,
			.embedding_scale = NULL,
			.transformer_layer = transformer_layer_qwen3,
			.build_rope_cache = build_rope_cache_global,
		},
	DECLARE_LANGUAGE_MODEL_DEF(QWEN3, QWEN3_MOE)
};

ModelDef QWEN3VL_DEF = {
	.arch = ARCH_QWEN3VL,
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
		        .vision_end_token_id   = 151653, // <|vision_end|>
		        .vision_embed_token_id = 151654, // <|vision_pad|>
		},
	.interface =
		{
//			.build_system_prompt = build_system_prompt_qwen3,
			.tokenize_prompt = tokenize_bpe,
			.decode_token = decode_token_bpe,
			.prepare_next_token = prepare_next_token_standard,
			.embedding_scale = NULL,
			.transformer_layer = transformer_layer_qwen3,
			.build_vision_tokens = build_vision_tokens_qwen3vl,
			.process_image_vision = process_image_vision_qwen3vl,
			.build_rope_cache = build_rope_cache_dynamic,
		},
	DECLARE_LANGUAGE_MODEL_DEF(QWEN3, QWEN3)
};

ModelDef QWEN3VL_MOE_DEF = {
	.arch = ARCH_QWEN3VL_MOE,
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
		        .vision_end_token_id   = 151653, // <|vision_end|>
		        .vision_embed_token_id = 151654, // <|vision_pad|>
		},
	.interface =
		{
//			.build_system_prompt = build_system_prompt_qwen3,
			.tokenize_prompt = tokenize_bpe,
			.decode_token = decode_token_bpe,
			.prepare_next_token = prepare_next_token_standard,
			.embedding_scale = NULL,
			.transformer_layer = transformer_layer_qwen3,
			.build_vision_tokens = build_vision_tokens_qwen3vl,
			.process_image_vision = process_image_vision_qwen3vl,
			.build_rope_cache = build_rope_cache_dynamic,
		},
	DECLARE_LANGUAGE_MODEL_DEF(QWEN3, QWEN3_MOE)
};

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
	{"%s.attention.sliding_window", GGUF_METADATA_VALUE_TYPE_FLOAT32, offsetof(Model, attn_sliding_window), false, false},
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

ModelDef GEMMA3_DEF = {
	.arch = ARCH_GEMMA3,
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
			.tokenize_prompt = tokenize_sp,
			.decode_token = decode_token_sp,
			.prepare_next_token = prepare_next_token_standard,
			.embedding_scale = embedding_scale_gemma3,
			.transformer_layer = transformer_layer_gemma3,
			.build_vision_tokens = build_vision_tokens_gemma3,
			.process_image_vision = process_image_vision_gemma3,
			.build_rope_cache = build_rope_cache_shared,
		},
	DECLARE_LANGUAGE_MODEL_DEF(GEMMA3, GEMMA3)
};

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
	{"%s.attention.layer_norm_rms_epsilon", GGUF_METADATA_VALUE_TYPE_FLOAT32, offsetof(Model, norm_eps), false, false},
	{"%s.attention.key_length", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, head_dim), false, false},
	{"%s.rope.freq_base", GGUF_METADATA_VALUE_TYPE_FLOAT32, offsetof(Model, rope_freq_base), false, false},
	{"%s.attention.head_count_kv", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, num_kv_heads), false, false},
	{"%s.embedding_length_per_layer_input", GGUF_METADATA_VALUE_TYPE_FLOAT32, offsetof(Model, pli_dim), false, false},
	{"%s.attention.sliding_window", GGUF_METADATA_VALUE_TYPE_FLOAT32, offsetof(Model, attn_sliding_window), false, false},
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

ModelDef GEMMA3N_DEF = {
	.arch = ARCH_GEMMA3N,
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
		        .vision_end_token_id   = -1,
		        .vision_embed_token_id = -1,
		},
	.interface =
		{
			.tokenize_prompt = tokenize_sp,
			.process_prompt = process_prompt_gemma3n,
			.decode_token = decode_token_sp,
			.prepare_next_token = prepare_next_token_gemma3n,
			.embedding_scale = embedding_scale_gemma3,
			.transformer_layer = transformer_layer_gemma3n,
			.build_rope_cache = build_rope_cache_shared,
		},
	DECLARE_LANGUAGE_MODEL_DEF(GEMMA3N, GEMMA3N)
};


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

static const MetadataDef GEMMA3_CLIP_METADATA_DEFS[] = {
	DECLARE_BASE_VISION_METADATA_DEFS
};

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

ModelDef GEMMA3_CLIP_DEF = {
	.arch = ARCH_CLIP_VISION,
	.projector = VISION_PROJECTOR_GEMMA3,
	.name = "Gemma3-clip",
	.params =
		{
			.proj_scale_factor = 4,
		},
	.interface = {},
	DECLARE_VISION_MODEL_DEF(GEMMA3_CLIP)
};

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
	{"%s.vision.spatial_merge_size", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(VisionModel, spatial_merge_size), false, false},
	{"%s.vision.is_deepstack_layers", GGUF_METADATA_VALUE_TYPE_BOOL, offsetof(VisionModel, is_deepstack_layers), true, false},
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

ModelDef QWEN3VL_CLIP_DEF = {
	.arch = ARCH_CLIP_VISION,
	.projector = VISION_PROJECTOR_QWEN3VL,
	.name = "Qwen3-VL-clip",
	.params =
		{
//			.proj_scale_factor = 1,
		},
	.interface = {},
	DECLARE_VISION_MODEL_DEF(QWEN3VL_CLIP)
};
