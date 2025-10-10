#include <stddef.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "model.h"
#include "engine.h"
#include "tokenize.h"
#include "threadpool.h"

static const TensorDef QWEN3_GLOBAL_TENSORS[] = {
	{"token_embd.weight", offsetof(Model, token_embd), FLAG_NONE},
	{"output_norm.weight", offsetof(Model, output_norm), FLAG_NONE},
	{"output.weight", offsetof(Model, output), FLAG_OPTIONAL},
};

static const TensorDef QWEN3_LAYER_TENSORS[] = {
	{"blk.%u.attn_q.weight", offsetof(layer_weights, attn_q), FLAG_NONE},
	{"blk.%u.attn_k.weight", offsetof(layer_weights, attn_k), FLAG_NONE},
	{"blk.%u.attn_v.weight", offsetof(layer_weights, attn_v), FLAG_NONE},
	{"blk.%u.attn_norm.weight", offsetof(layer_weights, attn_norm), FLAG_NONE},
	{"blk.%u.attn_q_norm.weight", offsetof(layer_weights, attn_q_norm), FLAG_NONE},
	{"blk.%u.attn_k_norm.weight", offsetof(layer_weights, attn_k_norm), FLAG_NONE},
	{"blk.%u.attn_output.weight", offsetof(layer_weights, attn_out), FLAG_NONE},
	{"blk.%u.ffn_norm.weight", offsetof(layer_weights, ffn_norm), FLAG_NONE},
	{"blk.%u.ffn_gate.weight", offsetof(layer_weights, ffn_gate), FLAG_DENSE_ONLY},
	{"blk.%u.ffn_up.weight", offsetof(layer_weights, ffn_up), FLAG_DENSE_ONLY},
	{"blk.%u.ffn_down.weight", offsetof(layer_weights, ffn_down), FLAG_DENSE_ONLY},
	{"blk.%u.ffn_gate_inp.weight", offsetof(layer_weights, ffn_gate_inp), FLAG_MOE_ONLY},
	{"blk.%u.ffn_gate_exps.weight", offsetof(layer_weights, ffn_gate_exps), FLAG_MOE_ONLY},
	{"blk.%u.ffn_down_exps.weight", offsetof(layer_weights, ffn_down_exps), FLAG_MOE_ONLY},
	{"blk.%u.ffn_up_exps.weight", offsetof(layer_weights, ffn_up_exps), FLAG_MOE_ONLY},
};

static const MetadataDef QWEN3_METADATA_DEFS[] = {
	{"%s.context_length", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, seq_length), false},
	{"%s.embedding_length", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, embed_dim), false},
	{"%s.block_count", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, num_layers), false},
	{"%s.feed_forward_length", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, ffn_dim), false},
	{"%s.attention.head_count", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, num_heads), false},
	{"%s.attention.layer_norm_rms_epsilon", GGUF_METADATA_VALUE_TYPE_FLOAT32, offsetof(Model, norm_eps), false},
	{"%s.attention.key_length", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, head_dim), false},
	{"%s.rope.freq_base", GGUF_METADATA_VALUE_TYPE_FLOAT32, offsetof(Model, rope_freq_base), false},
	{"%s.attention.head_count_kv", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, num_kv_heads), false},
	{"%s.expert_count", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, expert_count), true},
	{"%s.expert_used_count", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, expert_used_count), true},
	{"%s.expert_feed_forward_length", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, expert_ffn_dim), true},
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
};

ModelDef QWEN3_DEF = {
	.arch = ARCH_QWEN3,
	.name = "Qwen3",
	.params =
		{
			.sot_token_id = 151644,
			.eot_token_id = 151645,
			.newline_token_id = 198,
			.role_user_token_id = 872,
			.role_model_token_id = 77091,
		},
	.interface =
		{
			.tokenize_prompt = tokenize_bpe,
			.process_prompt = process_prompt_standard,
			.token_out = token_out_utf8_stream,
			.prepare_next_token = prepare_next_token_standard,
			.embedding_scale = NULL,
			.transformer_layer = transformer_layer_qwen3,
		},
	.metadata_defs = QWEN3_METADATA_DEFS,
	.num_metadata_defs = ARRAY_SIZE(QWEN3_METADATA_DEFS),
	.global_tensors = QWEN3_GLOBAL_TENSORS,
	.num_global_tensors = ARRAY_SIZE(QWEN3_GLOBAL_TENSORS),
	.layer_tensors = QWEN3_LAYER_TENSORS,
	.num_layer_tensors = ARRAY_SIZE(QWEN3_LAYER_TENSORS),
	.buffer_defs = QWEN3_BUFFERS,
	.num_buffer_defs = ARRAY_SIZE(QWEN3_BUFFERS),
};

static const TensorDef GEMMA3_GLOBAL_TENSORS[] = {
	{"token_embd.weight", offsetof(Model, token_embd), FLAG_NONE},
	{"output_norm.weight", offsetof(Model, output_norm), FLAG_NONE},
	{"output.weight", offsetof(Model, output), FLAG_OPTIONAL},
};

static const TensorDef GEMMA3_LAYER_TENSORS[] = {
	{"blk.%u.attn_q.weight", offsetof(layer_weights, attn_q), FLAG_NONE},
	{"blk.%u.attn_k.weight", offsetof(layer_weights, attn_k), FLAG_NONE},
	{"blk.%u.attn_v.weight", offsetof(layer_weights, attn_v), FLAG_NONE},
	{"blk.%u.attn_norm.weight", offsetof(layer_weights, attn_norm), FLAG_NONE},
	{"blk.%u.attn_q_norm.weight", offsetof(layer_weights, attn_q_norm), FLAG_NONE},
	{"blk.%u.attn_k_norm.weight", offsetof(layer_weights, attn_k_norm), FLAG_NONE},
	{"blk.%u.attn_output.weight", offsetof(layer_weights, attn_out), FLAG_NONE},
	{"blk.%u.ffn_norm.weight", offsetof(layer_weights, ffn_norm), FLAG_NONE},
	{"blk.%u.ffn_gate.weight", offsetof(layer_weights, ffn_gate), FLAG_NONE},
	{"blk.%u.ffn_up.weight", offsetof(layer_weights, ffn_up), FLAG_NONE},
	{"blk.%u.ffn_down.weight", offsetof(layer_weights, ffn_down), FLAG_NONE},
	{"blk.%u.post_attention_norm.weight", offsetof(layer_weights, post_attn_norm), FLAG_NONE},
	{"blk.%u.post_ffw_norm.weight", offsetof(layer_weights, post_ffw_norm), FLAG_NONE},
};

static const MetadataDef GEMMA3_METADATA_DEFS[] = {
	{"%s.context_length", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, seq_length), false},
	{"%s.embedding_length", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, embed_dim), false},
	{"%s.block_count", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, num_layers), false},
	{"%s.feed_forward_length", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, ffn_dim), false},
	{"%s.attention.head_count", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, num_heads), false},
	{"%s.attention.layer_norm_rms_epsilon", GGUF_METADATA_VALUE_TYPE_FLOAT32, offsetof(Model, norm_eps), false},
	{"%s.attention.key_length", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, head_dim), false},
	{"%s.rope.freq_base", GGUF_METADATA_VALUE_TYPE_FLOAT32, offsetof(Model, rope_freq_base), false},
	{"%s.attention.head_count_kv", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, num_kv_heads), false},
	{"%s.attention.sliding_window", GGUF_METADATA_VALUE_TYPE_FLOAT32, offsetof(Model, attn_sliding_window), false},
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
		},
	.interface =
		{
			.tokenize_prompt = tokenize_sp,
			.process_prompt = process_prompt_standard,
			.token_out = token_out_sp,
			.prepare_next_token = prepare_next_token_standard,
			.embedding_scale = embedding_scale_gemma3,
			.transformer_layer = transformer_layer_gemma3,
		},
	.metadata_defs = GEMMA3_METADATA_DEFS,
	.num_metadata_defs = ARRAY_SIZE(GEMMA3_METADATA_DEFS),
	.global_tensors = GEMMA3_GLOBAL_TENSORS,
	.num_global_tensors = ARRAY_SIZE(GEMMA3_GLOBAL_TENSORS),
	.layer_tensors = GEMMA3_LAYER_TENSORS,
	.num_layer_tensors = ARRAY_SIZE(GEMMA3_LAYER_TENSORS),
	.buffer_defs = GEMMA3_BUFFERS,
	.num_buffer_defs = ARRAY_SIZE(GEMMA3_BUFFERS),
};

static const TensorDef GEMMA3N_GLOBAL_TENSORS[] = {
	{"token_embd.weight", offsetof(Model, token_embd), FLAG_NONE},
	{"output_norm.weight", offsetof(Model, output_norm), FLAG_NONE},
	{"output.weight", offsetof(Model, output), FLAG_OPTIONAL},
	{"altup_proj.weight", offsetof(Model, altup_proj), FLAG_NONE},
	{"altup_unembd_proj.weight", offsetof(Model, altup_unembd_proj), FLAG_NONE},
	{"per_layer_model_proj.weight", offsetof(Model, per_layer_model_proj), FLAG_NONE},
	{"per_layer_proj_norm.weight", offsetof(Model, per_layer_proj_norm), FLAG_NONE},
	{"per_layer_token_embd.weight", offsetof(Model, per_layer_token_embd), FLAG_NONE}};

static const TensorDef GEMMA3N_LAYER_TENSORS[] = {
	{"blk.%u.attn_q.weight", offsetof(layer_weights, attn_q), FLAG_NONE},
	{"blk.%u.attn_k.weight", offsetof(layer_weights, attn_k), FLAG_NONE},
	{"blk.%u.attn_v.weight", offsetof(layer_weights, attn_v), FLAG_NONE},
	{"blk.%u.attn_norm.weight", offsetof(layer_weights, attn_norm), FLAG_NONE},
	{"blk.%u.attn_q_norm.weight", offsetof(layer_weights, attn_q_norm), FLAG_NONE},
	{"blk.%u.attn_k_norm.weight", offsetof(layer_weights, attn_k_norm), FLAG_NONE},
	{"blk.%u.attn_output.weight", offsetof(layer_weights, attn_out), FLAG_NONE},
	{"blk.%u.ffn_norm.weight", offsetof(layer_weights, ffn_norm), FLAG_NONE},
	{"blk.%u.ffn_gate.weight", offsetof(layer_weights, ffn_gate), FLAG_NONE},
	{"blk.%u.ffn_up.weight", offsetof(layer_weights, ffn_up), FLAG_NONE},
	{"blk.%u.ffn_down.weight", offsetof(layer_weights, ffn_down), FLAG_NONE},
	{"blk.%u.post_attention_norm.weight", offsetof(layer_weights, post_attn_norm), FLAG_NONE},
	{"blk.%u.post_ffw_norm.weight", offsetof(layer_weights, post_ffw_norm), FLAG_NONE},
	{"blk.%u.altup_correct_coef.weight", offsetof(layer_weights, altup_correct_coef), FLAG_NONE},
	{"blk.%u.altup_correct_scale.weight", offsetof(layer_weights, altup_correct_scale), FLAG_NONE},
	{"blk.%u.altup_predict_coef.weight", offsetof(layer_weights, altup_predict_coef), FLAG_NONE},
	{"blk.%u.altup_router.weight", offsetof(layer_weights, altup_router), FLAG_NONE},
	{"blk.%u.altup_router_norm.weight", offsetof(layer_weights, altup_router_norm), FLAG_NONE},
	{"blk.%u.inp_gate.weight", offsetof(layer_weights, inp_gate), FLAG_NONE},
	{"blk.%u.laurel_l.weight", offsetof(layer_weights, laurel_l), FLAG_NONE},
	{"blk.%u.laurel_post_norm.weight", offsetof(layer_weights, laurel_post_norm), FLAG_NONE},
	{"blk.%u.laurel_r.weight", offsetof(layer_weights, laurel_r), FLAG_NONE},
	{"blk.%u.proj.weight", offsetof(layer_weights, proj), FLAG_NONE},
	{"blk.%u.post_norm.weight", offsetof(layer_weights, post_norm), FLAG_NONE},
};

static const MetadataDef GEMMA3N_METADATA_DEFS[] = {
	{"%s.context_length", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, seq_length), false},
	{"%s.embedding_length", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, embed_dim), false},
	{"%s.block_count", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, num_layers), false},
	{"%s.attention.head_count", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, num_heads), false},
	{"%s.attention.layer_norm_rms_epsilon", GGUF_METADATA_VALUE_TYPE_FLOAT32, offsetof(Model, norm_eps), false},
	{"%s.attention.key_length", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, head_dim), false},
	{"%s.rope.freq_base", GGUF_METADATA_VALUE_TYPE_FLOAT32, offsetof(Model, rope_freq_base), false},
	{"%s.attention.head_count_kv", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, num_kv_heads), false},
	{"%s.embedding_length_per_layer_input", GGUF_METADATA_VALUE_TYPE_FLOAT32, offsetof(Model, pli_dim), false},
	{"%s.attention.sliding_window", GGUF_METADATA_VALUE_TYPE_FLOAT32, offsetof(Model, attn_sliding_window), false},
	{"%s.altup.num_inputs", GGUF_METADATA_VALUE_TYPE_FLOAT32, offsetof(Model, altup_num_inputs), false},
	// This one is optional, with a fallback
	{"%s.rope.scaling.factor", GGUF_METADATA_VALUE_TYPE_FLOAT32, offsetof(Model, rope_scale_factor), true},
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
		},
	.interface =
		{
			.tokenize_prompt = tokenize_sp,
			.process_prompt = process_prompt_gemma3n,
			.token_out = token_out_sp,
			.prepare_next_token = prepare_next_token_gemma3n,
			.embedding_scale = embedding_scale_gemma3,
			.transformer_layer = transformer_layer_gemma3n,
		},
	.metadata_defs = GEMMA3N_METADATA_DEFS,
	.num_metadata_defs = ARRAY_SIZE(GEMMA3N_METADATA_DEFS),
	.global_tensors = GEMMA3N_GLOBAL_TENSORS,
	.num_global_tensors = ARRAY_SIZE(GEMMA3N_GLOBAL_TENSORS),
	.layer_tensors = GEMMA3N_LAYER_TENSORS,
	.num_layer_tensors = ARRAY_SIZE(GEMMA3N_LAYER_TENSORS),
	.buffer_defs = GEMMA3N_BUFFERS,
	.num_buffer_defs = ARRAY_SIZE(GEMMA3N_BUFFERS),
};
