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

	{"tokenizer.ggml.eos_token_id", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, eos_token_id), true},
	{"tokenizer.ggml.bos_token_id", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, bos_token_id), true},
	{"tokenizer.ggml.unknown_token_id", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, unk_token_id), true},
	{"tokenizer.ggml.pad_token_id", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, pad_token_id), true},
	{"tokenizer.ggml.add_bos_token", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, add_bos_token), true},
	{"tokenizer.ggml.add_eos_token", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, add_eos_token), true},
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
	.tokenize_defs = &QWEN3_TOKENIZE_DEF,
	.struct_size = sizeof(Model),
	.layers_offset = offsetof(Model, layers),
	.layer_struct_size = sizeof(LayerWeights),
	.num_layers_offset = offsetof(Model, num_layers),
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
	.layer_tensors = QWEN3_MOE_LAYER_TENSORS,
	.num_layer_tensors = ARRAY_SIZE(QWEN3_MOE_LAYER_TENSORS),
	.buffer_defs = QWEN3_BUFFERS,
	.tokenize_defs = &QWEN3_TOKENIZE_DEF,
	.num_buffer_defs = ARRAY_SIZE(QWEN3_BUFFERS),
	.struct_size = sizeof(Model),
	.layers_offset = offsetof(Model, layers),
	.layer_struct_size = sizeof(LayerWeights),
	.num_layers_offset = offsetof(Model, num_layers),
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
	.tokenize_defs = &QWEN3_TOKENIZE_DEF,
	.struct_size = sizeof(Model),
	.layers_offset = offsetof(Model, layers),
	.layer_struct_size = sizeof(LayerWeights),
	.num_layers_offset = offsetof(Model, num_layers),
};

static const TensorDef GEMMA3_GLOBAL_TENSORS[] = {
	{"token_embd.weight", offsetof(Model, token_embd), FLAG_NONE},
	{"output_norm.weight", offsetof(Model, output_norm), FLAG_NONE},
	{"output.weight", offsetof(Model, output), FLAG_OPTIONAL},
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

	{"tokenizer.ggml.eos_token_id", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, eos_token_id), true},
	{"tokenizer.ggml.bos_token_id", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, bos_token_id), true},
	{"tokenizer.ggml.unknown_token_id", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, unk_token_id), true},
	{"tokenizer.ggml.pad_token_id", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, pad_token_id), true},
	{"tokenizer.ggml.add_bos_token", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, add_bos_token), true},
	{"tokenizer.ggml.add_eos_token", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, add_eos_token), true},
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
	.tokenize_defs = &GEMMA3_TOKENIZE_DEF,
	.num_buffer_defs = ARRAY_SIZE(GEMMA3_BUFFERS),
	.struct_size = sizeof(Model),
	.layers_offset = offsetof(Model, layers),
	.layer_struct_size = sizeof(LayerWeights),
	.num_layers_offset = offsetof(Model, num_layers),
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

	{"tokenizer.ggml.eos_token_id", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, eos_token_id), true},
	{"tokenizer.ggml.bos_token_id", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, bos_token_id), true},
	{"tokenizer.ggml.unknown_token_id", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, unk_token_id), true},
	{"tokenizer.ggml.pad_token_id", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, pad_token_id), true},
	{"tokenizer.ggml.add_bos_token", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, add_bos_token), true},
	{"tokenizer.ggml.add_eos_token", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, add_eos_token), true},
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
	.tokenize_defs = &GEMMA3_TOKENIZE_DEF,
	.num_buffer_defs = ARRAY_SIZE(GEMMA3N_BUFFERS),
	.struct_size = sizeof(Model),
	.layers_offset = offsetof(Model, layers),
	.layer_struct_size = sizeof(LayerWeights),
	.num_layers_offset = offsetof(Model, num_layers),
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
	{"%s.vision.projection_dim", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(VisionModel, projection_dim), false},
	{"%s.vision.image_size", GGUF_METADATA_VALUE_TYPE_FLOAT32, offsetof(VisionModel, image_size), false},
	{"%s.vision.patch_size", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(VisionModel, patch_size), false},
	{"%s.vision.embedding_length", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(VisionModel, embed_dim), false},
	{"%s.vision.block_count", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(VisionModel, num_layers), false},
	{"%s.vision.feed_forward_length", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(VisionModel, ffn_dim), false},
	{"%s.vision.attention.head_count", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(VisionModel, num_heads), false},
	{"%s.vision.attention.layer_norm_epsilon", GGUF_METADATA_VALUE_TYPE_FLOAT32, offsetof(VisionModel, norm_eps),
	 false},
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
	.arch = ARCH_GEMMA3_CLIP,
	.name = "Gemma-3-clip",
	.params =
		{
			.proj_scale_factor = 4,
		},
	.interface = {},
	.metadata_defs = GEMMA3_CLIP_METADATA_DEFS,
	.num_metadata_defs = ARRAY_SIZE(GEMMA3_CLIP_METADATA_DEFS),
	.global_tensors = GEMMA3_CLIP_GLOBAL_TENSORS,
	.num_global_tensors = ARRAY_SIZE(GEMMA3_CLIP_GLOBAL_TENSORS),
	.layer_tensors = GEMMA3_CLIP_LAYER_TENSORS,
	.num_layer_tensors = ARRAY_SIZE(GEMMA3_CLIP_LAYER_TENSORS),
	.buffer_defs = GEMMA3_CLIP_BUFFERS,
	.num_buffer_defs = ARRAY_SIZE(GEMMA3_CLIP_BUFFERS),
	.struct_size = sizeof(VisionModel),
	.layers_offset = offsetof(VisionModel, layers),
	.layer_struct_size = sizeof(VisionLayerWeights),
	.num_layers_offset = offsetof(VisionModel, num_layers),
};
