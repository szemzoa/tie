#include <inttypes.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include "model.h"
#include "gguf.h"
#include "main.h"
#include "threadpool.h"
#include "engine.h"
#include "math_dispatch.h"


int load_tensor(struct ctx_t *ctx, char *name, Tensor *tensor, int use_mmap)
{
	gguf_tensor *ggtensor;

	if ((ggtensor = get_tensor(ctx, name)) == NULL)
		return -1;

	tensor->mem.type = ggtensor->type;
	tensor->size_in_bytes = ggtensor->size;

	if (use_mmap) {
		// Point directly into the mapped file data
		tensor->mem.data = ggtensor->data;
		tensor->is_mmaped = true;
	} else {
		// Allocate new memory and read the tensor data from the file
		tensor->mem.data = aligned_alloc(128, ggtensor->size);
		if (!tensor->mem.data) {
			printf("%s OOM\n", __FUNCTION__);
			return -1;
		}

		pread(ctx->fd, tensor->mem.data, ggtensor->size, ggtensor->offset + ctx->tensor_data_offset);
		tensor->is_mmaped = false;
	}

	ctx->tensor_loaded++;

	return 0;
}

int model_load(struct ctx_t *ctx, int use_mmap, int context_length)
{
	char name_buffer[256];

	ctx->model = malloc(sizeof(Model));
	if (!ctx->model) {
		perror("Failed to allocate model");
		return -1;
	}
	ctx->model->arch = ARCH_UNKNOWN;

	ctx->model->arch_name = gguf_get_metadata_string(ctx, "general.architecture");

	if (strstr(ctx->model->arch_name, "gemma3"))
		ctx->model->arch = ARCH_GEMMA3;

	if (strstr(ctx->model->arch_name, "qwen3") || strstr(ctx->model->arch_name, "qwen3moe"))
		ctx->model->arch = ARCH_QWEN3;

	if (ctx->model->arch == ARCH_UNKNOWN) {
		perror("Failed to detect model");
		free(ctx->model);
		return -1;
	}

#define LOAD_MODEL_PARAM(field, metadata_name)                                                                         \
	snprintf(name_buffer, sizeof(name_buffer), "%s.%s", ctx->model->arch_name, metadata_name);                     \
	if (gguf_get_metadata_value(ctx, name_buffer, &ctx->model->field) != 0) {                                      \
		fprintf(stderr, "Failed to load %s\n", name_buffer);                                                   \
		goto _metadata_load_error;                                                                             \
	}

	LOAD_MODEL_PARAM(embed_dim, "embedding_length");
	LOAD_MODEL_PARAM(num_layers, "block_count");
	LOAD_MODEL_PARAM(num_heads, "attention.head_count");
	LOAD_MODEL_PARAM(num_kv_heads, "attention.head_count_kv");
	LOAD_MODEL_PARAM(head_dim, "attention.key_length");
	LOAD_MODEL_PARAM(ffn_dim, "feed_forward_length");
	LOAD_MODEL_PARAM(rope_freq_base, "rope.freq_base");
	if (context_length == 0) {
		LOAD_MODEL_PARAM(seq_length, "context_length");
	} else {
		ctx->model->seq_length = context_length;
	}
	printf("set context_length: %u\n", ctx->model->seq_length);
	LOAD_MODEL_PARAM(norm_eps, "attention.layer_norm_rms_epsilon");
	LOAD_MODEL_PARAM(eos_token, "embedding_length");

	/* Only GEMMA3 */
	snprintf(name_buffer, sizeof(name_buffer), "%s.rope.scaling.factor", ctx->model->arch_name);
	gguf_get_metadata_value(ctx, name_buffer, &ctx->model->rope_scale_factor);
	if (ctx->model->rope_scale_factor == 0.0f)
		ctx->model->rope_scale_factor = 1.0f;

	/* Only GEMMA3 */
	snprintf(name_buffer, sizeof(name_buffer), "%s.attention.sliding_window", ctx->model->arch_name);
	gguf_get_metadata_value(ctx, name_buffer, &ctx->model->attn_sliding_window);


	snprintf(name_buffer, sizeof(name_buffer), "%s.expert_count", ctx->model->arch_name);
	if (gguf_get_metadata_value(ctx, name_buffer, &ctx->model->expert_count) != 0) {
		ctx->model->is_moe = 0;
	} else {
		ctx->model->is_moe = 1;

		LOAD_MODEL_PARAM(expert_used_count, "expert_used_count");
		LOAD_MODEL_PARAM(expert_ffn_dim, "expert_feed_forward_length");
	}
#undef LOAD_MODEL_PARAM

	gguf_get_metadata_value(ctx, "tokenizer.ggml.eos_token_id", &ctx->model->eos_token);
	gguf_get_metadata_value(ctx, "tokenizer.ggml.bos_token_id", &ctx->model->bos_token);
	gguf_get_metadata_value(ctx, "tokenizer.ggml.unknown_token_id", &ctx->model->unk_token);
	gguf_get_metadata_value(ctx, "tokenizer.ggml.pad_token_id", &ctx->model->pad_token);
	gguf_get_metadata_value(ctx, "tokenizer.ggml.add_bos_token", &ctx->model->add_bos_token);
	gguf_get_metadata_value(ctx, "tokenizer.ggml.add_eos_token", &ctx->model->add_eos_token);

	/* load tokens */
	if (gguf_metadata_read_token_embeds(ctx, "tokenizer.ggml.tokens", ctx->model->arch == ARCH_GEMMA3 ? 0 : 1)
	    != 0) {
		free(ctx->model);
		perror("Failed to read tokens_embed array");
		return -1;
	}

	/* load token types */
	if (gguf_metadata_read_token_types(ctx, "tokenizer.ggml.token_type", ctx->model->arch == ARCH_GEMMA3 ? 0 : 1)
	    != 0) {
		free(ctx->model);
		perror("Failed to read ggml.type array");
		return -1;
	}

	printf("Found %u special tokens\n", special_tokens.count);
/*
	for (int i = 0; i < special_tokens.count; i++) {
	    char buf[256];
	    memset(buf, 0, 256);
	    memcpy(buf, special_tokens.specials[i].text, special_tokens.specials[i].length);

	    printf("#%u token_id = %u, token_string: %s\n", i, special_tokens.specials[i].token_id, buf);
	}
*/

	/* load token merges */
	if (ctx->model->arch == ARCH_QWEN3) {
		if (gguf_metadata_read_token_merges(ctx, "tokenizer.ggml.merges") != 0) {
			free(ctx->model);
			perror("Failed to read ggml.merges array");
			return -1;
		}
	}

	/* load token scores */
	if (ctx->model->arch == ARCH_GEMMA3) {
		if (gguf_metadata_read_token_scores(ctx, "tokenizer.ggml.scores") != 0) {
			free(ctx->model);
			perror("Failed to read ggml.scores array");
			return -1;
		}
	}

	ctx->tensor_loaded = 0;

	if (load_tensor(ctx, "token_embd.weight", &ctx->model->token_embd, use_mmap) != 0) {
		fprintf(stderr, "Failed to load token_embd.weight\n");
		free(ctx->model);
		return -1;
	}

	if (load_tensor(ctx, "output_norm.weight", &ctx->model->output_norm, use_mmap) != 0) {
		fprintf(stderr, "Failed to load token_embd.weight\n");
		free(ctx->model);
		return -1;
	}

	if (load_tensor(ctx, "output.weight", &ctx->model->output, use_mmap) != 0) {
		printf("Model using tied embedding, no output weight\n");
	}

	if ((ctx->model->layers = calloc(ctx->model->num_layers, sizeof(layer_weights))) == NULL) {
		free(ctx->model);
		perror("Failed to allocate layer_weights array");
		free(ctx->model);
		return -1;
	}

#define LOAD_LAYER_WEIGHT(field, name_fmt_str)                                                                         \
	snprintf(name_buffer, sizeof(name_buffer), name_fmt_str, block_idx);                                           \
	if (load_tensor(ctx, name_buffer, &ctx->model->layers[block_idx].field, use_mmap) != 0) {                      \
		fprintf(stderr, "Failed to load %s\n", name_buffer);                                                   \
		goto _tensor_load_error;                                                                               \
	}

	for (uint32_t block_idx = 0; block_idx < ctx->model->num_layers; block_idx++) {
		LOAD_LAYER_WEIGHT(attn_q, "blk.%u.attn_q.weight");
		LOAD_LAYER_WEIGHT(attn_k, "blk.%u.attn_k.weight");
		LOAD_LAYER_WEIGHT(attn_v, "blk.%u.attn_v.weight");
		LOAD_LAYER_WEIGHT(attn_norm, "blk.%u.attn_norm.weight");
		LOAD_LAYER_WEIGHT(attn_q_norm, "blk.%u.attn_q_norm.weight");
		LOAD_LAYER_WEIGHT(attn_k_norm, "blk.%u.attn_k_norm.weight");
		LOAD_LAYER_WEIGHT(attn_out, "blk.%u.attn_output.weight");
		LOAD_LAYER_WEIGHT(ffn_norm, "blk.%u.ffn_norm.weight");

		if (ctx->model->is_moe == 0) {
			LOAD_LAYER_WEIGHT(ffn_gate, "blk.%u.ffn_gate.weight");
			LOAD_LAYER_WEIGHT(ffn_up, "blk.%u.ffn_up.weight");
			LOAD_LAYER_WEIGHT(ffn_down, "blk.%u.ffn_down.weight");
		} else {
			LOAD_LAYER_WEIGHT(ffn_gate_inp, "blk.%u.ffn_gate_inp.weight");
			LOAD_LAYER_WEIGHT(ffn_gate_exps, "blk.%u.ffn_gate_exps.weight");
			LOAD_LAYER_WEIGHT(ffn_down_exps, "blk.%u.ffn_down_exps.weight");
			LOAD_LAYER_WEIGHT(ffn_up_exps, "blk.%u.ffn_up_exps.weight");
		}

		if (ctx->model->arch == ARCH_GEMMA3) {
			LOAD_LAYER_WEIGHT(post_attn_norm, "blk.%u.post_attention_norm.weight");
			LOAD_LAYER_WEIGHT(post_ffw_norm, "blk.%u.post_ffw_norm.weight");
		}
	}
#undef LOAD_LAYER_WEIGHT

#ifdef DEBUG_TENSORS
	dump_tensors(ctx);
#endif
	printf("Loaded %llu/%llu tensors\n", ctx->tensor_loaded, ctx->tensor_count);
	return 0;

_tensor_load_error:
	free(ctx->model->layers);

_metadata_load_error:
	free(ctx->model);

	return -1;
}

static inline void *xaligned_alloc(size_t alignment, size_t size_bytes)
{
	// aligned_alloc requires size % alignment == 0
	size_t padded = (size_bytes + (alignment - 1)) & ~(alignment - 1);

	void *p = NULL;
	if (posix_memalign(&p, alignment, padded) != 0)
		return NULL;

	return p;
}

static inline void create_internal_memory(MemType *m, ggml_type t, size_t nelems)
{
	m->type = t;
	size_t total = nelems * ggml_type_size(t);
	void *ptr = xaligned_alloc(32, total);
	if (!ptr) {
		fprintf(stderr, "ERROR: alloc %zu bytes for MemType failed\n", total);
		exit(1);
	}
	m->data = ptr;
}

static inline void release_internal_memory(MemType *m)
{
	free(m->data);
}

int model_init(struct ctx_t *ctx, float yarn_scale_factor, float repetiton_penality)
{
	char *general_name = gguf_get_metadata_string(ctx, "general.name");

	int q_dim = ctx->model->num_heads * ctx->model->head_dim;
	int kv_dim = ctx->model->num_kv_heads * ctx->model->head_dim;

	ctx->model->yarn_scale_factor = yarn_scale_factor;
	ctx->model->repetition_penalty = repetiton_penality;
	ctx->model->attn_scale = 1.0f / sqrtf(ctx->model->head_dim);

	printf("Initializing %s %s model with the following configuration:\n", general_name,
	       ctx->model->is_moe == 0 ? "dense" : "MoE");
	printf("Embed Dim: %d, Layers: %d, Heads: %d, KV Heads: %d, Head Dim: %d\n", ctx->model->embed_dim,
	       ctx->model->num_layers, ctx->model->num_heads, ctx->model->num_kv_heads, ctx->model->head_dim);
	printf("FFN Dim: %d, Rope Base: %.1f, Seq Len: %d, Vocab: %llu\n", ctx->model->ffn_dim,
	       ctx->model->rope_freq_base, ctx->model->seq_length, ctx->model->vocab_size);
	printf("Yarn Scale: %.2f, eps: %f, rope_scale: %.1f, sliding_window: %u\n", ctx->model->yarn_scale_factor,
	       ctx->model->norm_eps, ctx->model->rope_scale_factor, ctx->model->attn_sliding_window);

	if (ctx->model->is_moe == 1) {
		printf("Expert Count: %d, Expert Used Count: %d, Expert FFN Dim: %d\n", ctx->model->expert_count,
		       ctx->model->expert_used_count, ctx->model->expert_ffn_dim);
	}

	switch (ctx->model->arch) {
	    case ARCH_QWEN3:
		ctx->interface.tokenize_prompt = tokenize_bpe;
		ctx->interface.token_out = token_out_qwen3;
		ctx->model->sot_token = 151644;
		ctx->model->eot_token = 151645;
		ctx->model->newline_token = 198;
		ctx->model->role_user_token = 872;
		ctx->model->role_model_token = 77091;

		ctx->interface.embedding_scale = NULL;
		ctx->interface.activation = dispatch_swiglu_activation;
		break;

	    case ARCH_GEMMA3:
		ctx->interface.tokenize_prompt = tokenize_sp;
		ctx->interface.token_out = token_out_gemma3;
		ctx->model->sot_token = 105;
		ctx->model->eot_token = 106;
		ctx->model->eos_token = 106;
		ctx->model->newline_token = 107;
		ctx->model->role_user_token = 2364;
		ctx->model->role_model_token = 4368;

		ctx->interface.embedding_scale = embedding_scale_gemma3;
		ctx->interface.activation = dispatch_geglu_activation;
		break;
	}

	// Initialize SiLU lookup table
	silu_table_init();

	// Initialize RoPE cache
	ctx->rope_cache_local = malloc(sizeof(rope_cache_t));
	ctx->rope_cache_global = malloc(sizeof(rope_cache_t));
	if (!ctx->rope_cache_local || !ctx->rope_cache_global) {
		perror("Failed to allocate rope_cache");
		return -1;
	}

	rope_cache_init(ctx, ctx->rope_cache_global, ctx->model->seq_length, ctx->model->head_dim,
			ctx->model->rope_freq_base);

	/* Gemma3 local rope cache */
	if (ctx->model->arch == ARCH_GEMMA3) {
	    rope_cache_init(ctx, ctx->rope_cache_local, ctx->model->seq_length, ctx->model->head_dim, 10000.0f);
	}

	printf("Initializing KV cache\n");
	ctx->kv_cache = calloc(ctx->model->num_layers, sizeof(LayerKVCache));
	if (!ctx->kv_cache) {
		perror("Failed to allocate LayerKVCache array");
		free(ctx->rope_cache_local);
		free(ctx->rope_cache_global);
		return -1;
	}

	long long k_elements_per_layer =
		(long long)ctx->model->seq_length * ctx->model->num_kv_heads * ctx->model->head_dim;

	printf("KV cache elements per layer: %lld\n", k_elements_per_layer);
	for (int i = 0; i < ctx->model->num_layers; i++) {
		create_internal_memory(&ctx->kv_cache[i].k, GGML_TYPE_BF16, k_elements_per_layer);
		create_internal_memory(&ctx->kv_cache[i].v, GGML_TYPE_BF16, k_elements_per_layer);

		if (!ctx->kv_cache[i].k.data || !ctx->kv_cache[i].v.data) {
			perror("Failed to allocate K or V cache for a layer");
			for (int j = 0; j < i; ++j) {
				release_internal_memory(&ctx->kv_cache[j].k);
				release_internal_memory(&ctx->kv_cache[j].v);
			}
			free(ctx->kv_cache);
			free(ctx->rope_cache_local);
			free(ctx->rope_cache_global);
			return -1;
		}
	}
	printf("KV cache uses: %lld MB\n",
	       (k_elements_per_layer * sizeof(uint16_t)) * 2 * ctx->model->num_layers / 1024 / 1024);

	/* create internal memories */
	if (ctx->model->arch == ARCH_GEMMA3)
		create_internal_memory(&ctx->mem.residual_stratch, INTERNAL_MEMORY_TYPE,
				       ctx->model->embed_dim * MAX_PROMPT_BATCH_SIZE);

	create_internal_memory(&ctx->mem.hidden_state, INTERNAL_MEMORY_TYPE,
			       ctx->model->embed_dim * MAX_PROMPT_BATCH_SIZE);

	create_internal_memory(&ctx->mem.normed_qkv_input, INTERNAL_MEMORY_TYPE,
			       ctx->model->embed_dim * MAX_PROMPT_BATCH_SIZE);

	create_internal_memory(&ctx->mem.Q, INTERNAL_MEMORY_TYPE, q_dim * MAX_PROMPT_BATCH_SIZE);
	create_internal_memory(&ctx->mem.K, INTERNAL_MEMORY_TYPE, kv_dim * MAX_PROMPT_BATCH_SIZE);
	create_internal_memory(&ctx->mem.V, INTERNAL_MEMORY_TYPE, kv_dim * MAX_PROMPT_BATCH_SIZE);

	create_internal_memory(&ctx->mem.attn_output, INTERNAL_MEMORY_TYPE, q_dim * MAX_PROMPT_BATCH_SIZE);
	create_internal_memory(&ctx->mem.attn_proj_output, INTERNAL_MEMORY_TYPE,
			       ctx->model->embed_dim * MAX_PROMPT_BATCH_SIZE);
	create_internal_memory(&ctx->mem.normed_ffn_input, INTERNAL_MEMORY_TYPE,
			       ctx->model->embed_dim * MAX_PROMPT_BATCH_SIZE);

	create_internal_memory(&ctx->mem.gate_proj_output, INTERNAL_MEMORY_TYPE,
			       ctx->model->ffn_dim * MAX_PROMPT_BATCH_SIZE);
	create_internal_memory(&ctx->mem.up_proj_output, INTERNAL_MEMORY_TYPE,
			       ctx->model->ffn_dim * MAX_PROMPT_BATCH_SIZE);
	create_internal_memory(&ctx->mem.ffn_down_output, INTERNAL_MEMORY_TYPE,
			       ctx->model->embed_dim * MAX_PROMPT_BATCH_SIZE);

	create_internal_memory(&ctx->mem.logits, GGML_TYPE_F32, ctx->model->vocab_size);

	if (ctx->model->is_moe == 1) {
		create_internal_memory(&ctx->mem.expert_scores, GGML_TYPE_F32, ctx->model->expert_count);
		create_internal_memory(&ctx->mem.expert_out_fp32, GGML_TYPE_F32, ctx->model->embed_dim);

		ctx->mem.ffn_hidden1_scratch = aligned_alloc(32, ctx->model->expert_count * sizeof(MemType));
		ctx->mem.ffn_hidden2_scratch = aligned_alloc(32, ctx->model->expert_count * sizeof(MemType));
		ctx->mem.expert_outputs = aligned_alloc(32, ctx->model->expert_count * sizeof(MemType));

		for (int i = 0; i < ctx->model->expert_count; i++) {
			create_internal_memory(&ctx->mem.ffn_hidden1_scratch[i], GGML_TYPE_BF16,
					       ctx->model->expert_ffn_dim);
			create_internal_memory(&ctx->mem.ffn_hidden2_scratch[i], GGML_TYPE_BF16,
					       ctx->model->expert_ffn_dim);
			create_internal_memory(&ctx->mem.expert_outputs[i], GGML_TYPE_F32, ctx->model->embed_dim);
		}
	}

	ctx->mem.q_head_fp32_scratch = malloc(thread_pool->num_threads * sizeof(MemType));

	for (int t = 0; t < thread_pool->num_threads; t++) {

		create_internal_memory(&ctx->mem.q_head_fp32_scratch[t], GGML_TYPE_F32, ctx->model->head_dim);

		if ((ctx->mem.attn_scores_buffer[t] =
			     aligned_alloc(32, (long long)ctx->model->seq_length * sizeof(float)))
		    == NULL)

			goto _init_error_free_kv_cache;
	}

	ctx->kv_pos = 0;
	return 0;

_init_error_free_kv_cache:
	for (int i = 0; i < ctx->model->num_layers; i++) {
		release_internal_memory(&ctx->kv_cache[i].k);
		release_internal_memory(&ctx->kv_cache[i].v);
	}

	free(ctx->kv_cache);
	free(ctx->rope_cache_local);
	free(ctx->rope_cache_global);

	return -1;
}

void model_cleanup(struct ctx_t *ctx, int use_mmap)
{
	if (ctx->model->arch == ARCH_GEMMA3)
		release_internal_memory(&ctx->mem.residual_stratch);

	release_internal_memory(&ctx->mem.hidden_state);
	release_internal_memory(&ctx->mem.normed_qkv_input);
	release_internal_memory(&ctx->mem.Q);
	release_internal_memory(&ctx->mem.K);
	release_internal_memory(&ctx->mem.V);
	release_internal_memory(&ctx->mem.attn_output);
	release_internal_memory(&ctx->mem.attn_proj_output);
	release_internal_memory(&ctx->mem.normed_ffn_input);
	release_internal_memory(&ctx->mem.gate_proj_output);
	release_internal_memory(&ctx->mem.up_proj_output);
	release_internal_memory(&ctx->mem.ffn_down_output);
	release_internal_memory(&ctx->mem.logits);

	for (int t = 0; t < thread_pool->num_threads; t++) {
		release_internal_memory(&ctx->mem.q_head_fp32_scratch[t]);
		free(ctx->mem.attn_scores_buffer[t]);
	}

	free(ctx->mem.q_head_fp32_scratch);

	if (ctx->model->is_moe == 1) {

		for (int i = 0; i < thread_pool->num_threads; i++) {
			release_internal_memory(&ctx->mem.ffn_hidden1_scratch[i]);
			release_internal_memory(&ctx->mem.ffn_hidden2_scratch[i]);
			release_internal_memory(&ctx->mem.expert_outputs[i]);
		}

		free(ctx->mem.ffn_hidden1_scratch);
		free(ctx->mem.ffn_hidden2_scratch);
		free(ctx->mem.expert_outputs);

		release_internal_memory(&ctx->mem.expert_out_fp32);
		release_internal_memory(&ctx->mem.expert_scores);
	}

	thread_pool_destroy(thread_pool);

	for (uint32_t i = 0; i < ctx->model->num_layers; i++) {
		release_internal_memory(&ctx->kv_cache[i].k);
		release_internal_memory(&ctx->kv_cache[i].v);
	}
	free(ctx->kv_cache);

	if (use_mmap == 1) {

		free(ctx->model->token_embd.mem.data);
		free(ctx->model->output_norm.mem.data);

		if (ctx->model->output.mem.data)
			free(ctx->model->output.mem.data);

		for (uint32_t block_idx = 0; block_idx < ctx->model->num_layers; block_idx++) {
			free(ctx->model->layers[block_idx].attn_q.mem.data);
			free(ctx->model->layers[block_idx].attn_k.mem.data);
			free(ctx->model->layers[block_idx].attn_v.mem.data);
			free(ctx->model->layers[block_idx].attn_norm.mem.data);
			free(ctx->model->layers[block_idx].attn_q_norm.mem.data);
			free(ctx->model->layers[block_idx].attn_k_norm.mem.data);
			free(ctx->model->layers[block_idx].attn_out.mem.data);
			free(ctx->model->layers[block_idx].ffn_gate.mem.data);
			free(ctx->model->layers[block_idx].ffn_up.mem.data);
			free(ctx->model->layers[block_idx].ffn_down.mem.data);
			free(ctx->model->layers[block_idx].ffn_norm.mem.data);

			if (ctx->model->arch == ARCH_GEMMA3) {
				free(ctx->model->layers[block_idx].post_attn_norm.mem.data);
				free(ctx->model->layers[block_idx].post_ffw_norm.mem.data);
			}
		}
	}

	free(ctx->model->layers);
	free(ctx->model);
	free(ctx->rope_cache_local);
	free(ctx->rope_cache_global);

	free(ctx->tokenizer.token_table);
	free(ctx->tokenizer.token_lens);
	free(ctx->tokenizer.token_types);
	free(ctx->tokenizer.token_scores);

	free_trie(ctx->tokenizer.root);
	free_string_pool(ctx->tokenizer.pool);
}
