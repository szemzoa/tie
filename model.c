#include <inttypes.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "gguf.h"
#include "threadpool.h"
#include "main.h"
#include "maths.h"
#include "engine.h"

int model_create(struct ctx_t *ctx)
{
	char tensor_name_buffer[256];
	uint64_t tensor_mapped = 0;
	uint64_t size64;
	void *raw_weight_ptr;

	ctx->model = malloc(sizeof(Qwen3Model));
	if (!ctx->model) {
		perror("Failed to allocate Qwen3Model");
		return -1;
	}

	gguf_get_metadata_value(ctx, "qwen3.embedding_length", &ctx->model->embed_dim);
	gguf_get_metadata_value(ctx, "qwen3.block_count", &ctx->model->num_layers);
	gguf_get_metadata_value(ctx, "qwen3.attention.head_count", &ctx->model->num_heads);
	gguf_get_metadata_value(ctx, "qwen3.attention.head_count_kv", &ctx->model->num_kv_heads);
	gguf_get_metadata_value(ctx, "qwen3.attention.key_length", &ctx->model->head_dim);
	gguf_get_metadata_value(ctx, "qwen3.feed_forward_length", &ctx->model->ffn_dim);
	gguf_get_metadata_value(ctx, "qwen3.rope.freq_base", (float *)&ctx->model->rope_freq_base);
	gguf_get_metadata_value(ctx, "qwen3.context_length", &ctx->model->seq_length);
	gguf_get_metadata_value(ctx, "qwen3.attention.layer_norm_rms_epsilon", (float *)&ctx->model->norm_eps);
	gguf_get_metadata_value(ctx, "tokenizer.ggml.eos_token_id", &ctx->model->eos_token);

	if (gguf_metadata_read_tokens_embed(ctx, "tokenizer.ggml.tokens") != 0) {
		free(ctx->model);
		perror("Failed to read tokens_embed array");
		return -1;
	}
	printf("Found %u special tokens\n", special_tokens.count);

	if (gguf_metadata_read_merges(ctx, "tokenizer.ggml.merges") != 0) {
		free(ctx->model);
		perror("Failed to read ggml.merges array");
		return -1;
	}

	if ((ctx->model->layers = calloc(ctx->model->num_layers, sizeof(LayerWeights))) == NULL) {
		free(ctx->model);
		perror("Failed to allocate LayerWeights array");
		return -1;
	}

	if ((raw_weight_ptr = get_tensor(ctx, "token_embd.weight", &size64)) == NULL) {
		fprintf(stderr, "Failed to load token_embd.weight\n");
		return -1;
	}
	if ((ctx->model->token_embd = convert_bf16_to_f32(raw_weight_ptr, size64 / 2)) == NULL) {
		fprintf(stderr, "Failed to load token_embd.weight\n");
		return -1;
	}
	tensor_mapped++;

	if ((ctx->model->output_norm = get_tensor(ctx, "output_norm.weight", &size64)) == NULL) {
		fprintf(stderr, "Failed to load output_norm.weight\n");
		goto _tensor_load_error;
	}
	tensor_mapped++;

#define LOAD_WEIGHT(field, name_fmt_str)                                                                               \
	snprintf(tensor_name_buffer, sizeof(tensor_name_buffer), name_fmt_str, block_idx);                             \
	if ((ctx->model->layers[block_idx].field = get_tensor(ctx, tensor_name_buffer, &size64)) == NULL) {            \
		goto _tensor_load_error;                                                                               \
	}                                                                                                              \
	tensor_mapped++;

	for (uint32_t block_idx = 0; block_idx < ctx->model->num_layers; block_idx++) {
		LOAD_WEIGHT(attn_q, "blk.%u.attn_q.weight");
		LOAD_WEIGHT(attn_k, "blk.%u.attn_k.weight");
		LOAD_WEIGHT(attn_v, "blk.%u.attn_v.weight");
		LOAD_WEIGHT(attn_norm, "blk.%u.attn_norm.weight");
		LOAD_WEIGHT(attn_q_norm, "blk.%u.attn_q_norm.weight");
		LOAD_WEIGHT(attn_k_norm, "blk.%u.attn_k_norm.weight");
		LOAD_WEIGHT(attn_out, "blk.%u.attn_output.weight");
		LOAD_WEIGHT(ffn_gate, "blk.%u.ffn_gate.weight");
		LOAD_WEIGHT(ffn_up, "blk.%u.ffn_up.weight");
		LOAD_WEIGHT(ffn_down, "blk.%u.ffn_down.weight");
		LOAD_WEIGHT(ffn_norm, "blk.%u.ffn_norm.weight");
	}

#undef LOAD_WEIGHT

	if (ctx->tensor_count != tensor_mapped) {
		printf("Tensor load failed!!!! load: %llu, all: %llu", tensor_mapped, ctx->tensor_count);
		goto _tensor_load_error;
	}

#ifdef DEBUG_TENSORS
	dump_tensors(ctx);
#endif

	return 0;

_tensor_load_error:
	free(ctx->model);
	free(ctx->model->layers);

	return -1;
}

#if 0
int compare_weights(char *filename, int file_size, int py_offset, int size, float *c_weights)
{
	int	  rc = 0;
	FILE *f	 = fopen(filename, "rb");
	if (!f) {
		printf("can't open python file: %s\n", filename);
		return -1;
	}

	// Assuming you know the size of the vocab from your model config
	float *python_weights = malloc(file_size * sizeof(float));
	fread(python_weights, sizeof(float), file_size, f);
	fclose(f);

	// Now, compare python_logits with your C-generated logits element by element
	for (int i = 0; i < size; i++) {
		float diff = fabsf(python_weights[py_offset + i] - c_weights[i]);
		if (diff > 1e-4) { // Use a small tolerance for floating point math
			printf("Mismatch at index %d! Python: %f, C: %f\n", i, python_weights[py_offset + i], c_weights[i]);
			rc = -1;
			break;
		}
	}

#if 1
	printf("validate %s python weights: \t", filename);
	for (int i = 0; i < 5 && i < size; i++)
		printf("%.10f ", python_weights[py_offset + i]);

	printf("\n");

	printf("validate %s c-code weights: \t", filename);
	for (int i = 0; i < 5 && i < size; i++)
		printf("%.10f ", c_weights[i]);

	printf("\n");
#endif

	if (rc == 0) {
		printf("%s match\n", filename);
	} else {
		printf("%s differ\n", filename);
	}

	free(python_weights);
	return rc;
}
#endif

void model_cleanup(struct ctx_t *ctx)
{
	free(ctx->mem.hidden_state);
	free(ctx->mem.normed_qkv_input);
	free(ctx->mem.Q);
	free(ctx->mem.K);
	free(ctx->mem.V);

	for (int t = 0; t < thread_pool->num_threads; t++)
		free(ctx->mem.attn_scores_buffer[t]);

	free(ctx->mem.attn_output);
	free(ctx->mem.attn_proj_output);
	free(ctx->mem.normed_ffn_input);
	free(ctx->mem.gate_proj_output);
	free(ctx->mem.up_proj_output);
	free(ctx->mem.ffn_down_output);
	free(ctx->mem.logits);

	thread_pool_destroy(thread_pool);

	for (uint32_t i = 0; i < ctx->model->num_layers; i++) {
		free(ctx->kv_cache[i].k);
		free(ctx->kv_cache[i].v);
	}
	free(ctx->kv_cache);

	free(ctx->model->layers);
	free(ctx->model->token_embd);
	free(ctx->model);
	free(ctx->rope_cache);
	free_trie(ctx->root);
	free_string_pool(ctx->pool);
}

int model_init(struct ctx_t *ctx, float yarn_scale_factor, float repetiton_penality)
{
	ctx->model->yarn_scale_factor = yarn_scale_factor;
	ctx->model->repetition_penalty = repetiton_penality;
	ctx->model->attn_scale = 1.0f / sqrtf(ctx->model->head_dim);

	printf("Initializing Qwen3 model with the following configuration:\n");
	printf("Embed Dim: %d, Layers: %d, Heads: %d, KV Heads: %d, Head Dim: %d\n", ctx->model->embed_dim,
	       ctx->model->num_layers, ctx->model->num_heads, ctx->model->num_kv_heads, ctx->model->head_dim);
	printf("FFN Dim: %d, Rope Base: %.1f, Seq Len: %d, Vocab: %llu, Yarn Scale: "
	       "%.2f, eps: %f\n",
	       ctx->model->ffn_dim, ctx->model->rope_freq_base, ctx->model->seq_length, ctx->model->vocab_size,
	       ctx->model->yarn_scale_factor, ctx->model->norm_eps);

	printf("Initializing Qwen3 model KV cache\n");
	ctx->kv_cache = calloc(ctx->model->num_layers, sizeof(LayerKVCache));
	if (!ctx->kv_cache) {
		perror("Failed to allocate LayerKVCache array");
		return -1;
	}

	long long k_elements_per_layer =
		(long long)ctx->model->seq_length * ctx->model->num_kv_heads * ctx->model->head_dim;

	printf("KV cache elements per layer: %lld\n", k_elements_per_layer);
	for (int i = 0; i < ctx->model->num_layers; i++) {
		ctx->kv_cache[i].k = calloc(k_elements_per_layer, sizeof(float));
		ctx->kv_cache[i].v = calloc(k_elements_per_layer, sizeof(float));

		if (!ctx->kv_cache[i].k || !ctx->kv_cache[i].v) {
			perror("Failed to allocate K or V cache for a layer");
			for (int j = 0; j < i; ++j) {
				free(ctx->kv_cache[j].k);
				free(ctx->kv_cache[j].v);
			}
			free(ctx->kv_cache);
			return -1;
		}
	}

	// Initialize SiLU lookup table
	silu_table_init();

	// Initialize RoPE cache
	ctx->rope_cache = malloc(sizeof(rope_cache_t));
	if (!ctx->rope_cache) {
		perror("Failed to allocate rope_cache");
		goto _init_error_free_kv_cache;
		return -1;
	}

	rope_cache_init(ctx, ctx->model->seq_length, ctx->model->head_dim, ctx->model->rope_freq_base);

	int q_dim = ctx->model->num_heads * ctx->model->head_dim;
	int kv_dim = ctx->model->num_kv_heads * ctx->model->head_dim;

	ctx->mem.hidden_state = aligned_alloc(32, ctx->model->embed_dim * sizeof(float) * MAX_PROMPT_BATCH_SIZE);
	ctx->mem.normed_qkv_input = aligned_alloc(32, ctx->model->embed_dim * sizeof(float) * MAX_PROMPT_BATCH_SIZE);
	ctx->mem.Q = aligned_alloc(32, q_dim * sizeof(float) * MAX_PROMPT_BATCH_SIZE);
	ctx->mem.K = aligned_alloc(32, kv_dim * sizeof(float) * MAX_PROMPT_BATCH_SIZE);
	ctx->mem.V = aligned_alloc(32, kv_dim * sizeof(float) * MAX_PROMPT_BATCH_SIZE);
	ctx->mem.attn_output = aligned_alloc(32, q_dim * sizeof(float) * MAX_PROMPT_BATCH_SIZE);
	ctx->mem.attn_proj_output = aligned_alloc(32, ctx->model->embed_dim * sizeof(float) * MAX_PROMPT_BATCH_SIZE);
	ctx->mem.normed_ffn_input = aligned_alloc(32, ctx->model->embed_dim * sizeof(float) * MAX_PROMPT_BATCH_SIZE);
	ctx->mem.gate_proj_output = aligned_alloc(32, ctx->model->ffn_dim * sizeof(float) * MAX_PROMPT_BATCH_SIZE);
	ctx->mem.up_proj_output = aligned_alloc(32, ctx->model->ffn_dim * sizeof(float) * MAX_PROMPT_BATCH_SIZE);
	ctx->mem.ffn_down_output = aligned_alloc(32, ctx->model->embed_dim * sizeof(float) * MAX_PROMPT_BATCH_SIZE);
	ctx->mem.logits = aligned_alloc(32, ctx->model->vocab_size * sizeof(float));

	for (int t = 0; t < thread_pool->num_threads; t++) {
		if ((ctx->mem.attn_scores_buffer[t] =
			     aligned_alloc(32, (long long)ctx->model->seq_length * sizeof(float)))
		    == NULL)
			goto _init_error_free_kv_cache;
	}

	if (!ctx->mem.attn_output || !ctx->mem.attn_proj_output || !ctx->mem.normed_ffn_input
	    || !ctx->mem.gate_proj_output || !ctx->mem.up_proj_output || !ctx->mem.ffn_down_output || !ctx->mem.Q
	    || !ctx->mem.K || !ctx->mem.V || !ctx->mem.normed_qkv_input || !ctx->mem.hidden_state || !ctx->mem.logits) {
		goto _init_error_free_kv_cache;
	}

	return 0;

_init_error_free_kv_cache:
	for (int i = 0; i < ctx->model->num_layers; i++) {
		free(ctx->kv_cache[i].k);
		free(ctx->kv_cache[i].v);
	}

	free(ctx->kv_cache);

	return -1;
}
