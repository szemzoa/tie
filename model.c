#include <inttypes.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#include "gguf.h"
#include "threadpool.h"
#include "main.h"
#include "maths.h"
#include "engine.h"


int load_tensor(struct ctx_t *ctx, char *name, Tensor *tensor, int use_mmap)
{
	gguf_tensor *ggtensor;

	if ((ggtensor = get_tensor(ctx, name)) == NULL)
		return -1;

	// Populate the Tensor struct
	tensor->type = ggtensor->type; // e.g., GGUF_TYPE_Q6_K
	tensor->size_in_bytes = ggtensor->size;

	if (use_mmap) {
		// Point directly into the mapped file data
		tensor->data = ggtensor->data;
		tensor->is_mmaped = true;
	} else {
		// Allocate new memory and read the tensor data from the file
		tensor->data = aligned_alloc(128, ggtensor->size);
		if (!tensor->data) {
			printf("%s OOM\n", __FUNCTION__);
			return -1;
		}

		pread(ctx->fd, tensor->data, ggtensor->size, ggtensor->offset + ctx->tensor_data_offset);
		tensor->is_mmaped = false;
	}

	ctx->tensor_loaded++;
	return 0;
}

int model_create(struct ctx_t *ctx, int use_mmap)
{
	char tensor_name_buffer[256];

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

	ctx->tensor_loaded = 0;

	if (load_tensor(ctx, "token_embd.weight", &ctx->model->token_embd, use_mmap) != 0) {
		fprintf(stderr, "Failed to load token_embd.weight\n");
		return -1;
	}

	if (load_tensor(ctx, "output_norm.weight", &ctx->model->output_norm, use_mmap) != 0) {
		fprintf(stderr, "Failed to load token_embd.weight\n");
		return -1;
	}

	if (load_tensor(ctx, "output.weight", &ctx->model->output, use_mmap) != 0) {
		printf("Model using tied embedding, no output weight\n");
	}

	if ((ctx->model->layers = calloc(ctx->model->num_layers, sizeof(layer_weights))) == NULL) {
		free(ctx->model);
		perror("Failed to allocate layer_weights array");
		return -1;
	}

#define LOAD_LAYER_WEIGHT(field, name_fmt_str)                                                                         \
	snprintf(tensor_name_buffer, sizeof(tensor_name_buffer), name_fmt_str, block_idx);                             \
	if (load_tensor(ctx, tensor_name_buffer, &ctx->model->layers[block_idx].field, use_mmap) != 0) {               \
		fprintf(stderr, "Failed to load %s\n", tensor_name_buffer);                                            \
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
		LOAD_LAYER_WEIGHT(ffn_gate, "blk.%u.ffn_gate.weight");
		LOAD_LAYER_WEIGHT(ffn_up, "blk.%u.ffn_up.weight");
		LOAD_LAYER_WEIGHT(ffn_down, "blk.%u.ffn_down.weight");
		LOAD_LAYER_WEIGHT(ffn_norm, "blk.%u.ffn_norm.weight");
	}

#undef LOAD_LAYER_WEIGHT

	if (ctx->tensor_count != ctx->tensor_loaded) {
		printf("Tensor load failed!!!! load: %llu, all: %llu", ctx->tensor_loaded, ctx->tensor_count);
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

void model_cleanup(struct ctx_t *ctx, int use_mmap)
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

	if (use_mmap == 1) {

		free(ctx->model->token_embd.data);
		free(ctx->model->output_norm.data);

		if (ctx->model->output.data)
			free(ctx->model->output.data);

		for (uint32_t block_idx = 0; block_idx < ctx->model->num_layers; block_idx++) {
			free(ctx->model->layers[block_idx].attn_q.data);
			free(ctx->model->layers[block_idx].attn_k.data);
			free(ctx->model->layers[block_idx].attn_v.data);
			free(ctx->model->layers[block_idx].attn_norm.data);
			free(ctx->model->layers[block_idx].attn_q_norm.data);
			free(ctx->model->layers[block_idx].attn_k_norm.data);
			free(ctx->model->layers[block_idx].attn_out.data);
			free(ctx->model->layers[block_idx].ffn_gate.data);
			free(ctx->model->layers[block_idx].ffn_up.data);
			free(ctx->model->layers[block_idx].ffn_down.data);
			free(ctx->model->layers[block_idx].ffn_norm.data);
		}
	}

	free(ctx->model->layers);
	free(ctx->model);
	free(ctx->rope_cache);

	free(ctx->token_table);
	free(ctx->token_lens);

	free_trie(ctx->root);
	free_string_pool(ctx->pool);
}

int model_init(struct ctx_t *ctx, float yarn_scale_factor, float repetiton_penality)
{
	ctx->model->yarn_scale_factor = yarn_scale_factor;
	ctx->model->repetition_penalty = repetiton_penality;
	ctx->model->attn_scale = 1.0f / sqrtf(ctx->model->head_dim);

	printf("Initializing %s model with the following configuration:\n", gguf_get_metadata_string(ctx, "general.architecture"));
	printf("Embed Dim: %d, Layers: %d, Heads: %d, KV Heads: %d, Head Dim: %d\n", ctx->model->embed_dim,
	       ctx->model->num_layers, ctx->model->num_heads, ctx->model->num_kv_heads, ctx->model->head_dim);
	printf("FFN Dim: %d, Rope Base: %.1f, Seq Len: %d, Vocab: %llu, Yarn Scale: "
	       "%.2f, eps: %f\n",
	       ctx->model->ffn_dim, ctx->model->rope_freq_base, ctx->model->seq_length, ctx->model->vocab_size,
	       ctx->model->yarn_scale_factor, ctx->model->norm_eps);


	// Initialize SiLU lookup table
	silu_table_init();

	// Initialize RoPE cache
	ctx->rope_cache = malloc(sizeof(rope_cache_t));
	if (!ctx->rope_cache) {
		perror("Failed to allocate rope_cache");
		return -1;
	}
	rope_cache_init(ctx, ctx->model->seq_length, ctx->model->head_dim, ctx->model->rope_freq_base);


	printf("Initializing KV cache\n");
	ctx->kv_cache = calloc(ctx->model->num_layers, sizeof(LayerKVCache));
	if (!ctx->kv_cache) {
		perror("Failed to allocate LayerKVCache array");
		free(ctx->rope_cache);
		return -1;
	}

	long long k_elements_per_layer =
		(long long)ctx->model->seq_length * ctx->model->num_kv_heads * ctx->model->head_dim;

	printf("KV cache elements per layer: %lld\n", k_elements_per_layer);
	for (int i = 0; i < ctx->model->num_layers; i++) {
		ctx->kv_cache[i].k = calloc(k_elements_per_layer, sizeof(uint16_t));
		ctx->kv_cache[i].v = calloc(k_elements_per_layer, sizeof(uint16_t));

		if (!ctx->kv_cache[i].k || !ctx->kv_cache[i].v) {
			perror("Failed to allocate K or V cache for a layer");
			for (int j = 0; j < i; ++j) {
				free(ctx->kv_cache[j].k);
				free(ctx->kv_cache[j].v);
			}
			free(ctx->kv_cache);
			free(ctx->rope_cache);
			return -1;
		}
	}
	printf("KV cache uses: %lld MB\n", (k_elements_per_layer * sizeof(uint16_t)) * 2 * ctx->model->num_layers / 1024 / 1024);


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
	free(ctx->rope_cache);

	return -1;
}
