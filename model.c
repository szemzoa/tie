#include <inttypes.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include "model.h"
#include "model_defs.h"
#include "gguf.h"
#include "main.h"
#include "threadpool.h"
#include "engine.h"
#include "math_dispatch.h"


#define MAX_CHUNK_SIZE (1LL << 30) // 1073741824 bytes
static ssize_t safe_pread(int fd, void *buf, uint64_t count, off_t offset)
{
	if (count == 0)
		return 0;

	ssize_t total_read = 0;
	uint8_t *ptr = (uint8_t *)buf;

	while (count > 0) {
		size_t chunk = (count > MAX_CHUNK_SIZE) ? MAX_CHUNK_SIZE : (size_t)count;
		ssize_t ret = pread(fd, ptr, chunk, offset);

		if (ret < 0) {
			return -1;
		} else if (ret == 0) {
			return total_read;
		}

		total_read += ret;
		ptr += ret;
		offset += ret;
		count -= ret;
	}

	return total_read;
}

static int model_load_tensor(struct ctx_t *ctx, char *name, Tensor *tensor, int use_mmap)
{
	gguf_tensor *ggtensor;

	if ((ggtensor = get_tensor(ctx, name)) == NULL) {
		printf("tensor not found: %s\n", name);
		return -1;
	}

	tensor->mem.type = ggtensor->type;
	tensor->size_in_bytes = ggtensor->size;

	if (use_mmap) {
		// Point directly into the mapped file data
		tensor->mem.data = ggtensor->data;
		tensor->is_mmaped = true;
	} else {
		// Allocate new memory and read the tensor data from the file
		tensor->mem.data = aligned_alloc(32, ggtensor->size);
		if (!tensor->mem.data) {
			printf("%s OOM\n", __FUNCTION__);
			return -1;
		}

		if (safe_pread(ctx->fd, tensor->mem.data, ggtensor->size, ggtensor->offset + ctx->tensor_data_offset)
		    != ggtensor->size) {
			printf("%s failed to read Tensor: %s\n", __FUNCTION__, name);
			return -1;
		}

		tensor->is_mmaped = false;
	}

	ctx->tensor_loaded++;
	return 0;
}

static const ModelDef *find_model_def(int arch)
{
	const ModelDef *def = NULL;

	switch (arch) {
	case ARCH_QWEN3:
		def = &QWEN3_DEF;
		break;

	case ARCH_GEMMA3:
		def = &GEMMA3_DEF;
		break;

	case ARCH_GEMMA3N:
		def = &GEMMA3N_DEF;
		break;
	default:
		fprintf(stderr, "Failed to detect model\n");
		break;
	}

	return def;
}

int model_load_weights(struct ctx_t *ctx, int use_mmap, int context_length)
{
	const ModelDef *def;
	char name_buffer[256];

	ctx->tensor_loaded = 0;

	if ((def = find_model_def(ctx->model->arch)) == NULL) {
		fprintf(stderr, "Failed to find model defininition for load weights\n");
		goto _model_load_error;
	}

	for (size_t i = 0; i < def->num_global_tensors; i++) {
		const TensorDef *tdef = &def->global_tensors[i];

		// Calculate the destination pointer into the main Model struct
		Tensor *dest_tensor = (Tensor *)((uint8_t *)ctx->model + tdef->offset);

		if (model_load_tensor(ctx, (char *)tdef->name_fmt, dest_tensor, use_mmap) != 0) {

			if (tdef->flags & FLAG_OPTIONAL) {
				// This is not an error, just print an info message
				printf("Info: Optional tensor '%s' not found.\n", tdef->name_fmt);
			} else {
				fprintf(stderr, "Failed to load required tensor %s\n", tdef->name_fmt);
				goto _model_load_error;
			}
		}
	}

	ctx->model->layers = calloc(ctx->model->num_layers, sizeof(layer_weights));

	for (uint32_t block_idx = 0; block_idx < ctx->model->num_layers; block_idx++) {
		for (size_t i = 0; i < def->num_layer_tensors; i++) {
			const TensorDef *tdef = &def->layer_tensors[i];

			if ((tdef->flags & FLAG_DENSE_ONLY) && ctx->model->is_moe)
				continue;

			if ((tdef->flags & FLAG_MOE_ONLY) && !ctx->model->is_moe)
				continue;

			layer_weights *layer = &ctx->model->layers[block_idx];
			Tensor *dest_tensor = (Tensor *)((uint8_t *)layer + tdef->offset);

			snprintf(name_buffer, sizeof(name_buffer), tdef->name_fmt, block_idx);
			if (model_load_tensor(ctx, name_buffer, dest_tensor, use_mmap) != 0) {

				if (tdef->flags & FLAG_OPTIONAL) {
					// This is not an error, just print an info message
					printf("Info: Optional tensor '%s' not found.\n", tdef->name_fmt);
				} else {
					fprintf(stderr, "Failed to load required tensor %s\n", tdef->name_fmt);
					goto _model_load_error_layer;
				}
			}
		}
	}

	return 0;

_model_load_error_layer:
	free(ctx->model->layers);

_model_load_error:
	return -1;
}

int model_load_metadata(struct ctx_t *ctx)
{
	const ModelDef *def;
	char key_buffer[256];

	if ((def = find_model_def(ctx->model->arch)) == NULL) {
		fprintf(stderr, "Failed to find model defininition for metadata\n");
		goto _model_load_metadata_error;
	}

	for (size_t i = 0; i < def->num_metadata_defs; i++) {
		const MetadataDef *mdef = &def->metadata_defs[i];

		snprintf(key_buffer, sizeof(key_buffer), mdef->key_fmt, ctx->model->arch_name);

		// Calculate the destination pointer in the Model struct
		void *dest_ptr = (uint8_t *)ctx->model + mdef->offset;

		if (gguf_get_metadata_value(ctx, key_buffer, dest_ptr) != 0) {
			if (!mdef->is_optional) {
				fprintf(stderr, "Failed to load required metadata key: %s\n", key_buffer);
				goto _model_load_metadata_error;
			}
		}
	}

	/* fallbacks */
	if (ctx->model->rope_scale_factor == 0.0f) {
		ctx->model->rope_scale_factor = 1.0f;
	}

	if (ctx->model->expert_count != 0) {
		ctx->model->is_moe = 1;
	} else {
		ctx->model->is_moe = 0;
	}

	printf("Info: %s model detected\n", ctx->model->is_moe == 0 ? "Dense" : "MoE");

	return 0;

_model_load_metadata_error:
	return -1;
}

int model_mem_init(struct ctx_t *ctx)
{
	const ModelDef *def;

	if ((def = find_model_def(ctx->model->arch)) == NULL) {
		fprintf(stderr, "Failed to find model defininition for memory init\n");
		return -1;
	}

	printf("Initializing memory buffers...\n");

	// Standard Buffers
	for (size_t i = 0; i < def->num_buffer_defs; i++) {
		const BufferDef *bdef = &def->buffer_defs[i];

		// Skip conditional buffers if the condition isn't met
		if ((bdef->flags & FLAG_MOE_ONLY) && !ctx->model->is_moe)
			continue;

		MemType *dest_buffer = (MemType *)((uint8_t *)&ctx->mem + bdef->offset);
		size_t size_multiplier = 0;

		// Resolve size type to runtime value
		switch (bdef->size_type) {

		case SIZE_EMBED_DIM:
			size_multiplier = ctx->model->embed_dim;
			break;
		case SIZE_VOCAB_SIZE:
			size_multiplier = ctx->model->vocab_size;
			break;
		case SIZE_FFN_DIM:
			size_multiplier = ctx->model->ffn_dim;
			break;
		case SIZE_Q_DIM:
			size_multiplier = ctx->model->num_heads * ctx->model->head_dim;
			break;
		case SIZE_KV_DIM:
			size_multiplier = ctx->model->num_kv_heads * ctx->model->head_dim;
			break;
		case SIZE_NUM_LAYERS_X_PLI_DIM:
			size_multiplier = ctx->model->num_layers * ctx->model->pli_dim;
			break;
			//            case SIZE_EXPERT_COUNT: size_multiplier = ctx->model->expert_count;
			//        	    break;
		}

		alloc_memtype(dest_buffer, bdef->type, size_multiplier * MAX_PROMPT_BATCH_SIZE);
	}

	// Explicit Block for Special MoE Array Buffers
	if (ctx->model->is_moe) {

		alloc_memtype(&ctx->mem.expert_scores, GGML_TYPE_F32, ctx->model->expert_count);
		alloc_memtype(&ctx->mem.expert_out_fp32, GGML_TYPE_F32, ctx->model->embed_dim);

		ctx->mem.ffn_hidden1_scratch = malloc(ctx->model->expert_count * sizeof(MemType));
		ctx->mem.ffn_hidden2_scratch = malloc(ctx->model->expert_count * sizeof(MemType));
		ctx->mem.expert_outputs = malloc(ctx->model->expert_count * sizeof(MemType));
		// ... null checks ...

		for (int i = 0; i < ctx->model->expert_count; i++) {
			alloc_memtype(&ctx->mem.ffn_hidden1_scratch[i], GGML_TYPE_BF16, ctx->model->expert_ffn_dim);
			alloc_memtype(&ctx->mem.ffn_hidden2_scratch[i], GGML_TYPE_BF16, ctx->model->expert_ffn_dim);
			alloc_memtype(&ctx->mem.expert_outputs[i], GGML_TYPE_F32, ctx->model->embed_dim);
		}
	}

	// Special Per-Thread Buffers
	ctx->mem.q_head_fp32_scratch = malloc(thread_pool->num_threads * sizeof(MemType));
	// ... null check ...

	for (int t = 0; t < thread_pool->num_threads; t++) {
		alloc_memtype(&ctx->mem.q_head_fp32_scratch[t], GGML_TYPE_F32, ctx->model->head_dim);
		ctx->mem.attn_scores_buffer[t] = aligned_alloc(32, (long long)ctx->model->seq_length * sizeof(float));
		// ... null check ...
	}

	// Special AltUp Buffers
	if (ctx->model->altup_num_inputs > 0) {

		ctx->mem.altup_hidden_states = malloc(ctx->model->altup_num_inputs * sizeof(MemType));
		ctx->mem.altup_predicted_states = malloc(ctx->model->altup_num_inputs * sizeof(MemType));

		for (int i = 0; i < ctx->model->altup_num_inputs; i++) {
			alloc_memtype(&ctx->mem.altup_hidden_states[i], GGML_TYPE_F32,
				      ctx->model->embed_dim * MAX_PROMPT_BATCH_SIZE);

			alloc_memtype(&ctx->mem.altup_predicted_states[i], GGML_TYPE_F32,
				      ctx->model->embed_dim * MAX_PROMPT_BATCH_SIZE);
		}
	}

	return 0;
}

int model_load(struct ctx_t *ctx, int use_mmap, int context_length)
{
	int detect_special;

	if ((ctx->model = malloc(sizeof(Model))) == NULL) {
		perror("Failed to allocate model");
		return -1;
	}

	ctx->model->arch = ARCH_UNKNOWN;
	ctx->model->shared_kv_layers = 0;
	ctx->model->final_logit_softcap = 0;
	ctx->model->weight_layout = LAYOUT_ROW_MAJOR;

	ctx->model->arch_name = gguf_get_metadata_string(ctx, "general.architecture");

	if (!strncmp(ctx->model->arch_name, "gemma3n", 7)) {
		ctx->model->arch = ARCH_GEMMA3N;
	} else if (strstr(ctx->model->arch_name, "gemma3")) {
		ctx->model->arch = ARCH_GEMMA3;
	} else if (strstr(ctx->model->arch_name, "qwen3") || strstr(ctx->model->arch_name, "qwen3moe")) {
		ctx->model->arch = ARCH_QWEN3;
	} else {
		perror("Failed to detect model");
		free(ctx->model);
		return -1;
	}

	if (model_load_metadata(ctx) != 0) {
		goto _metadata_load_error;
	}

	if (context_length != 0) {
		ctx->model->seq_length = context_length;
		printf("set context_length: %u\n", ctx->model->seq_length);
	}

	gguf_get_metadata_value(ctx, "tokenizer.ggml.eos_token_id", &ctx->model->eos_token_id);
	gguf_get_metadata_value(ctx, "tokenizer.ggml.bos_token_id", &ctx->model->bos_token_id);
	gguf_get_metadata_value(ctx, "tokenizer.ggml.unknown_token_id", &ctx->model->unk_token_id);
	gguf_get_metadata_value(ctx, "tokenizer.ggml.pad_token_id", &ctx->model->pad_token_id);
	gguf_get_metadata_value(ctx, "tokenizer.ggml.add_bos_token", &ctx->model->add_bos_token);
	gguf_get_metadata_value(ctx, "tokenizer.ggml.add_eos_token", &ctx->model->add_eos_token);

	detect_special = 0;
	if (ctx->model->arch == ARCH_QWEN3)
		detect_special = 1;

	/* load tokens */
	if (gguf_metadata_read_token_embeds(ctx, "tokenizer.ggml.tokens", detect_special) != 0) {
		free(ctx->model);
		perror("Failed to read tokens_embed array");
		return -1;
	}

	/* load token types */
	if (gguf_metadata_read_token_types(ctx, "tokenizer.ggml.token_type", detect_special) != 0) {
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
	if (detect_special == 0) {
		if (gguf_metadata_read_token_scores(ctx, "tokenizer.ggml.scores") != 0) {
			free(ctx->model);
			perror("Failed to read ggml.scores array");
			return -1;
		}
	}

	if (model_load_weights(ctx, use_mmap, context_length) != 0) {
		printf("Failed to load model weights\n");
		free(ctx->model);
		return -1;
	}

#ifdef DEBUG_TENSORS
	dump_tensors(ctx);
#endif

	printf("Loaded %llu/%llu tensors\n", ctx->tensor_loaded, ctx->tensor_count);
	return 0;

_metadata_load_error:
	free(ctx->model);

	return -1;
}

int init_KV_cache(struct ctx_t *ctx)
{
	printf("Initializing KV cache\n");

	ctx->kv_cache = calloc(ctx->model->num_layers, sizeof(LayerKVCache));

	if (!ctx->kv_cache) {
		perror("Failed to allocate LayerKVCache array");
		return -1;
	}

	long long k_elements_per_layer =
		(long long)ctx->model->seq_length * ctx->model->num_kv_heads * ctx->model->head_dim;

	printf("KV cache elements per layer: %lld\n", k_elements_per_layer);
	for (int i = 0; i < ctx->model->num_layers; i++) {
		alloc_memtype(&ctx->kv_cache[i].k, GGML_TYPE_BF16, k_elements_per_layer);
		alloc_memtype(&ctx->kv_cache[i].v, GGML_TYPE_BF16, k_elements_per_layer);

		if (!ctx->kv_cache[i].k.data || !ctx->kv_cache[i].v.data) {
			perror("Failed to allocate K or V cache for a layer");
			for (int j = 0; j < i; ++j) {
				free_memtype(&ctx->kv_cache[j].k);
				free_memtype(&ctx->kv_cache[j].v);
			}
			free(ctx->kv_cache);
			return -1;
		}
	}
	printf("KV cache uses: %lld MB\n",
	       (k_elements_per_layer * sizeof(uint16_t)) * 2 * ctx->model->num_layers / 1024 / 1024);

	ctx->kv_pos = 0;
	return 0;
}


int model_init(struct ctx_t *ctx, float yarn_scale_factor, float repetiton_penality)
{
	const ModelDef *def;
	char *general_name = gguf_get_metadata_string(ctx, "general.name");


	if ((def = find_model_def(ctx->model->arch)) == NULL) {
		perror("Failed to find model");
		return -1;
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

	ctx->model->name = def->name;
	ctx->model->yarn_scale_factor = yarn_scale_factor;
	ctx->model->repetition_penalty = repetiton_penality;
	ctx->model->interface = def->interface;
	ctx->model->sot_token_id = def->params.sot_token_id;
	ctx->model->eot_token_id = def->params.eot_token_id;
	ctx->model->eos_token_id = def->params.eos_token_id;
	ctx->model->newline_token_id = def->params.newline_token_id;
	ctx->model->role_user_token_id = def->params.role_user_token_id;
	ctx->model->role_model_token_id = def->params.role_model_token_id;
	ctx->model->shared_kv_layers = def->params.shared_kv_layers;
	ctx->model->final_logit_softcap = def->params.final_logit_softcap;

	switch (ctx->model->arch) {
	case ARCH_QWEN3:
		ctx->model->attn_scale = 1.0f / sqrtf((float)ctx->model->head_dim);

		rope_cache_init(ctx, ctx->rope_cache_global, ctx->model->seq_length, ctx->model->head_dim,
				ctx->model->rope_freq_base, ctx->model->yarn_scale_factor);
		break;

	case ARCH_GEMMA3:
		ctx->model->attn_scale = 1.0f;

		rope_cache_init(ctx, ctx->rope_cache_global, ctx->model->seq_length, ctx->model->head_dim,
				ctx->model->rope_freq_base, ctx->model->rope_scale_factor);

		rope_cache_init(ctx, ctx->rope_cache_local, ctx->model->seq_length, ctx->model->head_dim, 10000.0f, 1.0f);
		break;

	case ARCH_GEMMA3N:
		ctx->model->attn_scale = 1.0f;

		if (ctx->model->num_layers == 30) {
			ctx->model->shared_kv_layers = 10;
			ctx->model->ffn_dim = 8192;
		} else if (ctx->model->num_layers == 35) {
			ctx->model->shared_kv_layers = 15;
			ctx->model->ffn_dim = 16384;
		}

		rope_cache_init(ctx, ctx->rope_cache_global, ctx->model->seq_length, ctx->model->head_dim,
				ctx->model->rope_freq_base, 1.0f);

		rope_cache_init(ctx, ctx->rope_cache_local, ctx->model->seq_length, ctx->model->head_dim, 10000.0f, 1.0f);
		break;
	}

	if (init_KV_cache(ctx) != 0) {
		free(ctx->rope_cache_local);
		free(ctx->rope_cache_global);
		return -1;
	}

	if (model_mem_init(ctx) != 0) {
		for (int i = 0; i < ctx->model->num_layers; i++) {
			free_memtype(&ctx->kv_cache[i].k);
			free_memtype(&ctx->kv_cache[i].v);
		}

		free(ctx->kv_cache);
		free(ctx->rope_cache_local);
		free(ctx->rope_cache_global);

		return -1;
	}

	printf("Initialized %s %s model with the following configuration:\n", general_name,
	       ctx->model->is_moe == 0 ? "dense" : "MoE");
	printf("Embed Dim: %d, Layers: %d, Heads: %d, KV Heads: %d, Head Dim: %d, Shared KV layers: %u\n",
	       ctx->model->embed_dim, ctx->model->num_layers, ctx->model->num_heads, ctx->model->num_kv_heads,
	       ctx->model->head_dim, ctx->model->shared_kv_layers);
	printf("FFN Dim: %d, Rope Base: %.1f, Seq Len: %d, Vocab: %llu\n", ctx->model->ffn_dim,
	       ctx->model->rope_freq_base, ctx->model->seq_length, ctx->model->vocab_size);
	printf("Yarn Scale: %.2f, eps: %f, rope_scale: %.1f, sliding_window: %u\n", ctx->model->yarn_scale_factor,
	       ctx->model->norm_eps, ctx->model->rope_scale_factor, ctx->model->attn_sliding_window);
	if (ctx->model->is_moe == 1)
		printf("Expert Count: %d, Expert Used Count: %d, Expert FFN Dim: %d\n", ctx->model->expert_count,
		       ctx->model->expert_used_count, ctx->model->expert_ffn_dim);
	if (ctx->model->altup_num_inputs > 0)
		printf("Altup Num Inputs: %i\n", ctx->model->altup_num_inputs);

	return 0;
}

void model_cleanup(struct ctx_t *ctx, int use_mmap)
{
	const ModelDef *def = find_model_def(ctx->model->arch);
	if (!def) {
		fprintf(stderr, "Error: Could not find model definition for cleanup.\n");
		return;
	}

	printf("cleanup...\n");

	// Standard Buffers
	for (size_t i = 0; i < def->num_buffer_defs; i++) {
		const BufferDef *bdef = &def->buffer_defs[i];
		if ((bdef->flags & FLAG_MOE_ONLY) && !ctx->model->is_moe)
			continue;
		MemType *dest_buffer = (MemType *)((uint8_t *)&ctx->mem + bdef->offset);
		free_memtype(dest_buffer);
	}

	// MoE Buffers
	if (ctx->model->is_moe == 1) {
		for (int i = 0; i < ctx->model->expert_count; i++) {
			free_memtype(&ctx->mem.ffn_hidden1_scratch[i]);
			free_memtype(&ctx->mem.ffn_hidden2_scratch[i]);
			free_memtype(&ctx->mem.expert_outputs[i]);
		}
		free(ctx->mem.ffn_hidden1_scratch);
		free(ctx->mem.ffn_hidden2_scratch);
		free(ctx->mem.expert_outputs);

		free_memtype(&ctx->mem.expert_out_fp32);
		free_memtype(&ctx->mem.expert_scores);
	}

	// Per-Thread Buffers
	for (int t = 0; t < thread_pool->num_threads; t++) {
		free_memtype(&ctx->mem.q_head_fp32_scratch[t]);
		free(ctx->mem.attn_scores_buffer[t]);
	}
	free(ctx->mem.q_head_fp32_scratch);

	// AltUp Buffers
	if (ctx->model->altup_num_inputs > 0) {
		for (int i = 0; i < ctx->model->altup_num_inputs; i++) {
			free_memtype(&ctx->mem.altup_hidden_states[i]);
			free_memtype(&ctx->mem.altup_predicted_states[i]);
		}
		free(ctx->mem.altup_hidden_states);
		free(ctx->mem.altup_predicted_states);
	}

	// Loaded Weights
	if (use_mmap == 0) {
		// Global Tensors
		for (size_t i = 0; i < def->num_global_tensors; i++) {
			const TensorDef *tdef = &def->global_tensors[i];
			Tensor *dest_tensor = (Tensor *)((uint8_t *)ctx->model + tdef->offset);
			if (dest_tensor->mem.data) {
				free(dest_tensor->mem.data);
			}
		}

		// Layer Tensors
		for (uint32_t block_idx = 0; block_idx < ctx->model->num_layers; block_idx++) {
			for (size_t i = 0; i < def->num_layer_tensors; i++) {
				const TensorDef *tdef = &def->layer_tensors[i];
				if ((tdef->flags & FLAG_DENSE_ONLY) && ctx->model->is_moe)
					continue;
				if ((tdef->flags & FLAG_MOE_ONLY) && !ctx->model->is_moe)
					continue;

				layer_weights *layer = &ctx->model->layers[block_idx];
				Tensor *dest_tensor = (Tensor *)((uint8_t *)layer + tdef->offset);
				if (dest_tensor->mem.data) {
					free(dest_tensor->mem.data);
				}
			}
		}
	}
	free(ctx->model->layers);

	for (int i = 0; i < ctx->tensor_count; i++) {
		gguf_tensor *tensor = &ctx->tensors[i];
		free(tensor->name);
	}
	free(ctx->tensors);

	for (uint32_t i = 0; i < ctx->model->num_layers; i++) {
		free_memtype(&ctx->kv_cache[i].k);
		free_memtype(&ctx->kv_cache[i].v);
	}
	free(ctx->kv_cache);

	thread_pool_destroy(thread_pool);
	free(ctx->model);

	free(ctx->rope_cache_local->sin);
	free(ctx->rope_cache_local->cos);
	free(ctx->rope_cache_local);

	if (ctx->rope_cache_global) {
		free(ctx->rope_cache_global->sin);
		free(ctx->rope_cache_global->cos);
		free(ctx->rope_cache_global);
	}

	free(ctx->tokenizer.token_table);
	free(ctx->tokenizer.token_lens);
	free(ctx->tokenizer.token_types);
	free(ctx->tokenizer.token_scores);

	free_trie(ctx->tokenizer.root);
	free_string_pool(ctx->tokenizer.pool);
}
