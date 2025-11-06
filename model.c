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
#include "vision.h"

const struct arch_t known_archs[] = {
	{"qwen3", ARCH_QWEN3},	   {"qwen3moe", ARCH_QWEN3_MOE}, {"gemma3", ARCH_GEMMA3},
	{"gemma3n", ARCH_GEMMA3N}, {"clip", ARCH_CLIP_VISION},	 {"qwen3vl", ARCH_QWEN3VL},
};

const struct arch_t known_projectors[] = {{"gemma3", VISION_PROJECTOR_GEMMA3},
					  {"qwen3vl_merger", VISION_PROJECTOR_QWEN3VL}};

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

int detect_architecture(const char *model_name)
{
	const struct arch_t *best_match = NULL;
	size_t best_len = 0;

	for (int i = 0; i < ARRAY_SIZE(known_archs); i++) {
		const struct arch_t *current_arch = &known_archs[i];

		if (strstr(model_name, current_arch->name)) {
			size_t current_len = strlen(current_arch->name);

			if (current_len > best_len) {
				best_len = current_len;
				best_match = current_arch;
			}
		}
	}

	if (best_match) {
		return best_match->id;
	}

	return -1;
}

int detect_projector(const char *model_name)
{
	const struct arch_t *best_match = NULL;
	size_t best_len = 0;

	for (int i = 0; i < ARRAY_SIZE(known_projectors); i++) {
		const struct arch_t *current_projector = &known_projectors[i];

		if (strstr(model_name, current_projector->name)) {
			size_t current_len = strlen(current_projector->name);

			if (current_len > best_len) {
				best_len = current_len;
				best_match = current_projector;
			}
		}
	}

	if (best_match) {
		return best_match->id;
	}

	return -1;
}

ModelDef *find_projector_def(struct GGUFModel *gguf_model)
{
	ModelDef *def = NULL;
	int projector;

	projector = detect_projector(gguf_metadata_get_string(gguf_model, "clip.projector_type"));

	switch (projector) {

	case VISION_PROJECTOR_GEMMA3:
		def = &GEMMA3_CLIP_DEF;
		break;
	case VISION_PROJECTOR_QWEN3VL:
		def = &QWEN3VL_CLIP_DEF;
		break;
	default:
		fprintf(stderr, "Failed to detect vision model\n");
	}

	return def;
}

ModelDef *find_model_def(struct GGUFModel *gguf_model)
{
	ModelDef *def = NULL;

	switch (gguf_model->arch) {
	case ARCH_QWEN3:
		def = &QWEN3_DEF;
		break;
	case ARCH_QWEN3_MOE:
		def = &QWEN3_MOE_DEF;
		break;
	case ARCH_QWEN3VL:
		def = &QWEN3VL_DEF;
		break;
	case ARCH_GEMMA3:
		def = &GEMMA3_DEF;
		break;
	case ARCH_GEMMA3N:
		def = &GEMMA3N_DEF;
		break;
	case ARCH_CLIP_VISION:
		return find_projector_def(gguf_model);
		break;
	default:
		fprintf(stderr, "Failed to detect model\n");
	}

	return def;
}

int init_KV_cache(struct TIEContext *ctx)
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
		alloc_memtype(&ctx->kv_cache[i].k, KV_CACHE_TYPE, k_elements_per_layer);
		alloc_memtype(&ctx->kv_cache[i].v, KV_CACHE_TYPE, k_elements_per_layer);

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

static int model_load_tensor(struct GGUFModel *model, char *name, Tensor *tensor, int use_mmap)
{
	GGUFTensor *gguf_tensor;

	if ((gguf_tensor = gguf_get_tensor(model, name)) == NULL)
		return -1;

	tensor->mem.type = gguf_tensor->type;
	tensor->size_in_bytes = gguf_tensor->size;

	for (int i = 0; i < 4; i++)
		tensor->dimensions[i] = gguf_tensor->dimensions[i];

	if (use_mmap) {
		// Point directly into the mapped file data
		tensor->mem.data = gguf_tensor->data;
		tensor->is_mmaped = true;
	} else {
		// Allocate new memory and read the tensor data from the file
		tensor->mem.data = aligned_alloc(32, gguf_tensor->size);
		if (!tensor->mem.data) {
			printf("%s OOM\n", __FUNCTION__);
			return -1;
		}

		if (safe_pread(model->fd, tensor->mem.data, gguf_tensor->size,
			       gguf_tensor->offset + model->tensor_data_offset)
		    != gguf_tensor->size) {
			printf("%s failed to read Tensor: %s\n", __FUNCTION__, name);
			return -1;
		}

		tensor->is_mmaped = false;
	}

	model->tensor_loaded++;
	return 0;
}

int model_load_weights(struct GGUFModel *gguf, void *model_struct, const ModelDef *def, int use_mmap)
{
	char name_buffer[256];
	gguf->tensor_loaded = 0;

	// Load Global Tensors
	for (size_t i = 0; i < def->num_global_tensors; i++) {
		const TensorDef *tdef = &def->global_tensors[i];

		Tensor *dest_tensor = (Tensor *)((uint8_t *)model_struct + tdef->offset);

		if (model_load_tensor(gguf, (char *)tdef->name_fmt, dest_tensor, use_mmap) != 0) {
			if (tdef->flags & FLAG_OPTIONAL) {
				printf("Info: Optional tensor '%s' not found.\n", tdef->name_fmt);
			} else {
				fprintf(stderr, "Failed to load required tensor %s\n", tdef->name_fmt);
				return -1;
			}
		}
	}

	// Allocate and Load Layer Tensors
	uint32_t num_layers = *(uint32_t *)((uint8_t *)model_struct + def->num_layers_offset);

	// Get a pointer to the 'layers' member of the model_struct
	void **layers_array_ptr = (void **)((uint8_t *)model_struct + def->layers_offset);
	*layers_array_ptr = calloc(num_layers, def->layer_struct_size);
	if (!*layers_array_ptr) {
		perror("Failed to allocate layers array");
		return -1;
	}

	for (uint32_t i = 0; i < num_layers; i++) {
		for (size_t j = 0; j < def->num_layer_tensors; j++) {
			const TensorDef *tdef = &def->layer_tensors[j];

			// Get a pointer to the current layer struct
			void *current_layer = (uint8_t *)(*layers_array_ptr) + i * def->layer_struct_size;
			Tensor *dest_tensor = (Tensor *)((uint8_t *)current_layer + tdef->offset);

			snprintf(name_buffer, sizeof(name_buffer), tdef->name_fmt, i);
			if (model_load_tensor(gguf, name_buffer, dest_tensor, use_mmap) != 0) {

				if (tdef->flags & FLAG_OPTIONAL) {
					// This is not an error, just print an info message
					printf("Info: Optional tensor '%s' not found.\n", tdef->name_fmt);
				} else {
					fprintf(stderr, "Failed to load required tensor %s\n", tdef->name_fmt);
					return -1;
				}
			}
		}
	}

	printf("Loaded %llu/%llu tensors\n", gguf->tensor_loaded, gguf->tensor_count);
	return 0;
}

int model_load_metadata(struct GGUFModel *gguf, void *model, const ModelDef *def)
{
	char key_buffer[256];

	// Get the architecture name from the GGUF file being processed
	const char *arch_name = gguf_metadata_get_string(gguf, "general.architecture");
	if (!arch_name) {
		fprintf(stderr, "GGUF file is missing general.architecture metadata.\n");
		return -1;
	}

	// Loop through the definitions and populate the new struct
	for (size_t i = 0; i < def->num_metadata_defs; i++) {
		const MetadataDef *mdef = &def->metadata_defs[i];

		if (strstr(mdef->key_fmt, "%s")) {
			snprintf(key_buffer, sizeof(key_buffer), mdef->key_fmt, arch_name);
		} else {
			snprintf(key_buffer, sizeof(key_buffer), mdef->key_fmt, NULL);
		}

		// Calculate the destination pointer relative to the NEWLY allocated struct
		void *dest_ptr = (uint8_t *)model + mdef->offset;

		if (mdef->is_array) {
			uint64_t array_size = 0;

			void *array_data = gguf_metadata_get_array_typed(gguf, key_buffer, GGUF_METADATA_VALUE_TYPE_ARRAY, &array_size);

			if (array_data) {
				// Get the target ModelArray struct in our model
				ModelArray *target_array = (ModelArray *)((char *)dest_ptr);

				// Get total size in bytes
				size_t element_size = gguf_get_type_size(mdef->type);
				size_t total_bytes = element_size * array_size;

				// Allocate memory and copy the data to take ownership
				target_array->data = malloc(total_bytes);
				if (!target_array->data) { /* handle error */
					fprintf(stderr, "Failed to load required metadata key: %s\n", key_buffer);
		    			return -1;
				}
				memcpy(target_array->data, array_data, total_bytes);

				// Store the size and type
				target_array->size = array_size;
				target_array->type = mdef->type;
			} else {

				fprintf(stderr, "Failed to load required metadata key: %s\n", key_buffer);
				return -1;
			}
		} else {

			if (gguf_metadata_get_value(gguf, key_buffer, dest_ptr) != 0) {
				if (!mdef->is_optional) {
					fprintf(stderr, "Failed to load required metadata key: %s\n", key_buffer);
					return -1;
				}
			}
		}
	}

	return 0;
}

int model_init_mem_language(struct TIEContext *ctx, const ModelDef *def)
{
	printf("Initializing memory buffers...\n");

	// Standard Buffers
	for (size_t i = 0; i < def->num_buffer_defs; i++) {
		const BufferDef *bdef = &def->buffer_defs[i];

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
		default:
			return -1;
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
		// TODO null checks

		for (int i = 0; i < ctx->model->expert_count; i++) {
			alloc_memtype(&ctx->mem.ffn_hidden1_scratch[i], GGML_TYPE_BF16, ctx->model->expert_ffn_dim);
			alloc_memtype(&ctx->mem.ffn_hidden2_scratch[i], GGML_TYPE_BF16, ctx->model->expert_ffn_dim);
			alloc_memtype(&ctx->mem.expert_outputs[i], GGML_TYPE_F32, ctx->model->embed_dim);
		}
	}

	// Special Per-Thread Buffers
	ctx->mem.q_head_fp32_scratch = malloc(thread_pool->num_threads * sizeof(MemType));
	// TODO null checks

	for (int t = 0; t < thread_pool->num_threads; t++) {
		alloc_memtype(&ctx->mem.q_head_fp32_scratch[t], GGML_TYPE_F32, ctx->model->head_dim);
		ctx->mem.attn_scores_buffer[t] = aligned_alloc(32, (long long)ctx->model->seq_length * sizeof(float));
		// TODO null checks
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

int model_load(struct TIEContext *ctx, struct GGUFModel *gguf, void **model, const ModelDef *def, int use_mmap)
{
	// Allocate the destination model struct (e.g., sizeof(Model) or sizeof(VisionModel))
	void *model_struct = calloc(1, def->struct_size);
	if (!model_struct) {
		perror("Failed to allocate model struct");
		return -1;
	}

	// Assign the newly created and populated struct back to the caller's pointer
	*model = model_struct;

	if (model_load_metadata(gguf, *model, def) != 0) {
		printf("Failed to load model metadata\n");
		return -1;
	}

	if (model_load_weights(gguf, *model, def, use_mmap) != 0) {
		printf("Failed to load model weights\n");
		return -1;
	}

	const TokenizeDef *tdef = def->tokenize_defs;
	if (tdef != NULL) {

		/* load tokens */
		if (gguf_model_read_token_embeds(ctx, gguf, "tokenizer.ggml.tokens", tdef->token_detect_specials)
		    != 0) {
			perror("Failed to read GGUF tokens array");
			return -1;
		}
		printf("Loaded model tokens, vocab size %llu\n", ctx->model->vocab_size);

		/* load token types */
		if (gguf_model_read_token_types(ctx, gguf, "tokenizer.ggml.token_type", tdef->token_detect_specials)
		    != 0) {
			perror("Failed to read GGUF token type array");
			return -1;
		}
		printf("Found %u special tokens\n", special_tokens.count);

		/* load token merges */
		if (tdef->token_load_merges == 1) {
			if (gguf_model_read_token_merges(ctx, gguf, "tokenizer.ggml.merges") != 0) {
				free(ctx->model);
				perror("Failed to read GGUF token merges array");
				return -1;
			}
		}

		/* load token scores */
		if (tdef->token_load_scores == 1) {
			if (gguf_model_read_token_scores(ctx, gguf, "tokenizer.ggml.scores") != 0) {
				free(ctx->model);
				perror("Failed to read GGUF token scores array");
				return -1;
			}
		}
	}

	printf("%s model load success\n", def->name);

	return 0;
}

int model_language_init(struct TIEContext *ctx, Model *model, const ModelDef *def, float yarn_scale_factor,
			float repetiton_penality)
{
	ctx->model->yarn_scale_factor = yarn_scale_factor;
	ctx->model->repetition_penalty = repetiton_penality;

	ctx->model->interface = def->interface;
	ctx->model->is_moe = def->params.is_moe;
	ctx->model->sot_token_id = def->params.sot_token_id;
	ctx->model->eot_token_id = def->params.eot_token_id;
	ctx->model->eos_token_id = def->params.eos_token_id;
	ctx->model->newline_token_id = def->params.newline_token_id;
	ctx->model->role_user_token_id = def->params.role_user_token_id;
	ctx->model->role_model_token_id = def->params.role_model_token_id;
	ctx->model->shared_kv_layers = def->params.shared_kv_layers;
	ctx->model->final_logit_softcap = def->params.final_logit_softcap;


	switch (def->arch) {
	case ARCH_QWEN3:
	case ARCH_QWEN3_MOE:
	case ARCH_QWEN3VL:
		ctx->model->attn_scale = 1.0f / sqrtf((float)ctx->model->head_dim);

		rope_cache_init(ctx, ctx->rope_cache_global, ctx->model->seq_length, ctx->model->head_dim,
				ctx->model->rope_freq_base, ctx->model->yarn_scale_factor);
		break;

	case ARCH_GEMMA3:
		ctx->model->attn_scale = 1.0f;

		rope_cache_init(ctx, ctx->rope_cache_global, ctx->model->seq_length, ctx->model->head_dim,
				ctx->model->rope_freq_base, ctx->model->rope_scale_factor);

		rope_cache_init(ctx, ctx->rope_cache_local, ctx->model->seq_length, ctx->model->head_dim, 10000.0f,
				1.0f);
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

		rope_cache_init(ctx, ctx->rope_cache_local, ctx->model->seq_length, ctx->model->head_dim, 10000.0f,
				1.0f);
		break;

	default:
		printf("%s %s model not defined?\n", __FUNCTION__, def->name);
		break;
	}

	init_KV_cache(ctx);

	if (model_init_mem_language(ctx, def) != 0)
		return -1;

	return 0;
}

void model_language_cleanup(struct TIEContext *ctx, struct GGUFModel *model, ModelDef *def, int use_mmap)
{
	printf("cleanup language model.\n");

	// Standard Buffers
	for (size_t i = 0; i < def->num_buffer_defs; i++) {
		const BufferDef *bdef = &def->buffer_defs[i];

		MemType *dest_buffer = (MemType *)((uint8_t *)&ctx->mem + bdef->offset);
		free_memtype(dest_buffer);
	}

	for (size_t i = 0; i < def->num_metadata_defs; i++) {
		const MetadataDef *mdef = &def->metadata_defs[i];

		if (mdef->is_array) {
		    void *dest_ptr = (uint8_t *)model + mdef->offset;

		    ModelArray *target_array = (ModelArray *)((char *)dest_ptr);
		    if (target_array->data)
			free(target_array->data);
		}
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
			Tensor *dest_tensor = (Tensor *)((uint8_t *)model + tdef->offset);
			if (dest_tensor->mem.data) {
				free(dest_tensor->mem.data);
			}
		}

		// Layer Tensors
		for (uint32_t block_idx = 0; block_idx < ctx->model->num_layers; block_idx++) {
			for (size_t i = 0; i < def->num_layer_tensors; i++) {
				const TensorDef *tdef = &def->layer_tensors[i];

				LayerWeights *layer = &ctx->model->layers[block_idx];
				Tensor *dest_tensor = (Tensor *)((uint8_t *)layer + tdef->offset);
				if (dest_tensor->mem.data) {
					free(dest_tensor->mem.data);
				}
			}
		}
	}
	free(ctx->model->layers);

	for (int i = 0; i < model->tensor_count; i++) {
		GGUFTensor *tensor = &model->tensors[i];
		free(tensor->name);
	}
	free(model->tensors);


	/* free KV cache */
	for (uint32_t i = 0; i < ctx->model->num_layers; i++) {
		free_memtype(&ctx->kv_cache[i].k);
		free_memtype(&ctx->kv_cache[i].v);
	}
	free(ctx->kv_cache);

	free(model);

	/* free rope cache(s) */
	free(ctx->rope_cache_local->sin);
	free(ctx->rope_cache_local->cos);
	free(ctx->rope_cache_local);

	if (ctx->rope_cache_global) {
		free(ctx->rope_cache_global->sin);
		free(ctx->rope_cache_global->cos);
		free(ctx->rope_cache_global);
	}

	/* free tokenizer */
	free(ctx->tokenizer.token_table);
	free(ctx->tokenizer.token_lens);
	free(ctx->tokenizer.token_types);
	free(ctx->tokenizer.token_scores);

	free_trie(ctx->tokenizer.root);
	free_string_pool(ctx->tokenizer.pool);
}

int model_init_mem_vision(struct TIEContext *ctx, const ModelDef *def)
{
	VisionModel *vm = ctx->model_vision;
	// Number of patches along one side of the image
	const int num_patches_side = vm->image_size / vm->patch_size;
	// Total number of patches from the image
	const int num_patches = num_patches_side * num_patches_side;
	// The full sequence length for the ViT
	const int vision_seq_len = num_patches;
	// The final sequence length after downsampling/pooling
	const int pooled_patches_side = num_patches_side / vm->proj_scale_factor;
	const int pooled_seq_len = pooled_patches_side * pooled_patches_side;

	printf("Initializing vision memory buffers...\n");

	// Standard Buffers
	for (size_t i = 0; i < def->num_buffer_defs; i++) {
		const BufferDef *bdef = &def->buffer_defs[i];

		MemType *dest_buffer = (MemType *)((uint8_t *)&ctx->vision_mem + bdef->offset);
		size_t size_multiplier = 0;

		switch (bdef->size_type) {

		case SIZE_VISION_IMAGE_RAW: // 896 * 896 * 3
			size_multiplier = vm->image_size * vm->image_size * 3;
			break;

		case SIZE_VISION_PATCH_EMBEDS: // 4096 * 1152
			size_multiplier = num_patches * vm->embed_dim;
			break;

		case SIZE_VISION_SEQ_LEN_X_EMBED_DIM: // 4096 * 1152
			size_multiplier = vision_seq_len * vm->embed_dim;
			break;

		case SIZE_VISION_SEQ_LEN_X_FFN_DIM: // 4096 * 4304
			size_multiplier = vision_seq_len * vm->ffn_dim;
			break;

		case SIZE_VISION_SEQ_LEN_X_QKV_DIM_X3: // 4096 * 1152 * 3
			size_multiplier = vision_seq_len * vm->embed_dim * 3;
			break;

		case SIZE_VISION_POOLED_EMBEDS: // 256 * 1152
			size_multiplier = pooled_seq_len * vm->embed_dim;
			break;

		case SIZE_VISION_PROJ_EMBEDS: // 256 * 2560
			size_multiplier = pooled_seq_len * vm->projection_dim;
			break;

		default:
			fprintf(stderr, "Error: Unknown buffer size type %d\n", bdef->size_type);
			return -1;
		}

		alloc_memtype(dest_buffer, bdef->type, size_multiplier);
	}

	for (int t = 0; t < thread_pool->num_threads; t++) {
		ctx->vision_mem.attn_scores_buffer[t] = aligned_alloc(32, (long long)vision_seq_len * sizeof(float));
		// TODO null checks
	}

	return 0;
}

void model_vision_cleanup(struct TIEContext *ctx, struct GGUFModel *model, ModelDef *def, int use_mmap)
{
	printf("cleanup vision model.\n");

	// Standard Buffers
	for (size_t i = 0; i < def->num_buffer_defs; i++) {
		const BufferDef *bdef = &def->buffer_defs[i];

		MemType *dest_buffer = (MemType *)((uint8_t *)&ctx->vision_mem + bdef->offset);
		free_memtype(dest_buffer);
	}

	// Per-Thread Buffers
	for (int t = 0; t < thread_pool->num_threads; t++) {
		free(ctx->vision_mem.attn_scores_buffer[t]);
	}

//	printf("clean medata arrays\n"); fflush(stdout);
	for (size_t i = 0; i < def->num_metadata_defs; i++) {
		const MetadataDef *mdef = &def->metadata_defs[i];

//		printf("clean medata array %s\n", mdef->key_fmt); fflush(stdout);

		if (mdef->is_array) {
		    void *dest_ptr = (uint8_t *)model + mdef->offset;

		    ModelArray *target_array = (ModelArray *)((char *)dest_ptr);
		    if (target_array->data)
			free(target_array->data);
		}
	}

	// Loaded Weights
	if (use_mmap == 0) {
		// Global Tensors
		for (size_t i = 0; i < def->num_global_tensors; i++) {
			const TensorDef *tdef = &def->global_tensors[i];
			Tensor *dest_tensor = (Tensor *)((uint8_t *)model + tdef->offset);
			if (dest_tensor->mem.data) {
				free(dest_tensor->mem.data);
			}
		}

		// Layer Tensors
		for (uint32_t block_idx = 0; block_idx < ctx->model_vision->num_layers; block_idx++) {
			for (size_t i = 0; i < def->num_layer_tensors; i++) {
				const TensorDef *tdef = &def->layer_tensors[i];

				VisionLayerWeights *layer = &ctx->model_vision->layers[block_idx];
				Tensor *dest_tensor = (Tensor *)((uint8_t *)layer + tdef->offset);
				if (dest_tensor->mem.data) {
					free(dest_tensor->mem.data);
				}
			}
		}
	}
	free(ctx->model_vision->layers);

	for (int i = 0; i < model->tensor_count; i++) {
		GGUFTensor *tensor = &model->tensors[i];
		free(tensor->name);
	}
	free(model->tensors);

	free(model);
}

int model_vision_init(struct TIEContext *ctx, VisionModel *model_vision, const ModelDef *def)
{
	VisionModel *vm = model_vision;
//	struct GGUFModel *gguf = ctx->gguf_vision;
	float *mean = (float *)vm->image_mean.data;
	float *std = (float *)vm->image_std.data;

	vm->interface = def->interface;
	vm->proj_scale_factor = def->params.proj_scale_factor;

	if (vm->has_vision_encoder != 1) {
		printf("%s %s model has no vision encoder...\n", __FUNCTION__, def->name);
		return -1;
	}

	printf("%s init\n", def->name);

	switch (def->projector) {
	case VISION_PROJECTOR_GEMMA3: {
		vm->soi_token_id = 255999;
		vm->eoi_token_id = 256000;
		vm->image_soft_token_id = 262144;
	} break;

	case VISION_PROJECTOR_QWEN3VL: {
/*
		uint64_t num_layers = 0;
		uint8_t *dstack_layers = (uint8_t *)gguf_metadata_get_array_typed(
			gguf, "clip.vision.is_deepstack_layers", GGUF_METADATA_VALUE_TYPE_BOOL, &num_layers);

		if (dstack_layers != NULL) {
			printf("deepstack_layers num=%llu\n", (unsigned long long)num_layers);
			for (uint64_t i = 0; i < num_layers; i++) {
				printf("deepstack_layers: #%llu %s\n", (unsigned long long)i,
				       dstack_layers[i] == 0 ? "false" : "true");
			}
		}
*/
	} break;
	default:
		printf("%s %s model not defined?\n", __FUNCTION__, def->name);
		return -1;
	}

	model_vision->meta.image_size = vm->image_size;
	for (int i = 0; i < 3; i++) {
	    model_vision->meta.image_mean[i] = mean[i];
	    model_vision->meta.image_std[i] = std[i];
	}

#if 0
	printf("INFO: Checking for swapped FFN weights in vision model...\n");
	for (int i = 0; i < vm->num_layers; ++i) {
	    VisionLayerWeights *l = &vm->layers[i];

	    // If ffn_down's input dim is 1152, it's swapped.
	    if (l->ffn_down.dimensions[0] == vm->embed_dim) {
		printf("INFO: Swapping FFN up/down weights for layer %d\n", i);

    		// Swap the weight Tensors
    		Tensor temp_w = l->ffn_up;
    		l->ffn_up = l->ffn_down;
    		l->ffn_down = temp_w;

    		// Swap the bias Tensors
    		Tensor temp_b = l->ffn_up_bias;
    		l->ffn_up_bias = l->ffn_down_bias;
    		l->ffn_down_bias = temp_b;
	    }
	}
#endif

	if (model_init_mem_vision(ctx, def) != 0)
		return -1;

	return 0;
}

void language_model_info(struct TIEContext *ctx)
{
	printf("Initialized %s language model with the following configuration:\n", ctx->model->def->name);
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
}

void vision_model_info(struct TIEContext *ctx)
{
	printf("Initialized %s vision model with the following configuration:\n", ctx->model->def->name);
	printf("Image size: %d, Proj Dim: %d, Patch size: %d\n", ctx->model_vision->image_size,
	       ctx->model_vision->projection_dim, ctx->model_vision->patch_size);
	printf("Embed Dim: %d, FFN Dim: %d, Layers: %d, Heads: %d, eps: %f\n", ctx->model_vision->embed_dim,
	       ctx->model_vision->ffn_dim, ctx->model_vision->num_layers, ctx->model_vision->num_heads,
	       ctx->model_vision->norm_eps);
	printf("Proj Scale Factor: %d\n", ctx->model_vision->proj_scale_factor);
}
