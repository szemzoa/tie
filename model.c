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

extern void build_rope_cache_dynamic(struct TIEContext *ctx, size_t seq_len);

const ModelArch known_archs[] = {
	{"qwen3", ARCH_QWEN3, &QWEN3_DEF, NULL},
	{"qwen3moe", ARCH_QWEN3_MOE, &QWEN3_MOE_DEF, NULL},
	{"gemma3", ARCH_GEMMA3, &GEMMA3_DEF, NULL},
	{"gemma3n", ARCH_GEMMA3N, &GEMMA3N_DEF, NULL},
	{"clip", ARCH_CLIP_VISION, NULL, NULL},
	{"qwen3vl", ARCH_QWEN3VL, &QWEN3VL_DEF, NULL},
	{"qwen3vlmoe", ARCH_QWEN3VL_MOE, &QWEN3VL_MOE_DEF, NULL},
	{"qwen3", ARCH_DEEPSEEK_QWEN3, &DEEPSEEK_QWEN3_DEF, "DeepSeek"},
	{"granite", ARCH_GRANITE, &GRANITE_DEF, NULL},
	{"lfm2", ARCH_LFM2, &LFM2_DEF, NULL},
	{"lfm2moe", ARCH_LFM2_MOE, &LFM2_MOE_DEF, NULL},
};

const ModelArch known_projectors[] = {
	{"gemma3", VISION_PROJECTOR_GEMMA3, &GEMMA3_CLIP_DEF, NULL},
	{"qwen3vl_merger", VISION_PROJECTOR_QWEN3VL, &QWEN3VL_CLIP_DEF, NULL},
};

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

int detect_architecture(const char *model_name, const char *base_name)
{
	const ModelArch *best_match = NULL;
	size_t best_len = 0;

	for (int i = 0; i < ARRAY_SIZE(known_archs); i++) {
		const ModelArch *current_arch = &known_archs[i];

		if (strstr(model_name, current_arch->name)) {
			size_t current_len = strlen(current_arch->name);

			if (current_arch->base_name == NULL) {

				if (current_len > best_len) {
					best_len = current_len;
					best_match = current_arch;
				}

			} else {

				if (strstr(base_name, current_arch->base_name))
					return current_arch->id;
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
	const ModelArch *best_match = NULL;
	size_t best_len = 0;

	for (int i = 0; i < ARRAY_SIZE(known_projectors); i++) {
		const ModelArch *current_projector = &known_projectors[i];

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
	int projector = detect_projector(gguf_metadata_get_string(gguf_model, "clip.projector_type"));

	for (int i = 0; i < ARRAY_SIZE(known_projectors); i++) {
		if (projector == known_projectors[i].id)
			return known_projectors[i].def;
	}

	return NULL;
}

ModelDef *find_model_def(struct GGUFModel *gguf_model)
{
	for (int i = 0; i < ARRAY_SIZE(known_archs); i++) {
		if (gguf_model->arch == known_archs[i].id) {

			if (gguf_model->arch == ARCH_CLIP_VISION)
				return find_projector_def(gguf_model);

			return known_archs[i].def;
		}
	}

	return NULL;
}

int init_KV_cache(struct TIEContext *ctx)
{
	int attention_layers = 0;

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

		if (ctx->model->layer_types[i] == LAYER_TYPE_ATTENTION) {
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

			attention_layers++;

		} else {

			ctx->kv_cache[i].k.data = NULL;
			ctx->kv_cache[i].v.data = NULL;
		}
	}

	printf("KV cache uses: %lld MB on %u attention layers\n",
	       (k_elements_per_layer * sizeof(uint16_t)) * 2 * attention_layers / 1024 / 1024, attention_layers);

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
				//				printf("Info: Optional tensor '%s' not found.\n",
				// tdef->name_fmt);
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
					// printf("Info: Optional tensor '%s' not found.\n", tdef->name_fmt);
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

		// Calculate the destination pointer relative to the allocated struct
		void *dest_ptr = (uint8_t *)model + mdef->offset;

		if (mdef->is_array) {
			uint64_t array_size = 0;

			void *array_data = gguf_metadata_get_array_typed(gguf, key_buffer,
									 GGUF_METADATA_VALUE_TYPE_ARRAY, &array_size);

			if (array_data) {
				// Get the target ModelArray struct in our model
				ModelArray *target_array = (ModelArray *)((char *)dest_ptr);

				// Get total size in bytes
				size_t element_size = gguf_get_type_size(mdef->type);
				size_t total_bytes = element_size * array_size;

				// Allocate memory and copy the data
				target_array->data = malloc(total_bytes);
				if (!target_array->data) {
					fprintf(stderr, "Failed to load required metadata key: %s\n", key_buffer);
					return -1;
				}
				memcpy(target_array->data, array_data, total_bytes);

				// Store the size and type
				target_array->size = array_size;
				target_array->type = mdef->type;

			} else {
				if (!mdef->is_optional) {
					fprintf(stderr, "Failed to load required metadata key: %s\n", key_buffer);
					return -1;
				}
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
		size_t alloc_size = 0;

		// Resolve size type to runtime value
		switch (bdef->size_type) {

		case SIZE_EMBED_DIM:
			alloc_size = ctx->model->embed_dim * MAX_PROMPT_BATCH_SIZE;
			break;
		case SIZE_VOCAB_SIZE:
			alloc_size = ctx->model->vocab_size * MAX_PROMPT_BATCH_SIZE;
			break;
		case SIZE_FFN_DIM:
			alloc_size = ctx->model->ffn_dim * MAX_PROMPT_BATCH_SIZE;
			break;
		case SIZE_FFN_DIM_X_2:
			alloc_size = ctx->model->ffn_dim * 2 * MAX_PROMPT_BATCH_SIZE;
			break;
		case SIZE_Q_DIM:
			alloc_size = ctx->model->num_heads * ctx->model->head_dim * MAX_PROMPT_BATCH_SIZE;
			break;
		case SIZE_KV_DIM:
			alloc_size = ctx->model->num_kv_heads * ctx->model->head_dim * MAX_PROMPT_BATCH_SIZE;
			break;
		case SIZE_NUM_LAYERS_X_PLI_DIM:
			alloc_size = ctx->model->num_layers * ctx->model->pli_dim * MAX_PROMPT_BATCH_SIZE;
			break;
		case SIZE_POS_IDS:
			alloc_size = 3 * ctx->model->seq_length;
			break;
		case SIZE_ROPE_SIN:
			alloc_size = ctx->model->seq_length * ctx->model->head_dim * MAX_PROMPT_BATCH_SIZE;
			break;
		case SIZE_ROPE_COS:
			alloc_size = ctx->model->seq_length * ctx->model->head_dim * MAX_PROMPT_BATCH_SIZE;
			break;
		default:
			return -1;
		}

		alloc_memtype(dest_buffer, bdef->type, alloc_size);
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
			alloc_memtype(&ctx->mem.ffn_hidden1_scratch[i], GGML_TYPE_F32, ctx->model->expert_ffn_dim);
			alloc_memtype(&ctx->mem.ffn_hidden2_scratch[i], GGML_TYPE_F32, ctx->model->expert_ffn_dim);

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

	// Special Mamba-2 Buffers
	size_t total_conv_elements = 0;
	size_t total_ssm_elements = 0;
	int mamba_layer_count = 0;

	for (int i = 0; i < ctx->model->num_layers; i++) {
		if (ctx->model->layer_types[i] == LAYER_TYPE_MAMBA2) {
			// Granite: inner_size=3072, conv_kernel=4, state_size=128
			total_conv_elements += (size_t)ctx->model->ssm_inner_size * ctx->model->ssm_conv_kernel;
			total_ssm_elements += (size_t)ctx->model->ssm_inner_size * ctx->model->ssm_state_size;
			mamba_layer_count++;
		}
	}

	//  Allocate the Big Blocks (if needed)
	if (mamba_layer_count > 0) {
		alloc_memtype(&ctx->mem.mamba_conv_states_memory, GGML_TYPE_F32, total_conv_elements);
		alloc_memtype(&ctx->mem.mamba_ssm_states_memory, GGML_TYPE_F32, total_ssm_elements);

		// Zero them out (Crucial for Recurrent State!)
		memset(ctx->mem.mamba_conv_states_memory.data, 0, ctx->mem.mamba_conv_states_memory.n_bytes);
		memset(ctx->mem.mamba_ssm_states_memory.data, 0, ctx->mem.mamba_ssm_states_memory.n_bytes);

		// Allocate the array of structs
		ctx->mem.mamba_states = calloc(ctx->model->num_layers, sizeof(LayerMambaState));
		alloc_memtype(&ctx->mem.mamba_in_proj_output, GGML_TYPE_F32,
			      ctx->model->ssm_proj_dim * MAX_PROMPT_BATCH_SIZE);
	}

	// Distribute Memory Pointers
	float *conv_ptr = (float *)ctx->mem.mamba_conv_states_memory.data;
	float *ssm_ptr = (float *)ctx->mem.mamba_ssm_states_memory.data;

	for (int i = 0; i < ctx->model->num_layers; i++) {
		if (ctx->model->layer_types[i] == LAYER_TYPE_MAMBA2) {
			// Assign pointers
			ctx->mem.mamba_states[i].conv_state = conv_ptr;
			ctx->mem.mamba_states[i].ssm_state = ssm_ptr;
			ctx->mem.mamba_states[i].conv_pos = 0;

			// Advance pointers
			conv_ptr += (size_t)ctx->model->ssm_inner_size * ctx->model->ssm_conv_kernel;
			ssm_ptr += (size_t)ctx->model->ssm_inner_size * ctx->model->ssm_state_size;
		}
	}

	//  Allocate Shared Expert Buffer (Reuse across layers)
	if (ctx->model->expert_shared_ffn_dim > 0) {
		alloc_memtype(&ctx->mem.shared_expert_output, GGML_TYPE_F32, ctx->model->expert_shared_ffn_dim);
	}


	/* LFM2 ShortConv Buffers */
	int shortconv_layer_count = 0;
	for (int i = 0; i < ctx->model->num_layers; i++) {
		if (ctx->model->layer_types[i] == LAYER_TYPE_SHORTCONV) {
			shortconv_layer_count++;
		}
	}

	if (shortconv_layer_count > 0) {
		// Allocate the array of state containers (one per layer)
		ctx->mem.lfm2_conv_states = calloc(ctx->model->num_layers, sizeof(LayerConvState));
		if (!ctx->mem.lfm2_conv_states) {
			fprintf(stderr, "Failed to allocate lfm2_conv_states array\n");
			exit(1);
		}

		// Allocate the actual data buffer for each ShortConv layer
		size_t state_elements = ctx->model->embed_dim * ctx->model->conv_kernel_size;

		// We use standard float size here since cache is FP32
		size_t state_bytes = state_elements * sizeof(float);

		for (int i = 0; i < ctx->model->num_layers; i++) {
			if (ctx->model->layer_types[i] == LAYER_TYPE_SHORTCONV) {
				// Use your aligned allocator (xaligned_alloc) directly
				// Do NOT use alloc_memtype here unless LayerConvState has a MemType field.

				ctx->mem.lfm2_conv_states[i].buffer = xaligned_alloc(32, state_bytes);

				if (!ctx->mem.lfm2_conv_states[i].buffer) {
					fprintf(stderr, "Failed to allocate conv buffer for layer %d\n", i);
					exit(1);
				}

				// Initialize to zero (CRITICAL for convolution history)
				memset(ctx->mem.lfm2_conv_states[i].buffer, 0, state_bytes);

				ctx->mem.lfm2_conv_states[i].pos = 0;
			}
		}

		// Allocate Scratch Buffers (These use MemType, so alloc_memtype is correct)
		// MAX_PROMPT_BATCH_SIZE is required for batching
		alloc_memtype(&ctx->mem.sconv_in_proj_output, GGML_TYPE_F32,
			      3 * ctx->model->embed_dim * MAX_PROMPT_BATCH_SIZE);

		alloc_memtype(&ctx->mem.sconv_conv_output, GGML_TYPE_F32,
			      ctx->model->embed_dim * MAX_PROMPT_BATCH_SIZE);
	}

	return 0;
}

int model_load(struct TIEContext *ctx, struct GGUFModel *gguf, void **model, const ModelDef *def, int use_mmap)
{
	// Allocate the destination model struct
	void *model_struct = calloc(1, def->struct_size);
	if (!model_struct) {
		perror("Failed to allocate model struct");
		return -1;
	}

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
		// printf("Loaded model tokens, vocab size %llu\n", ctx->model->vocab_size);

		/* load token types */
		if (gguf_model_read_token_types(ctx, gguf, "tokenizer.ggml.token_type", tdef->token_detect_specials)
		    != 0) {
			perror("Failed to read GGUF token type array");
			return -1;
		}
		// printf("Found %u special tokens\n", special_tokens.count);

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


void build_rope_cache_global(struct TIEContext *ctx, size_t seq_len)
{
	ctx->model->rope_cache_global = malloc(sizeof(RopeCacheType));
	int rope_dim = ctx->model->rope_dimension_count;

	// Fallback if metadata is missing
	if (rope_dim == 0)
		rope_dim = ctx->model->head_dim;

	// TODO
	if (!ctx->model->rope_cache_global) {
		perror("Failed to allocate rope_cache");
		exit(EXIT_FAILURE);
	}

	rope_cache_init(ctx, ctx->model->rope_cache_global, seq_len, ctx->model->head_dim, rope_dim,
			ctx->model->rope_freq_base, ctx->model->yarn_scale_factor);
}

void build_rope_cache_shared(struct TIEContext *ctx, size_t seq_len)
{
	ctx->model->rope_cache_local = malloc(sizeof(RopeCacheType));
	ctx->model->rope_cache_global = malloc(sizeof(RopeCacheType));
	int rope_dim = ctx->model->rope_dimension_count;

	// Fallback if metadata is missing
	if (rope_dim == 0)
		rope_dim = ctx->model->head_dim;

	if (!ctx->model->rope_cache_local || !ctx->model->rope_cache_global) {
		perror("Failed to allocate rope_cache");
		exit(EXIT_FAILURE);
	}

	rope_cache_init(ctx, ctx->model->rope_cache_global, seq_len, ctx->model->head_dim, rope_dim,
			ctx->model->rope_freq_base, ctx->model->rope_scale_factor);

	rope_cache_init(ctx, ctx->model->rope_cache_local, seq_len, ctx->model->head_dim, rope_dim, 10000.0f, 1.0f);
}

int model_language_init(struct TIEContext *ctx, Model *model, const ModelDef *def, float yarn_scale_factor,
			float repetiton_penality)
{
	ctx->model->yarn_scale_factor = yarn_scale_factor;
	ctx->model->repetition_penalty = repetiton_penality;
	int attn_layer_idx = -1;
	int mamba_layer_idx = -1;
	uint32_t *kv_heads;


	printf("\n[%s] init\n", def->name);

	/* Init language model defaults */
	ctx->model->eos_token_id = def->params.eos_token_id;

	/* Fallbacks */
	if (ctx->model->rope_scale_factor == 0.0f)
		ctx->model->rope_scale_factor = 1.0f;

	ctx->model->context_size = ctx->model->seq_length;

	if (ctx->config.context_length != 0)
		ctx->model->seq_length = ctx->config.context_length;

	ctx->model->interface = def->interface;
	ctx->model->is_moe = def->params.is_moe;
	ctx->model->final_logit_softcap = def->params.final_logit_softcap;

	ctx->model->layer_types = malloc(ctx->model->num_layers * sizeof(int));

	for (int i = 0; i < ctx->model->num_layers; i++) {
		ctx->model->layer_types[i] = LAYER_TYPE_ATTENTION;
	}

	switch (def->arch) {
	case ARCH_QWEN3:
	case ARCH_QWEN3_MOE:
		ctx->model->attn_scale = 1.0f / sqrtf((float)ctx->model->head_dim);
		break;

	case ARCH_GEMMA3:
		ctx->model->attn_scale = 1.0f;
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
		break;

	case ARCH_QWEN3VL:
		ctx->model->attn_scale = 1.0f / sqrtf((float)ctx->model->head_dim);
		ctx->model->use_mrope = 1;
		break;

	case ARCH_QWEN3VL_MOE:
		ctx->model->attn_scale = 1.0f / sqrtf((float)ctx->model->head_dim);
		ctx->model->use_mrope = 1;
		break;

	case ARCH_DEEPSEEK_QWEN3:
		ctx->model->attn_scale = 1.0f / sqrtf((float)ctx->model->head_dim);
		ctx->model->add_bos_token = 1;
		break;

	case ARCH_GRANITE:
		ctx->model->layer_types = malloc(ctx->model->num_layers * sizeof(uint8_t));

		// Iterate the KV array
		kv_heads = (uint32_t *)ctx->model->attn_head_count_kv.data;
		if (ctx->model->attn_head_count_kv.size != ctx->model->num_layers) {
			printf("invalid number of layer types in model?\n");
			return -1;
		}

		/* update model layer types if layer is NOT attention */
		for (int i = 0; i < ctx->model->num_layers; i++) {
			if (kv_heads[i] == 0) {
				ctx->model->layer_types[i] = LAYER_TYPE_MAMBA2;
				mamba_layer_idx = i;
			} else {
				attn_layer_idx = i;
				ctx->model->layer_types[i] = LAYER_TYPE_ATTENTION;
			}
		}

		// Derive Attention Params (head_dim)
		if (attn_layer_idx != -1) {
			Tensor *k_weight = &ctx->model->layers[attn_layer_idx].attn_k;
			// Dimension[0] is usually input (embed_dim), Dimension[1] is output (total_k_dim)
			// But GGUF shapes are often transposed in metadata vs memory.
			int k_dim = (k_weight->dimensions[0] == ctx->model->embed_dim) ? k_weight->dimensions[1]
										       : k_weight->dimensions[0];

			// Get KV heads from the array
			uint32_t *kv_heads_arr = (uint32_t *)ctx->model->attn_head_count_kv.data;
			int n_kv = kv_heads_arr[attn_layer_idx];

			ctx->model->head_dim = k_dim / n_kv; // 512 / 4 = 128
			ctx->model->num_kv_heads = n_kv;     // 4

			printf("Derived Attention Params: head_dim=%d, kv_heads=%d\n", ctx->model->head_dim,
			       ctx->model->num_kv_heads);
		}

		// Derive Mamba Params (ssm_proj_dim)
		if (mamba_layer_idx != -1) {
			Tensor *in_proj = &ctx->model->layers[mamba_layer_idx].ssm_in;
			// Find the dimension that isn't embed_dim (1536)
			int ssm_out = (in_proj->dimensions[0] == ctx->model->embed_dim) ? in_proj->dimensions[1]
											: in_proj->dimensions[0];

			ctx->model->ssm_proj_dim = ssm_out; // 6448
			printf("Derived Mamba Params: ssm_proj_dim=%d\n", ssm_out);
		}

		printf("Attention scale=%.10f\n", ctx->model->attn_scale);
		break;

	case ARCH_LFM2:
	case ARCH_LFM2_MOE:

		// Iterate the KV array
		kv_heads = (uint32_t *)ctx->model->attn_head_count_kv.data;
		if (ctx->model->attn_head_count_kv.size != ctx->model->num_layers) {
			printf("invalid number of layer types in model?\n");
			return -1;
		}

		/* update model layer types if layer is NOT attention */
		for (int i = 0; i < ctx->model->num_layers; i++) {
			if (kv_heads[i] == 0) {
				ctx->model->layer_types[i] = LAYER_TYPE_SHORTCONV;
			} else {
				ctx->model->layer_types[i] = LAYER_TYPE_ATTENTION;
				attn_layer_idx = i;
			}
			//			printf("layer#%u is %s\n", i, ctx->model->layer_types[i] ==
			// LAYER_TYPE_SHORTCONV ? "SHORTCONV" : "ATTENTION");
		}

		// Derive Attention Params (head_dim)
		if (attn_layer_idx != -1) {
			Tensor *k_weight = &ctx->model->layers[attn_layer_idx].attn_k;
			// Dimension[0] is usually input (embed_dim), Dimension[1] is output (total_k_dim)
			// But GGUF shapes are often transposed in metadata vs memory.
			int k_dim = (k_weight->dimensions[0] == ctx->model->embed_dim) ? k_weight->dimensions[1]
										       : k_weight->dimensions[0];

			// Get KV heads from the array
			uint32_t *kv_heads_arr = (uint32_t *)ctx->model->attn_head_count_kv.data;
			int n_kv = kv_heads_arr[attn_layer_idx];

			ctx->model->head_dim = k_dim / n_kv;
			ctx->model->num_kv_heads = n_kv;

			printf("Derived Attention Params: head_dim=%d, kv_heads=%d\n", ctx->model->head_dim,
			       ctx->model->num_kv_heads);
		}

		if (ctx->model->conv_kernel_size == 0) {
			printf("WARN: Conv kernel size 0. Defaulting to 3.\n");
			ctx->model->conv_kernel_size = 3; // LFM2 default
		}

		ctx->model->attn_scale = 1.0f / sqrtf((float)ctx->model->head_dim);
		break;

	default:
		printf(CLR_RED "!!! %s: %s model not defined? !!!\n", __FUNCTION__, def->name);
		printf(CLR_RESET);
		break;
	}

	// Initialize RoPE cache
	printf("init: RoPE cache, length: %u, dim: %u\n", ctx->model->context_size, ctx->model->head_dim);
	ctx->model->interface.build_rope_cache(ctx, ctx->model->context_size);

	init_KV_cache(ctx);

	if (model_init_mem_language(ctx, def) != 0)
		return -1;

#if 0
	if (ctx->mem.mamba_states) {
    	    for (int l = 0; l < ctx->model->num_layers; l++) {
        	LayerMambaState *s = &ctx->mem.mamba_states[l];

        	// Reset Convolution State
        	size_t conv_size = ctx->model->ssm_inner_size * ctx->model->ssm_conv_kernel * sizeof(float);
        	memset(s->conv_state, 0, conv_size);

        	// Reset SSM State
        	size_t ssm_size = ctx->model->ssm_time_step_rank * 64 * ctx->model->ssm_state_size * sizeof(float);
        	memset(s->ssm_state, 0, ssm_size);

        	// Reset Ring Buffer Index
        	s->conv_pos = 0;
    	    }
	}
#endif

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

	/* LFM2 shortconv */
	if (ctx->mem.lfm2_conv_states) {
		for (int i = 0; i < ctx->model->num_layers; i++) {
			if (ctx->mem.lfm2_conv_states[i].buffer) {
				free(ctx->mem.lfm2_conv_states[i].buffer);
			}
		}
		free(ctx->mem.lfm2_conv_states);
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
	//	free(ctx->model->layers);

	free(ctx->model->layer_types);

	for (size_t i = 0; i < def->num_metadata_defs; i++) {
		const MetadataDef *mdef = &def->metadata_defs[i];

		if (mdef->is_array) {
			void *dest_ptr = (uint8_t *)model + mdef->offset;

			ModelArray *target_array = (ModelArray *)((char *)dest_ptr);
			if (target_array->data)
				free(target_array->data);
		}
	}

	for (int i = 0; i < model->tensor_count; i++) {
		GGUFTensor *tensor = &model->tensors[i];
		free(tensor->name);
	}
	free(model->tensors);

	/* free rope cache(s) */
	if (ctx->model->rope_cache_local) {
		free(ctx->model->rope_cache_local->sin);
		free(ctx->model->rope_cache_local->cos);
		free(ctx->model->rope_cache_local);
	}

	if (ctx->model->rope_cache_global) {
		free(ctx->model->rope_cache_global->sin);
		free(ctx->model->rope_cache_global->cos);
		free(ctx->model->rope_cache_global);
	}

	/* free KV cache */
	for (uint32_t i = 0; i < ctx->model->num_layers; i++) {
		if (ctx->model->layer_types[i] == LAYER_TYPE_ATTENTION) {
			free_memtype(&ctx->kv_cache[i].k);
			free_memtype(&ctx->kv_cache[i].v);
		}
	}
	free(ctx->kv_cache);

	free(model);

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
	MemLayoutVision *mem = &ctx->vision_mem;

	const int num_patches_side = vm->image_size / vm->patch_size;
	const int num_patches = num_patches_side * num_patches_side;
	const int vision_seq_len = num_patches;
	// The final sequence length after downsampling/pooling
	const int pooled_patches_side = vm->def->params.proj_scale_factor == 0
						? num_patches_side
						: num_patches_side / vm->def->params.proj_scale_factor;
	const int pooled_seq_len = pooled_patches_side * pooled_patches_side;
	const int embed_dim = vm->embed_dim;

	int max_head_dim = vm->embed_dim / vm->num_heads;
	int num_threads = thread_pool->num_threads;

	printf("Initializing vision memory buffers...\n");

	// Allocate the array of structs
	mem->attn_scratch = calloc(num_threads, sizeof(VisionAttnScratch));

	for (int t = 0; t < num_threads; t++) {
		VisionAttnScratch *s = &mem->attn_scratch[t];

		// Allocate max possible size for each buffer
		alloc_memtype(&s->q_head, GGML_TYPE_F32, num_patches * max_head_dim);
		alloc_memtype(&s->k_head, GGML_TYPE_F32, num_patches * max_head_dim);
		alloc_memtype(&s->v_head, GGML_TYPE_F32, num_patches * max_head_dim);
		alloc_memtype(&s->v_head_t, GGML_TYPE_F32, num_patches * num_patches);
		alloc_memtype(&s->scores, GGML_TYPE_F32, num_patches * num_patches);
		alloc_memtype(&s->output_head, GGML_TYPE_F32, num_patches * max_head_dim);
	}

	// Standard Buffers
	for (size_t i = 0; i < def->num_buffer_defs; i++) {
		const BufferDef *bdef = &def->buffer_defs[i];

		MemType *dest_buffer = (MemType *)((uint8_t *)&ctx->vision_mem + bdef->offset);
		size_t size_multiplier = 0;

		switch (bdef->size_type) {

		case SIZE_VISION_IMAGE_RAW: // 896 * 896 * 3
			size_multiplier = MAX_IMAGE_W * MAX_IMAGE_H * 3;
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

		case SIZE_VISION_MERGER_TEMP: {
			const int merge_factor = vm->spatial_merge_size * vm->spatial_merge_size;
			const int merged_seq_len = vision_seq_len / merge_factor;
			const int merged_dim = embed_dim * merge_factor;

			size_multiplier = (size_t)merged_seq_len * merged_dim;
		} break;

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

	if (vm->num_deepstack_layers > 0) {
		ctx->vision_mem.deepstack_features = calloc(vm->num_deepstack_layers, sizeof(MemType));
		if (!ctx->vision_mem.deepstack_features) {
			fprintf(stderr, "Failed to allocate deepstack feature array\n");
			return -1;
		}

		//  Get the size for one deepstack feature buffer
		size_t feature_size = pooled_seq_len * vm->projection_dim;

		// Loop and allocate the data buffer for each one
		for (int i = 0; i < vm->num_deepstack_layers; i++) {
			alloc_memtype(&ctx->vision_mem.deepstack_features[i], GGML_TYPE_F32, feature_size);

			if (!ctx->vision_mem.deepstack_features[i].data) {
				fprintf(stderr, "Failed to allocate data for deepstack feature %d\n", i);
				return -1;
			}
		}
	}

	return 0;
}

void model_vision_cleanup(struct TIEContext *ctx, struct GGUFModel *model, ModelDef *def, int use_mmap)
{
	MemLayoutVision *mem = &ctx->vision_mem;
	int num_threads = thread_pool->num_threads;

	printf("cleanup vision model.\n");

	for (int t = 0; t < num_threads; t++) {
		VisionAttnScratch *s = &mem->attn_scratch[t];

		// Allocate max possible size for each buffer
		free_memtype(&s->q_head);
		free_memtype(&s->k_head);
		free_memtype(&s->v_head);
		free_memtype(&s->v_head_t);
		free_memtype(&s->scores);
		free_memtype(&s->output_head);
	}

	free(mem->attn_scratch);

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

	for (size_t i = 0; i < def->num_metadata_defs; i++) {
		const MetadataDef *mdef = &def->metadata_defs[i];

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

	/* free mrope cache(s) */
	free(ctx->model_vision->mrope_cache.sin_table);
	free(ctx->model_vision->mrope_cache.cos_table);

	free(model);
}

int model_vision_init(struct TIEContext *ctx, VisionModel *model_vision, const ModelDef *def)
{
	VisionModel *vm = model_vision;
	vm->interface = def->interface;
	//	vm->proj_scale_factor = def->params.proj_scale_factor;

	if (vm->has_vision_encoder != 1) {
		printf("%s %s model has no vision encoder...\n", __FUNCTION__, def->name);
		return -1;
	}

	printf("\n[%s] init\n", def->name);

	switch (def->projector) {
	case VISION_PROJECTOR_GEMMA3: {
		//		vm->spatial_merge_size = 1;
	} break;

	case VISION_PROJECTOR_QWEN3VL: {
		MRopeCacheType *cache = &vm->mrope_cache;

		// This defines the MAXIMUM buffer size (e.g., 48x48 = 2304 patches)
		const int num_patches_side = vm->image_size / vm->patch_size;
		const int num_heads = vm->num_heads;
		const int head_dim = vm->embed_dim / num_heads;

		// Count DeepStack layers
		bool *is_deepstack = (bool *)vm->is_deepstack_layers.data;
		vm->num_deepstack_layers = 0;

		for (int i = 0; i < vm->num_layers; i++) {
			if (is_deepstack[i] == true)
				vm->num_deepstack_layers++;
		}

		// Build tables for the MAX sequence length
		const int max_seq_len = num_patches_side * num_patches_side;
		cache->num_elements = (size_t)max_seq_len * head_dim;

		// Use aligned allocation for future SIMD optimizations
		size_t size_bytes = cache->num_elements * sizeof(float);
		cache->cos_table = (float *)aligned_alloc(32, size_bytes);
		cache->sin_table = (float *)aligned_alloc(32, size_bytes);

		// Check for allocation failure
		if (!cache->cos_table || !cache->sin_table) {
			fprintf(stderr, "Failed to allocate Vision RoPE cache\n");
			exit(1);
		}

	} break;

	default:
		printf("%s %s model not defined?\n", __FUNCTION__, def->name);
		return -1;
	}

	model_vision->meta.image_size = vm->image_size;

	float *mean = (float *)vm->image_mean.data;
	float *std = (float *)vm->image_std.data;

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
	ModelParams *params = &ctx->model->def->params;

	printf("Initialized %s language model with the following configuration:\n", ctx->model->def->name);
	printf("Embed Dim: %d, Layers: %d, Heads: %d, KV Heads: %d, Head Dim: %d, Shared KV layers: %u\n",
	       ctx->model->embed_dim, ctx->model->num_layers, ctx->model->num_heads, ctx->model->num_kv_heads,
	       ctx->model->head_dim, ctx->model->shared_kv_layers);
	printf("FFN Dim: %d, Rope Base: %.1f, Seq Len: %d, Vocab: %llu\n", ctx->model->ffn_dim,
	       ctx->model->rope_freq_base, ctx->model->seq_length, ctx->model->vocab_size);
	printf("Yarn Scale: %.2f, Eps: %f, Rope_scale: %.1f, Sliding_window: %u\n", ctx->model->yarn_scale_factor,
	       ctx->model->norm_eps, ctx->model->rope_scale_factor, ctx->model->attn_sliding_window);
	if (ctx->model->is_moe == 1)
		printf("Expert Count: %d, Expert Used Count: %d, Expert FFN Dim: %d, Expert shared FFN Dim: %d, Expert leading dense layers: %d\n",
		       ctx->model->expert_count, ctx->model->expert_used_count, ctx->model->expert_ffn_dim,
		       ctx->model->expert_shared_ffn_dim, ctx->model->expert_leading_dense_layers);
	printf("Embedding scale: %.5f, Residual_scale: %.05f, Logit scale: %.5f\n", ctx->model->embedding_scale,
	       ctx->model->residual_scale, ctx->model->logit_scale);

	if (ctx->model->altup_num_inputs > 0)
		printf("Altup Num Inputs: %i\n", ctx->model->altup_num_inputs);

	if (ctx->model->use_mrope == 1) {
		printf("M-RoPE sections: ");
		uint32_t *sect = (uint32_t *)ctx->model->mrope_sections.data;

		for (int i = 0; i < ctx->model->mrope_sections.size; i++)
			printf("[%u]", sect[i]);

		printf("\n");
	}

	if (params->is_hybrid == 1) {
		printf("SSM: conv_kernel: %u, state_size: %u, group_count: %u, inner_size: %u, time_step_rank: %u\n",
		       ctx->model->ssm_conv_kernel, ctx->model->ssm_state_size, ctx->model->ssm_group_count,
		       ctx->model->ssm_inner_size, ctx->model->ssm_time_step_rank);
	}

	printf("Context_length: %u\n", ctx->model->seq_length);
}

void vision_model_info(struct TIEContext *ctx)
{
	printf("Initialized %s vision model with the following configuration:\n", ctx->model->def->name);
	printf("Image size: %d, Proj Dim: %d, Patch size: %d\n", ctx->model_vision->image_size,
	       ctx->model_vision->projection_dim, ctx->model_vision->patch_size);
	printf("Embed Dim: %d, FFN Dim: %d, Layers: %d, Heads: %d, eps: %f\n", ctx->model_vision->embed_dim,
	       ctx->model_vision->ffn_dim, ctx->model_vision->num_layers, ctx->model_vision->num_heads,
	       ctx->model_vision->norm_eps);
	printf("Proj Scale Factor: %d, Deepstack layers: %u\n", ctx->model_vision->def->params.proj_scale_factor,
	       ctx->model_vision->num_deepstack_layers);
}
