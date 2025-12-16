#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "engine.h"
#include "model.h"
#include "math_dispatch.h"
#include "threadpool.h"

float silu_table[SILU_TABLE_SIZE];

void engine_alloc(struct TIEContext *ctx, int num_threads)
{
#ifdef CONFIG_ENABLE_AVX2
	__builtin_cpu_init();
	if (!__builtin_cpu_supports("avx2")) {
		fprintf(stderr, "AVX2 not supported!\n");
		exit(EXIT_FAILURE);
	}
#endif
	printf("Create thread_pool: %u threads\n", num_threads);
	if ((thread_pool = thread_pool_create(num_threads)) == NULL) {
		free(ctx);
		exit(EXIT_FAILURE);
	}

	/* Initialize tokenizer */
	ctx->tokenizer.root = create_node();
	ctx->tokenizer.pool = create_string_pool(1024 * 1024 * 4);

	// Initialize SiLU lookup table
	printf("init: SiLU table\n");
	silu_table_init();

	srand(time(NULL));

	math_dispatch_init();

	//	debug_init("debug_dump.log");

	return;
}

void engine_release(struct TIEContext *ctx)
{
	gguf_model_close(ctx->gguf_text);
	model_language_cleanup(ctx, ctx->gguf_text, ctx->model->def, ctx->config.use_mmap);

	if (ctx->gguf_vision != NULL) {
		gguf_model_close(ctx->gguf_vision);
		model_vision_cleanup(ctx, ctx->gguf_vision, ctx->model_vision->def, ctx->config.use_mmap);
	}

	tools_release(ctx);

	thread_pool_destroy(thread_pool);
	free(ctx);

	//	debug_close();
}

#if 1
#define DEBUG_MEMTYPE_NUM 10
void debug_memtype_f32(MemType *mem, char *name, int layer_idx, int debug_offset)
{
	float *p = mem->data;


	printf("--- layer%d_%s --- offset: %u [", layer_idx, name, debug_offset);
	for (int i = 0; i < DEBUG_MEMTYPE_NUM; i++) {
		if (i < DEBUG_MEMTYPE_NUM - 1) {
			printf("%.10f, ", p[i + debug_offset]);
		} else {
			printf("%.10f ...", p[i + debug_offset]);
		}
	}
	printf("]\n");
}

void debug_check_tensor(const char *name, MemType *mem, int len, int layer, int batch_idx)
{
	float *ptr = mem->data;

	double maxv = -INFINITY, minv = INFINITY;
	int has_nan = 0, has_inf = 0;
	for (int i = 0; i < len; ++i) {
		float v = ptr[i];
		if (isnan(v)) {
			has_nan = 1;
			break;
		}
		if (isinf(v)) {
			has_inf = 1;
			break;
		}
		if (v > maxv)
			maxv = v;
		if (v < minv)
			minv = v;
	}

	if (has_nan || has_inf || maxv > 1e30 || minv < -1e30) {
		fprintf(stderr, "[NAN/INF] layer=%d batch=%d var=%s nan=%d inf=%d max=%g min=%g\n", layer, batch_idx,
			name, has_nan, has_inf, maxv, minv);

		for (int i = 0; i < 8 && i < len; ++i)
			fprintf(stderr, "%g ", ptr[i]);
		fprintf(stderr, "\n");

		abort();
	}
}

int compare_tensor_with_file(const char *ref_filename, const float *your_c_tensor, long num_elements, float tolerance,
			     int debug_offset)
{
	//	return 0;

	FILE *file = fopen(ref_filename, "rb");
	if (file == NULL) {
		fprintf(stderr, "ERROR: Could not open reference file %s\n", ref_filename);
		return 1;
	}

	float *ref_tensor = (float *)malloc(num_elements * sizeof(float));
	if (ref_tensor == NULL) {
		fprintf(stderr, "ERROR: Could not allocate memory for reference tensor\n");
		fclose(file);
		return 1;
	}

	long elements_read = fread(ref_tensor, sizeof(float), num_elements, file);
	fclose(file);

	if (elements_read != num_elements) {
		fprintf(stderr, "ERROR: Read mismatch. Expected %ld elements, got %ld\n", num_elements, elements_read);
		free(ref_tensor);
		return 1;
	}

	printf("File: %s first %u elements from %u: ", ref_filename, DEBUG_MEMTYPE_NUM, debug_offset);
	for (int i = 0; i < DEBUG_MEMTYPE_NUM; i++) {
		if (i < DEBUG_MEMTYPE_NUM - 1) {
			printf("%.10f, ", ref_tensor[i + debug_offset]);
		} else {
			printf("%.10f ...", ref_tensor[i + debug_offset]);
		}
	}
	printf("\n");

	int mismatch_found = 0;
	for (long i = 0; i < num_elements; i++) {
		float diff = fabsf(your_c_tensor[i] - ref_tensor[i]);

		if (diff > tolerance) {
			fprintf(stderr, "\n--- MISMATCH DETECTED - tolerance: %.10f---\n", tolerance);
			fprintf(stderr, "File: %s\n", ref_filename);
			fprintf(stderr, "Index: %ld (elements: %ld)\n", i, num_elements);
			fprintf(stderr, "  C Value:   %f\n", your_c_tensor[i]);
			fprintf(stderr, "  Ref Value: %f\n", ref_tensor[i]);
			fprintf(stderr, "  Difference: %f\n", diff);

			mismatch_found = 1;
			break;
		}
	}

	free(ref_tensor);

	if (!mismatch_found) {
		printf("SUCCESS: %s matches the reference. Tolerance: %.10f\n", ref_filename, tolerance);
	} else {
		exit(EXIT_FAILURE);
	}

	return mismatch_found;
}

int load_tensor_from_file(const char *ref_filename, float *dest_tensor, long num_elements)
{
	FILE *file = fopen(ref_filename, "rb");
	if (file == NULL) {
		fprintf(stderr, "ERROR: Could not open file %s for loading.\n", ref_filename);
		return 1;
	}

	long elements_read = fread(dest_tensor, sizeof(float), num_elements, file);
	fclose(file);

	if (elements_read != num_elements) {
		fprintf(stderr, "ERROR: Read mismatch. Expected %ld elements, got %ld\n", num_elements, elements_read);
		return 1;
	}

	printf("SUCCESS: Loaded %ld elements from %s.\n", elements_read, ref_filename);
	return 0;
}
#endif


// A fast, high-precision expf approximation by Nicol N. Schraudolph.
inline float expf_fast(float x)
{
	// Clamp input to a safe range to avoid overflow/underflow
	x = fmaxf(-88.37626266479492f, fminf(88.37626266479492f, x));

	// Union for type-punning to access the float's bits as an integer
	union {
		float f;
		int i;
	} u;

	// The core of the approximation
	u.i = (int)(12102203.0f * x + 1065353216.0f);

	// A polynomial correction term for better accuracy
	float p = u.f;
	p = p * (1.9875691500E-4f * p + 1.3981999507E-3f);
	p = p * p + p * (8.3334519073E-3f * p + 4.1665795894E-2f);
	p = p * p + p * (1.6666665459E-1f * p + 5.0000001201E-1f);

	return u.f + u.f * p;
}

void *xaligned_alloc(size_t alignment, size_t size_bytes)
{
	size_t padded = (size_bytes + (alignment - 1)) & ~(alignment - 1);

	void *p = NULL;
	if (posix_memalign(&p, alignment, padded) != 0)
		return NULL;

	return p;
}

void alloc_memtype(MemType *m, GGMLType t, size_t nelems)
{
	m->type = t;
	size_t total = nelems * ggml_type_size(t);
	void *ptr = xaligned_alloc(32, total);
	if (!ptr) {
		fprintf(stderr, "ERROR: alloc %zu bytes for MemType failed\n", total);
		exit(1);
	}
	m->n_bytes = total;
	m->data = ptr;
}

void free_memtype(MemType *m)
{
	free(m->data);
}

MemType mem_slice(MemType *buffer, size_t offset_elements)
{
	size_t element_size = ggml_type_size(buffer->type);
	MemType slice = {.type = buffer->type, .data = (uint8_t *)buffer->data + (offset_elements * element_size)};
	return slice;
}

void kv_cache_reset(struct TIEContext *ctx)
{
	int kv_dim = ctx->model->num_kv_heads * ctx->model->head_dim;

	for (int i = 0; i < ctx->model->num_layers; i++) {
		memset(ctx->kv_cache[i].k.data, 0,
		       ctx->model->seq_length * kv_dim * ggml_type_size(ctx->kv_cache[i].k.type));
		memset(ctx->kv_cache[i].v.data, 0,
		       ctx->model->seq_length * kv_dim * ggml_type_size(ctx->kv_cache[i].v.type));
	}
}

static inline void *get_kv_head(const MemType *cache, int t, int kv_head_idx, int head_dim, int num_kv_heads,
				int ring_size, int sink_len)
{
	size_t element_size = ggml_type_size(cache->type);
	int physical_t;

	// SINK + RING MAPPING
	if (t < sink_len) {
		physical_t = t;
	} else {
		int rolling_capacity = ring_size - sink_len;
		int offset = t - sink_len;
		physical_t = sink_len + (offset % rolling_capacity);
	}

	size_t kv_dim = (size_t)num_kv_heads * head_dim;
	size_t offset_elements = (size_t)physical_t * kv_dim + (size_t)kv_head_idx * head_dim;

	return (uint8_t *)cache->data + offset_elements * element_size;
}


void silu_table_init()
{
	for (int i = 0; i < SILU_TABLE_SIZE; ++i) {
		float x = SILU_X_MIN + (SILU_X_MAX - SILU_X_MIN) * i / (SILU_TABLE_SIZE - 1);
		silu_table[i] = x / (1.0f + expf_fast(-x));
	}
}

float silu_lookup(float x)
{
	if (!isfinite(x)) {
		// If x is not a valid number, we can't use the lookup table.
		return x / (1.0f + expf_fast(-x));
	}

	if (x <= SILU_X_MIN)
		return x / (1.0f + expf_fast(-x));
	if (x >= SILU_X_MAX)
		return x / (1.0f + expf_fast(-x));

	float range = (SILU_X_MAX - SILU_X_MIN);
	float position = (x - SILU_X_MIN) * (SILU_TABLE_SIZE - 1) / range;

	int idx = (int)position;
	float frac = position - idx;

	// Clamp to prevent out-of-bounds read
	if (idx >= SILU_TABLE_SIZE - 1)
		idx = SILU_TABLE_SIZE - 2;

	return fmaf(silu_table[idx + 1] - silu_table[idx], frac, silu_table[idx]);
}

void rope_cache_init(struct TIEContext *ctx, RopeCacheType *rope_cache, int max_pos, int head_dim, int rope_dim,
		     float base, float scale)
{
	float scaled_pos;

	rope_cache->max_pos = max_pos;
	rope_cache->head_dim = head_dim;
	rope_cache->rope_dim = rope_dim;

	// only need cache for the ROTATED dimensions (rope_dim / 2)
	int cache_elements = max_pos * (rope_dim / 2);

	rope_cache->sin = aligned_alloc(32, sizeof(float) * cache_elements);
	rope_cache->cos = aligned_alloc(32, sizeof(float) * cache_elements);

	printf("RoPE init base=%.01f, scale=%f, rot_dim=%d\n", base, scale, rope_dim);

	for (int pos = 0; pos < max_pos; ++pos) {
		scaled_pos = (float)pos * scale;

		for (int i = 0; i < rope_dim / 2; ++i) {

			float exponent = (2.0f * (float)i) / (float)rope_dim;

			float inv_freq = 1.0f / powf(base, exponent);
			float angle = scaled_pos * inv_freq;

			rope_cache->sin[pos * (rope_dim / 2) + i] = sinf(angle);
			rope_cache->cos[pos * (rope_dim / 2) + i] = cosf(angle);
		}
	}
}

void build_rope_cache_dynamic(struct TIEContext *ctx, size_t seq_len)
{
	printf("%s seq_len: %zu\n", __FUNCTION__, seq_len);

	ctx->model->rope_cache_global = malloc(sizeof(RopeCacheType));
	if (!ctx->model->rope_cache_global) {
		perror("Failed to allocate rope_cache");
		exit(EXIT_FAILURE);
	}

	RopeCacheType *rope_cache = ctx->model->rope_cache_global;

	// Allocate the *data buffers
	size_t head_dim = ctx->model->head_dim;
	size_t max_tokens = seq_len; // max context length
	size_t buffer_bytes = sizeof(float) * max_tokens * head_dim;

	rope_cache->sin = aligned_alloc(32, buffer_bytes);
	rope_cache->cos = aligned_alloc(32, buffer_bytes);
	rope_cache->max_pos = max_tokens;

	if (!rope_cache->sin || !rope_cache->cos) {
		perror("Failed to allocate dynamic rope_cache buffers");
		exit(EXIT_FAILURE);
	}
}

// Qwen3-VL Interleaved M-RoPE logic
// Shuffles H and W frequencies into the T frequency buffer.
static void apply_interleaved_mrope(float *freqs_t, const float *freqs_h, const float *freqs_w,
				    const int *mrope_sections, int seq_len, int half_head_dim)
{
	// Copy H frequencies
	for (int i = 0; i < mrope_sections[1]; i++) { // Loop 0..19
		int t_idx = i * 3 + 1;		      // 1, 4, 7, ...
		if (t_idx >= half_head_dim)
			break;
		for (int j = 0; j < seq_len; j++) {
			freqs_t[j * half_head_dim + t_idx] = freqs_h[j * half_head_dim + t_idx];
		}
	}

	// Copy W frequencies
	for (int i = 0; i < mrope_sections[2]; i++) { // Loop 0..19
		int t_idx = i * 3 + 2;		      // 2, 5, 8, ...
		if (t_idx >= half_head_dim)
			break;
		for (int j = 0; j < seq_len; j++) {
			freqs_t[j * half_head_dim + t_idx] = freqs_w[j * half_head_dim + t_idx];
		}
	}
}

/* Builds the M-RoPE cos/sin tables */
void text_rope_cache_init(struct TIEContext *ctx, int seq_len, int start_pos)
{
	Model *lm = ctx->model;
	RopeCacheType *rope_cache = ctx->model->rope_cache_global;
	MemLayout *mem = &ctx->mem;

	const int head_dim = lm->head_dim;
	const int half_head_dim = head_dim / 2;
	const float rope_base = lm->rope_freq_base;

	// Calculate inv_freq
	float inv_freq[half_head_dim];
	const float log_rope_base = logf(rope_base);

	for (int i = 0; i < head_dim; i += 2) { // Iterate 0, 2, ... 126
		float dim_val = (float)i / (float)head_dim;
		inv_freq[i / 2] = expf(-dim_val * log_rope_base); // Fills inv_freq[0...63]
	}

	// Get the position ID buffers (filled by build_mrope_position_ids)
	const int max_len = ctx->model->seq_length;

	int *pos_t_data = (int *)mem->pos_ids.data;
	int *pos_h_data = pos_t_data + max_len;
	int *pos_w_data = pos_h_data + max_len;

	float *cos_table = (float *)rope_cache->cos;
	float *sin_table = (float *)rope_cache->sin;

	// Temporary buffers for T, H, W frequencies
	float *freqs_t = malloc(seq_len * half_head_dim * sizeof(float));
	float *freqs_h = malloc(seq_len * half_head_dim * sizeof(float));
	float *freqs_w = malloc(seq_len * half_head_dim * sizeof(float));
	if (!freqs_t || !freqs_h || !freqs_w) {
		fprintf(stderr, "Failed to alloc temp freq buffers for M-RoPE\n");
		return;
	}

	// Calculate T, H, W frequencies
	for (int i = 0; i < seq_len; i++) {
		for (int j = 0; j < half_head_dim; j++) {
			freqs_t[i * half_head_dim + j] = (float)pos_t_data[i] * inv_freq[j];
			freqs_h[i * half_head_dim + j] = (float)pos_h_data[i] * inv_freq[j];
			freqs_w[i * half_head_dim + j] = (float)pos_w_data[i] * inv_freq[j];
		}
	}

	// Apply Interleaved M-RoPE
	// This shuffles H/W frequencies into the T buffer (freqs_t)
	int *mrope_sections = (int *)lm->mrope_sections.data;
	apply_interleaved_mrope(freqs_t, freqs_h, freqs_w, mrope_sections, seq_len, half_head_dim);

	free(freqs_h);
	free(freqs_w);

	// Finalize cos/sin tables
	for (int i = 0; i < seq_len; i++) {

		int absolute_pos = start_pos + i;
		if (absolute_pos >= rope_cache->max_pos)
			continue;

		float *cos_row = cos_table + (size_t)absolute_pos * head_dim;
		float *sin_row = sin_table + (size_t)absolute_pos * head_dim;

		float *freqs_t_row = freqs_t + i * half_head_dim;

		memcpy(cos_row, freqs_t_row, half_head_dim * sizeof(float));
		memcpy(cos_row + half_head_dim, freqs_t_row, half_head_dim * sizeof(float));

		memcpy(sin_row, freqs_t_row, half_head_dim * sizeof(float));
		memcpy(sin_row + half_head_dim, freqs_t_row, half_head_dim * sizeof(float));

		for (int j = 0; j < head_dim; j++) {
			cos_row[j] = cosf(cos_row[j]);
			sin_row[j] = sinf(sin_row[j]);
		}
	}

	free(freqs_t);
}

// Calculates and writes the M-RoPE cos/sin values for a single new token at the specified position pos
void text_rope_cache_extend(struct TIEContext *ctx, int pos)
{
	Model *lm = ctx->model;
	RopeCacheType *rope_cache = ctx->model->rope_cache_global;

	//    printf("%s pos: %u\n", __FUNCTION__, pos);

	if (pos >= rope_cache->max_pos) {
		fprintf(stderr, "WARN: RoPE cache exhausted at pos %d. RoPE will be incorrect.\n", pos);
		return;
	}

	const int head_dim = lm->head_dim;	// 128
	const int half_head_dim = head_dim / 2; // 64
	const float rope_base = lm->rope_freq_base;

	// Calculate inv_freq
	float inv_freq[half_head_dim];
	const float log_rope_base = logf(rope_base);
	for (int i = 0; i < head_dim; i += 2) {
		float dim_val = (float)i / (float)head_dim;
		inv_freq[i / 2] = expf(-dim_val * log_rope_base);
	}

	// Calculate Frequencies for this one token
	// A new token is a text token, so its position is (T=pos, H=0, W=0)
	float freqs_t[half_head_dim];
	float freqs_h[half_head_dim];
	float freqs_w[half_head_dim];

	for (int j = 0; j < half_head_dim; j++) {
		freqs_t[j] = (float)pos * inv_freq[j];
		freqs_h[j] = (float)pos * inv_freq[j];
		freqs_w[j] = (float)pos * inv_freq[j];
	}

	// Interleaved M-RoPE
	int *mrope_sections = (int *)lm->mrope_sections.data;
	apply_interleaved_mrope(freqs_t, freqs_h, freqs_w, mrope_sections,
				1, // seq_len is 1
				half_head_dim);

	// Finalize and write cos/sin tables *only for pos
	float *cos_row = (float *)rope_cache->cos + (size_t)pos * head_dim;
	float *sin_row = (float *)rope_cache->sin + (size_t)pos * head_dim;

	memcpy(cos_row, freqs_t, half_head_dim * sizeof(float));
	memcpy(cos_row + half_head_dim, freqs_t, half_head_dim * sizeof(float));

	memcpy(sin_row, freqs_t, half_head_dim * sizeof(float));
	memcpy(sin_row + half_head_dim, freqs_t, half_head_dim * sizeof(float));

	for (int j = 0; j < head_dim; j++) {
		cos_row[j] = cosf(cos_row[j]);
		sin_row[j] = sinf(sin_row[j]);
	}
}

/* Fills the 3D M-RoPE position ID buffer based on the final token sequence */
void build_mrope_position_ids(struct TIEContext *ctx, const int *prompt_tokens, size_t prompt_len, bool has_image,
			      int start_pos, int h_patches_in, int w_patches_in)
{
	MemType *pos_id_buf = &ctx->mem.pos_ids;
	memset((void *)pos_id_buf->data, 0, pos_id_buf->n_bytes);
	ModelDef *def = ctx->model->def;

	// Get base pointers for each dimension (T, H, W)
	const int max_len = ctx->model->seq_length;
	int *pos_t = (int *)pos_id_buf->data;
	int *pos_h = pos_t + max_len;
	int *pos_w = pos_h + max_len;

	int text_pos_counter = start_pos;
	int image_patch_counter = 0;

	int h_patches = 0;
	int w_patches = 0;
	int num_image_patches = 0;

	if (has_image && ctx->model_vision) {
		// Calculate the *merged* grid size, which is what the LLM sees.
		h_patches = h_patches_in / ctx->model_vision->spatial_merge_size;
		w_patches = w_patches_in / ctx->model_vision->spatial_merge_size;

		// Calculate total expected patches
		num_image_patches = h_patches * w_patches;
	}

	// Loop over the final token list
	for (int i = 0; i < prompt_len; i++) {
		int token = prompt_tokens[i];

		// Check for the <vision_pad> token
		if (has_image && token == def->params.vision_embed_token_id) {
			// image patch token
			int h = image_patch_counter / w_patches;
			int w = image_patch_counter % w_patches;

			// text_pos_counter as the base for all.
			pos_t[i] = text_pos_counter; // T is 0 + counter
			pos_h[i] = h + text_pos_counter;
			pos_w[i] = w + text_pos_counter;

			image_patch_counter++;
			text_pos_counter++;

		} else {
			// normal text token
			pos_t[i] = text_pos_counter;
			pos_h[i] = text_pos_counter;
			pos_w[i] = text_pos_counter;

			text_pos_counter++;
		}
	}

	if (has_image && image_patch_counter != num_image_patches) {
		fprintf(stderr, "WARNING: M-RoPE position mismatch. Expected %d patches, counted %d\n",
			num_image_patches, image_patch_counter);
	}
}

static inline bool is_token_masked(struct TIEContext *ctx, AttentionType type, int abs_pos, int t)
{
	// Global attention sees everything
	if (type == ATTN_TYPE_GLOBAL)
		return false;

	// LOCAL / SLIDING WINDOW LOGIC
	// Gemma-3N (Centered Window)
	if (ctx->gguf_text->arch == ARCH_GEMMA3) {
		int window = ctx->model->attn_sliding_window;
		int left = (window - 1) / 2;
		int right = window / 2;
		int dist = abs_pos - t;

		// Allow if within [pos - left, pos + right]
		if ((dist >= 0 && dist <= left) || (dist < 0 && -dist <= right)) {
			return false; // Not masked (Visible)
		}
		return true; // Masked
	}

	// Standard Causal Sliding Window
	// [pos - window, pos]
	if (ctx->model->attn_sliding_window > 0) {
		int start_window = abs_pos - (int)ctx->model->attn_sliding_window;
		if (t >= start_window)
			return false; // Not masked
		return true;	      // Masked
	}

	// No mask
	return false;
}

void attention_worker(void *arg)
{
	attention_worker_task_t *task = (attention_worker_task_t *)arg;
	struct TIEContext *ctx = task->ctx;
	AttentionType attn_type = task->attn_type;
	int sink_len = task->sink_len;

	int q_dim = ctx->model->num_heads * ctx->model->head_dim;
	LayerKVCache *cache = &ctx->kv_cache[task->kv_source_layer_idx];

	// scratch buffers
	float *attn_scores_buffer = ctx->mem.attn_scores_buffer[task->thread_id];
	MemType *q_head_fp32_scratch = &ctx->mem.q_head_fp32_scratch[task->thread_id];

	int ring_size = ctx->model->seq_length;
	int rolling_capacity = ring_size - sink_len;

	//	printf("ATTN_SCALE: %.5f\n", ctx->model->attn_scale);

	for (int i = task->token_start_idx; i < task->token_end_idx; i++) {
		int absolute_pos = task->batch_start_pos + i;

		// CALCULATE MEMORY RANGES
		int sink_end = (absolute_pos < sink_len) ? absolute_pos + 1 : sink_len;

		int rolling_start = absolute_pos - rolling_capacity + 1;
		if (rolling_start < sink_len)
			rolling_start = sink_len;

		for (int h = task->head_start; h < task->head_end; h++) {

			// QUERY PREP
			MemType q_head_slice = mem_slice(&ctx->mem.Q, (size_t)i * q_dim + h * ctx->model->head_dim);
			MemType out_head_slice =
				mem_slice(&ctx->mem.attn_output, (size_t)i * q_dim + h * ctx->model->head_dim);

			dispatch_convert(&q_head_slice, q_head_fp32_scratch, ctx->model->head_dim);

			// Q-Scaling (Gemma-3)
			if (ctx->gguf_text->arch == ARCH_GEMMA3) {
				float q_scale = 1.0f / sqrtf((float)ctx->model->head_dim);
				float *q_fp32_data = (float *)q_head_fp32_scratch->data;
				for (int k = 0; k < ctx->model->head_dim; k++)
					q_fp32_data[k] *= q_scale;
			}

			int kv_head_idx = h / (ctx->model->num_heads / ctx->model->num_kv_heads);
			int compact_idx = 0;

			// LOOP 1: SINK (0..sink_len)
			for (int t = 0; t < sink_end; t++) {

				// Check mask
				if (!is_token_masked(ctx, attn_type, absolute_pos, t)) {
					void *k_head = get_kv_head(&cache->k, t, kv_head_idx, ctx->model->head_dim,
								   ctx->model->num_kv_heads, ring_size, sink_len);
					MemType k_head_slice = {.type = cache->k.type, .data = k_head};
					float score = dispatch_dot_product(q_head_fp32_scratch, &k_head_slice,
									   ctx->model->head_dim);

					attn_scores_buffer[compact_idx] = score * ctx->model->attn_scale;
				} else {
					attn_scores_buffer[compact_idx] = -INFINITY;
				}
				compact_idx++;
			}

			// LOOP 2: RING (Rolling History)
			for (int t = rolling_start; t <= absolute_pos; t++) {
				// Check mask
				if (!is_token_masked(ctx, attn_type, absolute_pos, t)) {
					void *k_head = get_kv_head(&cache->k, t, kv_head_idx, ctx->model->head_dim,
								   ctx->model->num_kv_heads, ring_size, sink_len);
					MemType k_head_slice = {.type = cache->k.type, .data = k_head};
					float score = dispatch_dot_product(q_head_fp32_scratch, &k_head_slice,
									   ctx->model->head_dim);

					attn_scores_buffer[compact_idx] = score * ctx->model->attn_scale;
				} else {
					attn_scores_buffer[compact_idx] = -INFINITY;
				}
				compact_idx++;
			}

			// SOFTMAX
			int num_scores = compact_idx;
			float max_score = -INFINITY;
			for (int k = 0; k < num_scores; k++) {
				if (attn_scores_buffer[k] > max_score)
					max_score = attn_scores_buffer[k];
			}

			float sum_exp = 0.0f;
			for (int k = 0; k < num_scores; k++) {
				if (attn_scores_buffer[k] == -INFINITY) {
					attn_scores_buffer[k] = 0.0f;
				} else {
					float val = expf(attn_scores_buffer[k] - max_score);
					attn_scores_buffer[k] = val;
					sum_exp += val;
				}
			}

			float inv_sum = 1.0f / (sum_exp > 1e-10f ? sum_exp : 1.0f);
			for (int k = 0; k < num_scores; k++)
				attn_scores_buffer[k] *= inv_sum;

			// WEIGHTED SUM
			memset(out_head_slice.data, 0, ctx->model->head_dim * ggml_type_size(out_head_slice.type));
			compact_idx = 0;

			// Sum Sink
			for (int t = 0; t < sink_end; t++) {
				float weight = attn_scores_buffer[compact_idx++];
				if (weight < 1e-10f)
					continue;

				void *v_head = get_kv_head(&cache->v, t, kv_head_idx, ctx->model->head_dim,
							   ctx->model->num_kv_heads, ring_size, sink_len);
				MemType v_slice = {.type = cache->v.type, .data = v_head};
				dispatch_accumulate_weighted_V(&v_slice, &out_head_slice, weight, ctx->model->head_dim);
			}

			// Sum Ring
			for (int t = rolling_start; t <= absolute_pos; t++) {
				float weight = attn_scores_buffer[compact_idx++];
				if (weight < 1e-10f)
					continue;

				void *v_head = get_kv_head(&cache->v, t, kv_head_idx, ctx->model->head_dim,
							   ctx->model->num_kv_heads, ring_size, sink_len);
				MemType v_slice = {.type = cache->v.type, .data = v_head};
				dispatch_accumulate_weighted_V(&v_slice, &out_head_slice, weight, ctx->model->head_dim);
			}
		}
	}
}

void attention(struct TIEContext *ctx, int batch_len, int layer_idx, int kv_source_layer_idx, int start_pos,
	       AttentionType attn_type, attention_fn worker, int sink_len)
{
	int num_threads = thread_pool->num_threads;
	attention_worker_task_t tasks[num_threads];

	// Parallelize over HEADS
	if (batch_len == 1) {
		int heads_per_thread = (ctx->model->num_heads + num_threads - 1) / num_threads;

		for (int t = 0; t < num_threads; t++) {
			int head_start = t * heads_per_thread;
			int head_end = head_start + heads_per_thread;
			if (head_end > ctx->model->num_heads)
				head_end = ctx->model->num_heads;
			if (head_start >= head_end)
				break;

			tasks[t] = (attention_worker_task_t){
				.ctx = ctx,
				.layer_idx = layer_idx,
				.kv_source_layer_idx = kv_source_layer_idx,
				.batch_start_pos = start_pos,
				.batch_len = batch_len,
				.attn_type = attn_type,
				.thread_id = t,
				.token_start_idx = 0,	  // Process the single token at index 0
				.token_end_idx = 1,	  //
				.head_start = head_start, // Each thread gets a different slice of heads
				.head_end = head_end,	  //
				.sink_len = sink_len,
			};
			thread_pool_submit(thread_pool, worker, &tasks[t]);
		}

		thread_pool_wait(thread_pool);
		return;
	}

	// Parallelize over TOKENS
	int tokens_per_thread = (batch_len + num_threads - 1) / num_threads;

	for (int t = 0; t < num_threads; t++) {
		int token_start = t * tokens_per_thread;
		int token_end = token_start + tokens_per_thread;
		if (token_end > batch_len)
			token_end = batch_len;
		if (token_start >= token_end)
			break;

		tasks[t] = (attention_worker_task_t){
			.ctx = ctx,
			.layer_idx = layer_idx,
			.kv_source_layer_idx = kv_source_layer_idx,
			.batch_start_pos = start_pos,
			.batch_len = batch_len,
			.attn_type = attn_type,
			.thread_id = t,
			.token_start_idx = token_start,	   // Each thread gets a different slice of tokens
			.token_end_idx = token_end,	   //
			.head_start = 0,		   // Each thread processes all heads
			.head_end = ctx->model->num_heads, //
			.sink_len = sink_len,
		};
		thread_pool_submit(thread_pool, worker, &tasks[t]);
	}

	thread_pool_wait(thread_pool);
}

void softmax(float *x, int size)
{
	if (size == 0)
		return;

	// Find max value for numerical stability
	float max_val = x[0];
	for (int i = 1; i < size; i++) {
		if (x[i] > max_val) {
			max_val = x[i];
		}
	}

	// Calculate exponentials and sum
	float sum = 0.0f;
	for (int i = 0; i < size; i++) {
		x[i] = expf(x[i] - max_val);
		sum = fmaf(x[i], 1.0f, sum);
	}

	// Normalize
	for (int i = 0; i < size; i++) {
		x[i] /= sum;
	}
}

// Finds the top-k experts from the router logits
void find_top_k(const float *router_logits, int expert_count, int k, ExpertChoice *top_k)
{
	for (int i = 0; i < k; ++i) {
		top_k[i] = (ExpertChoice){.index = -1, .score = -INFINITY};
	}

	for (int i = 0; i < expert_count; ++i) {
		float score = router_logits[i];
		if (score > top_k[k - 1].score) {
			// This expert is better than the worst of our current top-k
			top_k[k - 1] = (ExpertChoice){.index = i, .score = score};
			// Simple insertion sort to maintain the small top_k array
			for (int j = k - 2; j >= 0; --j) {
				if (top_k[j + 1].score > top_k[j].score) {
					ExpertChoice temp = top_k[j];
					top_k[j] = top_k[j + 1];
					top_k[j + 1] = temp;
				} else {
					break;
				}
			}
		}
	}
}

void process_expert_task(void *arg)
{
	expert_task_t *task = (expert_task_t *)arg;
	struct TIEContext *ctx = task->ctx;
	int expert_idx = task->expert_idx;
	LayerWeights *l = &ctx->model->layers[task->layer_idx];

	// Use thread-specific scratch buffers
	MemType *ffn_hidden1 = &ctx->mem.ffn_hidden1_scratch[task->thread_id];
	MemType *ffn_hidden2 = &ctx->mem.ffn_hidden2_scratch[task->thread_id];
	MemType *expert_out = &ctx->mem.expert_outputs[task->thread_id];
	MemType *normed_input = &task->normed_input;

	size_t up_gate_block_size_bytes = ggml_block_size(l->ffn_up_exps.mem.type);
	size_t up_gate_blocks_per_row = ctx->model->embed_dim / QK_K;
	size_t up_gate_blocks_per_matrix = ctx->model->expert_ffn_dim * up_gate_blocks_per_row;
	size_t up_gate_matrix_size_bytes = up_gate_blocks_per_matrix * up_gate_block_size_bytes;

	// Pre-calculate sizes for the DOWN expert matrices
	size_t down_block_size_bytes = ggml_block_size(l->ffn_down_exps.mem.type);
	size_t down_blocks_per_row = ctx->model->expert_ffn_dim / QK_K;
	size_t down_blocks_per_matrix = ctx->model->embed_dim * down_blocks_per_row;
	size_t down_matrix_size_bytes = down_blocks_per_matrix * down_block_size_bytes;

	// Calculate separate byte offsets for the selected expert
	size_t up_gate_offset_bytes = (size_t)expert_idx * up_gate_matrix_size_bytes;
	size_t down_offset_bytes = (size_t)expert_idx * down_matrix_size_bytes;

	// Create temporary Tensor structs pointing to the correct data slices
	Tensor expert_gate = {.mem.type = l->ffn_gate_exps.mem.type,
			      .mem.data = (uint8_t *)l->ffn_gate_exps.mem.data + up_gate_offset_bytes};
	Tensor expert_up = {.mem.type = l->ffn_up_exps.mem.type,
			    .mem.data = (uint8_t *)l->ffn_up_exps.mem.data + up_gate_offset_bytes};
	Tensor expert_down = {.mem.type = l->ffn_down_exps.mem.type,
			      .mem.data = (uint8_t *)l->ffn_down_exps.mem.data + down_offset_bytes};

	// FFN forward pass
	dispatch_mat_vec(ctx, normed_input, &expert_gate, ffn_hidden1, ctx->model->embed_dim,
			 ctx->model->expert_ffn_dim, false);
	dispatch_mat_vec(ctx, normed_input, &expert_up, ffn_hidden2, ctx->model->embed_dim, ctx->model->expert_ffn_dim,
			 false);

	// SiLU
	dispatch_swiglu_activation(ffn_hidden1, ffn_hidden2, ctx->model->expert_ffn_dim);

	// Down-projection
	dispatch_mat_vec(ctx, ffn_hidden1, &expert_down, expert_out, ctx->model->expert_ffn_dim, ctx->model->embed_dim,
			 false);
}

void prepare_next_token_standard(struct TIEContext *ctx, int next_token)
{
	if (ctx->model->use_mrope == 1) {
		// Build the cos/sin table for the new token at ctx->kv_pos
		text_rope_cache_extend(ctx, ctx->kv_pos);
	}

	MemType hidden_state_slice = mem_slice(&ctx->mem.hidden_state, 0);

	dispatch_embedding_row(&ctx->model->token_embd, next_token, &hidden_state_slice, ctx->model->embed_dim);

	if (ctx->model->interface.embedding_scale != NULL)
		ctx->model->interface.embedding_scale(ctx, &hidden_state_slice);
}

void process_embeddings(struct TIEContext *ctx, MemType *embeddings, size_t n_tokens)
{
	if (n_tokens == 0)
		return;

	MemType hidden_state_slice = mem_slice(&ctx->mem.hidden_state, (n_tokens - 1) * ctx->model->embed_dim);

	// Run the transformer layers.
	for (int l = 0; l < ctx->model->num_layers; l++) {
		ctx->model->interface.transformer_layer(ctx, l, n_tokens);
	}

	MemType hidden_state_first_slice = mem_slice(&ctx->mem.hidden_state, 0);
	memcpy(hidden_state_first_slice.data, hidden_state_slice.data,
	       ctx->model->embed_dim * ggml_type_size(ctx->mem.hidden_state.type));

	// Update the KV cache position.
	ctx->kv_pos += n_tokens;
}
