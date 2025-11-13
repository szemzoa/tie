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

#define DEBUG_MEMTYPE_NUM 10
void debug_memtype_f32(MemType *mem, char *name, int layer_idx, int debug_offset)
{
	float *p = mem->data;

	//	return;

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

#if 0
static void debug_check_tensor(const char *name, MemType *mem, int len, int layer, int batch_idx)
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

int compare_tensor_with_file(const char *ref_filename, const float *your_c_tensor, long num_elements, float tolerance, int debug_offset)
{
	FILE *file = fopen(ref_filename, "rb");
	if (file == NULL) {
		fprintf(stderr, "ERROR: Could not open reference file %s\n", ref_filename);
		return 1; // Return error
	}

	// Allocate memory to hold the reference data
	float *ref_tensor = (float *)malloc(num_elements * sizeof(float));
	if (ref_tensor == NULL) {
		fprintf(stderr, "ERROR: Could not allocate memory for reference tensor\n");
		fclose(file);
		return 1;
	}

	// Read the data from the file
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

	// --- The Comparison ---
	int mismatch_found = 0;
	for (long i = 0; i < num_elements; i++) {
		float diff = fabsf(your_c_tensor[i] - ref_tensor[i]);

		if (diff > tolerance) {
			fprintf(stderr, "\n--- MISMATCH DETECTED ---\n");
			fprintf(stderr, "File: %s\n", ref_filename);
			fprintf(stderr, "Index: %ld (elements: %ld)\n", i, num_elements);
			fprintf(stderr, "  C Value:   %f\n", your_c_tensor[i]);
			fprintf(stderr, "  Ref Value: %f\n", ref_tensor[i]);
			fprintf(stderr, "  Difference: %f\n", diff);

			mismatch_found = 1;
			break; // Stop at the first error
		}
	}

	free(ref_tensor); // Clean up

	if (!mismatch_found) {
		printf("SUCCESS: %s matches the reference.\n", ref_filename);
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

static inline void *xaligned_alloc(size_t alignment, size_t size_bytes)
{
	// aligned_alloc requires size % alignment == 0
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
	m->data = ptr;
}

void free_memtype(MemType *m)
{
	free(m->data);
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

void silu_table_init()
{
	for (int i = 0; i < SILU_TABLE_SIZE; ++i) {
		float x = SILU_X_MIN + (SILU_X_MAX - SILU_X_MIN) * i / (SILU_TABLE_SIZE - 1);
		silu_table[i] = x / (1.0f + expf_fast(-x));
	}
}

void rope_cache_init(struct TIEContext *ctx, RopeCacheType *rope_cache, int max_pos, int head_dim, float base,
		     float scale)
{
	float scaled_pos;

	rope_cache->max_pos = max_pos;
	rope_cache->head_dim = head_dim;

	rope_cache->sin = aligned_alloc(32, sizeof(float) * max_pos * (head_dim / 2));
	rope_cache->cos = aligned_alloc(32, sizeof(float) * max_pos * (head_dim / 2));

	printf("RoPE init base=%.01f, scale=%f\n", base, scale);

	for (int pos = 0; pos < max_pos; ++pos) {

		scaled_pos = (float)pos * scale;

		for (int i = 0; i < head_dim / 2; ++i) {
			float exponent = (2.0f * (float)i) / (float)head_dim;
			float inv_freq = 1.0f / powf(base, exponent);
			float angle = scaled_pos * inv_freq;

			// Store sin and cos
			rope_cache->sin[pos * (head_dim / 2) + i] = sinf(angle);
			rope_cache->cos[pos * (head_dim / 2) + i] = cosf(angle);
		}
	}
}

float silu_lookup(float x)
{
	if (!isfinite(x)) {
		// If x is not a valid number, we can't use the lookup table.
		// Fallback to a direct, safe calculation or return 0.
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

	// Use FMA for better interpolation precision
	return fmaf(silu_table[idx + 1] - silu_table[idx], frac, silu_table[idx]);
}

MemType mem_slice(MemType *buffer, size_t offset_elements)
{
	size_t element_size = ggml_type_size(buffer->type);
	MemType slice = {.type = buffer->type, .data = (uint8_t *)buffer->data + (offset_elements * element_size)};
	return slice;
}

void embedding_scale_gemma3(struct TIEContext *ctx, MemType *hidden_state_slice)
{
	float *hidden_data_fp32;

	float scale = sqrtf((float)ctx->model->embed_dim);
	hidden_data_fp32 = (float *)hidden_state_slice->data;
	for (int j = 0; j < ctx->model->embed_dim; j++) {
		float val = hidden_data_fp32[j];
		val *= scale;
		hidden_data_fp32[j] = val;
	}
}

static inline void *get_kv_head(const MemType *cache, int t, int kv_head_idx, int head_dim, int num_kv_heads)
{
	size_t element_size = ggml_type_size(cache->type);
	size_t kv_dim = (size_t)num_kv_heads * head_dim;
	size_t offset_elements = (size_t)t * kv_dim + (size_t)kv_head_idx * head_dim;
	return (uint8_t *)cache->data + offset_elements * element_size;
}

void attention_worker(void *arg)
{
	attention_worker_task_t *task = (attention_worker_task_t *)arg;
	struct TIEContext *ctx = task->ctx;
	AttentionType attn_type = task->attn_type;

	// Get model dimensions and memory type info
	int q_dim = ctx->model->num_heads * ctx->model->head_dim;
	LayerKVCache *cache = &ctx->kv_cache[task->kv_source_layer_idx];

	// Get this thread's private scratch buffers
	float *attn_scores_buffer = ctx->mem.attn_scores_buffer[task->thread_id];
	MemType *q_head_fp32_scratch = &ctx->mem.q_head_fp32_scratch[task->thread_id];

	// Loop over the assigned range of tokens
	for (int i = task->token_start_idx; i < task->token_end_idx; i++) {
		int absolute_pos = task->batch_start_pos + i;

		// Loop over the assigned range of heads
		for (int h = task->head_start; h < task->head_end; h++) {

			// Calculate the pointer to the source Q-head
			MemType q_head_slice = mem_slice(&ctx->mem.Q, (size_t)i * q_dim + h * ctx->model->head_dim);
			MemType out_head_slice =
				mem_slice(&ctx->mem.attn_output, (size_t)i * q_dim + h * ctx->model->head_dim);

			// Convert the source Q-head to FP32 in our scratch buffer
			dispatch_convert(&q_head_slice, q_head_fp32_scratch, ctx->model->head_dim);

			// PRE-SCALE THE QUERY
			if (ctx->gguf_text->arch == ARCH_GEMMA3) {

				float q_scale = 1.0f / sqrtf((float)ctx->model->head_dim);
				float *q_fp32_data = (float *)q_head_fp32_scratch->data;
				for (int i = 0; i < ctx->model->head_dim; i++) {
					q_fp32_data[i] *= q_scale;
				}
			}

			// Calculate pointer to the destination output head
			int kv_head_idx = h / (ctx->model->num_heads / ctx->model->num_kv_heads);

			// Calculate Raw Attention Scores
			for (int t = 0; t <= absolute_pos; t++) {

				bool can_attend = false;

				if (attn_type == ATTN_TYPE_GLOBAL) {
					// Global attention only needs the standard causal mask,
					// which the loop condition (t <= absolute_pos) already enforces.
					can_attend = true;
				} else {
					// Local attention needs the causal mask AND the sliding window check.
					int sliding_window_start =
						(int)absolute_pos - (int)ctx->model->attn_sliding_window;
					if (t >= sliding_window_start) {
						can_attend = true;
					}
				}

				if (can_attend) {
					void *k_head = get_kv_head(&cache->k, t, kv_head_idx, ctx->model->head_dim,
								   ctx->model->num_kv_heads);

					MemType k_head_slice = {.type = cache->k.type, .data = (void *)k_head};

					float score = dispatch_dot_product(q_head_fp32_scratch, &k_head_slice,
									   ctx->model->head_dim);

					attn_scores_buffer[t] = score * ctx->model->attn_scale;

				} else {
					attn_scores_buffer[t] = -INFINITY;
				}
			}

			bool all_scores_are_masked = true;
			for (int t = 0; t <= absolute_pos; t++) {
				if (attn_scores_buffer[t] > -INFINITY) {
					all_scores_are_masked = false;
					break;
				}
			}

			if (all_scores_are_masked && absolute_pos > 0) {
				printf("WARNING: All keys were masked for query at absolute_pos %d in layer %d!\n",
				       absolute_pos, task->layer_idx);
			}

			// Calculate Softmax
			// Softmax is also over the full history up to the absolute_pos
			float max_score = -INFINITY;
			for (int t = 0; t <= absolute_pos; t++) {
				if (attn_scores_buffer[t] > max_score)
					max_score = attn_scores_buffer[t];
			}

			float sum_exp = 0.0f;
			for (int t = 0; t <= absolute_pos; t++) {
				float val = expf(attn_scores_buffer[t] - max_score);
				attn_scores_buffer[t] = val;
				sum_exp += val;
			}

			float inv_sum_exp = 1.0f / sum_exp;
			for (int t = 0; t <= absolute_pos; t++) {
				attn_scores_buffer[t] *= inv_sum_exp;
			}

			// Calculate Weighted Sum of V-vectors
			memset(out_head_slice.data, 0, ctx->model->head_dim * ggml_type_size(out_head_slice.type));

			for (int t = 0; t <= absolute_pos; t++) {
				void *v_head_ptr = get_kv_head(&cache->v, t, kv_head_idx, ctx->model->head_dim,
							       ctx->model->num_kv_heads);

				float attention_weight = attn_scores_buffer[t];

				// Create a MemType view of the data from the KV cache
				MemType v_head_slice = {.type = cache->v.type, .data = (void *)v_head_ptr};

				dispatch_accumulate_weighted_V(&v_head_slice, &out_head_slice, attention_weight,
							       ctx->model->head_dim);
			}
		}
	}

	free(task);
}

void attention_worker_gemma3n(void *arg)
{
	attention_worker_task_t *task = (attention_worker_task_t *)arg;
	struct TIEContext *ctx = task->ctx;
	AttentionType attn_type = task->attn_type;

	int q_dim = ctx->model->num_heads * ctx->model->head_dim;
	LayerKVCache *cache = &ctx->kv_cache[task->kv_source_layer_idx];

	float *attn_scores_buffer = ctx->mem.attn_scores_buffer[task->thread_id];
	MemType *q_head_fp32_scratch = &ctx->mem.q_head_fp32_scratch[task->thread_id];

	for (int i = task->token_start_idx; i < task->token_end_idx; i++) {
		int absolute_pos = task->batch_start_pos + i;

		// All threads must know the full context length of the current batch.
		// Use task->batch_len, which is the total size of the batch.
		// Do NOT use task->token_end_idx, which is just the chunk size for this thread.
		int max_kv_len = task->batch_start_pos + task->batch_len;

		for (int h = task->head_start; h < task->head_end; h++) {

			MemType q_head_slice = mem_slice(&ctx->mem.Q, (size_t)i * q_dim + h * ctx->model->head_dim);
			MemType out_head_slice =
				mem_slice(&ctx->mem.attn_output, (size_t)i * q_dim + h * ctx->model->head_dim);

			dispatch_convert(&q_head_slice, q_head_fp32_scratch, ctx->model->head_dim);

			int kv_head_idx = h / (ctx->model->num_heads / ctx->model->num_kv_heads);

			// Calculate scores using the correct loop bounds and masking
			for (int t = 0; t < max_kv_len; t++) {
				bool can_attend = false;

				if (attn_type == ATTN_TYPE_GLOBAL) {

					can_attend = true;

				} else { // Gemma-3N's Centered Sliding Window
					int sliding_window_size = ctx->model->attn_sliding_window;
					int left_window_size = (sliding_window_size - 1) / 2;
					int right_window_size = sliding_window_size / 2;
					int dist = absolute_pos - t;

					if ((dist >= 0 && dist <= left_window_size)
					    || (dist < 0 && -dist <= right_window_size)) {
						can_attend = true;
					}
				}

				if (can_attend) {
					void *k_head = get_kv_head(&cache->k, t, kv_head_idx, ctx->model->head_dim,
								   ctx->model->num_kv_heads);
					MemType k_head_slice = {.type = cache->k.type, .data = k_head};
					float score = dispatch_dot_product(q_head_fp32_scratch, &k_head_slice,
									   ctx->model->head_dim);
					attn_scores_buffer[t] = score * ctx->model->attn_scale;
				} else {
					attn_scores_buffer[t] = -INFINITY;
				}
			}

			// Calculate Softmax over the correct number of tokens
			float max_score = -INFINITY;
			for (int t = 0; t < max_kv_len; t++) {
				if (attn_scores_buffer[t] > max_score)
					max_score = attn_scores_buffer[t];
			}

			float sum_exp = 0.0;
			for (int t = 0; t < max_kv_len; t++) {
				if (attn_scores_buffer[t] > -INFINITY) {
					float val = expf(attn_scores_buffer[t] - max_score);
					attn_scores_buffer[t] = val;
					sum_exp += val;
				} else {
					attn_scores_buffer[t] = 0.0f;
				}
			}

			float inv_sum_exp = 1.0f / (sum_exp > 1e-6 ? (float)sum_exp : 1.0f);
			for (int t = 0; t < max_kv_len; t++) {
				attn_scores_buffer[t] *= inv_sum_exp;
			}

			// Calculate Weighted Sum of V-vectors
			memset(out_head_slice.data, 0, ctx->model->head_dim * ggml_type_size(out_head_slice.type));
			for (int t = 0; t < max_kv_len; t++) {
				if (attn_scores_buffer[t] > 1e-6f) {
					void *v_head_ptr = get_kv_head(&cache->v, t, kv_head_idx, ctx->model->head_dim,
								       ctx->model->num_kv_heads);
					MemType v_head_slice = {.type = cache->v.type, .data = v_head_ptr};
					dispatch_accumulate_weighted_V(&v_head_slice, &out_head_slice,
								       attn_scores_buffer[t], ctx->model->head_dim);
				}
			}
		}
	}

	free(task);
}

void attention(struct TIEContext *ctx, int batch_len, int layer_idx, int kv_source_layer_idx, int start_pos,
	       AttentionType attn_type, attention_fn worker)
{
	int num_threads = thread_pool->num_threads;

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

			attention_worker_task_t *task = malloc(sizeof(attention_worker_task_t));
			*task = (attention_worker_task_t){
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
			};
			thread_pool_submit(thread_pool, worker, task);
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

		attention_worker_task_t *task = malloc(sizeof(attention_worker_task_t));
		*task = (attention_worker_task_t){
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
		};
		thread_pool_submit(thread_pool, worker, task);
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

	free(task);
}

int transformer_layer_qwen3(struct TIEContext *ctx, int layer_idx, int batch_len)
{
	LayerWeights *l = &ctx->model->layers[layer_idx];
	int kv_dim = ctx->model->num_kv_heads * ctx->model->head_dim;
	int q_dim = ctx->model->num_heads * ctx->model->head_dim;
	AttentionType attn_type = ATTN_TYPE_GLOBAL;
	RopeCacheType *active_rope_cache = ctx->rope_cache_global;

	// The absolute starting position for this batch
	int start_pos = ctx->kv_pos;

	// ============ Attention Block ============
	// RMSNorm on input
	for (int i = 0; i < batch_len; i++) {
		size_t offset = (size_t)i * ctx->model->embed_dim;

		// Create slices for the specific token being processed
		MemType hidden_state_slice = mem_slice(&ctx->mem.hidden_state, offset);
		MemType normed_input_slice = mem_slice(&ctx->mem.normed_qkv_input, offset);

		dispatch_rms_norm(&hidden_state_slice, &l->attn_norm, &normed_input_slice, ctx->model->embed_dim,
				  ctx->model->norm_eps);
	}

	// Compute Q/K/V Matrices
	dispatch_mat_mat(ctx, &ctx->mem.normed_qkv_input, &l->attn_q, &ctx->mem.Q, batch_len, ctx->model->embed_dim,
			 q_dim, true);

	dispatch_mat_mat(ctx, &ctx->mem.normed_qkv_input, &l->attn_k, &ctx->mem.K, batch_len, ctx->model->embed_dim,
			 kv_dim, true);

	dispatch_mat_mat(ctx, &ctx->mem.normed_qkv_input, &l->attn_v, &ctx->mem.V, batch_len, ctx->model->embed_dim,
			 kv_dim, true);

	// Apply RoPE
	for (int i = 0; i < batch_len; i++) {
		// The absolute position for the current token in the batch
		int absolute_pos = start_pos + i;

		for (int h = 0; h < ctx->model->num_heads; h++) {
			MemType Q_slice = mem_slice(&ctx->mem.Q, (size_t)i * q_dim + h * ctx->model->head_dim);

			if (l->attn_q_norm.mem.data) {
				dispatch_rms_norm(&Q_slice, &l->attn_q_norm, &Q_slice, ctx->model->head_dim,
						  ctx->model->norm_eps);
			}

			// Use the absolute position for RoPE
			dispatch_apply_rope_cache(active_rope_cache, &Q_slice, absolute_pos, ctx->model->head_dim);
		}

		for (int h = 0; h < ctx->model->num_kv_heads; h++) {
			MemType K_slice = mem_slice(&ctx->mem.K, (size_t)i * kv_dim + h * ctx->model->head_dim);

			if (l->attn_k_norm.mem.data) {
				dispatch_rms_norm(&K_slice, &l->attn_k_norm, &K_slice, ctx->model->head_dim,
						  ctx->model->norm_eps);
			}

			// Use the absolute position for RoPE
			dispatch_apply_rope_cache(active_rope_cache, &K_slice, absolute_pos, ctx->model->head_dim);
		}
	}

	// Store K/V to cache
	dispatch_store_KV_cache(ctx, layer_idx, start_pos, batch_len);

	// Multi-Head Attention Calculation
	attention(ctx, batch_len, layer_idx, layer_idx, start_pos, attn_type, attention_worker);

	// Output projection
	dispatch_mat_mat(ctx, &ctx->mem.attn_output, &l->attn_out, &ctx->mem.attn_proj_output, batch_len, q_dim,
			 ctx->model->embed_dim, true);

	// Add residual
	dispatch_apply_residual(&ctx->mem.hidden_state, &ctx->mem.attn_proj_output, batch_len * ctx->model->embed_dim);

	// ============ FFN Block ============
	// RMSNorm
	for (int i = 0; i < batch_len; i++) {
		size_t offset = (size_t)i * ctx->model->embed_dim;

		// Create slices for the specific token being processed
		MemType hidden_state_slice = mem_slice(&ctx->mem.hidden_state, offset);
		MemType normed_ffn_input_slice = mem_slice(&ctx->mem.normed_ffn_input, offset);

		dispatch_rms_norm(&hidden_state_slice, &l->ffn_norm, &normed_ffn_input_slice, ctx->model->embed_dim,
				  ctx->model->norm_eps);
	}

	// MoE
	if (ctx->model->is_moe) {

		// Get the size of a single quantization block for the expert tensors
		size_t block_size_bytes = ggml_block_size(l->ffn_up_exps.mem.type);
		if (block_size_bytes == 0) {
			return -1;
		}

		for (int i = 0; i < batch_len; i++) {

			// Pointers for the current token
			MemType normed_input_for_token_i =
				mem_slice(&ctx->mem.normed_ffn_input, (size_t)i * ctx->model->embed_dim);

			// Create a slice for the destination buffer
			MemType ffn_out_slice = mem_slice(&ctx->mem.ffn_down_output, (size_t)i * ctx->model->embed_dim);

			// Use a per-thread scratch buffer for FP32 accumulation
			MemType *ffn_out_fp32_scratch = &ctx->mem.expert_out_fp32;
			float *ffn_out_fp32_token_buffer = ctx->mem.expert_out_fp32.data;

			// Route, Select, and Gate
			// The input type for the router is the intermediate type, but the output scores are always
			// FP32.
			dispatch_mat_vec(ctx, &normed_input_for_token_i, &l->ffn_gate_inp, &ctx->mem.expert_scores,
					 ctx->model->embed_dim, ctx->model->expert_count, false);

			ExpertChoice top_experts[ctx->model->expert_used_count];
			find_top_k((float *)ctx->mem.expert_scores.data, ctx->model->expert_count,
				   ctx->model->expert_used_count, top_experts);

			float gate_values[ctx->model->expert_used_count];

			for (int j = 0; j < ctx->model->expert_used_count; j++)
				gate_values[j] = top_experts[j].score;

			softmax(gate_values, ctx->model->expert_used_count);

			// Parallel Expert Processing
			for (int j = 0; j < ctx->model->expert_used_count; j++) {
				expert_task_t *task = malloc(sizeof(expert_task_t));
				*task = (expert_task_t){
					.ctx = ctx,
					.thread_id = j,
					.layer_idx = layer_idx,
					.expert_idx = top_experts[j].index,
					.normed_input = normed_input_for_token_i,
				};
				thread_pool_submit(thread_pool, process_expert_task, task);
			}
			thread_pool_wait(thread_pool);

			// Accumulate results in the FP32 temporary buffer
			memset(ffn_out_fp32_scratch->data, 0, ctx->model->embed_dim * sizeof(float));

			for (int j = 0; j < ctx->model->expert_used_count; j++) {
				float gate_val = gate_values[j];
				float *expert_result = ctx->mem.expert_outputs[j].data;
				for (int k = 0; k < ctx->model->embed_dim; k++) {
					ffn_out_fp32_token_buffer[k] += gate_val * expert_result[k];
				}
			}

			// Convert the final FP32 result to the destination format
			dispatch_convert(ffn_out_fp32_scratch, &ffn_out_slice, ctx->model->embed_dim);
		}

	} else { // DENSE FFN

		// Gate + Up projections
		dispatch_mat_mat(ctx, &ctx->mem.normed_ffn_input, &l->ffn_gate, &ctx->mem.gate_proj_output, batch_len,
				 ctx->model->embed_dim, ctx->model->ffn_dim, true);
		dispatch_mat_mat(ctx, &ctx->mem.normed_ffn_input, &l->ffn_up, &ctx->mem.up_proj_output, batch_len,
				 ctx->model->embed_dim, ctx->model->ffn_dim, true);

		/* Call the interface activation function */
		dispatch_swiglu_activation(&ctx->mem.gate_proj_output, &ctx->mem.up_proj_output,
					   batch_len * ctx->model->ffn_dim);


		// Down projection
		dispatch_mat_mat(ctx, &ctx->mem.gate_proj_output, &l->ffn_down, &ctx->mem.ffn_down_output, batch_len,
				 ctx->model->ffn_dim, ctx->model->embed_dim, true);
	}

	// Add residual
	dispatch_apply_residual(&ctx->mem.hidden_state, &ctx->mem.ffn_down_output, batch_len * ctx->model->embed_dim);

	return 0;
}

int transformer_layer_gemma3(struct TIEContext *ctx, int layer_idx, int batch_len)
{
	LayerWeights *l = &ctx->model->layers[layer_idx];
	int kv_dim = ctx->model->num_kv_heads * ctx->model->head_dim;
	int q_dim = ctx->model->num_heads * ctx->model->head_dim;
	AttentionType attn_type;
	RopeCacheType *active_rope_cache;

	// The absolute starting position for this batch
	int start_pos = ctx->kv_pos;

	// Determine the attention and rope cache type for the current layer
	if ((layer_idx + 1) % 6 == 0) {
		attn_type = ATTN_TYPE_GLOBAL;
		active_rope_cache = ctx->rope_cache_global; // Use the global cache
	} else {
		attn_type = ATTN_TYPE_LOCAL;
		active_rope_cache = ctx->rope_cache_local; // Use the local cache
	}

	// Save the residual for the attention block.
	// We use a scratch buffer to hold the original hidden_state.
	memcpy(ctx->mem.residual_stratch.data, ctx->mem.hidden_state.data,
	       batch_len * ctx->model->embed_dim * sizeof(float));

	// ============ Attention Block ============
	// RMSNorm on input
	for (int i = 0; i < batch_len; i++) {
		size_t offset = (size_t)i * ctx->model->embed_dim;

		// Create slices for the specific token being processed
		MemType hidden_state_slice = mem_slice(&ctx->mem.hidden_state, offset);
		MemType normed_input_slice = mem_slice(&ctx->mem.normed_qkv_input, offset);

		dispatch_rms_norm(&hidden_state_slice, &l->attn_norm, &normed_input_slice, ctx->model->embed_dim,
				  ctx->model->norm_eps);
	}

	// Compute Q/K/V Matrices
	dispatch_mat_mat(ctx, &ctx->mem.normed_qkv_input, &l->attn_q, &ctx->mem.Q, batch_len, ctx->model->embed_dim,
			 q_dim, true);

	dispatch_mat_mat(ctx, &ctx->mem.normed_qkv_input, &l->attn_k, &ctx->mem.K, batch_len, ctx->model->embed_dim,
			 kv_dim, true);

	dispatch_mat_mat(ctx, &ctx->mem.normed_qkv_input, &l->attn_v, &ctx->mem.V, batch_len, ctx->model->embed_dim,
			 kv_dim, true);

	// Apply RoPE
	for (int i = 0; i < batch_len; i++) {
		// The absolute position for the current token in the batch
		int absolute_pos = start_pos + i;

		for (int h = 0; h < ctx->model->num_heads; h++) {
			MemType Q_slice = mem_slice(&ctx->mem.Q, (size_t)i * q_dim + h * ctx->model->head_dim);

			if (l->attn_q_norm.mem.data) {
				dispatch_rms_norm(&Q_slice, &l->attn_q_norm, &Q_slice, ctx->model->head_dim,
						  ctx->model->norm_eps);
			}

			// Use the absolute position for RoPE
			dispatch_apply_rope_cache(active_rope_cache, &Q_slice, absolute_pos, ctx->model->head_dim);
		}

		for (int h = 0; h < ctx->model->num_kv_heads; h++) {
			MemType K_slice = mem_slice(&ctx->mem.K, (size_t)i * kv_dim + h * ctx->model->head_dim);

			if (l->attn_k_norm.mem.data) {
				dispatch_rms_norm(&K_slice, &l->attn_k_norm, &K_slice, ctx->model->head_dim,
						  ctx->model->norm_eps);
			}

			// Use the absolute position for RoPE
			dispatch_apply_rope_cache(active_rope_cache, &K_slice, absolute_pos, ctx->model->head_dim);
		}
	}

	// Store K/V to cache
	dispatch_store_KV_cache(ctx, layer_idx, start_pos, batch_len);

	// Multi-Head Attention Calculation
	attention(ctx, batch_len, layer_idx, layer_idx, start_pos, attn_type, attention_worker);

	// Output projection
	dispatch_mat_mat(ctx, &ctx->mem.attn_output, &l->attn_out, &ctx->mem.attn_proj_output, batch_len, q_dim,
			 ctx->model->embed_dim, true);

	// Apply POST-ATTENTION norm.
	for (int i = 0; i < batch_len; i++) {
		size_t offset = (size_t)i * ctx->model->embed_dim;

		MemType attn_proj_slice = mem_slice(&ctx->mem.attn_proj_output, offset);

		dispatch_rms_norm(&attn_proj_slice, &l->post_attn_norm, &attn_proj_slice, ctx->model->embed_dim,
				  ctx->model->norm_eps);
	}

	// Add the attention residual.
	// Add the post-normed attention output to the original residual we saved.
	// The result is stored in `hidden_state`.
	dispatch_apply_residual(&ctx->mem.residual_stratch, &ctx->mem.attn_proj_output,
				batch_len * ctx->model->embed_dim);

	memcpy(ctx->mem.hidden_state.data, ctx->mem.residual_stratch.data,
	       batch_len * ctx->model->embed_dim * sizeof(float));


	// ============ FFN Block ============
	// Save the residual for the FFN block.
	memcpy(ctx->mem.residual_stratch.data, ctx->mem.hidden_state.data,
	       batch_len * ctx->model->embed_dim * sizeof(float));

	// RMSNorm
	for (int i = 0; i < batch_len; i++) {
		size_t offset = (size_t)i * ctx->model->embed_dim;

		// Create slices for the specific token being processed
		MemType hidden_state_slice = mem_slice(&ctx->mem.hidden_state, offset);
		MemType normed_ffn_input_slice = mem_slice(&ctx->mem.normed_ffn_input, offset);

		dispatch_rms_norm(&hidden_state_slice, &l->ffn_norm, &normed_ffn_input_slice, ctx->model->embed_dim,
				  ctx->model->norm_eps);
	}

	// Gate + Up projections
	dispatch_mat_mat(ctx, &ctx->mem.normed_ffn_input, &l->ffn_gate, &ctx->mem.gate_proj_output, batch_len,
			 ctx->model->embed_dim, ctx->model->ffn_dim, true);
	dispatch_mat_mat(ctx, &ctx->mem.normed_ffn_input, &l->ffn_up, &ctx->mem.up_proj_output, batch_len,
			 ctx->model->embed_dim, ctx->model->ffn_dim, true);

	/* Call the interface activation function */
	dispatch_geglu_activation(&ctx->mem.gate_proj_output, &ctx->mem.up_proj_output,
				  batch_len * ctx->model->ffn_dim);

	// Down projection
	dispatch_mat_mat(ctx, &ctx->mem.gate_proj_output, &l->ffn_down, &ctx->mem.ffn_down_output, batch_len,
			 ctx->model->ffn_dim, ctx->model->embed_dim, true);

	// Apply POST-FFN norm.
	for (int i = 0; i < batch_len; i++) {
		size_t offset = (size_t)i * ctx->model->embed_dim;
		MemType ffn_down_slice = mem_slice(&ctx->mem.ffn_down_output, offset);

		dispatch_rms_norm(&ffn_down_slice, &l->post_ffw_norm, &ffn_down_slice, ctx->model->embed_dim,
				  ctx->model->norm_eps);
	}

	// Second Residual Add.
	// Add the post-normed FFN output to the residual we saved
	dispatch_apply_residual(&ctx->mem.residual_stratch, &ctx->mem.ffn_down_output,
				batch_len * ctx->model->embed_dim);

	// Copy the correct final result back to hidden_state for the next layer.
	memcpy(ctx->mem.hidden_state.data, ctx->mem.residual_stratch.data,
	       batch_len * ctx->model->embed_dim * sizeof(float));

	return 0;
}

// Implements the full LAUREL block logic for a given input.
void dispatch_laurel(struct TIEContext *ctx, MemType *output, const MemType *input, LayerWeights *l, int batch_len)
{
	const int embed_dim = ctx->model->embed_dim;
	const int laurel_rank = 64; // From the config

	MemType laurel_left;
	alloc_memtype(&laurel_left, GGML_TYPE_F32, batch_len * laurel_rank);

	MemType laurel_right;
	alloc_memtype(&laurel_right, GGML_TYPE_F32, batch_len * embed_dim);

	// Left projection: input @ laurel_l.weight
	dispatch_mat_mat(ctx, input, &l->laurel_l, &laurel_left, batch_len, embed_dim, laurel_rank, true);

	// Right projection: laurel_left @ laurel_r.weight
	dispatch_mat_mat(ctx, &laurel_left, &l->laurel_r, &laurel_right, batch_len, laurel_rank, embed_dim, true);

	// Post-LAUREL normalization
	for (int i = 0; i < batch_len; i++) {
		MemType slice = mem_slice(&laurel_right, i * embed_dim);
		dispatch_rms_norm(&slice, &l->laurel_post_norm, &slice, embed_dim, ctx->model->norm_eps);
	}

	// Final residual connection: output = input + normed_result
	dispatch_apply_residual_to_buffer(input, &laurel_right, output, batch_len * embed_dim);

	free_memtype(&laurel_left);
	free_memtype(&laurel_right);
}

// This can be optimized with AVX2.
void dispatch_subtract_to_buffer(const MemType *src1, const MemType *src2, MemType *dest, int size)
{
	const float *src1_data = (const float *)src1->data;
	const float *src2_data = (const float *)src2->data;
	float *dest_data = (float *)dest->data;

	// A simple loop for element-wise addition.
	for (int i = 0; i < size; i++) {
		dest_data[i] = src1_data[i] - src2_data[i];
	}
}

// This can be optimized with AVX2.
void dispatch_elementwise_mul(MemType *dest, const MemType *src1, const MemType *src2, int size)
{
	float *dest_data = (float *)dest->data;
	const float *src1_data = (const float *)src1->data;
	const float *src2_data = (const float *)src2->data;

	for (int i = 0; i < size; i++) {
		dest_data[i] = src1_data[i] * src2_data[i];
	}
}

// This can be optimized with AVX2.
void dispatch_elementwise_mul_tensor(MemType *dest, const MemType *src1, const Tensor *src2, int batch_len,
				     int embed_dim)
{
	float *dest_data = (float *)dest->data;
	const float *src1_data = (const float *)src1->data;
	const float *src2_data = (const float *)src2->mem.data;

	int size = batch_len * embed_dim;

	for (int i = 0; i < size; i++) {
		// Use the modulo operator to repeat/broadcast the smaller src2 vector
		// for each token in the batch.
		dest_data[i] = src1_data[i] * src2_data[i % embed_dim];
	}
}

void dispatch_rms_norm_weightless(MemType *tensor, int size, float eps)
{
	float *x = (float *)tensor->data;

	// Unrolled accumulation for faster reduction
	float ss0 = 0.0f, ss1 = 0.0f, ss2 = 0.0f, ss3 = 0.0f;
	int i = 0;
	for (; i <= size - 4; i += 4) {
		ss0 = fmaf(x[i + 0], x[i + 0], ss0);
		ss1 = fmaf(x[i + 1], x[i + 1], ss1);
		ss2 = fmaf(x[i + 2], x[i + 2], ss2);
		ss3 = fmaf(x[i + 3], x[i + 3], ss3);
	}
	float ss = ss0 + ss1 + ss2 + ss3;
	for (; i < size; i++) {
		ss = fmaf(x[i], x[i], ss);
	}

	float inv_rms = 1.0f / sqrtf(ss / size + eps);

	// Apply scale
	for (int i = 0; i < size; ++i) {
		x[i] *= inv_rms;
	}
}

void dispatch_gaussian_topk(MemType *gate_tensor, int size)
{
	float *data = (float *)gate_tensor->data;
	const float f_sparsity_std_mul = 1.6448533535f; // Corresponds to the 95th percentile

	if (size == 0)
		return;

	// Numerically Stable Single-Pass Algorithm (Welford's)
	float mean = 0.0;
	float m2 = 0.0;
	float delta;

	for (int i = 0; i < size; i++) {
		delta = data[i] - mean;
		mean += delta / (i + 1);
		m2 += delta * (data[i] - mean);
	}

	float variance = m2 / size; // Population variance
	float std_dev = sqrt(variance);

	// Determine the cutoff value
	float cutoff = mean + std_dev * f_sparsity_std_mul;

	// Apply the ReLU-like sparsity: output = max(0, input - cutoff)
	for (int i = 0; i < size; i++) {
		data[i] = (data[i] > cutoff) ? (data[i] - cutoff) : 0.0f;
	}
}

inline float gelu_fast(float x)
{
	return 0.5f * x * (1.0f + tanhf(0.79788456f * x * (1.0f + 0.044715f * x * x)));
}

// GELU activation function to a tensor in-place.
// This can be optimized with AVX2.
void dispatch_gelu_inplace(MemType *tensor, int size)
{
	float *data = (float *)tensor->data;
	for (int i = 0; i < size; i++) {
		data[i] = gelu_fast(data[i]);
	}
}

void dispatch_softcap_logits(MemType *logits, int size, float cap)
{
	if (cap <= 0.0f)
		return;

	float *data = (float *)logits->data;
	float inv_cap = 1.0f / cap;

	for (int i = 0; i < size; i++) {
		data[i] = tanhf(data[i] * inv_cap) * cap;
	}
}

void dispatch_apply_residual_to_buffer(const MemType *src1, const MemType *src2, MemType *dest, int size)
{
	const float *src1_data = (const float *)src1->data;
	const float *src2_data = (const float *)src2->data;
	float *dest_data = (float *)dest->data;

	for (int i = 0; i < size; i++) {
		dest_data[i] = src1_data[i] + src2_data[i];
	}
}

void dispatch_altup_predict(struct TIEContext *ctx, MemType *predicted_states, MemType *input_states, LayerWeights *l,
			    int batch_len, int active_idx)
{
	const int embed_dim = ctx->model->embed_dim;
	const int num_altup = ctx->model->altup_num_inputs;
	MemType *active_state_in = &input_states[active_idx];

	MemType normed_for_router, modalities, prediction_coefs;
	alloc_memtype(&normed_for_router, GGML_TYPE_F32, batch_len * embed_dim);
	alloc_memtype(&modalities, GGML_TYPE_F32, batch_len * num_altup);
	alloc_memtype(&prediction_coefs, GGML_TYPE_F32, batch_len * (num_altup * num_altup));


	for (int i = 0; i < batch_len; i++) {
		size_t offset = (size_t)i * embed_dim;
		MemType active_slice = mem_slice(active_state_in, offset);
		MemType normed_slice = mem_slice(&normed_for_router, offset);

		// Normalize one token at a time
		dispatch_rms_norm(&active_slice, &l->altup_router_norm, &normed_slice, embed_dim, ctx->model->norm_eps);

		// Scale one token at a time
		float scale = 1.0f / (float)embed_dim;
		float *norm_data = (float *)normed_slice.data;
		for (int d = 0; d < embed_dim; d++) {
			norm_data[d] *= scale;
		}
	}

	dispatch_mat_mat(ctx, &normed_for_router, &l->altup_router, &modalities, batch_len, embed_dim, num_altup, true);

	float *mod = (float *)modalities.data;
	for (int i = 0; i < batch_len * num_altup; i++)
		mod[i] = tanhf(mod[i]);
	dispatch_mat_mat(ctx, &modalities, &l->altup_predict_coef, &prediction_coefs, batch_len, num_altup,
			 num_altup * num_altup, true);

	// Use Coefficients to Mix States
	const float *coefs_data = (const float *)prediction_coefs.data;
	MemType mixed_states;
	alloc_memtype(&mixed_states, GGML_TYPE_F32, batch_len * num_altup * embed_dim);

	// Re-order the loops to create a [state][token][dim] memory layout.
	// This makes the data for each state contiguous across the entire batch.
	for (int i = 0; i < num_altup; i++) {	      // For each OUTPUT state
		for (int t = 0; t < batch_len; t++) { // For each token
			const float *coefs_for_token = coefs_data + t * (num_altup * num_altup);
			float *out_vec = (float *)mixed_states.data + (i * batch_len + t) * embed_dim;
			memset(out_vec, 0, embed_dim * sizeof(float));

			for (int j = 0; j < num_altup; j++) { // For each INPUT state
				const float *in_vec = (const float *)input_states[j].data + t * embed_dim;
				const float coef = coefs_for_token[i * num_altup + j];
				for (int d = 0; d < embed_dim; d++) {
					out_vec[d] += coef * in_vec[d];
				}
			}
		}
	}

	// Final Residual Connection
	for (int i = 0; i < num_altup; i++) {
		// Slice points to the start of the contiguous data for state `i`.
		MemType slice = mem_slice(&mixed_states, (size_t)i * batch_len * embed_dim);
		dispatch_apply_residual_to_buffer(&input_states[i], &slice, &predicted_states[i],
						  batch_len * embed_dim);
	}

	free_memtype(&normed_for_router);
	free_memtype(&modalities);
	free_memtype(&prediction_coefs);
	free_memtype(&mixed_states);
}

// corrected_states: Output: The final 4 corrected states for this layer
// predictions: Input: The 4 states from the 'P' step
// final_active_output: Input: The result of the 'A' step
void dispatch_altup_correct(struct TIEContext *ctx, MemType *corrected_states, const MemType *predictions,
			    MemType *final_active_output, LayerWeights *l, int batch_len, int active_idx)
{
	const int embed_dim = ctx->model->embed_dim;
	const int num_altup = ctx->model->altup_num_inputs;

	MemType modalities, correction_coefs, innovation;
	alloc_memtype(&modalities, GGML_TYPE_F32, batch_len * num_altup);
	alloc_memtype(&correction_coefs, GGML_TYPE_F32, batch_len * num_altup);
	alloc_memtype(&innovation, GGML_TYPE_F32, batch_len * embed_dim);

	// Create a single buffer for all per-token calculations
	MemType temp_token_vec;
	alloc_memtype(&temp_token_vec, GGML_TYPE_F32, embed_dim);
	float *temp_data = (float *)temp_token_vec.data;

	// Process the Corrector Path ONE TOKEN AT A TIME
	for (int i = 0; i < batch_len; i++) {
		size_t offset = (size_t)i * embed_dim;
		MemType active_slice = mem_slice(final_active_output, offset);

		// Normalize the single active token's vector
		dispatch_rms_norm(&active_slice, &l->altup_router_norm, &temp_token_vec, embed_dim,
				  ctx->model->norm_eps);

		// Scale the result
		float scale = 1.0f / (float)embed_dim;
		for (int d = 0; d < embed_dim; d++) {
			temp_data[d] *= scale;
		}

		// Compute modalities for this single token
		MemType modalities_slice = mem_slice(&modalities, i * num_altup);
		dispatch_mat_vec(ctx, &temp_token_vec, &l->altup_router, &modalities_slice, embed_dim, num_altup, true);

		// Apply tanh to this token's modalities
		float *mod_slice_data = (float *)modalities_slice.data;
		for (int d = 0; d < num_altup; d++) {
			mod_slice_data[d] = tanhf(mod_slice_data[d]);
		}
	}

	// Compute correction coefficients for the whole batch
	dispatch_mat_mat(ctx, &modalities, &l->altup_correct_coef, &correction_coefs, batch_len, num_altup, num_altup,
			 true);

	// Add 1.0 to all coefficients
	float *coef = (float *)correction_coefs.data;
	for (int i = 0; i < batch_len * num_altup; i++) {
		coef[i] += 1.0f;
	}

	// Calculate the "innovation" vector for the whole batch
	dispatch_subtract_to_buffer(final_active_output, &predictions[active_idx], &innovation, batch_len * embed_dim);

	// Apply the correction to all states
	const float *final_coefs_data = (const float *)correction_coefs.data;
	for (int i = 0; i < num_altup; i++) {
		for (int t = 0; t < batch_len; t++) {
			const float *innovation_vec = (const float *)innovation.data + t * embed_dim;
			const float *prediction_vec = (const float *)predictions[i].data + t * embed_dim;
			float *corrected_vec = (float *)corrected_states[i].data + t * embed_dim;
			const float coef_for_this_state = final_coefs_data[t * num_altup + i];
			for (int d = 0; d < embed_dim; d++) {
				corrected_vec[d] = prediction_vec[d] + innovation_vec[d] * coef_for_this_state;
			}
		}
	}

	free_memtype(&temp_token_vec);
	free_memtype(&modalities);
	free_memtype(&correction_coefs);
	free_memtype(&innovation);
}

// This function calculates the final refinement_residual
void dispatch_pli_gating(struct TIEContext *ctx,
			 MemType *refinement_residual,	 // The final output buffer
			 const MemType *active_state,	 // Input: The active state from the 'Corrected' array
			 const MemType *per_layer_input, // Input: The 256-dim PLI vector for this layer
			 LayerWeights *l, int batch_len)
{
	const int embed_dim = ctx->model->embed_dim;
	const int pli_dim = ctx->model->pli_dim;
	MemType scaled_active_state, gated_state, gelu_state, modulated_state, projected_state;

	alloc_memtype(&scaled_active_state, GGML_TYPE_F32, batch_len * embed_dim);
	alloc_memtype(&gated_state, GGML_TYPE_F32, batch_len * pli_dim);
	alloc_memtype(&gelu_state, GGML_TYPE_F32, batch_len * pli_dim);
	alloc_memtype(&modulated_state, GGML_TYPE_F32, batch_len * pli_dim);
	alloc_memtype(&projected_state, GGML_TYPE_F32, batch_len * embed_dim);

	// Scale the Active State
	// This is an element-wise multiplication by the altup_correct_scale vector
	dispatch_elementwise_mul_tensor(&scaled_active_state, active_state, &l->altup_correct_scale, batch_len,
					embed_dim);

	// Gate Projection (2048-dim -> 256-dim)
	dispatch_mat_mat(ctx, &scaled_active_state, &l->inp_gate, &gated_state, batch_len, embed_dim, pli_dim, true);

	// GeLU Activation
	// The GeLU for this gate is special; it doesn't have an 'up' projection, so we pass NULL or a dummy.
	// Let's create a temporary dummy buffer for the 'up' projection if needed.
	memcpy(gelu_state.data, gated_state.data, batch_len * pli_dim * ggml_type_size(gated_state.type));
	dispatch_gelu_inplace(&gelu_state, batch_len * pli_dim);

	// Modulate by Per-Layer Input
	// Element-wise multiplication: gelu_state * per_layer_input
	dispatch_elementwise_mul(&modulated_state, &gelu_state, per_layer_input, batch_len * pli_dim);

	// Project Back (256-dim -> 2048-dim)
	dispatch_mat_mat(ctx, &modulated_state, &l->proj, &projected_state, batch_len, pli_dim, embed_dim, true);

	// Final Norm
	// The output is the final refinement_residual
	for (int i = 0; i < batch_len; i++) {
		MemType projected_slice = mem_slice(&projected_state, i * embed_dim);
		MemType dest_slice = mem_slice(refinement_residual, i * embed_dim);

		// Apply the norm to each token's vector individually
		dispatch_rms_norm(&projected_slice, &l->post_norm, &dest_slice, embed_dim, ctx->model->norm_eps);
	}

	free_memtype(&scaled_active_state);
	free_memtype(&gated_state);
	free_memtype(&gelu_state);
	free_memtype(&modulated_state);
	free_memtype(&projected_state);
}

/* Extracts all per-layer inputs for a specific layer from the main PLI buffer.
 * The source buffer has a layout of [batch_len][num_layers][pli_dim].
 * The destination buffer will have a contiguous layout of [batch_len][pli_dim].
 */
void get_per_layer_input_for_layer(MemType *dest_buffer,    // Pre-allocated destination buffer
				   MemType *src_pli_buffer, // The main [batch_len, num_layers, pli_dim] buffer
				   int layer_idx, int batch_len, int num_layers, int pli_dim)
{
	// Loop through each token in the batch
	for (int i = 0; i < batch_len; i++) {
		// Calculate the offset to find the source data for this token at the specified layer
		size_t src_offset = (i * num_layers + layer_idx) * pli_dim;
		MemType src_slice = mem_slice(src_pli_buffer, src_offset);

		// Calculate the offset in the destination buffer
		size_t dest_offset = i * pli_dim;
		MemType dest_slice = mem_slice(dest_buffer, dest_offset);

		// Copy the pli_dim (e.g., 256) floats from the source to the destination
		memcpy(dest_slice.data, src_slice.data, pli_dim * ggml_type_size(dest_slice.type));
	}
}

void prepare_next_token_standard(struct TIEContext *ctx, int next_token)
{
	// Create a slice for this token's position in the hidden_state buffer
	MemType hidden_state_slice = mem_slice(&ctx->mem.hidden_state, 0);

	dispatch_embedding_row(&ctx->model->token_embd, next_token, &hidden_state_slice, ctx->model->embed_dim);

	if (ctx->model->interface.embedding_scale != NULL)
		ctx->model->interface.embedding_scale(ctx, &hidden_state_slice);
}

void prepare_next_token_gemma3n(struct TIEContext *ctx, int next_token)
{
	const int embed_dim = ctx->model->embed_dim;
	const int pli_dim = ctx->model->pli_dim;
	const int num_layers = ctx->model->num_layers;

	// Create a temporary buffer for the single token's scaled embedding
	MemType single_token_embedding;
	alloc_memtype(&single_token_embedding, GGML_TYPE_F32, embed_dim);

	// Prepare the altup_hidden_states for the next step
	dispatch_embedding_row(&ctx->model->token_embd, next_token, &single_token_embedding, embed_dim);
	ctx->model->interface.embedding_scale(ctx, &single_token_embedding);

	// Copy the base embedding to the first state
	memcpy(ctx->mem.altup_hidden_states[0].data, single_token_embedding.data,
	       embed_dim * ggml_type_size(single_token_embedding.type));

	// Create the other 3 parallel states for this single token
	create_altup_parallel_states(ctx, &ctx->mem.altup_hidden_states[0], 1, &ctx->model->altup_proj,
				     ctx->mem.altup_hidden_states);

	// Prepare the per_layer_inputs for the next step
	MemType pli_from_lookup, pli_from_proj;
	alloc_memtype(&pli_from_lookup, GGML_TYPE_F32, num_layers * pli_dim);
	alloc_memtype(&pli_from_proj, GGML_TYPE_F32, num_layers * pli_dim);

	// Get the raw PLI for the single token
	calculate_and_deinterleave_pli_raw(ctx, &next_token, 1, &pli_from_lookup);

	// Project the main embedding
	dispatch_mat_mat(ctx, &single_token_embedding, &ctx->model->per_layer_model_proj, &pli_from_proj, 1, embed_dim,
			 num_layers * pli_dim, true);

	// Scale and Norm the projected component
	float scale = 1.0f / sqrtf((float)embed_dim);
	float *proj_data = (float *)pli_from_proj.data;
	for (size_t i = 0; i < (size_t)num_layers * pli_dim; i++) {
		proj_data[i] *= scale;
	}

	for (int i = 0; i < num_layers; i++) {
		MemType slice = mem_slice(&pli_from_proj, i * pli_dim);
		dispatch_rms_norm(&slice, &ctx->model->per_layer_proj_norm, &slice, pli_dim, ctx->model->norm_eps);
	}

	// Add the two sources into the final persistent buffer.
	dispatch_apply_residual_to_buffer(&pli_from_lookup, &pli_from_proj, &ctx->mem.per_layer_inputs,
					  num_layers * pli_dim);

	// Apply the final scale factor
	scale = 1.0f / sqrtf(2.0f);
	float *final_data = (float *)ctx->mem.per_layer_inputs.data;
	for (size_t i = 0; i < (size_t)num_layers * pli_dim; i++) {
		final_data[i] *= scale;
	}

	free_memtype(&single_token_embedding);
	free_memtype(&pli_from_lookup);
	free_memtype(&pli_from_proj);
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

// Process prompt tokens (Qwen3, Gemma3)
void process_prompt_standard(struct TIEContext *ctx, int *prompt_tokens, size_t prompt_len)
{
	MemType hidden_state_slice;

	for (int i = 0; i < prompt_len; i++) {
		// Create a slice for this token's position in the hidden_state buffer
		hidden_state_slice = mem_slice(&ctx->mem.hidden_state, i * ctx->model->embed_dim);
		dispatch_embedding_row(&ctx->model->token_embd, prompt_tokens[i], &hidden_state_slice,
				       ctx->model->embed_dim);

		if (ctx->model->interface.embedding_scale != NULL)
			ctx->model->interface.embedding_scale(ctx, &hidden_state_slice);
	}

	process_embeddings(ctx, &ctx->mem.hidden_state, prompt_len);
}


// Process prompt tokens (Gemma-3n)
void process_prompt_gemma3n(struct TIEContext *ctx, int *prompt_tokens, size_t prompt_len)
{
	const int embed_dim = ctx->model->embed_dim;
	const int pli_dim = ctx->model->pli_dim;
	const int num_layers = ctx->model->num_layers;

	// Allocate all necessary buffers
	MemType scaled_token_embeddings;
	alloc_memtype(&scaled_token_embeddings, GGML_TYPE_F32, prompt_len * embed_dim);

	MemType pli_from_lookup_scaled;
	alloc_memtype(&pli_from_lookup_scaled, GGML_TYPE_F32, prompt_len * num_layers * pli_dim);

	MemType pli_from_projection;
	alloc_memtype(&pli_from_projection, GGML_TYPE_F32, prompt_len * num_layers * pli_dim);

	// Temporary buffer for de-interleaving the projected PLI
	MemType pli_from_projection_deinterleaved;
	alloc_memtype(&pli_from_projection_deinterleaved, GGML_TYPE_F32, prompt_len * num_layers * pli_dim);

	// Get the raw token embeddings
	for (int i = 0; i < prompt_len; i++) {
		MemType slice = mem_slice(&scaled_token_embeddings, i * embed_dim);
		dispatch_embedding_row(&ctx->model->token_embd, prompt_tokens[i], &slice, embed_dim);
	}

	// Scale the embeddings
	for (int i = 0; i < prompt_len; i++) {
		MemType slice = mem_slice(&scaled_token_embeddings, i * embed_dim);
		ctx->model->interface.embedding_scale(ctx, &slice);
	}

	// Create parallel AltUp states
	memcpy(ctx->mem.altup_hidden_states[0].data, scaled_token_embeddings.data,
	       prompt_len * embed_dim * ggml_type_size(scaled_token_embeddings.type));
	create_altup_parallel_states(ctx, &ctx->mem.altup_hidden_states[0], prompt_len, &ctx->model->altup_proj,
				     ctx->mem.altup_hidden_states);

	// Look up PLI from its dedicated table (creates [B, L, D] layout)
	calculate_and_deinterleave_pli_raw(ctx, prompt_tokens, prompt_len, &pli_from_lookup_scaled);

	// Project the main embeddings (creates flat [B, L*D] layout)
	dispatch_mat_mat(ctx, &scaled_token_embeddings, &ctx->model->per_layer_model_proj, &pli_from_projection,
			 prompt_len, embed_dim, num_layers * pli_dim, true);

	// Scale and Norm the projected embeddings (still in flat layout)
	float scale = 1.0f / sqrtf((float)embed_dim);
	float *proj_data = (float *)pli_from_projection.data;
	for (size_t i = 0; i < prompt_len * num_layers * pli_dim; i++) {
		proj_data[i] *= scale;
	}

	int num_vectors_to_norm = prompt_len * num_layers;
	for (int i = 0; i < num_vectors_to_norm; i++) {
		MemType vector_slice = mem_slice(&pli_from_projection, i * pli_dim);
		dispatch_rms_norm(&vector_slice, &ctx->model->per_layer_proj_norm, &vector_slice, pli_dim,
				  ctx->model->norm_eps);
	}

	// De-interleave the projected PLI to match the lookup PLI's layout
	for (int i = 0; i < prompt_len; i++) {
		for (int l = 0; l < num_layers; l++) {
			size_t src_offset = (i * num_layers + l) * pli_dim;
			MemType src_slice = mem_slice(&pli_from_projection, src_offset);

			size_t dest_offset = (i * num_layers + l) * pli_dim;
			MemType dest_slice = mem_slice(&pli_from_projection_deinterleaved, dest_offset);

			memcpy(dest_slice.data, src_slice.data, pli_dim * ggml_type_size(dest_slice.type));
		}
	}

	// Add the two sources together (now with matching layouts)
	dispatch_apply_residual_to_buffer(&pli_from_lookup_scaled, &pli_from_projection_deinterleaved,
					  &ctx->mem.per_layer_inputs, prompt_len * num_layers * pli_dim);

	// Apply the final scale
	scale = 1.0f / sqrtf(2.0f);
	float *final_data = (float *)ctx->mem.per_layer_inputs.data;
	for (size_t i = 0; i < prompt_len * num_layers * pli_dim; i++) {
		final_data[i] *= scale;
	}

	for (int l = 0; l < num_layers; l++) {
		ctx->model->interface.transformer_layer(ctx, l, prompt_len);
	}

	ctx->kv_pos += prompt_len;

	free_memtype(&scaled_token_embeddings);
	free_memtype(&pli_from_lookup_scaled);
	free_memtype(&pli_from_projection);
	free_memtype(&pli_from_projection_deinterleaved); // Don't forget to free the new buffer
}

// Function to calculate per-token magnitude
void calculate_per_token_magnitude(float *magnitudes, const MemType *state, size_t num_tokens, int dim)
{
	for (size_t t = 0; t < num_tokens; t++) {
		const float *token_vec = (const float *)state->data + t * dim;
		float ss = 0.0;
		for (int i = 0; i < dim; i++) {
			ss += token_vec[i] * token_vec[i];
		}

		// Divide by the dimension 'dim' to calculate the mean of squares, then take the square root.
		magnitudes[t] = sqrt(ss / dim);
	}
}

// Function to create the parallel states
void create_altup_parallel_states(struct TIEContext *ctx, MemType *base_state, size_t prompt_len,
				  Tensor *altup_proj_tensor, MemType *destination)
{
	const int embed_dim = ctx->model->embed_dim;

	// Get a pointer to the single, large altup_proj tensor
	size_t matrix_size_bytes = (size_t)embed_dim * embed_dim * ggml_type_size(altup_proj_tensor->mem.type);

	// Loop and create the other parallel states (state[1] through state[3])
	for (int i = 1; i < ctx->model->altup_num_inputs; i++) {
		MemType *dest_state = &destination[i];

		// Create a temporary "view" of the i-th matrix within the flat tensor
		Tensor altup_proj_slice = *altup_proj_tensor;
		size_t offset = (size_t)(i - 1) * matrix_size_bytes;
		altup_proj_slice.mem.data = (uint8_t *)altup_proj_tensor->mem.data + offset;

		// Project: dest_state = base_state @ altup_projection_slice
		dispatch_mat_mat(ctx, base_state, &altup_proj_slice, dest_state, prompt_len, embed_dim, embed_dim,
				 true);

		for (size_t t = 0; t < prompt_len; t++) {
			const float *base_vec = (const float *)base_state->data + t * embed_dim;
			float *dest_vec = (float *)dest_state->data + t * embed_dim;

			// Calculate sum of squares for both vectors
			float ss_base = 0.0;
			float ss_dest = 0.0;
			for (int d = 0; d < embed_dim; d++) {
				ss_base += base_vec[d] * base_vec[d];
				ss_dest += dest_vec[d] * dest_vec[d];
			}

			// Calculate the final scaling factor directly
			const float scale_factor = sqrtf(ss_base / (ss_dest + 1e-12f));

			// Apply the scaling
			for (int d = 0; d < embed_dim; d++) {
				dest_vec[d] *= scale_factor;
			}
		}
	}
}

void calculate_and_deinterleave_pli_raw(struct TIEContext *ctx, int *prompt_tokens, size_t prompt_len,
					MemType *dest_buffer)
{
	const int pli_dim = ctx->model->pli_dim;
	const int num_layers = ctx->model->num_layers;

	// Allocate a temporary buffer to hold the giant (e.g., 7680-dim) vector for ONE token.
	MemType temp_pli_vector;
	alloc_memtype(&temp_pli_vector, GGML_TYPE_F32, num_layers * pli_dim);

	// Loop through each token in the prompt.
	for (int i = 0; i < prompt_len; i++) {
		int token_id = prompt_tokens[i];

		// Look up the full (e.g., 7680-dim) vector for the current token from the PLI embedding table.
		dispatch_embedding_row(&ctx->model->per_layer_token_embd, token_id, &temp_pli_vector,
				       num_layers * pli_dim);

		// De-interleave the temporary vector into the final destination buffer.
		// This loop takes the stacked vector (e.g., [layer0_data, layer1_data, ...])
		// and distributes it into the final [token, layer, dim] layout.
		for (int l = 0; l < num_layers; l++) {
			// Source: The l-th slice of the temporary vector.
			MemType src_slice = mem_slice(&temp_pli_vector, l * pli_dim);

			// Destination: The correct spot in the final buffer for token `i` at layer `l`.
			size_t dest_offset = (i * num_layers * pli_dim) + (l * pli_dim);
			MemType dest_slice = mem_slice(dest_buffer, dest_offset);

			// Copy the 256 floats for this layer.
			memcpy(dest_slice.data, src_slice.data, pli_dim * ggml_type_size(dest_slice.type));

			float scale = sqrtf(256.0f);
			float *data = (float *)dest_slice.data;
			for (size_t d = 0; d < pli_dim; d++) {
				data[d] *= scale;
			}
		}
	}

	free_memtype(&temp_pli_vector);
}

void post_process_altup_states(struct TIEContext *ctx,
			       MemType *final_hidden_state, // Output: The single, final vector
			       MemType *final_altup_states, // Input: The array of 4 states
			       size_t n_tokens)
{
	size_t last_token_idx = n_tokens - 1;
	const int embed_dim = ctx->model->embed_dim;
	const int num_altup = ctx->model->altup_num_inputs;

	// The active index is always 0 for Gemma-3N
	const int active_idx = 0;

	// Temporary buffer to hold the final versions of each state before averaging
	MemType temp_states[num_altup];
	for (int i = 0; i < num_altup; i++) {
		alloc_memtype(&temp_states[i], GGML_TYPE_F32, embed_dim);
	}

	//  The active state (index 0) is our "base". Copy its last token's data directly.
	MemType base_state_slice = mem_slice(&final_altup_states[active_idx], last_token_idx * embed_dim);
	memcpy(temp_states[active_idx].data, base_state_slice.data, embed_dim * sizeof(float));

	// Un-Project and Rescale the INACTIVE states (1, 2, 3)
	float target_magnitude;
	calculate_per_token_magnitude(&target_magnitude, &base_state_slice, 1, embed_dim);

	size_t matrix_size_bytes =
		(size_t)embed_dim * embed_dim * ggml_type_size(ctx->model->altup_unembd_proj.mem.type);

	for (int i = 0; i < num_altup - 1; i++) {
		// The inactive states are 1, 2, 3. The projection matrices are 0, 1, 2.
		int inactive_state_idx = i + 1;
		int proj_matrix_idx = i;

		MemType *dest_state = &temp_states[inactive_state_idx];
		MemType src_slice = mem_slice(&final_altup_states[inactive_state_idx], last_token_idx * embed_dim);

		Tensor unembd_proj_slice = ctx->model->altup_unembd_proj;
		unembd_proj_slice.mem.data =
			(uint8_t *)unembd_proj_slice.mem.data + (size_t)proj_matrix_idx * matrix_size_bytes;

		dispatch_mat_vec(ctx, &src_slice, &unembd_proj_slice, dest_state, embed_dim, embed_dim, false);

		float new_magnitude;
		calculate_per_token_magnitude(&new_magnitude, dest_state, 1, embed_dim);
		float scale_factor = target_magnitude / (new_magnitude + 1e-12f);

		float *dest_data = (float *)dest_state->data;
		for (int d = 0; d < embed_dim; d++) {
			dest_data[d] *= scale_factor;
		}
	}

	// Average All States into the final_hidden_state buffer
	float inv_num_altup = 1.0f / (float)num_altup;
	float *final_data = (float *)final_hidden_state->data;
	memset(final_data, 0, embed_dim * sizeof(float));

	for (int i = 0; i < num_altup; i++) {
		const float *src_data = (const float *)temp_states[i].data;
		for (int d = 0; d < embed_dim; d++) {
			final_data[d] += src_data[d]; // Sum first...
		}
	}

	// Scale once at the end.
	for (int d = 0; d < embed_dim; d++) {
		final_data[d] *= inv_num_altup;
	}

	for (int i = 0; i < num_altup; i++) {
		free_memtype(&temp_states[i]);
	}
}

int transformer_layer_gemma3n(struct TIEContext *ctx, int layer_idx, int batch_len)
{
	LayerWeights *l = &ctx->model->layers[layer_idx];
	int kv_dim = ctx->model->num_kv_heads * ctx->model->head_dim;
	int q_dim = ctx->model->num_heads * ctx->model->head_dim;
	AttentionType attn_type = ATTN_TYPE_LOCAL;
	RopeCacheType *active_rope_cache = ctx->rope_cache_local;
	const int pli_dim = ctx->model->pli_dim;
	const int embed_dim = ctx->model->embed_dim;
	const int DATA_ACTIVE_IDX = 0; // The first state is used for the main computation path


	// The absolute starting position for this batch
	int start_pos = ctx->kv_pos;

	// Determine the attention and rope cache type for the current layer
	if ((layer_idx + 1) % 5 == 0) {
		attn_type = ATTN_TYPE_GLOBAL;
		active_rope_cache = ctx->rope_cache_global; // Use the global cache
	}

	// Gemma-3n KV sharing logic
	int first_kv_shared_layer_idx = ctx->model->num_layers - ctx->model->shared_kv_layers;
	bool is_kv_shared_layer = (layer_idx >= first_kv_shared_layer_idx);

	int kv_source_layer_idx = layer_idx;
	bool store_full_length_kv = false;

	if (is_kv_shared_layer) {
		// For shared layers, find the last non-shared layer of the same type
		int my_type = attn_type;
		kv_source_layer_idx = -1;
		for (int i = first_kv_shared_layer_idx - 1; i >= 0; --i) {
			int prev_type = ((i + 1) % 5 == 0) ? ATTN_TYPE_GLOBAL : ATTN_TYPE_LOCAL;
			if (prev_type == my_type) {
				kv_source_layer_idx = i;
				break;
			}
		}
		if (kv_source_layer_idx < 0) {
			fprintf(stderr, "[ERROR] No kv_source_layer_idx found for layer %d type %d\n", layer_idx,
				my_type);
			exit(1);
		}
	} else {
		// For non-shared layers, decide if this is the last non-shared layer of its type
		int my_type = attn_type;
		store_full_length_kv = true;
		for (int i = layer_idx + 1; i < first_kv_shared_layer_idx; ++i) {
			int next_type = ((i + 1) % 5 == 0) ? ATTN_TYPE_GLOBAL : ATTN_TYPE_LOCAL;
			if (next_type == my_type) {
				store_full_length_kv = false;
				break;
			}
		}
	}

	dispatch_altup_predict(ctx, ctx->mem.altup_predicted_states, ctx->mem.altup_hidden_states, l, batch_len,
			       DATA_ACTIVE_IDX);

	// Save First Residual: residual_1 = active_prediction.
	memcpy(ctx->mem.residual_stratch.data, ctx->mem.altup_predicted_states[DATA_ACTIVE_IDX].data,
	       batch_len * ctx->model->embed_dim * sizeof(float));

	// ============ Attention Block ============
	// RMSNorm on input
	for (int i = 0; i < batch_len; i++) {
		size_t offset = (size_t)i * ctx->model->embed_dim;

		// Create slices for the specific token being processed
		MemType input_slice = mem_slice(&ctx->mem.altup_predicted_states[DATA_ACTIVE_IDX], offset);
		MemType normed_slice = mem_slice(&ctx->mem.normed_qkv_input, offset);

		dispatch_rms_norm(&input_slice, &l->attn_norm, &normed_slice, ctx->model->embed_dim,
				  ctx->model->norm_eps);
	}

	MemType laurel_output;
	alloc_memtype(&laurel_output, GGML_TYPE_F32, batch_len * ctx->model->embed_dim);

	dispatch_laurel(ctx, &laurel_output, &ctx->mem.normed_qkv_input, l, batch_len);

	// Compute Q/K/V Matrices
	dispatch_mat_mat(ctx, &ctx->mem.normed_qkv_input, &l->attn_q, &ctx->mem.Q, batch_len, ctx->model->embed_dim,
			 q_dim, true);

	if (!is_kv_shared_layer) {
		// For non shared layers, compute K and V normally
		dispatch_mat_mat(ctx, &ctx->mem.normed_qkv_input, &l->attn_k, &ctx->mem.K, batch_len,
				 ctx->model->embed_dim, kv_dim, true);

		dispatch_mat_mat(ctx, &ctx->mem.normed_qkv_input, &l->attn_v, &ctx->mem.V, batch_len,
				 ctx->model->embed_dim, kv_dim, true);
	}

	// Apply RoPE
	for (int i = 0; i < batch_len; i++) {
		// The absolute position for the current token in the batch
		int absolute_pos = start_pos + i;

		for (int h = 0; h < ctx->model->num_heads; h++) {
			MemType Q_slice = mem_slice(&ctx->mem.Q, (size_t)i * q_dim + h * ctx->model->head_dim);

			if (l->attn_q_norm.mem.data) {
				dispatch_rms_norm(&Q_slice, &l->attn_q_norm, &Q_slice, ctx->model->head_dim,
						  ctx->model->norm_eps);
			}

			// Use the absolute position for RoPE
			dispatch_apply_rope_cache(active_rope_cache, &Q_slice, absolute_pos, ctx->model->head_dim);
		}

		if (!is_kv_shared_layer) {

			for (int h = 0; h < ctx->model->num_kv_heads; h++) {
				MemType K_slice = mem_slice(&ctx->mem.K, (size_t)i * kv_dim + h * ctx->model->head_dim);

				if (l->attn_k_norm.mem.data) {
					dispatch_rms_norm(&K_slice, &l->attn_k_norm, &K_slice, ctx->model->head_dim,
							  ctx->model->norm_eps);
				}

				// Use the absolute position for RoPE
				dispatch_apply_rope_cache(active_rope_cache, &K_slice, absolute_pos,
							  ctx->model->head_dim);
			}

			for (int h = 0; h < ctx->model->num_kv_heads; h++) {
				MemType V_slice = mem_slice(&ctx->mem.V, (size_t)i * kv_dim + h * ctx->model->head_dim);

				dispatch_rms_norm_weightless(&V_slice, ctx->model->head_dim, ctx->model->norm_eps);
			}
		}
	}

	// Store K/V to cache
	if (!is_kv_shared_layer) {
		dispatch_store_KV_cache(ctx, layer_idx, start_pos, batch_len);
	}

	// Multi-Head Attention Calculation
	attention(ctx, batch_len, layer_idx, kv_source_layer_idx, start_pos, attn_type, attention_worker_gemma3n);

	// Output projection
	dispatch_mat_mat(ctx, &ctx->mem.attn_output, &l->attn_out, &ctx->mem.attn_proj_output, batch_len, q_dim,
			 ctx->model->embed_dim, true);

	// Apply POST-ATTENTION norm
	for (int i = 0; i < batch_len; i++) {
		size_t offset = (size_t)i * ctx->model->embed_dim;

		MemType attn_proj_slice = mem_slice(&ctx->mem.attn_proj_output, offset);

		dispatch_rms_norm(&attn_proj_slice, &l->post_attn_norm, &attn_proj_slice, ctx->model->embed_dim,
				  ctx->model->norm_eps);
	}

	// Add the post-normed attention output to the original residual
	dispatch_apply_residual(&ctx->mem.residual_stratch, &ctx->mem.attn_proj_output,
				batch_len * ctx->model->embed_dim);

	// Mix LAUREL and Attention: Compute attn_laurel_mix = (attn_with_residual + laurel_out) * (1.0f / sqrtf(2.0f)).
	MemType attn_laurel_mix;
	alloc_memtype(&attn_laurel_mix, GGML_TYPE_F32, batch_len * ctx->model->embed_dim);

	float scale = 1.0f / sqrtf(2.0f);
	for (int i = 0; i < batch_len; i++) {
		size_t offset = (size_t)i * ctx->model->embed_dim;

		MemType attn_laurel_mix_slice = mem_slice(&attn_laurel_mix, offset);
		MemType residual_stratch_slice = mem_slice(&ctx->mem.residual_stratch, offset);
		MemType laurel_output_slice = mem_slice(&laurel_output, offset);

		float *attn_laurel_mix_out = (float *)attn_laurel_mix_slice.data;
		float *residual_stratch_data = (float *)residual_stratch_slice.data;
		float *laurel_output_data = (float *)laurel_output_slice.data;

		for (int d = 0; d < ctx->model->embed_dim; d++)
			attn_laurel_mix_out[d] = (residual_stratch_data[d] + laurel_output_data[d]) * scale;
	}

	// Save Second Residual: residual_2 = attn_laurel_mix.
	memcpy(ctx->mem.residual_stratch.data, attn_laurel_mix.data, batch_len * ctx->model->embed_dim * sizeof(float));

	// ============ FFN Block ============
	// RMSNorm
	for (int i = 0; i < batch_len; i++) {
		size_t offset = (size_t)i * ctx->model->embed_dim;

		// Create slices for the specific token being processed
		MemType attn_laurel_mix_slice = mem_slice(&attn_laurel_mix, offset);
		MemType normed_ffn_input_slice = mem_slice(&ctx->mem.normed_ffn_input, offset);

		dispatch_rms_norm(&attn_laurel_mix_slice, &l->ffn_norm, &normed_ffn_input_slice, ctx->model->embed_dim,
				  ctx->model->norm_eps);
	}

	// Gate + Up projections
	dispatch_mat_mat(ctx, &ctx->mem.normed_ffn_input, &l->ffn_gate, &ctx->mem.gate_proj_output, batch_len,
			 ctx->model->embed_dim, ctx->model->ffn_dim, true);
	dispatch_mat_mat(ctx, &ctx->mem.normed_ffn_input, &l->ffn_up, &ctx->mem.up_proj_output, batch_len,
			 ctx->model->embed_dim, ctx->model->ffn_dim, true);

	if (layer_idx < 10) {
		for (int i = 0; i < batch_len; i++) {
			// Get a slice for the current token's gate vector
			MemType gate_slice = mem_slice(&ctx->mem.gate_proj_output, i * ctx->model->ffn_dim);

			// Apply sparsity to this token's slice only
			dispatch_gaussian_topk(&gate_slice, ctx->model->ffn_dim);
		}
	}

	/* Call the interface activation function */
	dispatch_geglu_activation(&ctx->mem.gate_proj_output, &ctx->mem.up_proj_output,
				  batch_len * ctx->model->ffn_dim);

	// Down projection
	dispatch_mat_mat(ctx, &ctx->mem.gate_proj_output, &l->ffn_down, &ctx->mem.ffn_down_output, batch_len,
			 ctx->model->ffn_dim, ctx->model->embed_dim, true);

	// Apply POST-FFN norm.
	for (int i = 0; i < batch_len; i++) {
		size_t offset = (size_t)i * ctx->model->embed_dim;
		MemType ffn_down_slice = mem_slice(&ctx->mem.ffn_down_output, offset);

		dispatch_rms_norm(&ffn_down_slice, &l->post_ffw_norm, &ffn_down_slice, ctx->model->embed_dim,
				  ctx->model->norm_eps);
	}

	// Second Residual Add.
	// Add the post-normed FFN output to the residual
	dispatch_apply_residual(&ctx->mem.residual_stratch, &ctx->mem.ffn_down_output,
				batch_len * ctx->model->embed_dim);

	// Create a temporary buffer to build the final output for THIS layer
	MemType final_layer_output[ctx->model->altup_num_inputs];
	for (int i = 0; i < ctx->model->altup_num_inputs; i++) {
		alloc_memtype(&final_layer_output[i], GGML_TYPE_F32, batch_len * embed_dim);
	}

	// ctx->mem.residual_stratch is the final active output.
	dispatch_altup_correct(ctx, final_layer_output, ctx->mem.altup_predicted_states, &ctx->mem.residual_stratch, l,
			       batch_len, DATA_ACTIVE_IDX);

	// The Final Gating and Refinement
	MemType refinement_residual;
	alloc_memtype(&refinement_residual, GGML_TYPE_F32, batch_len * embed_dim);

	// Create a buffer to hold the contiguous per-layer inputs for this layer
	MemType per_layer_input_for_layer;
	alloc_memtype(&per_layer_input_for_layer, GGML_TYPE_F32, batch_len * pli_dim);

	// Gather the scattered PLI data
	get_per_layer_input_for_layer(&per_layer_input_for_layer, &ctx->mem.per_layer_inputs, layer_idx, batch_len,
				      ctx->model->num_layers, pli_dim);

	// Calculate the refinement_residual using the active slice from the corrected states
	dispatch_pli_gating(
		ctx, &refinement_residual,
		&final_layer_output[DATA_ACTIVE_IDX], // The input is ALWAYS the active data path's corrected state
		&per_layer_input_for_layer, l, batch_len);

	// Apply the refinement to the INACTIVE states
	for (int j = 0; j < ctx->model->altup_num_inputs; j++) {
		if (j == DATA_ACTIVE_IDX)
			continue;

		dispatch_apply_residual(&final_layer_output[j], &refinement_residual, batch_len * embed_dim);
	}

	// Commit the temporary buffer back to the main state
	for (int i = 0; i < ctx->model->altup_num_inputs; i++) {
		memcpy(ctx->mem.altup_hidden_states[i].data, final_layer_output[i].data,
		       batch_len * embed_dim * ggml_type_size(final_layer_output[i].type));

		free_memtype(&final_layer_output[i]);
	}

	free_memtype(&refinement_residual);
	free_memtype(&per_layer_input_for_layer);

	return 0;
};
