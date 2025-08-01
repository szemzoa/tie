#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "engine.h"
#include "model.h"
#include "maths.h"
#include "threadpool.h"

typedef struct {
	struct ctx_t *ctx;
	int layer_idx;
	int pos;
	int head_start;
	int head_end;
	float *attn_scores_buffer;
} attention_task_t;

/**
 * @brief The data required for a single thread to process a chunk of attention.
 * Now includes the starting position of the batch.
 */
typedef struct {
	struct ctx_t *ctx;
	int layer_idx;
	int start_token_idx; // The first token in the batch this thread will process
	int end_token_idx;   // The last token (exclusive)
	int batch_start_pos; // The absolute position of the beginning of the prompt batch
	int thread_id;	     // The ID used to get a unique scratch buffer
} attention_batch_task_t;

typedef struct {
	int index;
	float score;
} ExpertChoice;

float silu_table[SILU_TABLE_SIZE];


void reset_kv_cache(struct ctx_t *ctx)
{
	for (int i = 0; i < ctx->model->num_layers; i++) {
		memset(ctx->kv_cache[i].k, 0,
		       ctx->model->seq_length * ctx->model->num_kv_heads * ctx->model->head_dim * sizeof(uint16_t));
		memset(ctx->kv_cache[i].v, 0,
		       ctx->model->seq_length * ctx->model->num_kv_heads * ctx->model->head_dim * sizeof(uint16_t));
	}
}

void silu_table_init()
{
	for (int i = 0; i < SILU_TABLE_SIZE; ++i) {
		float x = SILU_X_MIN + (SILU_X_MAX - SILU_X_MIN) * i / (SILU_TABLE_SIZE - 1);
		silu_table[i] = x / (1.0f + expf(-x));
	}
}

void rope_cache_init(struct ctx_t *ctx, int max_pos, int head_dim, float base)
{
	rope_cache_t *rope_cache = ctx->rope_cache;
	rope_cache->max_pos = max_pos;
	rope_cache->head_dim = head_dim;

	rope_cache->sin = aligned_alloc(32, sizeof(float) * max_pos * (head_dim / 2));
	rope_cache->cos = aligned_alloc(32, sizeof(float) * max_pos * (head_dim / 2));

	for (int pos = 0; pos < max_pos; ++pos) {
		float scaled_pos = (float)pos * ctx->model->yarn_scale_factor;

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

inline float silu_lookup(float x)
{
	if (!isfinite(x)) {
		// If x is not a valid number, we can't use the lookup table.
		// Fallback to a direct, safe calculation or return 0.
		// A direct calculation is safer if the result is needed.
		return x / (1.0f + expf(-x));
	}

	if (x <= SILU_X_MIN)
		return x / (1.0f + expf(-x));
	if (x >= SILU_X_MAX)
		return x / (1.0f + expf(-x));

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

static void apply_rope_cache(struct ctx_t *ctx, float *x, int pos, int head_dim)
{
#ifdef CONFIG_ENABLE_AVX2
	if (__builtin_cpu_supports("avx2")) {
		return apply_rope_cache_avx2(ctx, x, pos, head_dim);
	}
#endif
	rope_cache_t *rope_cache = ctx->rope_cache;
	int h_dim_half = head_dim / 2;

	if (pos >= rope_cache->max_pos) {
		fprintf(stderr, "Position %d exceeds rope cache max_pos %d\n", pos, rope_cache->max_pos);
		return;
	}

	const float *sin_vals = rope_cache->sin + pos * h_dim_half;
	const float *cos_vals = rope_cache->cos + pos * h_dim_half;

	int i = 0;
	for (; i <= h_dim_half - 2; i += 2) {
		float x_r0 = x[i], x_i0 = x[i + h_dim_half];
		float x_r1 = x[i + 1], x_i1 = x[i + 1 + h_dim_half];

		float sin0 = sin_vals[i], cos0 = cos_vals[i];
		float sin1 = sin_vals[i + 1], cos1 = cos_vals[i + 1];

		x[i] = fmaf(x_r0, cos0, -x_i0 * sin0);
		x[i + h_dim_half] = fmaf(x_r0, sin0, x_i0 * cos0);

		x[i + 1] = fmaf(x_r1, cos1, -x_i1 * sin1);
		x[i + 1 + h_dim_half] = fmaf(x_r1, sin1, x_i1 * cos1);
	}

	// Handle tail if odd head_dim
	for (; i < h_dim_half; ++i) {
		float x_real = x[i];
		float x_imag = x[i + h_dim_half];
		float sin = sin_vals[i], cos = cos_vals[i];

		x[i] = fmaf(x_real, cos, -x_imag * sin);
		x[i + h_dim_half] = fmaf(x_real, sin, x_imag * cos);
	}
}

static inline uint16_t *get_kv_head(uint16_t *cache_base, int t, int kv_head_idx, int head_dim, int num_kv_heads)
{
	// Total vector size for each timestep = num_kv_heads * head_dim
	int kv_dim = num_kv_heads * head_dim;
	return cache_base + (long long)t * kv_dim + kv_head_idx * head_dim;
}

static inline float add_residual(float acc, float res)
{
	return fmaf(res, 1.0f, acc);
}

static void attention_task(void *arg)
{
	attention_task_t *task = (attention_task_t *)arg;
	struct ctx_t *ctx = task->ctx;
	int layer_idx = task->layer_idx;
	int current_pos = task->pos;
	LayerKVCache *cache = &ctx->kv_cache[layer_idx];
	float *attn_scores_buffer = task->attn_scores_buffer;

	for (int h = task->head_start; h < task->head_end; h++) {
		// Get pointers to the current Q head and the corresponding output buffer position
		float *q_head = ctx->mem.Q + h * ctx->model->head_dim;
		float *out_head = ctx->mem.attn_output + h * ctx->model->head_dim;

		// Determine the KV head index for this Q head (for GQA)
		int kv_head_idx = h / (ctx->model->num_heads / ctx->model->num_kv_heads);

		// 1. Calculate Raw Attention Scores
		// Calculate the dot product of the current FP32 Q-head with all previous
		// BF16 K-heads in the cache
		for (int t = 0; t <= current_pos; t++) {
			// Read from K cache, which stores BF16
			uint16_t *k_head =
				get_kv_head(cache->k, t, kv_head_idx, ctx->model->head_dim, ctx->model->num_kv_heads);
			// Use the dot product for FP32 and BF16 vectors
			float score = dot_product_f32_bf16(q_head, k_head, ctx->model->head_dim);
			attn_scores_buffer[t] = score * ctx->model->attn_scale;
		}

		// 2. Calculate Softmax
		float max_score = -INFINITY;
		for (int t = 0; t <= current_pos; t++) {
			if (attn_scores_buffer[t] > max_score) {
				max_score = attn_scores_buffer[t];
			}
		}

		float sum_exp = 0.0f;
		for (int t = 0; t <= current_pos; t++) {
			// Exponentiate with max score subtraction for numerical stability
			float val = expf(attn_scores_buffer[t] - max_score);
			attn_scores_buffer[t] = val;	    // Store the exponentiated value
			sum_exp = fmaf(val, 1.0f, sum_exp); // Accumulate sum using FMA
		}

		float inv_sum_exp = 1.0f / sum_exp;

		// 3. Calculate Weighted Sum of V-vectors
		for (int t = 0; t <= current_pos; t++) {
			// Read from V cache, which stores BF16
			uint16_t *v_head =
				get_kv_head(cache->v, t, kv_head_idx, ctx->model->head_dim, ctx->model->num_kv_heads);
			// Normalize the score to get the final attention weight
			float attention_weight = attn_scores_buffer[t] * inv_sum_exp;

			// Accumulate weighted V-vector values into the output head buffer
			//			for (int i = 0; i < ctx->model->head_dim; i++) {
			//				out_head[i] = fmaf(attention_weight, bf16_to_fp32(v_head[i]),
			// out_head[i]);
			//			}
			accumulate_weighted_fp32_bf16(out_head, attention_weight, v_head, ctx->model->head_dim);
		}
	}

	free(task);
}

/**
 * @brief The function executed by each thread for batched attention.
 *
 * This version correctly calculates the absolute position of each token to handle
 * multi-turn conversation context.
 */
static void attention_batch_task(void *arg)
{
	attention_batch_task_t *task = (attention_batch_task_t *)arg;
	struct ctx_t *ctx = task->ctx;
	int layer_idx = task->layer_idx;
	int batch_start_pos = task->batch_start_pos;

	// Get model dimensions
	int num_heads = ctx->model->num_heads;
	int head_dim = ctx->model->head_dim;
	int num_kv_heads = ctx->model->num_kv_heads;
	int q_dim = num_heads * head_dim;
	int kv_dim = num_kv_heads * head_dim;

	// Each thread gets its own temporary score buffer
	float *attn_scores_buffer = ctx->mem.attn_scores_buffer[task->thread_id];

	// Each thread processes a *range* of query tokens from the prompt batch
	for (int i = task->start_token_idx; i < task->end_token_idx; i++) {
		// Calculate the token's absolute position in the entire conversation
		int absolute_pos = batch_start_pos + i;

		for (int h = 0; h < num_heads; h++) {
			float *q_head = ctx->mem.Q + (long long)i * q_dim + h * head_dim;
			float *out_head = ctx->mem.attn_output + (long long)i * q_dim + h * head_dim;
			int kv_head_idx = h / (num_heads / num_kv_heads);

			// 1. Calculate Raw Attention Scores
			// The Q-head dots with ALL previous K-heads, up to its absolute_pos
			for (int t = 0; t <= absolute_pos; t++) {
				uint16_t *k_head =
					ctx->kv_cache[layer_idx].k + (long long)t * kv_dim + kv_head_idx * head_dim;
				float score = dot_product_f32_bf16(q_head, k_head, head_dim);
				attn_scores_buffer[t] = score * ctx->model->attn_scale;
			}

			// 2. Calculate Softmax
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

			// 3. Calculate Weighted Sum of V-vectors
			memset(out_head, 0, head_dim * sizeof(float));
			for (int t = 0; t <= absolute_pos; t++) {
				uint16_t *v_head =
					ctx->kv_cache[layer_idx].v + (long long)t * kv_dim + kv_head_idx * head_dim;
				float attention_weight = attn_scores_buffer[t];

				//				for (int j = 0; j < head_dim; j++) {
				//					out_head[j] += attention_weight *
				// bf16_to_fp32(v_head[j]);
				//				}
				accumulate_weighted_fp32_bf16(out_head, attention_weight, v_head, head_dim);
			}
		}
	}
	free(task);
}

void attention_parallel(struct ctx_t *ctx, int layer_idx, int current_pos, bool use_threads)
{
	int q_dim = ctx->model->num_heads * ctx->model->head_dim;

	memset(ctx->mem.attn_output, 0, q_dim * sizeof(float));

	if (use_threads == 0) {
		LayerKVCache *cache = &ctx->kv_cache[layer_idx];

		// Loop over all query heads
		for (int h = 0; h < ctx->model->num_heads; h++) {
			float *q_head = ctx->mem.Q + h * ctx->model->head_dim;
			float *out_head = ctx->mem.attn_output + h * ctx->model->head_dim;
			int kv_head_idx = h / (ctx->model->num_heads / ctx->model->num_kv_heads);

			// 1. Calculate Raw Attention Scores
			for (int t = 0; t <= current_pos; t++) {
				uint16_t *k_head = get_kv_head(cache->k, t, kv_head_idx, ctx->model->head_dim,
							       ctx->model->num_kv_heads);
				float score = dot_product_f32_bf16(q_head, k_head, ctx->model->head_dim);
				ctx->mem.attn_scores_buffer[0][t] =
					score * ctx->model->attn_scale; // Use first buffer for sequential mode
			}

			// 2. Calculate Softmax
			float max_score = -INFINITY;
			for (int t = 0; t <= current_pos; t++) {
				if (ctx->mem.attn_scores_buffer[0][t] > max_score) {
					max_score = ctx->mem.attn_scores_buffer[0][t];
				}
			}

			float sum_exp = 0.0f;
			for (int t = 0; t <= current_pos; t++) {
				float val = expf(ctx->mem.attn_scores_buffer[0][t] - max_score);
				ctx->mem.attn_scores_buffer[0][t] = val;
				sum_exp = fmaf(val, 1.0f, sum_exp);
			}

			float inv_sum_exp = 1.0f / sum_exp;

			// 3. Calculate Weighted Sum of V-vectors
			memset(out_head, 0, ctx->model->head_dim * sizeof(float));
			for (int t = 0; t <= current_pos; t++) {
				uint16_t *v_head = get_kv_head(cache->v, t, kv_head_idx, ctx->model->head_dim,
							       ctx->model->num_kv_heads);
				float attention_weight = ctx->mem.attn_scores_buffer[0][t] * inv_sum_exp;

				//				for (int i = 0; i < ctx->model->head_dim; i++) {
				//					out_head[i] = fmaf(attention_weight,
				// bf16_to_fp32(v_head[i]), out_head[i]);
				//				}
				accumulate_weighted_fp32_bf16(out_head, attention_weight, v_head, ctx->model->head_dim);
			}
		}

		return;
	}

	int num_threads = thread_pool->num_threads;
	int heads_per_thread = (ctx->model->num_heads + num_threads - 1) / num_threads;

	for (int t = 0; t < num_threads; t++) {
		int head_start = t * heads_per_thread;
		int head_end = head_start + heads_per_thread;
		if (head_end > ctx->model->num_heads) {
			head_end = ctx->model->num_heads;
		}
		if (head_start >= head_end) {
			break;
		}

		attention_task_t *task = malloc(sizeof(attention_task_t));
		if (!task) {
			fprintf(stderr, "ERROR: Failed to allocate memory for attention_task\n");
			continue;
		}

		*task = (attention_task_t){.ctx = ctx,
					   .layer_idx = layer_idx,
					   .pos = current_pos,
					   .head_start = head_start,
					   .head_end = head_end,
					   .attn_scores_buffer = ctx->mem.attn_scores_buffer[t]};

		thread_pool_submit(thread_pool, attention_task, task);
	}

	thread_pool_wait(thread_pool);
}

void attention_batch(struct ctx_t *ctx, int batch_len, int layer_idx, int start_pos, bool use_threads)
{
	if (use_threads == 0) {
		// Get model dimensions
		int num_heads = ctx->model->num_heads;
		int head_dim = ctx->model->head_dim;
		int num_kv_heads = ctx->model->num_kv_heads;
		int q_dim = num_heads * head_dim;
		int kv_dim = num_kv_heads * head_dim;

		// The temporary buffer for one token's attention scores.
		// Reuse this buffer for each token in the batch.
		float *attn_scores_buffer = ctx->mem.attn_scores_buffer[0];

		// Iterate over each token in the prompt
		// This is the "query" token (the one that is "looking").
		for (int i = 0; i < batch_len; i++) {
			// The absolute position of the current query token
			int absolute_pos = start_pos + i;

			// Loop over each query head for the current token
			for (int h = 0; h < num_heads; h++) {
				// Get pointers to the current token's Q head and its corresponding output
				float *q_head = ctx->mem.Q + (long long)i * q_dim + h * head_dim;
				float *out_head = ctx->mem.attn_output + (long long)i * q_dim + h * head_dim;

				int kv_head_idx = h / (num_heads / num_kv_heads);

				// 1. Calculate Raw Attention Scores
				// The current Q-head dots with all *previous* K-heads in the cache.
				for (int t = 0; t <= absolute_pos; t++) {
					// This is the "key" token (the one being looked at).
					uint16_t *k_head = ctx->kv_cache[layer_idx].k + (long long)t * kv_dim
							   + kv_head_idx * head_dim;
					float score = dot_product_f32_bf16(q_head, k_head, head_dim);
					attn_scores_buffer[t] = score * ctx->model->attn_scale;
				}

				// 2. Calculate Softmax (for scores 0 to current_pos)
				float max_score = -INFINITY;
				for (int t = 0; t <= absolute_pos; t++) {
					if (attn_scores_buffer[t] > max_score) {
						max_score = attn_scores_buffer[t];
					}
				}
				float sum_exp = 0.0f;
				for (int t = 0; t <= absolute_pos; t++) {
					float val = expf(attn_scores_buffer[t] - max_score);
					attn_scores_buffer[t] = val;
					sum_exp += val;
				}
				float inv_sum_exp = 1.0f / sum_exp;
				// Normalize scores to get final attention weights
				for (int t = 0; t <= absolute_pos; t++) {
					attn_scores_buffer[t] *= inv_sum_exp;
				}

				// 3. Calculate Weighted Sum of V-vectors
				// Zero out the output head before accumulating
				memset(out_head, 0, head_dim * sizeof(float));
				for (int t = 0; t <= absolute_pos; t++) {
					uint16_t *v_head = ctx->kv_cache[layer_idx].v + (long long)t * kv_dim
							   + kv_head_idx * head_dim;
					float attention_weight = attn_scores_buffer[t];

					// Accumulate weighted V-vectors
					//				for (int i = 0; i < head_dim; i++) {
					//					out_head[i] += attention_weight *
					// bf16_to_fp32(v_head[i]);
					//				}
					accumulate_weighted_fp32_bf16(out_head, attention_weight, v_head, head_dim);
				}
			}
		}

		return;
	}

	int num_threads = thread_pool->num_threads;
	int tokens_per_thread = (batch_len + num_threads - 1) / num_threads;

	for (int t = 0; t < num_threads; t++) {
		int start_token = t * tokens_per_thread;
		int end_token = start_token + tokens_per_thread;

		if (start_token >= batch_len)
			break;
		if (end_token > batch_len)
			end_token = batch_len;

		attention_batch_task_t *task = malloc(sizeof(attention_batch_task_t));
		*task = (attention_batch_task_t){.ctx = ctx,
						 .layer_idx = layer_idx,
						 .start_token_idx = start_token,
						 .end_token_idx = end_token,
						 .batch_start_pos = start_pos, // Pass the absolute starting position
						 .thread_id = t};
		thread_pool_submit(thread_pool, attention_batch_task, task);
	}
	thread_pool_wait(thread_pool);
}

void attention_unified(struct ctx_t *ctx, int batch_len, int layer_idx, int start_pos, bool use_threads)
{
	if (batch_len == 1) {
		attention_parallel(ctx, layer_idx, start_pos, use_threads);
	} else {
		attention_batch(ctx, batch_len, layer_idx, start_pos, use_threads);
	}
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
		sum += x[i];
	}
	// Normalize
	for (int i = 0; i < size; i++) {
		x[i] /= sum;
	}
}

// Comparison function for qsort
int compare_experts(const void *a, const void *b)
{
	float score_a = ((ExpertChoice *)a)->score;
	float score_b = ((ExpertChoice *)b)->score;
	if (score_a < score_b)
		return 1;
	if (score_a > score_b)
		return -1;
	return 0;
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

size_t get_ggml_block_size(int type)
{
	switch (type) {
	case GGML_TYPE_Q4_K:
		return sizeof(block_q4_k);
	case GGML_TYPE_Q6_K:
		return sizeof(block_q6_k);
	default:
		printf("FATAL: MoE operates on unsupported tensor type %d\n", type);
		return 0;
	}
}

void process_expert_task(void *arg)
{
	expert_task_t *task = (expert_task_t *)arg;
	struct ctx_t *ctx = task->ctx;
	int expert_idx = task->expert_idx;
	layer_weights *l = &ctx->model->layers[task->layer_idx];

	// Use thread-specific scratch buffers
	float *ffn_hidden1 = ctx->mem.ffn_hidden1_scratch[task->thread_id];
	float *ffn_hidden2 = ctx->mem.ffn_hidden2_scratch[task->thread_id];
	float *expert_out = ctx->mem.expert_out_scratch[task->thread_id];

	size_t up_gate_block_size_bytes = get_ggml_block_size(l->ffn_up_exps.type);
	size_t up_gate_blocks_per_row = ctx->model->embed_dim / QK_K;
	size_t up_gate_blocks_per_matrix = ctx->model->expert_ffn_dim * up_gate_blocks_per_row;
	size_t up_gate_matrix_size_bytes = up_gate_blocks_per_matrix * up_gate_block_size_bytes;

	// Pre-calculate sizes for the DOWN expert matrices
	size_t down_block_size_bytes = get_ggml_block_size(l->ffn_down_exps.type);
	size_t down_blocks_per_row = ctx->model->expert_ffn_dim / QK_K;
	size_t down_blocks_per_matrix = ctx->model->embed_dim * down_blocks_per_row;
	size_t down_matrix_size_bytes = down_blocks_per_matrix * down_block_size_bytes;

	// Calculate separate byte offsets for the selected expert
	size_t up_gate_offset_bytes = (size_t)expert_idx * up_gate_matrix_size_bytes;
	size_t down_offset_bytes = (size_t)expert_idx * down_matrix_size_bytes;

	// Create temporary Tensor structs pointing to the correct data slices
	Tensor expert_gate = {.type = l->ffn_gate_exps.type,
			      .data = (uint8_t *)l->ffn_gate_exps.data + up_gate_offset_bytes};
	Tensor expert_up = {.type = l->ffn_up_exps.type, .data = (uint8_t *)l->ffn_up_exps.data + up_gate_offset_bytes};
	Tensor expert_down = {.type = l->ffn_down_exps.type,
			      .data = (uint8_t *)l->ffn_down_exps.data + down_offset_bytes};

	// FFN forward pass
	parallel_mat_vec_unified(task->normed_input, &expert_gate, ffn_hidden1, ctx->model->embed_dim,
				 ctx->model->expert_ffn_dim, false);
	parallel_mat_vec_unified(task->normed_input, &expert_up, ffn_hidden2, ctx->model->embed_dim,
				 ctx->model->expert_ffn_dim, false);

	// SiLU
	for (int k = 0; k < ctx->model->expert_ffn_dim; k++) {
		ffn_hidden1[k] = silu_lookup(ffn_hidden1[k]) * ffn_hidden2[k];
	}

	// Down-projection
	parallel_mat_vec_unified(ffn_hidden1, &expert_down, expert_out, ctx->model->expert_ffn_dim,
				 ctx->model->embed_dim, false);

	// The final result is copied to the shared output buffer passed in the task
	memcpy(task->output_buffer, expert_out, ctx->model->embed_dim * sizeof(float));

	free(task);
}

int transformer_layer(struct ctx_t *ctx, int layer_idx, int batch_len, bool use_threads)
{
	float *x = ctx->mem.hidden_state;
	layer_weights *l = &ctx->model->layers[layer_idx];
	LayerKVCache *cache = &ctx->kv_cache[layer_idx];
	int kv_dim = ctx->model->num_kv_heads * ctx->model->head_dim;
	int q_dim = ctx->model->num_heads * ctx->model->head_dim;

	// The absolute starting position for this batch
	int start_pos = ctx->kv_pos;

	// ============ Attention Block ============
	// Step 1: RMSNorm on input
	for (int i = 0; i < batch_len; i++) {
		rms_norm(ctx->mem.normed_qkv_input + (long long)i * ctx->model->embed_dim,
			 x + (long long)i * ctx->model->embed_dim, l->attn_norm.data, ctx->model->embed_dim,
			 ctx->model->norm_eps);
	}

	// Step 2: Compute Q/K/V Matrices
	parallel_mat_mat_unified(ctx->mem.normed_qkv_input, &l->attn_q, ctx->mem.Q, batch_len, ctx->model->embed_dim,
				 q_dim, use_threads);
	parallel_mat_mat_unified(ctx->mem.normed_qkv_input, &l->attn_k, ctx->mem.K, batch_len, ctx->model->embed_dim,
				 kv_dim, use_threads);
	parallel_mat_mat_unified(ctx->mem.normed_qkv_input, &l->attn_v, ctx->mem.V, batch_len, ctx->model->embed_dim,
				 kv_dim, use_threads);

	// Step 3: Apply RoPE
	for (int i = 0; i < batch_len; i++) {
		// The absolute position for the current token in the batch
		int absolute_pos = start_pos + i;

		float *q_token_row = ctx->mem.Q + (long long)i * q_dim;
		float *k_token_row = ctx->mem.K + (long long)i * kv_dim;

		for (int h = 0; h < ctx->model->num_heads; h++) {
			float *q_head = q_token_row + h * ctx->model->head_dim;
			if (l->attn_q_norm.data) {
				rms_norm(q_head, q_head, l->attn_q_norm.data, ctx->model->head_dim,
					 ctx->model->norm_eps);
			}
			// Use the absolute position for RoPE
			apply_rope_cache(ctx, q_head, absolute_pos, ctx->model->head_dim);
		}
		for (int h = 0; h < ctx->model->num_kv_heads; h++) {
			float *k_head = k_token_row + h * ctx->model->head_dim;
			if (l->attn_k_norm.data) {
				rms_norm(k_head, k_head, l->attn_k_norm.data, ctx->model->head_dim,
					 ctx->model->norm_eps);
			}
			// Use the absolute position for RoPE
			apply_rope_cache(ctx, k_head, absolute_pos, ctx->model->head_dim);
		}
	}

	// Step 4: Store K/V to cache
	for (int i = 0; i < batch_len * kv_dim; i++) {
		cache->k[start_pos * kv_dim + i] = fp32_to_bf16(ctx->mem.K[i]);
		cache->v[start_pos * kv_dim + i] = fp32_to_bf16(ctx->mem.V[i]);
	}

	// Step 5: Multi-Head Attention Calculation
	attention_unified(ctx, batch_len, layer_idx, start_pos, use_threads);

	// Step 6: Output projection and residual add
	parallel_mat_mat_unified(ctx->mem.attn_output, &l->attn_out, ctx->mem.attn_proj_output, batch_len, q_dim,
				 ctx->model->embed_dim, use_threads);

	for (long long i = 0; i < (long long)batch_len * ctx->model->embed_dim; i++) {
		x[i] = add_residual(x[i], ctx->mem.attn_proj_output[i]);
	}

	// ============ FFN Block ============
	// Step 1: RMSNorm
	for (int i = 0; i < batch_len; i++) {
		rms_norm(ctx->mem.normed_ffn_input + (long long)i * ctx->model->embed_dim,
			 x + (long long)i * ctx->model->embed_dim, l->ffn_norm.data, ctx->model->embed_dim,
			 ctx->model->norm_eps);
	}

	// A buffer to hold the final output of the FFN block for the entire batch
	float *ffn_batch_output = ctx->mem.ffn_down_output; // Reuse existing buffer

	// MoE
	if (ctx->model->is_moe) {
		// Get the size of a single quantization block for the expert tensors
		size_t block_size_bytes = get_ggml_block_size(l->ffn_up_exps.type);
		if (block_size_bytes == 0) {
			return -1;
		}

		// A temporary buffer to hold the outputs of all experts before accumulation
		float (*expert_outputs)[ctx->model->embed_dim] =
			malloc(ctx->model->expert_used_count * sizeof(*expert_outputs));

		if (expert_outputs == NULL) {
			printf("FATAL: MoE expert_outputs OOM\n");
			return -1;
		}

		for (int i = 0; i < batch_len; i++) {
			float *normed_input_for_token_i =
				ctx->mem.normed_ffn_input + (long long)i * ctx->model->embed_dim;
			float *ffn_out_for_token_i = ffn_batch_output + (long long)i * ctx->model->embed_dim;

			// 1. Route
			parallel_mat_vec_unified(normed_input_for_token_i, &l->ffn_gate_inp, ctx->mem.expert_scores,
						 ctx->model->embed_dim, ctx->model->expert_count, false);

			// 2. Select: Find the top experts for this token
			ExpertChoice top_experts[ctx->model->expert_used_count];
			find_top_k(ctx->mem.expert_scores, ctx->model->expert_count, ctx->model->expert_used_count,
				   top_experts);

			// 3. Gate: Apply softmax to the top scores to get the final weights
			float gate_values[ctx->model->expert_used_count];
			for (int j = 0; j < ctx->model->expert_used_count; j++) {
				gate_values[j] = top_experts[j].score;
			}
			softmax(gate_values, ctx->model->expert_used_count);

			for (int j = 0; j < ctx->model->expert_used_count; j++) {
				expert_task_t *task = malloc(sizeof(expert_task_t));
				*task = (expert_task_t){.ctx = ctx,
							.thread_id = j,
							.layer_idx = layer_idx,
							.expert_idx = top_experts[j].index,
							.normed_input = normed_input_for_token_i,
							.output_buffer = expert_outputs[j]};
				thread_pool_submit(thread_pool, process_expert_task, task);
			}
			thread_pool_wait(thread_pool);

			// Accumulation
			memset(ffn_out_for_token_i, 0, ctx->model->embed_dim * sizeof(float));

			for (int j = 0; j < ctx->model->expert_used_count; j++) {
				float gate_val = gate_values[j];
				for (int k = 0; k < ctx->model->embed_dim; k++) {
					ffn_out_for_token_i[k] += gate_val * expert_outputs[j][k];
				}
			}
		}

		free(expert_outputs);

	} else { // DENSE FFN

		// Step 2: Gate + Up projections
		parallel_mat_mat_unified(ctx->mem.normed_ffn_input, &l->ffn_gate, ctx->mem.gate_proj_output, batch_len,
					 ctx->model->embed_dim, ctx->model->ffn_dim, use_threads);
		parallel_mat_mat_unified(ctx->mem.normed_ffn_input, &l->ffn_up, ctx->mem.up_proj_output, batch_len,
					 ctx->model->embed_dim, ctx->model->ffn_dim, use_threads);

		// Step 3: SwiGLU activation
		for (long long i = 0; i < (long long)batch_len * ctx->model->ffn_dim; i++) {
			ctx->mem.gate_proj_output[i] =
				silu_lookup(ctx->mem.gate_proj_output[i]) * ctx->mem.up_proj_output[i];
		}

		// Step 4: Down projection
		parallel_mat_mat_unified(ctx->mem.gate_proj_output, &l->ffn_down, ffn_batch_output, batch_len,
					 ctx->model->ffn_dim, ctx->model->embed_dim, use_threads);
	}

	// Step 5: Final residual connection
	for (long long i = 0; i < (long long)batch_len * ctx->model->embed_dim; i++) {
		x[i] = add_residual(x[i], ffn_batch_output[i]);
	}

	return 0;
}

/**
 * @brief Retrieves a single token's embedding vector from a tensor.
 *
 * This function acts as a dispatcher. It checks the tensor's type and calls
 * the appropriate internal logic to dequantize or convert the embedding row
 * into a standard float32 vector.
 *
 * @param tensor The token embedding tensor.
 * @param row_index The token ID whose embedding vector to retrieve.
 * @param dest The destination float buffer to write the vector to.
 * @param embed_dim The dimension of the embedding vector.
 */
void get_embedding_row(const Tensor *tensor, int row_index, float *dest, int embed_dim)
{
	switch (tensor->type) {

	case GGML_TYPE_Q4_K: {
		int blocks_per_row = embed_dim / QK_K;
		block_q4_k *src = (block_q4_k *)tensor->data;
		long long row_block_offset = (long long)row_index * blocks_per_row;

		for (int block_idx = 0; block_idx < blocks_per_row; block_idx++) {
			dequantize_row_q4_k(&src[row_block_offset + block_idx], dest + block_idx * QK_K, QK_K);
		}
		break;
	}

	case GGML_TYPE_Q6_K: {
		int blocks_per_row = embed_dim / QK_K;

		block_q6_k *src = (block_q6_k *)tensor->data;
		long long row_block_offset = (long long)row_index * blocks_per_row;

		for (int block_idx = 0; block_idx < blocks_per_row; block_idx++) {
			dequantize_row_q6_k(&src[row_block_offset + block_idx], dest + block_idx * QK_K, QK_K);
		}
		break;
	}

	case GGML_TYPE_F32: {
		float *src = (float *)tensor->data;
		long long row_offset = (long long)row_index * embed_dim;

		memcpy(dest, src + row_offset, embed_dim * sizeof(float));
		break;
	}

	case GGML_TYPE_BF16: {
		uint16_t *src = (uint16_t *)tensor->data;
		long long row_offset = (long long)row_index * embed_dim;

		for (int i = 0; i < embed_dim; i++) {
			dest[i] = bf16_to_fp32(src[row_offset + i]);
		}
		break;
	}

	default:
		fprintf(stderr, "Error: Unsupported tensor type %d for embedding lookup\n", tensor->type);
		break;
	}
}
