#include <inttypes.h>
#include <stdio.h>
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

float silu_table[SILU_TABLE_SIZE];

void reset_kv_cache(struct ctx_t *ctx)
{
	for (int i = 0; i < ctx->model->num_layers; i++) {
		memset(ctx->kv_cache[i].k, 0,
		       ctx->model->seq_length * ctx->model->num_kv_heads * ctx->model->head_dim * sizeof(float));
		memset(ctx->kv_cache[i].v, 0,
		       ctx->model->seq_length * ctx->model->num_kv_heads * ctx->model->head_dim * sizeof(float));
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

	rope_cache->sin = malloc(sizeof(float) * max_pos * (head_dim / 2));
	rope_cache->cos = malloc(sizeof(float) * max_pos * (head_dim / 2));

	for (int pos = 0; pos < max_pos; ++pos) {
		float scaled_pos = (float)pos * ctx->model->yarn_scale_factor;

		for (int i = 0; i < head_dim / 2; ++i) {
			// FIX: Use passed-in head_dim, not model's embed_dim
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

void rms_norm(float *__restrict o, const float *__restrict x, const float *__restrict weight, int size, float eps)
{
#ifdef CONFIG_ENABLE_AVX2
	if (__builtin_cpu_supports("avx2")) {
		return rms_norm_avx2(o, x, weight, size, eps);
	}
#endif
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

	// Apply weight and scale
	for (int i = 0; i < size; ++i) {
		o[i] = x[i] * weight[i] * inv_rms;
	}
}

static void apply_rope_cache(struct ctx_t *ctx, float *x, int pos, int head_dim)
{
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

/*
 *    pointer to cache->k or cache->v
 *    time step (token position)
 *    kv head index
 *    dimension per head
 *    number of KV heads (usually 8 for GQA)
 */
static inline float *get_kv_head(float *cache_base, int t, int kv_head_idx, int head_dim, int num_kv_heads)
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

		// --- 1. Calculate Raw Attention Scores ---
		// Calculate the dot product of the current FP32 Q-head with all previous
		// FP32 K-heads in the cache
		for (int t = 0; t <= current_pos; t++) {
			// Read from K cache, which stores FP32
			float *k_head =
				get_kv_head(cache->k, t, kv_head_idx, ctx->model->head_dim, ctx->model->num_kv_heads);
			// Use the new dot product for two FP32 vectors
			float score = dot_product_f32(q_head, k_head, ctx->model->head_dim);
			attn_scores_buffer[t] = score * ctx->model->attn_scale;
		}

		// --- 2. Calculate Softmax ---
		// This block numerically stabilizes softmax by subtracting the max score
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

		// --- 3. Calculate Weighted Sum of V-vectors ---
		for (int t = 0; t <= current_pos; t++) {
			// Read from V cache, which stores FP32
			float *v_head =
				get_kv_head(cache->v, t, kv_head_idx, ctx->model->head_dim, ctx->model->num_kv_heads);
			// Normalize the score to get the final attention weight
			float attention_weight = attn_scores_buffer[t] * inv_sum_exp;

			// Accumulate weighted FP32 V-vector values into the output head buffer
			for (int i = 0; i < ctx->model->head_dim; i++) {
				out_head[i] = fmaf(attention_weight, v_head[i], out_head[i]);
			}
		}
	}

	free(task);
}

/**
 * @brief Performs batched, causal self-attention for prompt processing.
 *
 * This function calculates attention for an entire sequence (batch) of tokens at once.
 * It correctly handles the causal nature of the attention mechanism, where each token
 * can only attend to itself and the tokens that came before it.
 *
 * @param ctx The main context struct.
 * @param mem The batched memory workspace.
 * @param batch_len The number of tokens in the prompt.
 * @param layer_idx The current transformer layer index.
 */
void attention_batch_sequential(struct ctx_t *ctx, int batch_len, int layer_idx, int start_pos)
{
	// Get model dimensions
	int num_heads = ctx->model->num_heads;
	int head_dim = ctx->model->head_dim;
	int num_kv_heads = ctx->model->num_kv_heads;
	int q_dim = num_heads * head_dim;
	int kv_dim = num_kv_heads * head_dim;

	// The temporary buffer for one token's attention scores.
	// We reuse this buffer for each token in the batch.
	float *attn_scores_buffer = ctx->mem.attn_scores_buffer[0];

	// --- Main loop: Iterate over each token in the prompt ---
	// This is the "query" token (the one that is "looking").
	for (int i = 0; i < batch_len; i++) {
		// The absolute position of the current query token
		int absolute_pos = start_pos + i;

		// --- Loop over each query head for the current token ---
		for (int h = 0; h < num_heads; h++) {
			// Get pointers to the current token's Q head and its corresponding output
			float *q_head = ctx->mem.Q + (long long)i * q_dim + h * head_dim;
			float *out_head = ctx->mem.attn_output + (long long)i * q_dim + h * head_dim;

			int kv_head_idx = h / (num_heads / num_kv_heads);

			// 1. --- Calculate Raw Attention Scores ---
			// The current Q-head dots with all *previous* K-heads in the cache.
			for (int t = 0; t <= absolute_pos; t++) {
				// This is the "key" token (the one being looked at).
				float *k_head =
					ctx->kv_cache[layer_idx].k + (long long)t * kv_dim + kv_head_idx * head_dim;
				float score = dot_product_f32(q_head, k_head, head_dim);
				attn_scores_buffer[t] = score * ctx->model->attn_scale;
			}

			// 2. --- Calculate Softmax (for scores 0 to current_pos) ---
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

			// 3. --- Calculate Weighted Sum of V-vectors ---
			// Zero out the output head before accumulating
			memset(out_head, 0, head_dim * sizeof(float));
			for (int t = 0; t <= absolute_pos; t++) {
				float *v_head =
					ctx->kv_cache[layer_idx].v + (long long)t * kv_dim + kv_head_idx * head_dim;
				float attention_weight = attn_scores_buffer[t];

				// Accumulate weighted V-vectors
				for (int i = 0; i < head_dim; i++) {
					out_head[i] += attention_weight * v_head[i];
				}
			}
		}
	}
}

/**
 * @brief The function executed by each thread for batched attention.
 *
 * This version correctly calculates the absolute position of each token to handle
 * multi-turn conversation context.
 */
static void process_attention_batch_task(void *arg)
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
		// **THE CRITICAL FIX IS HERE**
		// Calculate the token's absolute position in the entire conversation
		int absolute_pos = batch_start_pos + i;

		for (int h = 0; h < num_heads; h++) {
			float *q_head = ctx->mem.Q + (long long)i * q_dim + h * head_dim;
			float *out_head = ctx->mem.attn_output + (long long)i * q_dim + h * head_dim;
			int kv_head_idx = h / (num_heads / num_kv_heads);

			// 1. --- Calculate Raw Attention Scores ---
			// The Q-head dots with ALL previous K-heads, up to its absolute_pos
			for (int t = 0; t <= absolute_pos; t++) {
				float *k_head =
					ctx->kv_cache[layer_idx].k + (long long)t * kv_dim + kv_head_idx * head_dim;
				float score = dot_product_f32(q_head, k_head, head_dim);
				attn_scores_buffer[t] = score * ctx->model->attn_scale;
			}

			// 2. --- Calculate Softmax ---
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

			// 3. --- Calculate Weighted Sum of V-vectors ---
			memset(out_head, 0, head_dim * sizeof(float));
			for (int t = 0; t <= absolute_pos; t++) {
				float *v_head =
					ctx->kv_cache[layer_idx].v + (long long)t * kv_dim + kv_head_idx * head_dim;
				float attention_weight = attn_scores_buffer[t];
				for (int j = 0; j < head_dim; j++) {
					out_head[j] += attention_weight * v_head[j];
				}
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

			// --- 1. Calculate Raw Attention Scores ---
			for (int t = 0; t <= current_pos; t++) {
				float *k_head = get_kv_head(cache->k, t, kv_head_idx, ctx->model->head_dim,
							    ctx->model->num_kv_heads);
				float score = dot_product_f32(q_head, k_head, ctx->model->head_dim);
				ctx->mem.attn_scores_buffer[0][t] =
					score * ctx->model->attn_scale; // Use first buffer for sequential mode
			}

			// --- 2. Calculate Softmax ---
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

			// --- 3. Calculate Weighted Sum of V-vectors ---
			memset(out_head, 0, ctx->model->head_dim * sizeof(float));
			for (int t = 0; t <= current_pos; t++) {
				float *v_head = get_kv_head(cache->v, t, kv_head_idx, ctx->model->head_dim,
							    ctx->model->num_kv_heads);
				float attention_weight = ctx->mem.attn_scores_buffer[0][t] * inv_sum_exp;
				for (int i = 0; i < ctx->model->head_dim; i++) {
					out_head[i] = fmaf(attention_weight, v_head[i], out_head[i]);
				}
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
		attention_batch_sequential(ctx, batch_len, layer_idx, start_pos);
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
		thread_pool_submit(thread_pool, process_attention_batch_task, task);
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

int transformer_layer_unified(struct ctx_t *ctx, int layer_idx, int batch_len, bool use_threads)
{
	float *x = ctx->mem.hidden_state;
	LayerWeights *l = &ctx->model->layers[layer_idx];
	LayerKVCache *cache = &ctx->kv_cache[layer_idx];
	int kv_dim = ctx->model->num_kv_heads * ctx->model->head_dim;
	int q_dim = ctx->model->num_heads * ctx->model->head_dim;

	// The absolute starting position for this batch
	int start_pos = ctx->kv_pos;

	// ============ Attention Block ============
	// Step 1: RMSNorm on input
	for (int i = 0; i < batch_len; i++) {
		rms_norm(ctx->mem.normed_qkv_input + (long long)i * ctx->model->embed_dim,
			 x + (long long)i * ctx->model->embed_dim, l->attn_norm, ctx->model->embed_dim,
			 ctx->model->norm_eps);
	}

	// Step 2: Compute Q/K/V Matrices
	parallel_mat_mat_bf16(ctx->mem.normed_qkv_input, l->attn_q, ctx->mem.Q, batch_len, ctx->model->embed_dim, q_dim,
			      use_threads);
	parallel_mat_mat_bf16(ctx->mem.normed_qkv_input, l->attn_k, ctx->mem.K, batch_len, ctx->model->embed_dim,
			      kv_dim, use_threads);
	parallel_mat_mat_bf16(ctx->mem.normed_qkv_input, l->attn_v, ctx->mem.V, batch_len, ctx->model->embed_dim,
			      kv_dim, use_threads);

	// Step 3: Apply RoPE
	for (int i = 0; i < batch_len; i++) {
		// The absolute position for the current token in the batch
		int absolute_pos = start_pos + i;

		float *q_token_row = ctx->mem.Q + (long long)i * q_dim;
		float *k_token_row = ctx->mem.K + (long long)i * kv_dim;

		for (int h = 0; h < ctx->model->num_heads; h++) {
			float *q_head = q_token_row + h * ctx->model->head_dim;
			if (l->attn_q_norm) {
				rms_norm(q_head, q_head, l->attn_q_norm, ctx->model->head_dim, ctx->model->norm_eps);
			}
			// Use the absolute position for RoPE
			apply_rope_cache(ctx, q_head, absolute_pos, ctx->model->head_dim);
		}
		for (int h = 0; h < ctx->model->num_kv_heads; h++) {
			float *k_head = k_token_row + h * ctx->model->head_dim;
			if (l->attn_k_norm) {
				rms_norm(k_head, k_head, l->attn_k_norm, ctx->model->head_dim, ctx->model->norm_eps);
			}
			// Use the absolute position for RoPE
			apply_rope_cache(ctx, k_head, absolute_pos, ctx->model->head_dim);
		}
	}

	// Step 4: Store K/V to cache
	memcpy(cache->k + (long long)start_pos * kv_dim, ctx->mem.K, (long long)batch_len * kv_dim * sizeof(float));
	memcpy(cache->v + (long long)start_pos * kv_dim, ctx->mem.V, (long long)batch_len * kv_dim * sizeof(float));

	// Step 5: Multi-Head Attention Calculation (Corrected)
	attention_unified(ctx, batch_len, layer_idx, start_pos, use_threads);

	// Step 6: Output projection and residual add
	parallel_mat_mat_bf16(ctx->mem.attn_output, l->attn_out, ctx->mem.attn_proj_output, batch_len, q_dim,
			      ctx->model->embed_dim, use_threads);

	for (long long i = 0; i < (long long)batch_len * ctx->model->embed_dim; i++) {
		x[i] = add_residual(x[i], ctx->mem.attn_proj_output[i]);
	}

	// ============ FFN Block ============
	// Step 1: RMSNorm
	for (int i = 0; i < batch_len; i++) {
		rms_norm(ctx->mem.normed_ffn_input + (long long)i * ctx->model->embed_dim,
			 x + (long long)i * ctx->model->embed_dim, l->ffn_norm, ctx->model->embed_dim,
			 ctx->model->norm_eps);
	}

	// Step 2: Gate + Up projections
	parallel_mat_mat_bf16(ctx->mem.normed_ffn_input, l->ffn_gate, ctx->mem.gate_proj_output, batch_len,
			      ctx->model->embed_dim, ctx->model->ffn_dim, use_threads);
	parallel_mat_mat_bf16(ctx->mem.normed_ffn_input, l->ffn_up, ctx->mem.up_proj_output, batch_len,
			      ctx->model->embed_dim, ctx->model->ffn_dim, use_threads);

	// Step 3: SwiGLU activation
	for (long long i = 0; i < (long long)batch_len * ctx->model->ffn_dim; i++) {
		ctx->mem.gate_proj_output[i] = silu_lookup(ctx->mem.gate_proj_output[i]) * ctx->mem.up_proj_output[i];
	}

	// Step 4: Down projection
	parallel_mat_mat_bf16(ctx->mem.gate_proj_output, l->ffn_down, ctx->mem.ffn_down_output, batch_len,
			      ctx->model->ffn_dim, ctx->model->embed_dim, use_threads);

	// Step 5: Residual add
	for (long long i = 0; i < (long long)batch_len * ctx->model->embed_dim; i++) {
		x[i] = add_residual(x[i], ctx->mem.ffn_down_output[i]);
	}

	return 0;
}
