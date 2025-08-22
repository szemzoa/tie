#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "engine.h"
#include "model.h"
#include "math_dispatch.h"
#include "threadpool.h"

typedef struct {
	struct ctx_t *ctx;
	int layer_idx;
	int pos;
	int head_start;
	int head_end;
	int thread_id; // The ID used to get a unique scratch buffer
} attention_task_t;

typedef struct {
	struct ctx_t *ctx;
	int layer_idx;
	int start_token_idx; // The first token in the batch this thread will process
	int end_token_idx;   // The last token (exclusive)
	int batch_start_pos; // The absolute position of the beginning of the prompt batch
	int thread_id;	     // The ID used to get a unique scratch buffer
} attention_batch_task_t;

typedef struct {
	struct ctx_t *ctx;
	int layer_idx;
	int thread_id;
	int expert_idx;
	MemType normed_input;
} expert_task_t;

typedef struct {
	int index;
	float score;
} ExpertChoice;

float silu_table[SILU_TABLE_SIZE];

void kv_cache_reset(struct ctx_t *ctx)
{
        int kv_dim = ctx->model->num_kv_heads * ctx->model->head_dim;

        for (int i = 0; i < ctx->model->num_layers; i++) {
                // Assume k and v have been allocated with the correct size and type
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

float silu_lookup(float x)
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

MemType mem_slice(MemType *buffer, size_t offset_elements)
{
	size_t element_size = ggml_type_size(buffer->type);
	MemType slice = {.type = buffer->type, .data = (uint8_t *)buffer->data + (offset_elements * element_size)};
	return slice;
}

static inline void *get_kv_head(const MemType *cache, int t, int kv_head_idx, int head_dim, int num_kv_heads)
{
        size_t element_size = ggml_type_size(cache->type);
        size_t kv_dim = (size_t)num_kv_heads * head_dim;
        size_t offset_elements = (size_t)t * kv_dim + (size_t)kv_head_idx * head_dim;
        return (uint8_t *)cache->data + offset_elements * element_size;
}

static void attention_task(void *arg)
{
	attention_task_t *task = (attention_task_t *)arg;
	struct ctx_t *ctx = task->ctx;
	int current_pos = task->pos;
	LayerKVCache *cache = &ctx->kv_cache[task->layer_idx];

	// 1. Get this thread's private scratch buffers
	MemType *q_head_fp32_scratch = &ctx->mem.q_head_fp32_scratch[task->thread_id];
	float *attn_scores_buffer = ctx->mem.attn_scores_buffer[task->thread_id];

	for (int h = task->head_start; h < task->head_end; h++) {
		// 2. Create slices for the specific Q and Output heads upfront
		MemType q_head_slice = mem_slice(&ctx->mem.Q, (size_t)h * ctx->model->head_dim);
		MemType out_head_slice = mem_slice(&ctx->mem.attn_output, (size_t)h * ctx->model->head_dim);

		// 3. Convert the source Q-head to FP32 using the new dispatch signature
		dispatch_convert(&q_head_slice, q_head_fp32_scratch, ctx->model->head_dim);

		// Determine the KV head index for this Q head (for GQA)
		int kv_head_idx = h / (ctx->model->num_heads / ctx->model->num_kv_heads);

		// 4. Calculate Raw Attention Scores
		// Calculate the dot product of the current FP32 Q-head with all previous K-heads in the cache
		for (int t = 0; t <= current_pos; t++) {
			void *k_head =
				get_kv_head(&cache->k, t, kv_head_idx, ctx->model->head_dim, ctx->model->num_kv_heads);

			MemType k_head_slice = {.type = GGML_TYPE_BF16, .data = (void *)k_head };

			float score = dispatch_dot_product(q_head_fp32_scratch, &k_head_slice, ctx->model->head_dim);
			attn_scores_buffer[t] = score * ctx->model->attn_scale;
		}

		// 5. Calculate Softmax
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

		// 6. Calculate Weighted Sum of V-vectors
		memset(out_head_slice.data, 0, ctx->model->head_dim * ggml_type_size(out_head_slice.type));

		for (int t = 0; t <= current_pos; t++) {
			void *v_head_ptr =
				get_kv_head(&cache->v, t, kv_head_idx, ctx->model->head_dim, ctx->model->num_kv_heads);

			float attention_weight = attn_scores_buffer[t] * inv_sum_exp;

			// 7. Create a MemType view of the data from the KV cache
			MemType v_head_slice = {.type = ctx->kv_cache[kv_head_idx].v.type, .data = (void *)v_head_ptr};

			dispatch_accumulate_weighted_V(&v_head_slice, &out_head_slice, attention_weight,
						       ctx->model->head_dim);
		}
	}

	free(task);
}

static void attention_batch_task(void *arg)
{
	attention_batch_task_t *task = (attention_batch_task_t *)arg;
	struct ctx_t *ctx = task->ctx;
	int batch_start_pos = task->batch_start_pos;

	// Get model dimensions and memory type info
	int q_dim = ctx->model->num_heads * ctx->model->head_dim;
	LayerKVCache *cache = &ctx->kv_cache[task->layer_idx];

	// Get this thread's private scratch buffers
	float *attn_scores_buffer = ctx->mem.attn_scores_buffer[task->thread_id];
	MemType *q_head_fp32_scratch = &ctx->mem.q_head_fp32_scratch[task->thread_id];

	// Each thread processes a *range* of query tokens from the prompt batch
	for (int i = task->start_token_idx; i < task->end_token_idx; i++) {
		int absolute_pos = batch_start_pos + i;

		for (int h = 0; h < ctx->model->num_heads; h++) {
			// 1. Calculate the pointer to the source Q-head
			MemType q_head_slice = mem_slice(&ctx->mem.Q, (size_t)i * q_dim + h * ctx->model->head_dim);
			MemType out_head_slice = mem_slice(&ctx->mem.attn_output, (size_t)i * q_dim + h * ctx->model->head_dim);

			// 2. Convert the source Q-head to FP32 in our scratch buffer
			dispatch_convert(&q_head_slice, q_head_fp32_scratch, ctx->model->head_dim);

			// 3. Calculate pointer to the destination output head
			int kv_head_idx = h / (ctx->model->num_heads / ctx->model->num_kv_heads);

			// 1. Calculate Raw Attention Scores
			for (int t = 0; t <= absolute_pos; t++) {
				void *k_head =
					get_kv_head(&cache->k, t, kv_head_idx, ctx->model->head_dim, ctx->model->num_kv_heads);

				MemType k_head_slice = { .type = GGML_TYPE_BF16, .data = (void *)k_head };

				float score = dispatch_dot_product(q_head_fp32_scratch, &k_head_slice, ctx->model->head_dim);
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
			memset(out_head_slice.data, 0, ctx->model->head_dim * ggml_type_size(out_head_slice.type));

			for (int t = 0; t <= absolute_pos; t++) {
				void *v_head_ptr =
					get_kv_head(&cache->v, t, kv_head_idx, ctx->model->head_dim, ctx->model->num_kv_heads);

				float attention_weight = attn_scores_buffer[t];

				// Create a MemType view of the data from the KV cache
				MemType v_head_slice = {.type = ctx->kv_cache[kv_head_idx].v.type, .data = (void *)v_head_ptr};

				dispatch_accumulate_weighted_V(&v_head_slice, &out_head_slice, attention_weight,
							       ctx->model->head_dim);
			}
		}
	}
	free(task);
}

void attention_parallel(struct ctx_t *ctx, int layer_idx, int current_pos)
{
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
					   .thread_id = t};
		thread_pool_submit(thread_pool, attention_task, task);
	}

	thread_pool_wait(thread_pool);
}

void attention_batch(struct ctx_t *ctx, int batch_len, int layer_idx, int start_pos)
{
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
		sum = fmaf(x[i], 1.0f, sum); // Accumulate sum using FMA
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
	struct ctx_t *ctx = task->ctx;
	int expert_idx = task->expert_idx;
	layer_weights *l = &ctx->model->layers[task->layer_idx];

	// Use thread-specific scratch buffers
	MemType *ffn_hidden1 = &ctx->mem.ffn_hidden1_scratch[task->thread_id];
	MemType *ffn_hidden2 = &ctx->mem.ffn_hidden2_scratch[task->thread_id];
	MemType *expert_out = &ctx->mem.expert_outputs[task->thread_id];
	MemType *normed_input = &task->normed_input;

	size_t up_gate_block_size_bytes = get_ggml_block_size(l->ffn_up_exps.mem.type);
	size_t up_gate_blocks_per_row = ctx->model->embed_dim / QK_K;
	size_t up_gate_blocks_per_matrix = ctx->model->expert_ffn_dim * up_gate_blocks_per_row;
	size_t up_gate_matrix_size_bytes = up_gate_blocks_per_matrix * up_gate_block_size_bytes;

	// Pre-calculate sizes for the DOWN expert matrices
	size_t down_block_size_bytes = get_ggml_block_size(l->ffn_down_exps.mem.type);
	size_t down_blocks_per_row = ctx->model->expert_ffn_dim / QK_K;
	size_t down_blocks_per_matrix = ctx->model->embed_dim * down_blocks_per_row;
	size_t down_matrix_size_bytes = down_blocks_per_matrix * down_block_size_bytes;

	// Calculate separate byte offsets for the selected expert
	size_t up_gate_offset_bytes = (size_t)expert_idx * up_gate_matrix_size_bytes;
	size_t down_offset_bytes = (size_t)expert_idx * down_matrix_size_bytes;

	// Create temporary Tensor structs pointing to the correct data slices
	Tensor expert_gate = {.mem.type = l->ffn_gate_exps.mem.type,
			      .mem.data = (uint8_t *)l->ffn_gate_exps.mem.data + up_gate_offset_bytes};
	Tensor expert_up = {.mem.type = l->ffn_up_exps.mem.type, .mem.data = (uint8_t *)l->ffn_up_exps.mem.data + up_gate_offset_bytes};
	Tensor expert_down = {.mem.type = l->ffn_down_exps.mem.type,
			      .mem.data = (uint8_t *)l->ffn_down_exps.mem.data + down_offset_bytes};

	// FFN forward pass
	dispatch_mat_vec(normed_input, &expert_gate, ffn_hidden1, ctx->model->embed_dim, ctx->model->expert_ffn_dim,
			 false);
	dispatch_mat_vec(normed_input, &expert_up, ffn_hidden2, ctx->model->embed_dim, ctx->model->expert_ffn_dim,
			 false);

	// SiLU
	dispatch_swiglu_activation(ffn_hidden1, ffn_hidden2, ctx->model->expert_ffn_dim);

	// Down-projection
	dispatch_mat_vec(ffn_hidden1, &expert_down, expert_out, ctx->model->expert_ffn_dim, ctx->model->embed_dim,
			 false);

	free(task);
}

int transformer_layer(struct ctx_t *ctx, int layer_idx, int batch_len)
{
	layer_weights *l = &ctx->model->layers[layer_idx];
	int kv_dim = ctx->model->num_kv_heads * ctx->model->head_dim;
	int q_dim = ctx->model->num_heads * ctx->model->head_dim;

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
	dispatch_mat_mat(&ctx->mem.normed_qkv_input, &l->attn_q, &ctx->mem.Q, batch_len, ctx->model->embed_dim, q_dim,
			 true);
	dispatch_mat_mat(&ctx->mem.normed_qkv_input, &l->attn_k, &ctx->mem.K, batch_len, ctx->model->embed_dim, kv_dim,
			 true);
	dispatch_mat_mat(&ctx->mem.normed_qkv_input, &l->attn_v, &ctx->mem.V, batch_len, ctx->model->embed_dim, kv_dim,
			 true);

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
			dispatch_apply_rope_cache(ctx, &Q_slice, absolute_pos, ctx->model->head_dim);
		}

		for (int h = 0; h < ctx->model->num_kv_heads; h++) {
			MemType K_slice = mem_slice(&ctx->mem.K, (size_t)i * kv_dim + h * ctx->model->head_dim);

			if (l->attn_k_norm.mem.data) {
				dispatch_rms_norm(&K_slice, &l->attn_k_norm, &K_slice, ctx->model->head_dim,
						  ctx->model->norm_eps);
			}
			// Use the absolute position for RoPE
			dispatch_apply_rope_cache(ctx, &K_slice, absolute_pos, ctx->model->head_dim);
		}
	}

	// Store K/V to cache
	dispatch_store_KV_cache(ctx, layer_idx, start_pos, batch_len);

	// Multi-Head Attention Calculation
	if (batch_len == 1) {
		attention_parallel(ctx, layer_idx, start_pos);
	} else {
		attention_batch(ctx, batch_len, layer_idx, start_pos);
	}

	// Output projection and residual add
	dispatch_mat_mat(&ctx->mem.attn_output, &l->attn_out, &ctx->mem.attn_proj_output, batch_len, q_dim,
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
		size_t block_size_bytes = get_ggml_block_size(l->ffn_up_exps.mem.type);
		if (block_size_bytes == 0) {
			return -1;
		}

		for (int i = 0; i < batch_len; i++) {

			// 1. Get type-aware pointers for the current token
			MemType normed_input_for_token_i =
				mem_slice(&ctx->mem.normed_ffn_input, (size_t)i * ctx->model->embed_dim);

			// Create a clean slice for the destination buffer.
			MemType ffn_out_slice = mem_slice(&ctx->mem.ffn_down_output, (size_t)i * ctx->model->embed_dim);

			// Use a per-thread scratch buffer for FP32 accumulation
			MemType *ffn_out_fp32_scratch = &ctx->mem.expert_out_fp32;
			float *ffn_out_fp32_token_buffer = ctx->mem.expert_out_fp32.data;

			// 2. Route, Select, and Gate
			// The input type for the router is the intermediate type, but the output scores are always
			// FP32.
			dispatch_mat_vec(&normed_input_for_token_i, &l->ffn_gate_inp, &ctx->mem.expert_scores,
					 ctx->model->embed_dim, ctx->model->expert_count, false);

			ExpertChoice top_experts[ctx->model->expert_used_count];
			find_top_k((float *)ctx->mem.expert_scores.data, ctx->model->expert_count,
				   ctx->model->expert_used_count, top_experts);

			float gate_values[ctx->model->expert_used_count];

			for (int j = 0; j < ctx->model->expert_used_count; j++)
				gate_values[j] = top_experts[j].score;

			softmax(gate_values, ctx->model->expert_used_count);

			// 3. Parallel Expert Processing
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

			// 4. Accumulate results in the FP32 temporary buffer
			memset(ffn_out_fp32_scratch->data, 0, ctx->model->embed_dim * sizeof(float));

			for (int j = 0; j < ctx->model->expert_used_count; j++) {
				float gate_val = gate_values[j];
				float *expert_result = ctx->mem.expert_outputs[j].data; // This is FP32
				for (int k = 0; k < ctx->model->embed_dim; k++) {
					ffn_out_fp32_token_buffer[k] += gate_val * expert_result[k];
				}
			}

			// 5. Convert the final FP32 result to the destination format (e.g., BF16)
			dispatch_convert(ffn_out_fp32_scratch, &ffn_out_slice, ctx->model->embed_dim);
		}

	} else { // DENSE FFN

		// Gate + Up projections
		dispatch_mat_mat(&ctx->mem.normed_ffn_input, &l->ffn_gate, &ctx->mem.gate_proj_output, batch_len,
				 ctx->model->embed_dim, ctx->model->ffn_dim, true);

		dispatch_mat_mat(&ctx->mem.normed_ffn_input, &l->ffn_up, &ctx->mem.up_proj_output, batch_len,
				 ctx->model->embed_dim, ctx->model->ffn_dim, true);

		// SwiGLU activation
		dispatch_swiglu_activation(&ctx->mem.gate_proj_output, &ctx->mem.up_proj_output,
					   batch_len * ctx->model->ffn_dim);

		// Down projection
		dispatch_mat_mat(&ctx->mem.gate_proj_output, &l->ffn_down, &ctx->mem.ffn_down_output, batch_len,
				 ctx->model->ffn_dim, ctx->model->embed_dim, true);
	}

	// Final residual connection
	dispatch_apply_residual(&ctx->mem.hidden_state, &ctx->mem.ffn_down_output, batch_len * ctx->model->embed_dim);

	return 0;
}
