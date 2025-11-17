#include <stdlib.h>
#include <stdio.h>
#include "math_dispatch.h"
#include "math_scalar.h"

#ifdef CONFIG_ENABLE_AVX2
#include "math_avx2.h"
#endif

// #define DEBUG_ACCEL

#ifdef DEBUG_ACCEL
#define debug_accel(...)                                                                                               \
	do {                                                                                                           \
		printf(__VA_ARGS__);                                                                                   \
	} while (0)
#else
#define debug_accel(...)
#endif

static void mat_vec_task(void *arg);
static void mat_mat_task(void *arg);

/*  The global dispatch tables, listing all available implementations */
embedding_row_dispatch_t EMBEDDING_ROW_DISPATCH_TABLE[] = {
#ifdef CONFIG_ENABLE_AVX2
	{GGML_TYPE_Q6_K, GGML_TYPE_BF16, get_embedding_row_q6k_bf16_avx2, 1},
	{GGML_TYPE_Q6_K, GGML_TYPE_F32, get_embedding_row_q6k_f32_avx2, 1},
	{GGML_TYPE_Q4_K, GGML_TYPE_F32, get_embedding_row_q4k_f32_avx2, 1},
	{GGML_TYPE_BF16, GGML_TYPE_BF16, get_embedding_row_bf16_bf16_avx2, 1},
	{GGML_TYPE_BF16, GGML_TYPE_F32, get_embedding_row_bf16_f32_avx2, 1},
#endif
	{GGML_TYPE_Q6_K, GGML_TYPE_BF16, get_embedding_row_q6k_bf16_scalar, 0},
	{GGML_TYPE_Q4_K, GGML_TYPE_BF16, get_embedding_row_q4k_bf16_scalar, 0},
	{GGML_TYPE_BF16, GGML_TYPE_BF16, get_embedding_row_bf16_bf16_scalar, 0},
	{GGML_TYPE_Q6_K, GGML_TYPE_F32, get_embedding_row_q6k_f32_scalar, 0},
	{GGML_TYPE_Q4_K, GGML_TYPE_F32, get_embedding_row_q4k_f32_scalar, 0},
	{GGML_TYPE_BF16, GGML_TYPE_F32, get_embedding_row_bf16_f32_scalar, 0},
	{GGML_TYPE_F32, GGML_TYPE_F32, get_embedding_row_f32_f32_scalar, 0},
};

rms_norm_dispatch_t RMS_NORM_DISPATCH_TABLE[] = {
#ifdef CONFIG_ENABLE_AVX2
	{GGML_TYPE_BF16, GGML_TYPE_F32, GGML_TYPE_BF16, rms_norm_bf16_f32_bf16_avx2, 1},
	{GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32, rms_norm_f32_f32_f32_avx2, 1},
#endif
	{GGML_TYPE_BF16, GGML_TYPE_F32, GGML_TYPE_BF16, rms_norm_bf16_f32_bf16_scalar, 0},
	{GGML_TYPE_BF16, GGML_TYPE_F32, GGML_TYPE_F32, rms_norm_bf16_f32_f32_scalar, 0},
	{GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32, rms_norm_f32_f32_f32_scalar, 0},
};

apply_rope_cache_dispatch_t APPLY_ROPE_CACHE_DISPATCH_TABLE[] = {
#ifdef CONFIG_ENABLE_AVX2
	{GGML_TYPE_BF16, apply_rope_cache_bf16_avx2, 1},
	{GGML_TYPE_F32, apply_rope_cache_f32_avx2, 1},
#endif
	{GGML_TYPE_BF16, apply_rope_cache_bf16_scalar, 0},
	{GGML_TYPE_F32, apply_rope_cache_f32_scalar, 0},
};

apply_mrope_cache_dispatch_t APPLY_MROPE_CACHE_DISPATCH_TABLE[] = {
	{GGML_TYPE_F32, apply_mrope_cache_f32_scalar, 0},
};

accumulate_weighted_V_dispatch_t ACCUMULATE_WEIGHTED_V_DISPATCH_TABLE[] = {
#ifdef CONFIG_ENABLE_AVX2
	{GGML_TYPE_BF16, GGML_TYPE_BF16, accumulate_weighted_V_bf16_bf16_avx2, 1},
	{GGML_TYPE_F32, GGML_TYPE_BF16, accumulate_weighted_V_f32_bf16_avx2, 1},
#endif
	{GGML_TYPE_BF16, GGML_TYPE_BF16, accumulate_weighted_V_bf16_bf16_scalar, 0},
	{GGML_TYPE_F32, GGML_TYPE_BF16, accumulate_weighted_V_f32_bf16_scalar, 0},
	{GGML_TYPE_F32, GGML_TYPE_F32, accumulate_weighted_V_f32_f32_scalar, 0},
};

store_KV_cache_dispatch_t STORE_KV_CACHE_DISPATCH_TABLE[] = {
#ifdef CONFIG_ENABLE_AVX2
	{GGML_TYPE_BF16, GGML_TYPE_BF16, store_KV_cache_bf16_bf16_avx2, 1},
	{GGML_TYPE_F32, GGML_TYPE_BF16, store_KV_cache_f32_bf16_avx2, 1},
#endif
	{GGML_TYPE_BF16, GGML_TYPE_BF16, store_KV_cache_bf16_bf16_scalar, 0},
	{GGML_TYPE_F32, GGML_TYPE_BF16, store_KV_cache_f32_bf16_scalar, 0},
	{GGML_TYPE_F32, GGML_TYPE_F32, store_KV_cache_f32_f32_scalar, 0},
};

apply_residual_dispatch_t APPLY_RESIDUAL_DISPATCH_TABLE[] = {
#ifdef CONFIG_ENABLE_AVX2
	{GGML_TYPE_BF16, GGML_TYPE_BF16, apply_residual_bf16_bf16_avx2, 1},
	{GGML_TYPE_F32, GGML_TYPE_F32, apply_residual_f32_f32_avx2, 1},
#endif
	{GGML_TYPE_BF16, GGML_TYPE_BF16, apply_residual_bf16_bf16_scalar, 0},
	{GGML_TYPE_F32, GGML_TYPE_BF16, apply_residual_f32_bf16_scalar, 0},
	{GGML_TYPE_F32, GGML_TYPE_F32, apply_residual_f32_f32_scalar, 0},
};

mat_vec_dispatch_t MAT_VEC_DISPATCH_TABLE[] = {
#ifdef CONFIG_ENABLE_AVX2
	{GGML_TYPE_BF16, GGML_TYPE_Q4_K, GGML_TYPE_BF16, mat_vec_row_bf16_q4k_bf16_avx2, 1},
	{GGML_TYPE_BF16, GGML_TYPE_Q6_K, GGML_TYPE_BF16, mat_vec_row_bf16_q6k_bf16_avx2, 1},
	{GGML_TYPE_BF16, GGML_TYPE_Q4_K, GGML_TYPE_F32, mat_vec_row_bf16_q4k_f32_avx2, 1},
	{GGML_TYPE_BF16, GGML_TYPE_Q6_K, GGML_TYPE_F32, mat_vec_row_bf16_q6k_f32_avx2, 1},
	{GGML_TYPE_F32, GGML_TYPE_Q4_K, GGML_TYPE_F32, mat_vec_row_f32_q4k_f32_avx2, 1},
	{GGML_TYPE_F32, GGML_TYPE_Q6_K, GGML_TYPE_F32, mat_vec_row_f32_q6k_f32_avx2, 1},
	{GGML_TYPE_F32, GGML_TYPE_BF16, GGML_TYPE_F32, mat_vec_row_f32_bf16_f32_avx2, 1},
	{GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32, mat_vec_row_f32_f32_f32_avx2, 1},
#endif
	{GGML_TYPE_BF16, GGML_TYPE_Q4_K, GGML_TYPE_BF16, mat_vec_row_bf16_q4k_bf16_scalar, 0},
	{GGML_TYPE_BF16, GGML_TYPE_Q6_K, GGML_TYPE_BF16, mat_vec_row_bf16_q6k_bf16_scalar, 0},
	{GGML_TYPE_BF16, GGML_TYPE_BF16, GGML_TYPE_BF16, mat_vec_row_bf16_bf16_bf16_scalar, 0},
	{GGML_TYPE_F32, GGML_TYPE_Q4_K, GGML_TYPE_BF16, mat_vec_row_f32_q4k_bf16_scalar, 0},
	{GGML_TYPE_F32, GGML_TYPE_Q6_K, GGML_TYPE_BF16, mat_vec_row_f32_q6k_bf16_scalar, 0},
	{GGML_TYPE_F32, GGML_TYPE_Q4_K, GGML_TYPE_F32, mat_vec_row_f32_q4k_f32_scalar, 0},
	{GGML_TYPE_F32, GGML_TYPE_Q6_K, GGML_TYPE_F32, mat_vec_row_f32_q6k_f32_scalar, 0},
	{GGML_TYPE_F32, GGML_TYPE_BF16, GGML_TYPE_F32, mat_vec_row_f32_bf16_f32_scalar, 0},

	{GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32, mat_vec_row_f32_f32_f32_scalar, 0},

	{GGML_TYPE_BF16, GGML_TYPE_Q6_K, GGML_TYPE_F32, mat_vec_row_bf16_q6k_f32_scalar, 0},
	{GGML_TYPE_BF16, GGML_TYPE_Q4_K, GGML_TYPE_F32, mat_vec_row_bf16_q4k_f32_scalar, 0},
	{GGML_TYPE_BF16, GGML_TYPE_BF16, GGML_TYPE_F32, mat_vec_row_bf16_bf16_f32_scalar, 0},
	{GGML_TYPE_BF16, GGML_TYPE_F32, GGML_TYPE_F32, mat_vec_row_bf16_f32_f32_scalar, 0},
	{GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F32, mat_vec_row_f32_f16_f32_scalar, 0},
};

swiglu_activation_dispatch_t SWIGLU_ACTIVATION_DISPATCH_TABLE[] = {
#ifdef CONFIG_ENABLE_AVX2
	{GGML_TYPE_BF16, GGML_TYPE_BF16, swiglu_activation_bf16_bf16_avx2, 1},
	{GGML_TYPE_F32, GGML_TYPE_F32, swiglu_activation_f32_f32_avx2, 1},
#endif
	{GGML_TYPE_BF16, GGML_TYPE_BF16, swiglu_activation_bf16_bf16_scalar, 0},
	{GGML_TYPE_F32, GGML_TYPE_F32, swiglu_activation_f32_f32_scalar, 0},
};

geglu_activation_dispatch_t GEGLU_ACTIVATION_DISPATCH_TABLE[] = {
#ifdef CONFIG_ENABLE_AVX2
	//	{GGML_TYPE_BF16, GGML_TYPE_BF16, swiglu_activation_bf16_bf16_avx2, 1},
	{GGML_TYPE_F32, GGML_TYPE_F32, geglu_activation_f32_f32_avx2, 1},
#endif
	//	{GGML_TYPE_BF16, GGML_TYPE_BF16, swiglu_activation_bf16_bf16_scalar, 0},
	{GGML_TYPE_F32, GGML_TYPE_F32, geglu_activation_f32_f32_scalar, 0},
};

convert_dispatch_t CONVERT_DISPATCH_TABLE[] = {
#ifdef CONFIG_ENABLE_AVX2
	{GGML_TYPE_BF16, GGML_TYPE_F32, convert_bf16_f32_avx2, 1},
	{GGML_TYPE_BF16, GGML_TYPE_BF16, convert_bf16_bf16_avx2, 1},
	{GGML_TYPE_F32, GGML_TYPE_BF16, convert_f32_bf16_avx2, 1},
	{GGML_TYPE_F32, GGML_TYPE_F32, convert_f32_f32_avx2, 1},
#endif
	{GGML_TYPE_BF16, GGML_TYPE_F32, convert_bf16_f32_scalar, 0},
	{GGML_TYPE_BF16, GGML_TYPE_BF16, convert_bf16_bf16_scalar, 0},
	{GGML_TYPE_F32, GGML_TYPE_BF16, convert_f32_bf16_scalar, 0},
	{GGML_TYPE_F32, GGML_TYPE_F32, convert_f32_f32_scalar, 0},
};

dot_product_dispatch_t DOT_PRODUCT_DISPATCH_TABLE[] = {
#ifdef CONFIG_ENABLE_AVX2
	{GGML_TYPE_F32, GGML_TYPE_BF16, dot_product_f32_bf16_avx2, 1},
	{GGML_TYPE_F32, GGML_TYPE_F32, dot_product_f32_f32_avx2, 1},
#endif
	{GGML_TYPE_F32, GGML_TYPE_BF16, dot_product_f32_bf16_scalar, 0},
	{GGML_TYPE_F32, GGML_TYPE_F32, dot_product_f32_f32_scalar, 0},
};

void dispatch_embedding_row(const Tensor *W, int row_index, MemType *O_slice, int embed_dim)
{
	for (int i = 0; i < ARRAY_SIZE(EMBEDDING_ROW_DISPATCH_TABLE); ++i) {
		embedding_row_dispatch_t *entry = &EMBEDDING_ROW_DISPATCH_TABLE[i];
		if (entry->input_type == W->mem.type && entry->output_type == O_slice->type) {
#ifdef DEBUG_ACCEL
			if (entry->accel == 0) {
				debug_accel("-- WARN: %s uses scalar function ---\n", __FUNCTION__);
			}
#endif
			entry->func(W, row_index, O_slice->data, embed_dim);
			return;
		}
	}

	fprintf(stderr, "FATAL: No EmbeddingRow implementation found for Tensor type %s and output type %s\n",
		gguf_get_type_name(W->mem.type), gguf_get_type_name(O_slice->type));

	exit(1);
}

void dispatch_rms_norm(const MemType *X_slice, const Tensor *W, MemType *O_slice, int size, float eps)
{
	for (int i = 0; i < ARRAY_SIZE(RMS_NORM_DISPATCH_TABLE); ++i) {
		rms_norm_dispatch_t *entry = &RMS_NORM_DISPATCH_TABLE[i];
		if (entry->input_type == X_slice->type && entry->tensor_type == W->mem.type
		    && entry->output_type == O_slice->type) {
#ifdef DEBUG_ACCEL
			if (entry->accel == 0) {
				debug_accel("-- WARN: %s uses scalar function ---\n", __FUNCTION__);
			}
#endif
			entry->func(O_slice->data, X_slice->data, W, size, eps);
			return;
		}
	}

	fprintf(stderr, "FATAL: No RMSNorm implementation found for input_type: %s, Tensor type: %s, output_type: %s\n",
		gguf_get_type_name(X_slice->type), gguf_get_type_name(W->mem.type), gguf_get_type_name(O_slice->type));
	exit(1);
}

static void mat_vec_task(void *arg)
{
	mat_vec_task_t *task = (mat_vec_task_t *)arg;
	const Tensor *tensor = task->W;

	task->mat_vec(task->X, tensor->mem.data, task->O, task->in_dim, task->start_row, task->end_row);

	free(task); // The worker is responsible for freeing its arguments.
}

void dispatch_mat_vec(struct TIEContext *ctx, const MemType *X, const Tensor *W, MemType *O, int in_dim, int out_dim,
		      int use_threads)
{
	int effective_in_dim = in_dim;
	int effective_out_dim = out_dim;

	// For transposed (Column-Major) weights, the logical output dimension (number of rows)
	// is the same as the logical input dimension of the *next* layer.
	if (ctx->model->weight_layout == LAYOUT_COL_MAJOR) {
		effective_in_dim = out_dim;
		effective_out_dim = in_dim;
	}

	int num_threads = thread_pool->num_threads;
	int rows_per_thread = (effective_out_dim + num_threads - 1) / num_threads;

	for (int i = 0; i < ARRAY_SIZE(MAT_VEC_DISPATCH_TABLE); ++i) {
		mat_vec_dispatch_t *entry = &MAT_VEC_DISPATCH_TABLE[i];

		if (entry->input_type == X->type && entry->output_type == O->type
		    && entry->tensor_type == W->mem.type) {
#ifdef DEBUG_ACCEL
			if (entry->accel == 0) {
				debug_accel("-- WARN: %s uses scalar function in: %s, weight: %s, out: %s ---\n",
					    __FUNCTION__, gguf_get_type_name(X->type), gguf_get_type_name(W->mem.type),
					    gguf_get_type_name(O->type));
			}
#endif

			if (use_threads == 0) {
				entry->mat_vec(X->data, W->mem.data, O->data, effective_in_dim, 0, effective_out_dim);
				return;
			}

			for (int t = 0; t < num_threads; t++) {
				int start_row = t * rows_per_thread;
				int end_row = start_row + rows_per_thread;

				if (start_row >= effective_out_dim)
					break;
				if (end_row > effective_out_dim)
					end_row = effective_out_dim;

				mat_vec_task_t *task = malloc(sizeof(mat_vec_task_t));
				if (!task) {
					printf("%s OOM\n", __FUNCTION__);
					exit(1);
				}

				*task = (mat_vec_task_t){.X = X->data,
							 .W = W,
							 .O = O->data,
							 .in_dim = effective_in_dim,
							 .start_row = start_row,
							 .end_row = end_row,
							 .mat_vec = entry->mat_vec};
				thread_pool_submit(thread_pool, mat_vec_task, task);
			}
			thread_pool_wait(thread_pool);
			return;
		}
	}

	fprintf(stderr,
		"FATAL: No MatVec implementation found for input type: %s, tensor type: %s and output type %s\n",
		gguf_get_type_name(X->type), gguf_get_type_name(W->mem.type), gguf_get_type_name(O->type));

	exit(1);
}

static void mat_mat_task(void *arg)
{
	mat_mat_task_t *task = (mat_mat_task_t *)arg;
	struct TIEContext *ctx = task->ctx;

	size_t x_element_size = ggml_type_size(task->X_type);
	size_t o_element_size = ggml_type_size(task->O_type);
	size_t x_row_stride_bytes = (size_t)task->in_dim * x_element_size;
	size_t o_row_stride_bytes = (size_t)task->out_dim * o_element_size;

	for (int i = task->start_row; i < task->end_row; ++i) {
		MemType x_row_slice = {.type = task->X_type, .data = (uint8_t *)task->X + i * x_row_stride_bytes};
		MemType o_row_slice = {.type = task->O_type, .data = (uint8_t *)task->O + i * o_row_stride_bytes};

		// Set use_threads to false since this is already inside a thread.
		dispatch_mat_vec(ctx, &x_row_slice, task->W, &o_row_slice, task->in_dim, task->out_dim, 0);
	}
	free(task);
}

void dispatch_mat_mat(struct TIEContext *ctx, const MemType *X, const Tensor *W, MemType *O, int batch_len, int in_dim,
		      int out_dim, int use_threads)
{
	if (batch_len == 1) {
		return dispatch_mat_vec(ctx, X, W, O, in_dim, out_dim, use_threads);
	}

	int num_threads = thread_pool->num_threads;
	int rows_per_thread = (batch_len + num_threads - 1) / num_threads;

	for (int t = 0; t < num_threads; t++) {
		int start_row = t * rows_per_thread;
		int end_row = start_row + rows_per_thread;
		if (end_row > batch_len)
			end_row = batch_len;
		if (start_row >= end_row)
			break;

		mat_mat_task_t *task = malloc(sizeof(mat_mat_task_t));
		*task = (mat_mat_task_t){.ctx = ctx,
					 .X = X->data,
					 .W = W,
					 .O = O->data,
					 .in_dim = in_dim,
					 .out_dim = out_dim,
					 .start_row = start_row,
					 .end_row = end_row,
					 .X_type = X->type,
					 .O_type = O->type};
		thread_pool_submit(thread_pool, mat_mat_task, task);
	}
	thread_pool_wait(thread_pool);
}


void dispatch_apply_rope_cache(RopeCacheType *rope_cache, MemType *X_slice, int pos, int head_dim)
{
	for (int i = 0; i < ARRAY_SIZE(APPLY_ROPE_CACHE_DISPATCH_TABLE); ++i) {
		apply_rope_cache_dispatch_t *entry = &APPLY_ROPE_CACHE_DISPATCH_TABLE[i];
		if (entry->input_type == X_slice->type) {
#ifdef DEBUG_ACCEL
			if (entry->accel == 0) {
				debug_accel("-- WARN: %s uses scalar function ---\n", __FUNCTION__);
			}
#endif
			entry->func(rope_cache, X_slice->data, pos, head_dim);
			return;
		}
	}

	fprintf(stderr, "FATAL: No ApplyRope implementation found for input type %s\n",
		gguf_get_type_name(X_slice->type));
	exit(1);
}

void dispatch_apply_mrope_cache(RopeCacheType *rope_cache, MemType *X_slice, int pos, int head_dim)
{
	for (int i = 0; i < ARRAY_SIZE(APPLY_MROPE_CACHE_DISPATCH_TABLE); ++i) {
		apply_mrope_cache_dispatch_t *entry = &APPLY_MROPE_CACHE_DISPATCH_TABLE[i];
		if (entry->input_type == X_slice->type) {
#ifdef DEBUG_ACCEL
			if (entry->accel == 0) {
				debug_accel("-- WARN: %s uses scalar function ---\n", __FUNCTION__);
			}
#endif
			entry->func(rope_cache, X_slice->data, pos, head_dim);
			return;
		}
	}

	fprintf(stderr, "FATAL: No ApplyM-Rope implementation found for input type %s\n",
		gguf_get_type_name(X_slice->type));
	exit(1);
}

void dispatch_accumulate_weighted_V(const MemType *V_slice, MemType *O_slice, float weight, int size)
{
	for (int i = 0; i < ARRAY_SIZE(ACCUMULATE_WEIGHTED_V_DISPATCH_TABLE); ++i) {
		accumulate_weighted_V_dispatch_t *entry = &ACCUMULATE_WEIGHTED_V_DISPATCH_TABLE[i];
		if (entry->value_type == V_slice->type && entry->output_type == O_slice->type) {
#ifdef DEBUG_ACCEL
			if (entry->accel == 0) {
				debug_accel("-- WARN: %s uses scalar function ---\n", __FUNCTION__);
			}
#endif
			entry->func(O_slice->data, weight, V_slice->data, size);
			return;
		}
	}

	fprintf(stderr, "FATAL: No AccumulateWeighted_V implementation found for value type %s and output type %s\n",
		gguf_get_type_name(V_slice->type), gguf_get_type_name(O_slice->type));
	exit(1);
}

void dispatch_store_KV_cache(struct TIEContext *ctx, int layer_idx, int start_pos, int batch_len)
{
	if (ctx->mem.K.type != ctx->mem.V.type) {
		fprintf(stderr, "FATAL: StoreKVCache K and V memories type mismatch\n");
		exit(1);
	}

	for (int i = 0; i < ARRAY_SIZE(STORE_KV_CACHE_DISPATCH_TABLE); ++i) {
		store_KV_cache_dispatch_t *entry = &STORE_KV_CACHE_DISPATCH_TABLE[i];
		if (entry->input_type == ctx->mem.K.type && entry->output_type == ctx->kv_cache[layer_idx].k.type) {
#ifdef DEBUG_ACCEL
			if (entry->accel == 0) {
				debug_accel("-- WARN: %s uses scalar function ---\n", __FUNCTION__);
			}
#endif
			entry->func(ctx, layer_idx, start_pos, batch_len);
			return;
		}
	}

	fprintf(stderr, "FATAL: No StoreKVCache implementation found for input type %s and output type %s\n",
		gguf_get_type_name(ctx->mem.K.type), gguf_get_type_name(ctx->kv_cache[layer_idx].k.type));

	exit(1);
}

void dispatch_apply_residual(MemType *acc, const MemType *residual, int size)
{
	for (int i = 0; i < ARRAY_SIZE(APPLY_RESIDUAL_DISPATCH_TABLE); ++i) {
		apply_residual_dispatch_t *entry = &APPLY_RESIDUAL_DISPATCH_TABLE[i];
		if (entry->input_type == residual->type && entry->output_type == acc->type) {
#ifdef DEBUG_ACCEL
			if (entry->accel == 0) {
				debug_accel("-- WARN: %s uses scalar function ---\n", __FUNCTION__);
			}
#endif
			entry->func(acc->data, residual->data, size);
			return;
		}
	}

	fprintf(stderr, "FATAL: No ApplyResidual implementation found for input_type: %s and output_type: %s\n",
		gguf_get_type_name(residual->type), gguf_get_type_name(acc->type));

	exit(1);
}

void dispatch_swiglu_activation(MemType *gate, MemType *up, int size)
{
	for (int i = 0; i < ARRAY_SIZE(SWIGLU_ACTIVATION_DISPATCH_TABLE); ++i) {
		swiglu_activation_dispatch_t *entry = &SWIGLU_ACTIVATION_DISPATCH_TABLE[i];
		if (entry->gate_type == gate->type && entry->up_type == up->type) {
#ifdef DEBUG_ACCEL
			if (entry->accel == 0) {
				debug_accel("-- WARN: %s uses scalar function ---\n", __FUNCTION__);
			}
#endif
			entry->func(gate->data, up->data, size);
			return;
		}
	}

	fprintf(stderr, "FATAL: No SwiGLU implementation found for gate type: %s and up type: %s\n",
		gguf_get_type_name(gate->type), gguf_get_type_name(up->type));
	exit(1);
}

void dispatch_geglu_activation(MemType *gate, MemType *up, int size)
{
	for (int i = 0; i < ARRAY_SIZE(GEGLU_ACTIVATION_DISPATCH_TABLE); ++i) {
		geglu_activation_dispatch_t *entry = &GEGLU_ACTIVATION_DISPATCH_TABLE[i];
		if (entry->gate_type == gate->type && entry->up_type == up->type) {
#ifdef DEBUG_ACCEL
			if (entry->accel == 0) {
				debug_accel("-- WARN: %s uses scalar function ---\n", __FUNCTION__);
			}
#endif
			entry->func(gate->data, up->data, size);
			return;
		}
	}

	fprintf(stderr, "FATAL: No GeGLU implementation found for gate type: %s and up type: %s\n",
		gguf_get_type_name(gate->type), gguf_get_type_name(up->type));
	exit(1);
}

void dispatch_convert(const MemType *src, MemType *dest, int size)
{
	for (int i = 0; i < ARRAY_SIZE(CONVERT_DISPATCH_TABLE); ++i) {
		convert_dispatch_t *entry = &CONVERT_DISPATCH_TABLE[i];
		if (entry->input_type == src->type && entry->output_type == dest->type) {
#ifdef DEBUG_ACCEL
			if (entry->accel == 0) {
				debug_accel("-- WARN: %s uses scalar function ---\n", __FUNCTION__);
			}
#endif
			entry->func(src->data, dest->data, size);
			return;
		}
	}

	fprintf(stderr, "FATAL: No Convert implementation found for input type %s and output type %s\n",
		gguf_get_type_name(src->type), gguf_get_type_name(dest->type));
	exit(1);
}

float dispatch_dot_product(const MemType *vec_a, const MemType *vec_b, int size)
{
	for (int i = 0; i < ARRAY_SIZE(DOT_PRODUCT_DISPATCH_TABLE); ++i) {
		dot_product_dispatch_t *entry = &DOT_PRODUCT_DISPATCH_TABLE[i];
		if (entry->type_a == vec_a->type && entry->type_b == vec_b->type) {
#ifdef DEBUG_ACCEL
			if (entry->accel == 0) {
				debug_accel("-- WARN: %s uses scalar function ---\n", __FUNCTION__);
			}
#endif
			return entry->func(vec_a->data, vec_b->data, size);
		}
	}

	// Handle error
	fprintf(stderr, "FATAL: No dot_product implementation found for types %s and %s\n",
		gguf_get_type_name(vec_a->type), gguf_get_type_name(vec_b->type));
	exit(1);
}
