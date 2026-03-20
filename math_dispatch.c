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

#define DISPATCH_MAX_GGML_TYPES 64

static mat_vec_fn CACHE_MAT_VEC[DISPATCH_MAX_GGML_TYPES][DISPATCH_MAX_GGML_TYPES][DISPATCH_MAX_GGML_TYPES];


static void mat_vec_task(void *arg);
static void mat_mat_task(void *arg);

// Helper for parallel execution
typedef void (*parallel_task_fn)(void *arg, int start, int end);


/*  The global dispatch tables, listing all available implementations */
embedding_row_dispatch_t EMBEDDING_ROW_DISPATCH_TABLE[] = {
#ifdef CONFIG_ENABLE_AVX2
	{GGML_TYPE_Q6_K, GGML_TYPE_F32, get_embedding_row_q6k_f32_avx2, 1},
	{GGML_TYPE_Q4_K, GGML_TYPE_F32, get_embedding_row_q4k_f32_avx2, 1},
	{GGML_TYPE_BF16, GGML_TYPE_F32, get_embedding_row_bf16_f32_avx2, 1},
#endif
	{GGML_TYPE_Q6_K, GGML_TYPE_F32, get_embedding_row_q6k_f32_scalar, 0},
	{GGML_TYPE_Q4_K, GGML_TYPE_F32, get_embedding_row_q4k_f32_scalar, 0},
	{GGML_TYPE_Q5_K, GGML_TYPE_F32, get_embedding_row_q5k_f32_scalar, 0},
	{GGML_TYPE_BF16, GGML_TYPE_F32, get_embedding_row_bf16_f32_scalar, 0},
	{GGML_TYPE_F32, GGML_TYPE_F32, get_embedding_row_f32_f32_scalar, 0},

	//	{GGML_TYPE_Q5_K, GGML_TYPE_F32, get_embedding_row_q5k_f32_scalar, 0},
	//	{GGML_TYPE_Q8_0, GGML_TYPE_F32, get_embedding_row_q8_0_f32_scalar, 0},
};

rms_norm_dispatch_t RMS_NORM_DISPATCH_TABLE[] = {
#ifdef CONFIG_ENABLE_AVX2
	{GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32, rms_norm_f32_f32_f32_avx2, 1},
#endif
	{GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32, rms_norm_f32_f32_f32_scalar, 0},
};

rms_norm_dispatch_t RMS_NORM_CENTERED_DISPATCH_TABLE[] = {
	{GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32, rms_norm_centered_f32_f32_f32_scalar, 0},
};

layer_norm_dispatch_t LAYER_NORM_DISPATCH_TABLE[] = {
#ifdef CONFIG_ENABLE_AVX2
	{GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32, layer_norm_f32_f32_f32_f32_avx2, 1},
#endif
	{GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32, layer_norm_f32_f32_f32_f32_scalar, 0},
};

apply_rope_cache_dispatch_t APPLY_ROPE_CACHE_DISPATCH_TABLE[] = {
#ifdef CONFIG_ENABLE_AVX2
	{GGML_TYPE_F32, apply_rope_cache_f32_avx2, 1},
#endif
	{GGML_TYPE_F32, apply_rope_cache_f32_scalar, 0},
};

apply_rope_cache_dispatch_t APPLY_ROPE_CACHE_INTERLEAVED_DISPATCH_TABLE[] = {
#ifdef CONFIG_ENABLE_AVX2
	{GGML_TYPE_F32, apply_rope_cache_interleaved_f32_avx2, 1},
#endif
	{GGML_TYPE_F32, apply_rope_cache_interleaved_f32_scalar, 0},
};

apply_mrope_cache_dispatch_t APPLY_MROPE_CACHE_DISPATCH_TABLE[] = {
	{GGML_TYPE_F32, apply_mrope_cache_f32_scalar, 0},
};

accumulate_weighted_V_dispatch_t ACCUMULATE_WEIGHTED_V_DISPATCH_TABLE[] = {
#ifdef CONFIG_ENABLE_AVX2
	{GGML_TYPE_F32, GGML_TYPE_BF16, accumulate_weighted_V_f32_bf16_avx2, 1},
#endif
	{GGML_TYPE_F32, GGML_TYPE_BF16, accumulate_weighted_V_f32_bf16_scalar, 0},
	{GGML_TYPE_F32, GGML_TYPE_F32, accumulate_weighted_V_f32_f32_scalar, 0},
};

store_KV_cache_dispatch_t STORE_KV_CACHE_DISPATCH_TABLE[] = {
#ifdef CONFIG_ENABLE_AVX2
	{GGML_TYPE_F32, GGML_TYPE_BF16, store_KV_cache_f32_bf16_avx2, 1},
#endif
	{GGML_TYPE_F32, GGML_TYPE_BF16, store_KV_cache_f32_bf16_scalar, 0},
	{GGML_TYPE_F32, GGML_TYPE_F32, store_KV_cache_f32_f32_scalar, 0},
};

apply_residual_dispatch_t APPLY_RESIDUAL_DISPATCH_TABLE[] = {
#ifdef CONFIG_ENABLE_AVX2
	{GGML_TYPE_F32, GGML_TYPE_F32, apply_residual_f32_f32_avx2, 1},
#endif
	{GGML_TYPE_F32, GGML_TYPE_F32, apply_residual_f32_f32_scalar, 0},
};

apply_residual_scaled_dispatch_t APPLY_RESIDUAL_SCALED_DISPATCH_TABLE[] = {
	{GGML_TYPE_F32, GGML_TYPE_F32, apply_residual_scaled_f32_f32_scalar, 0},
};

mat_vec_dispatch_t MAT_VEC_DISPATCH_TABLE[] = {
#ifdef CONFIG_ENABLE_AVX2
	{GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32, mat_vec_row_f32_f32_f32_avx2, 1},
	{GGML_TYPE_F32, GGML_TYPE_Q4_K, GGML_TYPE_F32, mat_vec_row_f32_q4k_f32_avx2, 1},
	{GGML_TYPE_F32, GGML_TYPE_Q5_K, GGML_TYPE_F32, mat_vec_row_f32_q5k_f32_avx2, 1},
	{GGML_TYPE_F32, GGML_TYPE_Q6_K, GGML_TYPE_F32, mat_vec_row_f32_q6k_f32_avx2, 1},
	{GGML_TYPE_F32, GGML_TYPE_BF16, GGML_TYPE_F32, mat_vec_row_f32_bf16_f32_avx2, 1},
	{GGML_TYPE_BF16, GGML_TYPE_Q4_K, GGML_TYPE_F32, mat_vec_row_bf16_q4k_f32_avx2, 1},
	{GGML_TYPE_BF16, GGML_TYPE_Q6_K, GGML_TYPE_F32, mat_vec_row_bf16_q6k_f32_avx2, 1},
#endif
	{GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32, mat_vec_row_f32_f32_f32_scalar, 0},
	{GGML_TYPE_F32, GGML_TYPE_Q4_K, GGML_TYPE_F32, mat_vec_row_f32_q4k_f32_scalar, 0},
	{GGML_TYPE_F32, GGML_TYPE_Q5_K, GGML_TYPE_F32, mat_vec_row_f32_q5k_f32_scalar, 0},
	{GGML_TYPE_F32, GGML_TYPE_Q6_K, GGML_TYPE_F32, mat_vec_row_f32_q6k_f32_scalar, 0},
	{GGML_TYPE_F32, GGML_TYPE_BF16, GGML_TYPE_F32, mat_vec_row_f32_bf16_f32_scalar, 0},
	{GGML_TYPE_BF16, GGML_TYPE_Q6_K, GGML_TYPE_F32, mat_vec_row_bf16_q6k_f32_scalar, 0},
	{GGML_TYPE_BF16, GGML_TYPE_Q4_K, GGML_TYPE_F32, mat_vec_row_bf16_q4k_f32_scalar, 0},
	{GGML_TYPE_BF16, GGML_TYPE_BF16, GGML_TYPE_F32, mat_vec_row_bf16_bf16_f32_scalar, 0},
	{GGML_TYPE_BF16, GGML_TYPE_F32, GGML_TYPE_F32, mat_vec_row_bf16_f32_f32_scalar, 0},
	{GGML_TYPE_F32, GGML_TYPE_Q4_K, GGML_TYPE_BF16, mat_vec_row_f32_q4k_bf16_scalar, 0},
	{GGML_TYPE_F32, GGML_TYPE_Q4_K, GGML_TYPE_BF16, mat_vec_row_f32_q4k_bf16_scalar, 0},

	{GGML_TYPE_F32, GGML_TYPE_Q8_0, GGML_TYPE_F32, mat_vec_row_f32_q8_0_f32_scalar, 0},

	/* column wise for Gemma3 vision */
	{GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F32, mat_vec_row_f32_f16_f32_scalar, 0},
};

swiglu_activation_dispatch_t SWIGLU_ACTIVATION_DISPATCH_TABLE[] = {
#ifdef CONFIG_ENABLE_AVX2
	{GGML_TYPE_BF16, GGML_TYPE_BF16, swiglu_activation_bf16_bf16_avx2, 1},
	{GGML_TYPE_F32, GGML_TYPE_F32, swiglu_activation_f32_f32_avx2, 1},
#endif
	{GGML_TYPE_F32, GGML_TYPE_F32, swiglu_activation_f32_f32_scalar, 0},
	{GGML_TYPE_BF16, GGML_TYPE_BF16, swiglu_activation_bf16_bf16_scalar, 0},
};

geglu_activation_dispatch_t GEGLU_ACTIVATION_DISPATCH_TABLE[] = {
#ifdef CONFIG_ENABLE_AVX2
	{GGML_TYPE_F32, GGML_TYPE_F32, geglu_activation_f32_f32_avx2, 1},
#endif
	{GGML_TYPE_F32, GGML_TYPE_F32, geglu_activation_f32_f32_scalar, 0},
};

convert_dispatch_t CONVERT_DISPATCH_TABLE[] = {
#ifdef CONFIG_ENABLE_AVX2
	{GGML_TYPE_BF16, GGML_TYPE_F32, convert_bf16_f32_avx2, 1},
	{GGML_TYPE_F32, GGML_TYPE_F32, convert_f32_f32_avx2, 1},

#endif
	{GGML_TYPE_BF16, GGML_TYPE_F32, convert_bf16_f32_scalar, 0},
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

conv_2d_dispatch_t CONV2D_DISPATCH_TABLE[] = {
#ifdef CONFIG_ENABLE_AVX2
	{GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32, conv_2d_f32_f32_f32_f32_avx2, 1},
#endif
	{GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32, conv_2d_f32_f32_f32_f32_scalar, 0},
};

transpose_dispatch_t TRANSPOSE_DISPATCH_TABLE[] = {
#ifdef CONFIG_ENABLE_AVX2
	{GGML_TYPE_F32, transpose_f32_avx2, 1},
#endif
	{GGML_TYPE_F32, transpose_f32_scalar, 0},
	{GGML_TYPE_BF16, transpose_bf16_scalar, 0},
};

l2_norm_dispatch_t L2_NORM_DISPATCH_TABLE[] = {
	{GGML_TYPE_F32, l2_norm_f32_scalar, 0},
};

/* -------------------------------------------------------------------------
 * Parallel Helpers
 * ------------------------------------------------------------------------- */
typedef struct {
	parallel_task_fn func;
	void *arg;
	int start;
	int end;
} parallel_worker_ctx_t;

static void parallel_worker_shim(void *arg)
{
	parallel_worker_ctx_t *ctx = (parallel_worker_ctx_t *)arg;
	ctx->func(ctx->arg, ctx->start, ctx->end);
}

void dispatch_parallel(parallel_task_fn func, void *arg, int total, int min_grain)
{
	if (total <= 0)
		return;

	if (thread_pool->num_threads <= 1 || total < min_grain) {
		func(arg, 0, total);
		return;
	}

	int num_threads = thread_pool->num_threads;
	int chunk_size = (total + num_threads - 1) / num_threads;

	parallel_worker_ctx_t tasks[num_threads];

	for (int t = 0; t < num_threads; t++) {
		int start = t * chunk_size;
		int end = start + chunk_size;
		if (start >= total)
			break;
		if (end > total)
			end = total;

		tasks[t].func = func;
		tasks[t].arg = arg;
		tasks[t].start = start;
		tasks[t].end = end;

		thread_pool_submit(thread_pool, parallel_worker_shim, &tasks[t]);
	}

	thread_pool_wait(thread_pool);
}

void dispatch_l2_norm(MemType *X_slice, int size, float eps)
{
	for (int i = 0; i < ARRAY_SIZE(L2_NORM_DISPATCH_TABLE); ++i) {
		l2_norm_dispatch_t *entry = &L2_NORM_DISPATCH_TABLE[i];
		if (entry->input_type == X_slice->type) {
			entry->func(X_slice->data, size, eps);
			return;
		}
	}
	fprintf(stderr, "FATAL: No L2Norm implementation found for input type %s\n", gguf_get_type_name(X_slice->type));
	exit(1);
}

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

/* -------------------------------------------------------------------------
 * RMS norm
 * ------------------------------------------------------------------------- */
typedef struct {
	rms_norm_fn func;
	MemType *O;
	const MemType *X;
	const Tensor *W;
	int size;
	float eps;
	int stride;
} rms_norm_batched_ctx_t;

static void rms_norm_batched_worker(void *arg, int start, int end)
{
	rms_norm_batched_ctx_t *ctx = (rms_norm_batched_ctx_t *)arg;
	size_t x_element_size = ggml_type_size(ctx->X->type);
	size_t o_element_size = ggml_type_size(ctx->O->type);

	// Calculate strides in bytes
	size_t x_stride_bytes = (size_t)ctx->stride * x_element_size;
	size_t o_stride_bytes = (size_t)ctx->stride * o_element_size;

	for (int i = start; i < end; i++) {
		void *x_ptr = (uint8_t *)ctx->X->data + (size_t)i * x_stride_bytes;
		void *o_ptr = (uint8_t *)ctx->O->data + (size_t)i * o_stride_bytes;

		ctx->func(o_ptr, x_ptr, ctx->W, ctx->size, ctx->eps);
	}
}

void dispatch_rms_norm_batched(const MemType *X, const Tensor *W, MemType *O, int size, float eps, int batch_len,
			       int stride)
{
	for (int i = 0; i < ARRAY_SIZE(RMS_NORM_DISPATCH_TABLE); ++i) {
		rms_norm_dispatch_t *entry = &RMS_NORM_DISPATCH_TABLE[i];
		if (entry->input_type == X->type && entry->tensor_type == W->mem.type
		    && entry->output_type == O->type) {
#ifdef DEBUG_ACCEL
			if (entry->accel == 0) {
				debug_accel("-- WARN: %s uses scalar function ---\n", __FUNCTION__);
			}
#endif
			rms_norm_batched_ctx_t args = {.func = entry->func,
						       .O = O,
						       .X = X,
						       .W = W,
						       .size = size,
						       .eps = eps,
						       .stride = stride};

			// Parallelize over batch
			dispatch_parallel(rms_norm_batched_worker, &args, batch_len, 1);
			return;
		}
	}

	fprintf(stderr, "FATAL: No RMSNorm implementation found for input_type: %s, Tensor type: %s, output_type: %s\n",
		gguf_get_type_name(X->type), gguf_get_type_name(W->mem.type), gguf_get_type_name(O->type));
	exit(1);
}

void dispatch_rms_norm_centered_batched(const MemType *X, const Tensor *W, MemType *O, int size, float eps,
					int batch_len, int stride)
{
	for (int i = 0; i < ARRAY_SIZE(RMS_NORM_CENTERED_DISPATCH_TABLE); ++i) {
		rms_norm_dispatch_t *entry = &RMS_NORM_CENTERED_DISPATCH_TABLE[i];
		if (entry->input_type == X->type && entry->tensor_type == W->mem.type
		    && entry->output_type == O->type) {
#ifdef DEBUG_ACCEL
			if (entry->accel == 0) {
				debug_accel("-- WARN: %s uses scalar function ---\n", __FUNCTION__);
			}
#endif
			rms_norm_batched_ctx_t args = {.func = entry->func,
						       .O = O,
						       .X = X,
						       .W = W,
						       .size = size,
						       .eps = eps,
						       .stride = stride};

			// Parallelize over batch
			dispatch_parallel(rms_norm_batched_worker, &args, batch_len, 1);
			return;
		}
	}

	fprintf(stderr,
		"FATAL: No Centered RMSNorm implementation found for input_type: %s, Tensor type: %s, output_type: %s\n",
		gguf_get_type_name(X->type), gguf_get_type_name(W->mem.type), gguf_get_type_name(O->type));
	exit(1);
}

void dispatch_rms_norm(const MemType *X_slice, const Tensor *W, MemType *O_slice, int size, float eps)
{
	// Forward to batched implementation with batch_len=1
	dispatch_rms_norm_batched(X_slice, W, O_slice, size, eps, 1, 0);
}

void dispatch_rms_norm_centered(const MemType *X_slice, const Tensor *W, MemType *O_slice, int size, float eps)
{
	// Forward to batched implementation with batch_len=1
	dispatch_rms_norm_centered_batched(X_slice, W, O_slice, size, eps, 1, 0);
}

static mat_vec_fn get_mat_vec_kernel(GGMLType X_type, GGMLType W_type, GGMLType O_type)
{
	mat_vec_fn func = CACHE_MAT_VEC[X_type][W_type][O_type];

	if (!func) {
		fprintf(stderr, "FATAL: No MatVec impl for input_type: %s x tensor_type: %s -> output_type: %s\n",
			gguf_get_type_name(X_type), gguf_get_type_name(W_type), gguf_get_type_name(O_type));
		exit(1);
	}
	return func;
}

static void mat_vec_task(void *arg)
{
	mat_vec_task_t *task = (mat_vec_task_t *)arg;
	const Tensor *tensor = task->W;

	task->mat_vec(task->X, tensor->mem.data, task->O, task->in_dim, task->start_row, task->end_row);
}

void dispatch_mat_vec(struct TIEContext *ctx, const MemType *X, const Tensor *W, MemType *O, int in_dim, int out_dim,
		      int use_threads)
{
	// O(1) Lookup
	mat_vec_fn func = get_mat_vec_kernel(X->type, W->mem.type, O->type);

	int effective_in_dim = in_dim;
	int effective_out_dim = out_dim;

#if 0
        if (ctx->model->weight_layout == LAYOUT_COL_MAJOR) {
	        effective_in_dim = out_dim;
	        effective_out_dim = in_dim;
        }
#endif

	if (!use_threads) {
		func(X->data, W->mem.data, O->data, effective_in_dim, 0, effective_out_dim);
		return;
	}

	// Threaded execution
	int num_threads = thread_pool->num_threads;
	int rows_per_thread = (effective_out_dim + num_threads - 1) / num_threads;

	mat_vec_task_t tasks[num_threads];

	for (int t = 0; t < num_threads; t++) {
		int start = t * rows_per_thread;
		int end = start + rows_per_thread;
		if (start >= effective_out_dim)
			break;
		if (end > effective_out_dim)
			end = effective_out_dim;

		tasks[t] = (mat_vec_task_t){.X = X->data,
					    .W = W,
					    .O = O->data,
					    .in_dim = effective_in_dim,
					    .start_row = start,
					    .end_row = end,
					    .mat_vec = func};
		thread_pool_submit(thread_pool, mat_vec_task, &tasks[t]);
	}

	thread_pool_wait(thread_pool);
}

static void mat_mat_task(void *arg)
{
	mat_mat_task_t *task = (mat_mat_task_t *)arg;

	size_t x_element_size = ggml_type_size(task->X_type);
	size_t o_element_size = ggml_type_size(task->O_type);
	size_t x_row_stride_bytes = (size_t)task->in_dim * x_element_size;
	size_t o_row_stride_bytes = (size_t)task->out_dim * o_element_size;

	for (int i = task->start_row; i < task->end_row; ++i) {
		void *x_row_data = (uint8_t *)task->X + i * x_row_stride_bytes;
		void *o_row_data = (uint8_t *)task->O + i * o_row_stride_bytes;

		// Direct kernel call
		task->mat_vec(x_row_data, task->W->mem.data, o_row_data, task->in_dim, 0, task->out_dim);
	}
}

void dispatch_mat_mat(struct TIEContext *ctx, const MemType *X, const Tensor *W, MemType *O, int batch_len, int in_dim,
		      int out_dim, int use_threads)
{
	if (batch_len == 1) {
		return dispatch_mat_vec(ctx, X, W, O, in_dim, out_dim, use_threads);
	}

	// Threading Heuristic: Don't thread small ops (like MoE gates or small layers)
	if (use_threads && ((long long)in_dim * out_dim < 32768)) {
		use_threads = 0;
	}

	mat_vec_fn func = get_mat_vec_kernel(X->type, W->mem.type, O->type);

	/* Support Serial Execution */
	if (!use_threads) {
		size_t x_el_size = ggml_type_size(X->type);
		size_t o_el_size = ggml_type_size(O->type);
		size_t x_stride = in_dim * x_el_size;
		size_t o_stride = out_dim * o_el_size;

		for (int i = 0; i < batch_len; ++i) {
			void *x_row = (uint8_t *)X->data + i * x_stride;
			void *o_row = (uint8_t *)O->data + i * o_stride;

			func(x_row, W->mem.data, o_row, in_dim, 0, out_dim);
		}
		return;
	}

	int num_threads = thread_pool->num_threads;
	int rows_per_thread = (batch_len + num_threads - 1) / num_threads;
	mat_mat_task_t tasks[num_threads];

	for (int t = 0; t < num_threads; t++) {
		int start_row = t * rows_per_thread;
		int end_row = start_row + rows_per_thread;
		if (end_row > batch_len)
			end_row = batch_len;
		if (start_row >= end_row)
			break;

		tasks[t] = (mat_mat_task_t){.ctx = ctx,
					    .X = X->data,
					    .W = W,
					    .O = O->data,
					    .in_dim = in_dim,
					    .out_dim = out_dim,
					    .start_row = start_row,
					    .end_row = end_row,
					    .X_type = X->type,
					    .O_type = O->type,
					    .mat_vec = func};
		thread_pool_submit(thread_pool, mat_mat_task, &tasks[t]);
	}

	thread_pool_wait(thread_pool);
}

/* -------------------------------------------------------------------------
 * RoPE Cache
 * ------------------------------------------------------------------------- */
typedef struct {
	apply_rope_cache_fn func;
	RopeCacheType *rope_cache;
	MemType *X;
	int start_pos;
	int head_dim;
	int stride; // Stride between tokens in ELEMENTS
} rope_batched_ctx_t;

static void rope_batched_worker(void *arg, int start, int end)
{
	rope_batched_ctx_t *ctx = (rope_batched_ctx_t *)arg;
	size_t element_size = ggml_type_size(ctx->X->type);
	size_t stride_bytes = (size_t)ctx->stride * element_size;

	for (int i = start; i < end; i++) {
		// Calculate position for this token
		int pos = ctx->start_pos + i;

		// Pointer math
		void *x_ptr = (uint8_t *)ctx->X->data + (size_t)i * stride_bytes;

		ctx->func(ctx->rope_cache, x_ptr, pos, ctx->head_dim);
	}
}

void dispatch_apply_rope_cache_batched(RopeCacheType *rope_cache, MemType *X, int start_pos, int head_dim,
				       int batch_len, int stride)
{
	for (int i = 0; i < ARRAY_SIZE(APPLY_ROPE_CACHE_DISPATCH_TABLE); ++i) {
		apply_rope_cache_dispatch_t *entry = &APPLY_ROPE_CACHE_DISPATCH_TABLE[i];
		if (entry->input_type == X->type) {
#ifdef DEBUG_ACCEL
			if (entry->accel == 0) {
				debug_accel("-- WARN: %s uses scalar function ---\n", __FUNCTION__);
			}
#endif
			rope_batched_ctx_t args = {.func = entry->func,
						   .rope_cache = rope_cache,
						   .X = X,
						   .start_pos = start_pos,
						   .head_dim = head_dim,
						   .stride = stride};

			dispatch_parallel(rope_batched_worker, &args, batch_len, 1);
			return;
		}
	}

	fprintf(stderr, "FATAL: No ApplyRope implementation found for input type %s\n", gguf_get_type_name(X->type));
	exit(1);
}

void dispatch_apply_rope_cache_interleaved_batched(RopeCacheType *rope_cache, MemType *X, int start_pos, int head_dim,
						   int batch_len, int stride)
{
	for (int i = 0; i < ARRAY_SIZE(APPLY_ROPE_CACHE_INTERLEAVED_DISPATCH_TABLE); ++i) {
		apply_rope_cache_dispatch_t *entry = &APPLY_ROPE_CACHE_INTERLEAVED_DISPATCH_TABLE[i];
		if (entry->input_type == X->type) {
#ifdef DEBUG_ACCEL
			if (entry->accel == 0) {
				debug_accel("-- WARN: %s uses scalar function ---\n", __FUNCTION__);
			}
#endif
			rope_batched_ctx_t args = {.func = entry->func,
						   .rope_cache = rope_cache,
						   .X = X,
						   .start_pos = start_pos,
						   .head_dim = head_dim,
						   .stride = stride};

			dispatch_parallel(rope_batched_worker, &args, batch_len, 1);
			return;
		}
	}

	fprintf(stderr, "FATAL: No ApplyRope implementation found for input type %s\n", gguf_get_type_name(X->type));
	exit(1);
}

void dispatch_apply_mrope_cache_batched(RopeCacheType *rope_cache, MemType *X, int start_pos, int head_dim,
					int batch_len, int stride)
{
	for (int i = 0; i < ARRAY_SIZE(APPLY_MROPE_CACHE_DISPATCH_TABLE); ++i) {
		apply_mrope_cache_dispatch_t *entry = &APPLY_MROPE_CACHE_DISPATCH_TABLE[i];
		if (entry->input_type == X->type) {
#ifdef DEBUG_ACCEL
			if (entry->accel == 0) {
				debug_accel("-- WARN: %s uses scalar function ---\n", __FUNCTION__);
			}
#endif
			rope_batched_ctx_t args = {.func = entry->func,
						   .rope_cache = rope_cache,
						   .X = X,
						   .start_pos = start_pos,
						   .head_dim = head_dim,
						   .stride = stride};

			dispatch_parallel(rope_batched_worker, &args, batch_len, 1);
			return;
		}
	}

	fprintf(stderr, "FATAL: No ApplyRope implementation found for input type %s\n", gguf_get_type_name(X->type));
	exit(1);
}

void dispatch_apply_rope_cache(RopeCacheType *rope_cache, MemType *X_slice, int pos, int head_dim)
{
	// Forward to batched with len=1
	dispatch_apply_rope_cache_batched(rope_cache, X_slice, pos, head_dim, 1, 0);
}

void dispatch_apply_rope_cache_interleaved(RopeCacheType *rope_cache, MemType *X_slice, int pos, int head_dim)
{
	// Forward to batched with len=1
	dispatch_apply_rope_cache_interleaved_batched(rope_cache, X_slice, pos, head_dim, 1, 0);
}

void dispatch_apply_mrope_cache(RopeCacheType *rope_cache, MemType *X_slice, int pos, int head_dim)
{
	// Forward to batched with len=1
	dispatch_apply_mrope_cache_batched(rope_cache, X_slice, pos, head_dim, 1, 0);
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

void dispatch_store_KV_cache(struct TIEContext *ctx, int layer_idx, int start_pos, int batch_len, int sink_len)
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
			entry->func(ctx, layer_idx, start_pos, batch_len, sink_len);
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

void dispatch_apply_residual_scaled(MemType *acc, const MemType *residual, int size, float scale)
{
	for (int i = 0; i < ARRAY_SIZE(APPLY_RESIDUAL_SCALED_DISPATCH_TABLE); ++i) {
		apply_residual_scaled_dispatch_t *entry = &APPLY_RESIDUAL_SCALED_DISPATCH_TABLE[i];
		if (entry->input_type == residual->type && entry->output_type == acc->type) {
#ifdef DEBUG_ACCEL
			if (entry->accel == 0) {
				debug_accel("-- WARN: %s uses scalar function ---\n", __FUNCTION__);
			}
#endif
			entry->func(acc->data, residual->data, size, scale);
			return;
		}
	}

	fprintf(stderr, "FATAL: No ApplyResidualScaled implementation found for input_type: %s and output_type: %s\n",
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

void dispatch_layer_norm(MemType *dest, const MemType *src, const Tensor *weight, const Tensor *bias, int size,
			 float eps)
{
	for (int i = 0; i < ARRAY_SIZE(LAYER_NORM_DISPATCH_TABLE); ++i) {
		layer_norm_dispatch_t *entry = &LAYER_NORM_DISPATCH_TABLE[i];
		if (entry->input_type == src->type && entry->output_type == dest->type
		    && entry->tensor_type == weight->mem.type && entry->bias_type == bias->mem.type) {
#ifdef DEBUG_ACCEL
			if (entry->accel == 0) {
				debug_accel("-- WARN: %s uses scalar function ---\n", __FUNCTION__);
			}
#endif
			entry->func(dest, src, weight, bias, size, eps);
			return;
		}
	}

	fprintf(stderr,
		"FATAL: No LayerNorm implementation found for input_type: %s, output_type: %s, tensor_type: %s, bias_type: %s\n",
		gguf_get_type_name(src->type), gguf_get_type_name(dest->type), gguf_get_type_name(weight->mem.type),
		gguf_get_type_name(bias->mem.type));
	exit(1);
}

void dispatch_transpose(MemType *dest, const MemType *src, int rows, int cols)
{
	for (int i = 0; i < ARRAY_SIZE(TRANSPOSE_DISPATCH_TABLE); ++i) {
		transpose_dispatch_t *entry = &TRANSPOSE_DISPATCH_TABLE[i];
		if (entry->type == src->type && entry->type == dest->type) {
#ifdef DEBUG_ACCEL
			if (entry->accel == 0) {
				debug_accel("-- WARN: %s uses scalar function ---\n", __FUNCTION__);
			}
#endif
			entry->func(dest, src, rows, cols);
			return;
		}
	}

	fprintf(stderr, "FATAL: No Transpose implementation found for type: %s\n", gguf_get_type_name(src->type));
	exit(1);
}

void dispatch_conv_2d(MemType *dest, const MemType *src_image, const Tensor *kernel_tensor, const Tensor *bias_tensor,
		      int H_in, int W_in, int stride, int padding)
{
	for (int i = 0; i < ARRAY_SIZE(CONV2D_DISPATCH_TABLE); ++i) {
		conv_2d_dispatch_t *entry = &CONV2D_DISPATCH_TABLE[i];
		if (entry->input_type == src_image->type && entry->kernel_type == kernel_tensor->mem.type
		    && entry->output_type == dest->type) {
#ifdef DEBUG_ACCEL
			if (entry->accel == 0) {
				debug_accel("-- WARN: %s uses scalar function ---\n", __FUNCTION__);
			}
#endif
			return entry->func(dest, src_image, kernel_tensor, bias_tensor, H_in, W_in, stride, padding);
		}
	}

	// Handle error
	fprintf(stderr, "FATAL: No conv_2d implementation found for input_types %s and tensor_type: %s\n",
		gguf_get_type_name(src_image->type), gguf_get_type_name(kernel_tensor->mem.type));
	exit(1);
}

void math_dispatch_init()
{
	printf("Initializing Math Dispatch Cache...\n");

	// Populate MAT_VEC Cache
	for (int i = 0; i < ARRAY_SIZE(MAT_VEC_DISPATCH_TABLE); ++i) {
		mat_vec_dispatch_t *e = &MAT_VEC_DISPATCH_TABLE[i];
		if (e->input_type < DISPATCH_MAX_GGML_TYPES && e->tensor_type < DISPATCH_MAX_GGML_TYPES
		    && e->output_type < DISPATCH_MAX_GGML_TYPES) {
			if (CACHE_MAT_VEC[e->input_type][e->tensor_type][e->output_type] == NULL) {
				CACHE_MAT_VEC[e->input_type][e->tensor_type][e->output_type] = e->mat_vec;
			}
		}
	}
}
